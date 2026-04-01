"""Tests for Getstream transport implementation.

Two focused tests:
1. Mock-based full participant lifecycle (join -> audio -> video -> leave)
2. Real integration: GetstreamTransportClient connects, sends audio+video,
   a raw SDK participant verifies reception and sends media back.
"""

import asyncio
import os
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from dotenv import load_dotenv
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_getstream.transport import (
    GetstreamParams,
    GetstreamTransportClient,
)

load_dotenv()


STREAM_API_KEY = os.environ.get("STREAM_API_KEY")
STREAM_API_SECRET = os.environ.get("STREAM_API_SECRET")
GETSTREAM_INTEGRATION_AVAILABLE = bool(STREAM_API_KEY and STREAM_API_SECRET)


class TestGetstreamParticipantLifecycle:
    """Mock-based test covering the full event lifecycle:
    join -> audio -> track add/publish -> track unpublish -> leave.
    """

    async def test_full_participant_session(
        self,
        create_client,
        make_participant,
        make_participant_event,
        make_track_published_event,
        make_track_unpublished_event,
        make_pcm_data,
    ):
        """Simulate a complete participant session from join to leave."""
        client = await create_client(video_in_enabled=True)
        user = make_participant("user-A", "session-1")

        # 1. Participant joins
        join_event = make_participant_event("user-A", "session-1")
        client._on_participant_joined(join_event)
        assert "user-A" in client._participants

        await client._async_on_participant_joined("user-A")
        client._callbacks.on_first_participant_joined.assert_called_once_with("user-A")

        # 2. Audio starts flowing
        mock_pcm = make_pcm_data("user-A")
        client._on_audio(mock_pcm)
        assert client._audio_queue.qsize() == 1
        assert "user-A" in client._audio_subscribed_participants

        # 3. Video track is added + published (two-phase resolution)
        client._on_track_added("video-track-1", "video", user)
        assert "video-track-1" in client._pending_tracks

        pub_event = make_track_published_event(
            "user-A", "session-1", TrackType.TRACK_TYPE_VIDEO
        )
        client._on_track_published(pub_event)
        assert "video-track-1" not in client._pending_tracks
        assert ("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO) in client._track_map

        # 4. Track is unpublished
        mock_task = MagicMock()
        client._video_subscriber_tasks["user-A:video-track-1"] = mock_task
        client._video_subscribed_participants.add("user-A")

        unpub_event = make_track_unpublished_event(
            "user-A", "session-1", TrackType.TRACK_TYPE_VIDEO
        )
        client._on_track_unpublished(unpub_event)
        mock_task.cancel.assert_called_once()
        assert "user-A:video-track-1" not in client._video_subscriber_tasks
        assert "user-A" not in client._video_subscribed_participants

        # 5. Participant leaves
        left_event = make_participant_event("user-A", "session-1")
        client._on_participant_left(left_event)
        assert "user-A" not in client._participants
        assert not client._other_participant_has_joined


@pytest.mark.skipif(
    not GETSTREAM_INTEGRATION_AVAILABLE,
    reason="Requires STREAM_API_KEY and STREAM_API_SECRET env vars and getstream[webrtc]",
)
@pytest.mark.integration
class TestGetstreamBidirectionalMedia:
    """Real integration test using GetstreamTransportClient.

    The bot connects via the actual transport client (connect/disconnect),
    publishes audio+video, and a raw SDK participant verifies reception.
    The raw participant also sends media back to verify the transport receives it.
    """

    async def test_simultaneous_audio_and_video_bidirectional(self, create_callbacks):
        """GetstreamTransportClient exchanges audio+video with a real participant."""
        from getstream import AsyncStream
        from getstream.models import UserRequest
        from getstream.video import rtc
        from getstream.video.rtc import AudioStreamTrack, PcmData
        from getstream.video.rtc.tracks import (
            SubscriptionConfig,
            TrackSubscriptionConfig,
        )

        call_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        bot_user_id = f"bot-{uuid.uuid4().hex[:6]}"
        human_user_id = f"human-{uuid.uuid4().hex[:6]}"

        bot_params = GetstreamParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,
            video_out_enabled=True,
            video_out_framerate=15,
        )
        bot_callbacks = create_callbacks()
        bot_client = GetstreamTransportClient(
            api_key=STREAM_API_KEY,
            api_secret=STREAM_API_SECRET,
            call_type="default",
            call_id=call_id,
            user_id=bot_user_id,
            params=bot_params,
            callbacks=bot_callbacks,
            transport_name="integration-test-bot",
        )

        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        bot_client._task_manager = task_manager

        bot_client._client = AsyncStream(
            api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET
        )
        await bot_client._client.upsert_users(UserRequest(id=bot_user_id, name="Bot"))

        await bot_client._client.upsert_users(
            UserRequest(id=human_user_id, name="Human"),
        )

        mock_start_frame = MagicMock()
        mock_start_frame.audio_out_sample_rate = 24000
        await bot_client.start(mock_start_frame)

        await bot_client.connect()

        try:
            assert bot_client._connected, "Bot should be connected"
            assert bot_client._audio_track is not None, "Bot should have an audio track"
            assert bot_client._video_track is not None, "Bot should have a video track"

            api_client = AsyncStream(
                api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET
            )
            call = api_client.video.call("default", call_id)

            sub_config = SubscriptionConfig(
                default=TrackSubscriptionConfig(track_types=[1, 2])
            )
            cm_human = await rtc.join(
                call,
                user_id=human_user_id,
                create=False,
                subscription_config=sub_config,
            )

            human_received_audio = []
            human_received_video_tracks = []

            @cm_human.on("audio")
            def on_human_audio(pcm_data):
                human_received_audio.append(pcm_data)

            @cm_human.on("track_added")
            def on_human_track_added(track_id, kind, user):
                if kind == "video" and user and user.user_id != human_user_id:
                    human_received_video_tracks.append(track_id)

            async with cm_human:
                await asyncio.sleep(2)

                human_audio_track = AudioStreamTrack(
                    sample_rate=24000, channels=1, format="s16"
                )
                await cm_human.add_tracks(audio=human_audio_track)
                await cm_human.republish_tracks()

                for _ in range(100):
                    num_samples = 480
                    t = np.linspace(0, 0.020, num_samples, endpoint=False)
                    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
                    pcm = PcmData(
                        samples=samples,
                        sample_rate=24000,
                        channels=1,
                        format="s16",
                    )
                    await bot_client._audio_track.write(pcm)

                for i in range(15):
                    value = (i * 30 + 50) % 256
                    rgb = np.full((120, 160, 3), fill_value=value, dtype=np.uint8)
                    bot_client._video_track.write(rgb.tobytes(), (160, 120), "RGB")

                for _ in range(50):
                    num_samples = 480
                    t = np.linspace(0, 0.020, num_samples, endpoint=False)
                    samples = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)
                    pcm = PcmData(
                        samples=samples,
                        sample_rate=24000,
                        channels=1,
                        format="s16",
                    )
                    await human_audio_track.write(pcm)

                deadline = time.time() + 20
                while time.time() < deadline:
                    human_got_audio = len(human_received_audio) > 0
                    human_got_video = len(human_received_video_tracks) > 0
                    bot_got_audio = bot_client._audio_queue.qsize() > 0
                    if human_got_audio and human_got_video and bot_got_audio:
                        break
                    await asyncio.sleep(0.5)

                assert len(human_received_audio) > 0, (
                    "Human did not receive audio from the bot transport"
                )
                pcm_from_bot = human_received_audio[0]
                assert hasattr(pcm_from_bot, "samples")
                assert len(pcm_from_bot.samples) > 0

                assert len(human_received_video_tracks) > 0, (
                    "Human did not receive video track from the bot transport"
                )

                assert bot_client._audio_queue.qsize() > 0, (
                    "Bot transport did not receive audio from human"
                )

                assert human_user_id in bot_client._participants, (
                    "Bot transport did not register the human participant"
                )

        finally:
            await bot_client.disconnect()
            assert not bot_client._connected, "Bot should be disconnected"
