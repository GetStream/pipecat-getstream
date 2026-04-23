import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

from pipecat_getstream.transport import (
    GetstreamCallbacks,
    GetstreamParams,
    GetstreamTransportClient,
)


@pytest.fixture()
def create_callbacks():
    """Factory that creates GetstreamCallbacks with all-AsyncMock handlers."""

    def _factory() -> "GetstreamCallbacks":
        return GetstreamCallbacks(
            on_connected=AsyncMock(),
            on_disconnected=AsyncMock(),
            on_before_disconnect=AsyncMock(),
            on_participant_joined=AsyncMock(),
            on_participant_left=AsyncMock(),
            on_audio_track_subscribed=AsyncMock(),
            on_audio_track_unsubscribed=AsyncMock(),
            on_video_track_subscribed=AsyncMock(),
            on_video_track_unsubscribed=AsyncMock(),
            on_custom_event=AsyncMock(),
            on_first_participant_joined=AsyncMock(),
            on_call_ended=AsyncMock(),
        )

    return _factory


@pytest.fixture()
def create_client(create_callbacks):
    """Factory that creates a GetstreamTransportClient with mocked internals."""

    async def _factory(
        video_in_enabled: bool = False,
        audio_in_enabled: bool = True,
    ) -> "GetstreamTransportClient":
        params = GetstreamParams(
            video_in_enabled=video_in_enabled,
            audio_in_enabled=audio_in_enabled,
        )
        callbacks = create_callbacks()
        client = GetstreamTransportClient(
            api_key="test-key",
            api_secret="test-secret",
            call_type="default",
            call_id="test-call",
            user_id="bot-user",
            params=params,
            callbacks=callbacks,
            transport_name="test-transport",
        )
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        client._task_manager = task_manager
        return client

    return _factory


@pytest.fixture()
def make_participant():
    """Factory that creates a mock Participant protobuf."""

    def _factory(user_id: str, session_id: str = "session-1"):
        p = MagicMock()
        p.user_id = user_id
        p.session_id = session_id
        return p

    return _factory


@pytest.fixture()
def make_participant_event(make_participant):
    """Factory that creates a mock ParticipantJoined/Left protobuf event."""

    def _factory(user_id: str, session_id: str = "session-1"):
        event = MagicMock()
        event.participant = make_participant(user_id, session_id)
        return event

    return _factory


@pytest.fixture()
def make_track_published_event(make_participant):
    """Factory that creates a mock TrackPublished protobuf event."""

    def _factory(user_id: str, session_id: str, track_type: int):
        event = MagicMock()
        event.user_id = user_id
        event.session_id = session_id
        event.type = track_type
        event.participant = make_participant(user_id, session_id)
        return event

    return _factory


@pytest.fixture()
def make_track_unpublished_event(make_participant):
    """Factory that creates a mock TrackUnpublished protobuf event."""

    def _factory(user_id: str, session_id: str, track_type: int):
        event = MagicMock()
        event.user_id = user_id
        event.session_id = session_id
        event.type = track_type
        event.cause = 0
        event.participant = make_participant(user_id, session_id)
        return event

    return _factory


@pytest.fixture()
def make_pcm_data(make_participant):
    """Factory that creates a mock PcmData with .participant attribute."""

    def _factory(user_id: str, session_id: str = "session-1"):
        pcm = MagicMock()
        pcm.participant = make_participant(user_id, session_id)
        return pcm

    return _factory
