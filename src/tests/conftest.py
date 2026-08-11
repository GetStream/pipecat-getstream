import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from getstream.video.rtc import PcmData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams
from pipecat.workers.runner import WorkerRunner

from pipecat_getstream.transport import (
    GetstreamCallbacks,
    GetstreamInputTransport,
    GetstreamParams,
    GetstreamTransport,
    GetstreamTransportClient,
)


class _OfflineClient(GetstreamTransportClient):
    """Client that skips the SFU join, so the input transport runs without a call."""

    async def connect(self):
        pass


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
    """Factory that creates a 20ms silent 48kHz PcmData chunk from a participant."""

    def _factory(
        user_id: str,
        session_id: str = "session-1",
        pts: int | None = None,
    ) -> PcmData:
        return PcmData(
            sample_rate=48000,
            format="s16",
            samples=np.zeros(960, dtype=np.int16),
            pts=pts,
            time_base=1 / 48000,
            participant=make_participant(user_id, session_id),
        )

    return _factory


@pytest.fixture()
async def run_pipeline():
    """Factory that runs processors in a PipelineWorker for the duration of the test."""
    running: list[tuple[PipelineWorker, asyncio.Task]] = []

    async def _factory(*processors: FrameProcessor) -> PipelineWorker:
        worker = PipelineWorker(
            Pipeline(list(processors)),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
        started = asyncio.Event()

        @worker.event_handler("on_pipeline_started")
        async def on_pipeline_started(_worker, _frame):
            started.set()

        runner = WorkerRunner(handle_sigint=False)
        run_task = asyncio.create_task(runner.run(worker))
        running.append((worker, run_task))
        await asyncio.wait_for(started.wait(), timeout=30)
        return worker

    yield _factory

    for worker, run_task in running:
        await worker.cancel()
        await run_task


@pytest.fixture()
async def input_transport(create_callbacks, run_pipeline):
    """A started GetstreamInputTransport's client and a downstream frame queue.

    The client skips the SFU join, so audio can be fed in with `client._on_audio()`
    and read back from the queue as the transport pushes it downstream.
    """
    params = GetstreamParams(audio_in_enabled=True, audio_in_sample_rate=48000)
    transport = GetstreamTransport(
        api_key="test-key",
        token="test-token",
        call_type="default",
        call_id="test-call",
        user_id="bot-user",
        params=params,
    )
    client = _OfflineClient(
        api_key="test-key",
        token="test-token",
        call_type="default",
        call_id="test-call",
        user_id="bot-user",
        params=params,
        callbacks=create_callbacks(),
        transport_name="test-transport",
    )
    input_t = GetstreamInputTransport(transport, client, params)

    received: asyncio.Queue = asyncio.Queue()
    sink = QueuedFrameProcessor(
        queue=received, queue_direction=FrameDirection.DOWNSTREAM
    )
    await run_pipeline(input_t, sink)

    return client, received
