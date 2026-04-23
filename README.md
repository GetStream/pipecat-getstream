# pipecat-getstream

[![PyPI](https://img.shields.io/pypi/v/pipecat-getstream.svg)](https://pypi.org/project/pipecat-getstream/)

A [GetStream.io](https://getstream.io/) WebRTC transport for [Pipecat](https://github.com/pipecat-ai/pipecat) — the open-source framework for building conversational AI agents.

This plugin enables bidirectional audio and video streaming between Pipecat pipelines and GetStream video calls, allowing you to build voice and multimodal AI agents that interact with users through GetStream's real-time communication infrastructure.

> **Maintained by [GetStream](https://github.com/GetStream).**

## Features

- **Bidirectional audio and video** — send and receive audio/video between your Pipecat agent and call participants
- **Participant lifecycle events** — react to participants joining, leaving, and subscribing/unsubscribing tracks
- **Interruption handling** — immediate audio buffer flushing when a user interrupts the agent
- **REST helper** — manage users, calls, and authentication tokens via the GetStream API
- **Custom events** — send and receive structured JSON events between the agent and call participants (up to 5KB per event)
- **Clock-based audio pacing** — drift-free real-time audio output

## Installation

```bash
pip install pipecat-getstream
```

## Quickstart

```python
import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat_getstream import GetstreamTransport
from pipecat_getstream.transport import GetstreamParams
from pipecat_getstream.utils import GetstreamRESTHelper


async def main():
    transport = GetstreamTransport(
        api_key="your-api-key",
        api_secret="your-api-secret",
        call_type="default",
        call_id="my-call-id",
        user_id="pipecat-bot",
        params=GetstreamParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,  # your STT service
        user_aggregator,  # LLM context aggregator
        llm,  # your LLM service
        tts,  # your TTS service
        transport.output(),
        assistant_aggregator,
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))
    await runner.run(task)


asyncio.run(main())
```

## Configuration

### GetstreamTransport

| Parameter    | Type              | Description                     |
|--------------|-------------------|---------------------------------|
| `api_key`    | `str`             | GetStream API key               |
| `api_secret` | `str`             | GetStream API secret            |
| `call_type`  | `str`             | Call type (e.g. `"default"`)    |
| `call_id`    | `str`             | Unique call identifier          |
| `user_id`    | `str`             | Bot/agent user ID               |
| `params`     | `GetstreamParams` | Transport parameters (optional) |

### GetstreamParams

Extends Pipecat's `TransportParams`. Commonly used options:

| Parameter               | Type   | Default | Description                     |
|-------------------------|--------|---------|---------------------------------|
| `audio_in_enabled`      | `bool` | `True`  | Receive audio from participants |
| `audio_out_enabled`     | `bool` | `True`  | Send agent audio                |
| `audio_in_sample_rate`  | `int`  | `16000` | Input audio sample rate         |
| `audio_out_sample_rate` | `int`  | `16000` | Output audio sample rate        |
| `video_in_enabled`      | `bool` | `False` | Receive video from participants |
| `video_out_enabled`     | `bool` | `False` | Send agent video                |
| `video_out_framerate`   | `int`  | `30`    | Output video frame rate         |
| `video_out_is_live`     | `bool` | `False` | Live video mode                 |

## REST Helper

`GetstreamRESTHelper` provides convenience methods for managing calls and users via the GetStream API.

```python
from pipecat_getstream.utils import GetstreamRESTHelper

helper = GetstreamRESTHelper(api_key="your-api-key", api_secret="your-api-secret")

# Create or update a user
await helper.create_user(user_id="demo-user", name="Demo User")

# Create or get a call
await helper.create_call(call_type="default", call_id="my-call", created_by_id="demo-user")

# Generate a JWT token for a user
token = helper.create_token(user_id="demo-user", expiration=3600)

# Delete a call
await helper.delete_call(call_type="default", call_id="my-call")
```

## Custom Events

Send structured events to everyone watching the call:

```python
await transport.send_custom_event({"type": "agent_state", "state": "thinking"})
```

Receive events from participants by registering `on_stream_custom_event`:

```python
@transport.event_handler("on_stream_custom_event")
async def on_stream_custom_event(transport, event):
    print(f"Got custom event: {event}")
```

Payloads are limited to 5KB. Events are delivered only to clients that are currently watching the call.

## Event Handlers

Register event handlers on the transport to respond to call lifecycle events:

```python
@transport.event_handler("on_connected")
async def on_connected(*args):
    print("Bot connected to the call")


@transport.event_handler("on_first_participant_joined")
async def on_first_participant_joined(transport, participant_id):
    print(f"First participant joined: {participant_id}")


@transport.event_handler("on_stream_custom_event")
async def on_first_participant_joined(transport, payload):
    print(f"Received custom event: {payload}")
```

### Available Events

| Event                         | Arguments                | Description                                          |
|-------------------------------|--------------------------|------------------------------------------------------|
| `on_connected`                | —                        | Bot has connected to the call                        |
| `on_disconnected`             | —                        | Bot has disconnected from the call                   |
| `on_before_disconnect`        | —                        | Called before the bot disconnects                    |
| `on_participant_connected`    | `participant_id`         | A participant has joined the call                    |
| `on_participant_disconnected` | `participant_id`         | A participant has left the call                      |
| `on_participant_left`         | `participant_id`         | A participant has left the call                      |
| `on_first_participant_joined` | `participant_id`         | The first participant has joined the call            |
| `on_audio_track_subscribed`   | `participant_id`         | Audio from a participant is now being received       |
| `on_audio_track_unsubscribed` | `participant_id`         | Audio from a participant is no longer being received |
| `on_video_track_subscribed`   | `participant_id`         | Video from a participant is now being received       |
| `on_video_track_unsubscribed` | `participant_id`         | Video from a participant is no longer being received |
| `on_data_received`            | `data`, `participant_id` | Custom data/event received from a participant        |
| `on_stream_custom_event`      | `event`                  | Custom event from a client watching the call         |

## Running the Example

The included `example.py` demonstrates a complete voice agent using Deepgram STT/TTS and Google LLM.

### 1. Set environment variables

```bash
export STREAM_BASE_URL="https://your-stream-app-url"
export STREAM_API_KEY="your-api-key"
export STREAM_API_SECRET="your-api-secret"
export DEEPGRAM_API_KEY="your-deepgram-key"
export GOOGLE_API_KEY="your-google-key"

# Optional
export STREAM_CALL_TYPE="default"
export STREAM_CALL_ID="my-call-id"
export STREAM_USER_ID="pipecat-bot"
```

### 2. Run

```bash
uv run example.py
```

The bot will join the call and automatically open a browser window so you can join as a participant.

## Compatibility

- **Python:** 3.10>=, <3.14
- **Pipecat:** `>= 0.0.108`
- **GetStream SDK:** `>= 3.3.0, < 4`

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
