# Changelog

# v0.4.1

## Bug Fixes

- Use the configured output video size for the placeholder frame (#24)

# v0.4.0

## Breaking Changes

- Minimum `getstream` version is now 6.1.1 (#23)

## Bug Fixes

- Resample input audio with a PyAV resampler per participant, so audio and pts of different participants do not mix and buffering drops to about 1 ms (#21)

# v0.3.0

## Breaking Changes

- Minimum Python version is now 3.11 (#19)
- Minimum `getstream` version is now 4.2.0 (#19)

## New Features

- Pass the received audio pts to `UserAudioRawFrame` as nanoseconds (#19)

# v0.2.0

## New Features

- Connect with a pre-minted scoped user token instead of the API secret (#6)

# v0.1.0

## New Features

- Initial release of the GetStream WebRTC transport for Pipecat
- Send and receive custom events between call participants (#1)
