class AudioCoreError(Exception):
    """Base exception for audio-core failures."""


class InvalidAudioInput(AudioCoreError):
    """Input is missing, unsafe, or is not a regular file."""


class AudioToolUnavailable(AudioCoreError):
    """Required external executable is unavailable."""


class AudioProbeError(AudioCoreError):
    """ffprobe could not decode or describe the input."""


class AudioNormalizeError(AudioCoreError):
    """ffmpeg normalization failed."""


class AudioAnalysisError(AudioCoreError):
    """Audio quality analysis failed."""


class AudioVadError(AudioCoreError):
    """Voice activity detection failed."""


class AudioVadDependencyError(AudioVadError):
    """The selected VAD backend is not installed."""


class AudioSegmentExportError(AudioCoreError):
    """Audio segment extraction failed."""


class InvalidStateTransition(AudioCoreError):
    """A job state transition is not allowed."""
