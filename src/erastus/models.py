from dataclasses import dataclass
from faster_whisper.transcribe import Segment
from typing import List


@dataclass
class SessionTranscription:
    """Container for a session's transcription data."""
    session_id: str
    participants: List[str]
    text_segments: List[Segment]