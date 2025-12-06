from faster_whisper.transcribe import TranscriptionInfo, Segment

class TranscriptionProcessor:
    @staticmethod
    def process_transcriptions(transcriptions: list[tuple[list[Segment], TranscriptionInfo]]) -> str:
        """
        Process multiple transcriptions and return a single formatted transcript string.

        Args:
            transcriptions: List of tuples (segments list, transcription info) per audio track.

        Returns:
            A combined, time-ordered transcript string where each segment is
            formatted with timestamps and a speaker label.
        """
        # Reorders all segments by start time
        all_segments: list[Segment] = []
        for i, (segments, transcription_info) in enumerate(transcriptions):
            speaker = f'speaker_{i+1}'
            for segment in segments:
                segment.text = TranscriptionProcessor.format_segment(segment, speaker)
                all_segments.append(segment)
        all_segments.sort(key=lambda seg: seg.start)

        # Concatenate all segments into a single transcript string
        full_transcript = "\n".join([seg.text for seg in all_segments])
        return full_transcript

    @staticmethod
    def format_segment(segment: Segment, speaker: str = 'unknown_speaker') -> str:
        """Create a readable line for a single transcript segment.

        Format: [start - end] speaker: text
        """
        return f"[{segment.start:.2f} - {segment.end:.2f}] {speaker}: {segment.text.strip()}"