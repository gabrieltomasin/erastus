from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import TranscriptionInfo, Segment
from pathlib import Path
from config import WHISPER_MODEL, USE_CUDA, LANGUAGE

class Transcriber:
    def __init__(self, model_size: str = WHISPER_MODEL, use_cuda: bool = USE_CUDA):
        self.batched_model = self.instantiate_batched_model(
            model_size=model_size,
            device="cuda" if use_cuda else "cpu",
            compute_type="int8_float16" if use_cuda else "int8"
        )

    def instantiate_batched_model(self, model_size: str, device: str, compute_type: str) -> BatchedInferencePipeline:
        """
        Instantiate a BatchedInferencePipeline with the specified model size, device, and compute type.

        Args:
            model_size (str): The size of the Whisper model to use (e.g., "large-v3-turbo").
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
            compute_type (str): The compute type for the model (e.g., "int8_float16" or "int8").

        Returns:
            BatchedInferencePipeline: An instance of BatchedInferencePipeline with the specified configuration.
        """
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        batched_model = BatchedInferencePipeline(model=model)
        return batched_model

    def transcribe_single_audio(self, audio_path: Path, batch_size: int = 16, language: str = LANGUAGE) -> tuple[list[Segment], TranscriptionInfo]:
        """
        Transcribe a single audio file.

        Args:
            audio_path (Path): The path to the audio file to transcribe.
            batch_size (int): The batch size for transcription.

        Returns:
            tuple[list[Segment], TranscriptionInfo]: A tuple containing the list of segments and transcription info.
        """
        segments, info = self.batched_model.transcribe(str(audio_path), batch_size=batch_size, language=language)
        return segments, info