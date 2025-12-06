import logging
from pathlib import Path
import argparse

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    UPLOAD_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    OUTPUT_DIR,
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    WHISPER_MODEL,
    USE_CUDA,
    LANGUAGE,
    detect_cuda,
)
from src.file_handler import FileHandler
from src.transcriber import Transcriber
from src.transcription_processor import TranscriptionProcessor
from src.summarizer import DeepSeekSummarizer

from faster_whisper.transcribe import TranscriptionInfo, Segment

def main(
    zip_path: str,
    model_size: str | None = None,
    use_cuda: bool | None = None,
    language: str | None = None,
    batch_size: int = 16,
    output_dir: Path | None = None,
):
    """Main processing function.

    Parameters can override values from `config.py` (model, device, language, etc.).
    """
    
    # 1. Inicializa componentes
    file_handler = FileHandler(UPLOAD_DIR, AUDIO_DIR)
    effective_model = model_size or WHISPER_MODEL
    effective_use_cuda = USE_CUDA if use_cuda is None else use_cuda
    transcriber = Transcriber(model_size=effective_model, use_cuda=effective_use_cuda)
    
    # Verifica API key
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY not found!")
        return
    
    summarizer = DeepSeekSummarizer(DEEPSEEK_API_KEY, DEEPSEEK_API_URL)
    
    try:
        # 2. Extract audio files
        logger.info(f"Extracting audio files from: {zip_path}")
        audio_files = file_handler.extract_audio_files(zip_path)
        
        if not audio_files:
            logger.error("No .flac audio files found in the zip!")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # 3. Transcribe audio files
        logger.info("Starting transcription...")
        transcripts: list[tuple[list[Segment], TranscriptionInfo]] = []
        for file in audio_files:
            logger.info(f"Transcribing: {file}") 
            transcript = transcriber.transcribe_single_audio(file, batch_size=batch_size, language=language or LANGUAGE)
            transcripts.append(transcript)
        
        # 4. Process transcriptions
        logger.info("Processing transcriptions...")
        aggregated = TranscriptionProcessor.process_transcriptions(transcripts)
        
        # Save full transcription
        effective_output_dir = output_dir or OUTPUT_DIR
        output_file = effective_output_dir / f"transcription_full_{Path(zip_path).stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(aggregated)
        
        logger.info(f"Transcript saved to: {output_file}")
        
        # 5. Generate summary via DeepSeek
        logger.info("Generating summary with DeepSeek...")
        summary = summarizer.summarize(aggregated)
        
        # Save summary
        summary_file = effective_output_dir / f"session_summary_{Path(zip_path).stem}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Summary saved to: {summary_file}")
        print("\n" + "="*50)
        print("SESSION SUMMARY:")
        print("="*50)
        print(summary)
        
    except FileNotFoundError:
        logger.error(f"File not found: {zip_path}")
        print(f"Error: The file '{zip_path}' was not found!")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
    
    finally:
        # 6. Limpeza (opcional)
        # file_handler.cleanup()
        pass

if __name__ == "__main__":
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Receives a .zip containing .flac audio files from a TTRPG session and generates a detailed summary using DeepSeek API.'
    )
    parser.add_argument(
        'zip_file',
        type=str,
        help='Path to the ZIP file containing audio files (ex: craig.zip or full/path/craig.zip)'
    )
    parser.add_argument('--model', type=str, default=None, help='Whisper model to use (overrides config)')
    parser.add_argument('--use-cuda', type=str, choices=['auto', 'true', 'false'], default=None,
                        help="Whether to use CUDA: 'true'|'false'|'auto' (default uses value from config or auto-detect)")
    parser.add_argument('--language', type=str, default=None,
                        help='Force transcription language (empty or omitted => auto-detect)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size used during transcription')
    parser.add_argument('--output-dir', type=str, default=None, help='Override output directory for generated transcripts/summaries')
    
    args = parser.parse_args()
    
    # Compute effective configuration from CLI + environment defaults
    model_size = args.model or WHISPER_MODEL

    if args.use_cuda is None:
        use_cuda = USE_CUDA
    else:
        val = args.use_cuda.strip().lower()
        if val == 'true':
            use_cuda = True
        elif val == 'false':
            use_cuda = False
        else:
            use_cuda = detect_cuda()

    language = args.language if args.language is not None else LANGUAGE
    batch_size = args.batch_size or 16
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # Create required directories using effective output_dir
    for dir_path in [UPLOAD_DIR, AUDIO_DIR, TRANSCRIPT_DIR, output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure the input file exists
    zip_path = Path(args.zip_file)
    if not zip_path.exists():
        print(f"Error: The file '{args.zip_file}' does not exist!")
        print("Usage: python main.py path/to/file.zip")
        exit(1)
    
    # Call the main function with the effective configuration
    main(
        str(zip_path),
        model_size=model_size,
        use_cuda=use_cuda,
        language=language,
        batch_size=batch_size,
        output_dir=output_dir,
    )