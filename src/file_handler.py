import zipfile
import shutil
from pathlib import Path
import logging

from config import SUPPORTED_AUDIO_FORMATS

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, upload_dir, audio_dir):
        self.upload_dir = Path(upload_dir)
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_file(self, file_bytes, filename):
        """Save an uploaded file to the upload directory.

        Args:
            file_bytes: Raw file bytes to write.
            filename: Target file name.

        Returns: Path to the saved file.
        """
        file_path = self.upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        return file_path
    
    def is_supported_audio_file(self, file_path: Path) -> bool:
        """Check if a file is a supported audio format.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if the file has a supported audio extension, False otherwise.
        """
        return file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    
    def get_single_audio_file(self, audio_path: str | Path) -> list[Path]:
        """Handle a single audio file input.
        
        Copies the audio file to the audio directory for processing.
        
        Args:
            audio_path: Path to the single audio file.
            
        Returns:
            A list containing the path to the copied audio file.
            
        Raises:
            ValueError: If the file format is not supported.
        """
        audio_path = Path(audio_path)
        
        if not self.is_supported_audio_file(audio_path):
            supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {supported}"
            )
        
        # Copy to audio directory for consistent processing
        dest_path = self.audio_dir / audio_path.name
        shutil.copy2(audio_path, dest_path)
        logger.info(f"Copied audio file: {audio_path.name}")
        
        return [dest_path]
    
    def extract_audio_files(self, zip_path, extensions: tuple[str, ...] | None = None):
        """Extract audio files from a ZIP archive.

        This will scan the provided ZIP and extract files with supported audio
        extensions into the configured audio directory.
        
        Args:
            zip_path: Path to the ZIP archive.
            extensions: Tuple of file extensions to extract. If None, uses
                SUPPORTED_AUDIO_FORMATS from config.

        Returns:
            A list of extracted paths (Path objects).
        """
        if extensions is None:
            extensions = SUPPORTED_AUDIO_FORMATS
            
        audio_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files inside the zip
            file_list = zip_ref.namelist()
            
            # Filter only audio files with supported extensions
            audio_files = [
                f for f in file_list 
                if any(f.lower().endswith(ext) for ext in extensions)
            ]
            
            # Extract the audio files
            for audio_file in audio_files:
                zip_ref.extract(audio_file, self.audio_dir)
                logger.info(f"Extracted: {audio_file}")
        
        return [self.audio_dir / f for f in audio_files]
    
    def cleanup(self):
        """Remove temporary audio files and recreate audio directory."""
        if self.audio_dir.exists():
            shutil.rmtree(self.audio_dir)
        self.audio_dir.mkdir(exist_ok=True)