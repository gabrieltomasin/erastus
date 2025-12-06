import zipfile
import shutil
from pathlib import Path
import logging

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
    
    def extract_audio_files(self, zip_path, target_ext=".flac"):
        """Extract only audio files from a ZIP archive.

        This will scan the provided ZIP and extract files ending with the
        `target_ext` into the configured audio directory.

        Returns a list of extracted paths (Path objects).
        """
        audio_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files inside the zip
            file_list = zip_ref.namelist()
            
            # Filter only audio files
            audio_files = [f for f in file_list if f.lower().endswith(target_ext)]
            
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