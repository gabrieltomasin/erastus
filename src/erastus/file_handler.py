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
        """Save uploaded file to the upload directory."""
        file_path = self.upload_dir / filename
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        return file_path
    
    def is_supported_audio_file(self, file_path: Path) -> bool:
        """Check if file has a supported audio extension."""
        return file_path.suffix.lower() in SUPPORTED_AUDIO_FORMATS
    
    def get_single_audio_file(self, audio_path: str | Path) -> list[Path]:
        """Copy a single audio file to the processing directory."""
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if not self.is_supported_audio_file(audio_path):
            supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
            raise ValueError(f"Unsupported format: {audio_path.suffix}. Try: {supported}")
        
        # Copy to audio dir, handle name collisions
        dest_path = self.audio_dir / audio_path.name
        if dest_path.exists():
            stem, suffix = audio_path.stem, audio_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = self.audio_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(audio_path, dest_path)
        logger.info(f"Copied: {audio_path.name}")
        
        return [dest_path]
    
    def extract_audio_files(self, zip_path, extensions=None):
        """Extract audio files from a ZIP archive."""
        if extensions is None:
            extensions = SUPPORTED_AUDIO_FORMATS
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            audio_files = [
                f for f in zf.namelist() 
                if any(f.lower().endswith(ext) for ext in extensions)
            ]
            
            for audio_file in audio_files:
                zf.extract(audio_file, self.audio_dir)
                logger.info(f"Extracted: {audio_file}")
        
        return [self.audio_dir / f for f in audio_files]
    
    def cleanup(self):
        """Remove temporary audio files and recreate audio directory."""
        if self.audio_dir.exists():
            shutil.rmtree(self.audio_dir)
        self.audio_dir.mkdir(exist_ok=True)