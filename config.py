import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
UPLOAD_DIR = TEMP_DIR / "uploads"
AUDIO_DIR = TEMP_DIR / "audios"
TRANSCRIPT_DIR = TEMP_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "outputs"

# Whisper config
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")

# USE_CUDA supports these values in env: 'true'|'false'|'auto' (default 'true').
# When set to auto, we attempt to detect GPU availability at runtime.
_USE_CUDA_RAW = os.getenv("USE_CUDA", "true").strip().lower()


#TODO include torch in requirements.txt
def _detect_cuda_via_runtime() -> bool:
	"""Try to detect whether CUDA is available.

	This is a best-effort check and will return False when detection tools
	are not available (e.g., torch not installed). We avoid expensive system
	calls for portability.
	"""
	# Prefer PyTorch detection if available
	try:
		import torch

		return torch.cuda.is_available()
	except Exception:
		pass

	# Fallback: check CUDA_VISIBLE_DEVICES env var
	if os.getenv("CUDA_VISIBLE_DEVICES"):
		val = os.getenv("CUDA_VISIBLE_DEVICES").strip()
		if val and val != "-1":
			return True

	# No clear signal for GPU
	return False


def _parse_use_cuda(raw: str | None) -> bool:
	if raw is None:
		raw = "auto"
	raw = raw.strip().lower()
	if raw in ("true", "1", "yes", "y"):
		return True
	if raw in ("false", "0", "no", "n"):
		return False
	# auto or unknown -> attempt detection
	return _detect_cuda_via_runtime()


USE_CUDA = _parse_use_cuda(_USE_CUDA_RAW)

LANGUAGE = os.getenv("LANGUAGE", None)

# Public alias for runtime CUDA detection so other modules can reuse it.
def detect_cuda() -> bool:
	return _detect_cuda_via_runtime()

# DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"