# Erastus — Audio transcribe & summarize for TTRPG sessions

Erastus is a small toolkit for extracting, transcribing and summarizing recorded tabletop RPG (TTRPG) session audio (e.g. Discord/Craig multi-track exports). It stitches transcripts, generates speaker-labeled text and produces a short session summary using an LLM backend.

This repository contains utilities around faster-whisper for audio transcription and a small wrapper to call an external summarization API (DeepSeek). The project is intentionally minimal and meant to be published as a Python library later.

Key features
- Extract audio files from a ZIP (Craig-style exports)
- Transcribe audio using faster-whisper / Whisper models
- Process and merge multi-track transcriptions into a single time-ordered transcript
- Produce a session summary by calling a chat-based summarizer API

Table of contents
- [Erastus — Audio transcribe \& summarize for TTRPG sessions](#erastus--audio-transcribe--summarize-for-ttrpg-sessions)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [License](#license)

Requirements
-----------

This project targets Python 3.10+ and uses heavy ML dependencies (faster-whisper, onnx/ctranslate2). If you plan to run transcription locally, verify your environment supports the chosen model and device.

See `requirements.txt` for a pinned developer environment.

Installation
------------

For development, create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Usage
-----

The repository currently has a command-line entry point implemented at `main.py`. It expects a ZIP file containing `.flac` audio recordings (Craig Discord export). Example:

```bash
python main.py path/to/craig.zip
```

The CLI supports several runtime overrides. Common options:

- `--model`: override the Whisper model used (default taken from environment or `config.py`).
- `--use-cuda`: `true` / `false` / `auto` — if `auto` the runtime tries to detect a GPU.
- `--language`: force a language code for transcription (omit to auto-detect).
- `--batch-size`: batch size for faster-whisper processing (default 16).
- `--output-dir`: override where transcript and summary files are written.

Example with overrides:

```bash
python main.py path/to/craig.zip --model large-v3-turbo --use-cuda auto --batch-size 16 --output-dir outputs/
```

The script will:

1. Extract .flac tracks from the ZIP
2. Transcribe each audio track
3. Merge segments into a single time-ordered transcript
4. Send the full transcript to a summarizer API and save the result


Configuration
-------------

Copy `.env.example` to a `.env` file and add your API key for the summarizer:

```bash
cp .env.example .env
# then edit .env and set DEEPSEEK_API_KEY
```

Check `config.py` for directory locations and default model/device configuration.

Important env vars (overridable by CLI):

- `DEEPSEEK_API_KEY` (required) — your DeepSeek API key; keep it secret.
- `WHISPER_MODEL` — default whisper model, e.g. `large-v3-turbo`.
- `USE_CUDA` — `true` / `false` / `auto` (default) — `auto` attempts to detect a GPU at runtime.
- `LANGUAGE` — default language code (empty = auto-detect).


License
-------

This project is released under the MIT License — see `LICENSE`.