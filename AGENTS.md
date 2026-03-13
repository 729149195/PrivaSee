# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

PrivaSee is a privacy risk analysis system with two main services:

| Service | Port | How to run |
|---------|------|------------|
| Frontend (Vite) | 5173 | `cd frontend && npm run dev` |
| Backend (Flask) | 5000 | `cd backend && python3 app.py --no-ocr --no-whisper` |
| Ollama (LLM) | 11434 | `ollama serve` (background) |

Standard commands (`npm run dev`, `npm run build`, `npm run lint`) are in `frontend/package.json`.

### Non-obvious caveats

- **Backend hardcoded path**: `backend/config.py` references `/home/zhangxiangxuan/桌面/Projects/PrivaSee`. Before starting the backend, create this directory structure: `sudo mkdir -p /home/zhangxiangxuan/桌面/Projects/PrivaSee/backend /home/zhangxiangxuan/桌面/Projects/PrivaSee/data /home/zhangxiangxuan/桌面/Projects/PrivaSee/models && sudo chown -R ubuntu:ubuntu /home/zhangxiangxuan`.
- **Backend optional services**: OCR and Whisper require PyTorch + GPU + large ML models. Use `--no-ocr --no-whisper` flags to skip them. MemoryStream also needs `torch` and `sentence-transformers`.
- **Ollama model**: The frontend defaults to `niels32167/qwen3-4b-instruct:latest`. For lightweight testing, pull `qwen3:0.6b` and select it in the UI model picker. Ollama needs `zstd` installed (`sudo apt-get install -y zstd`) before its install script will work.
- **Frontend lint**: ESLint reports pre-existing `no-unused-vars` errors across the codebase. These are not regressions.
- **Frontend proxy**: Vite proxies `/v1/*` and `/api/*` to Ollama (port 11434), and `/ocr-api/*`, `/whisper-api/*`, `/memory-api/*` to the backend (port 5000). Ensure both services are running before the frontend needs them.
