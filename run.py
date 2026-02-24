#!/usr/bin/env python3
"""
PicturaAI Server Launcher
Activates the venv and starts the FastAPI backend.
"""
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND = ROOT / "backend"
VENV_PYTHON = ROOT / "venv" / "Scripts" / "python.exe"

def main():
    print("""
  ╔════════════════════════════════════════════════════╗
  ║         PicturaAI — Neural Style Transfer           ║
  ║             v2.0 Production Server                 ║
  ╚════════════════════════════════════════════════════╝

  ► App:      http://localhost:8000/app
  ► API docs: http://localhost:8000/docs
  ► Press Ctrl+C to stop
    """)

    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    cmd = [python, "-m", "uvicorn", "main:app",
           "--host", "0.0.0.0",
           "--port", "8000",
           "--reload",
           "--reload-dir", str(BACKEND)]

    try:
        subprocess.run(cmd, cwd=str(BACKEND), check=True)
    except KeyboardInterrupt:
        print("\n  Server stopped.")

if __name__ == "__main__":
    main()
