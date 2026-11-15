import os
import sys
from pathlib import Path

# Tests import backend modules directly (main.py, nst_engine.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
