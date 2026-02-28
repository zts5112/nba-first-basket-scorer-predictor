"""Streamlit Cloud entrypoint — runs ui/app.py."""

import runpy
import sys
from pathlib import Path

# Ensure ui/ and src/ are importable
root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "ui"))

runpy.run_path(str(root / "ui" / "app.py"), run_name="__main__")
