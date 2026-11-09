"""
Root entrypoint for Streamlit Cloud deployments pointing to app.py.
Delegates directly to app/app.py.
"""
from pathlib import Path
import runpy
import sys

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

target_app = root_dir / "app" / "app.py"
runpy.run_path(str(target_app), run_name="__main__")
