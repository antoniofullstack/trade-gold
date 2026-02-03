import os
import subprocess
import sys
from pathlib import Path


def setup_repo(repo_dir="/content/trade-gold", install_deps=True):
    """
    Prepare repo for running in Google Colab.

    - Sets cwd to repo_dir
    - Adds repo_dir to sys.path
    - Optionally installs dependencies
    """
    repo_path = Path(repo_dir).resolve()
    os.chdir(repo_path)

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    if install_deps:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(repo_path / "requirements.txt")]
        )

    return repo_path
