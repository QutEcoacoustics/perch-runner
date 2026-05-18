import shutil
from pathlib import Path

import pytest
import tqdm

# Disable tqdm's monitor thread — it spawns a daemon thread that can
# interfere with subprocess-based test isolation.
tqdm.tqdm.monitor_interval = 0

# the folder in which we will create temporary directories for input, output 
MOUNTED_TEMP_PARENT = Path(__file__).resolve().parents[1] / "mounted"


@pytest.fixture
def workspace(tmp_path):
    """
    Creates a temporary workspace folder for each tests
    """
    # Keep artifacts on mounted storage so they are inspectable from the host.
    run_dir = MOUNTED_TEMP_PARENT / tmp_path.name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    source = run_dir / "input"
    source.mkdir(parents=True)
    output = run_dir / "output"
    output.mkdir(parents=True)
    yield source, output
    if run_dir.exists():
        shutil.rmtree(run_dir)

from .fixtures.embeddings import *  # noqa: F401
