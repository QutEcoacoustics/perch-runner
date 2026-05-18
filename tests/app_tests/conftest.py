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
    
    Rather than use tmp_path directly, we create a subdirectory in a mounted location. 
    This allows us to inspect the generated files during the test runs.
    tmp_path is only used to get a unique directory name (inside the mounted directory) based on the test function name.
    """

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
