import tqdm

# Disable tqdm's monitor thread — it spawns a daemon thread that can
# interfere with subprocess-based test isolation.
tqdm.tqdm.monitor_interval = 0

from .fixtures.embeddings import *  # noqa: F401
