"""Worker thread calculation based on available system RAM."""

import logging
import os

log = logging.getLogger(__name__)

# Estimated RAM per worker thread (model buffers + audio shard + embeddings).
# Based on empirical measurements: 8 threads used ~10GB peak on a 1-hour file,
# minus ~1.5GB base overhead = ~1.1GB per thread. Use 1.5GB to be safe.
RAM_PER_WORKER_GB = 1.5

# Base RAM needed regardless of worker count (TF runtime, model weights, Python)
RAM_BASE_GB = 2.0


def compute_workers(workers_config, available_ram_gb: float | None = None) -> int:
    """Determine the number of worker threads based on config and available RAM.

    Args:
        workers_config: int, or 'auto' (default). If 'auto', computes based on
            available system RAM.
        available_ram_gb: Override for testing. If None, reads from system.

    Returns:
        Number of worker threads (minimum 1, maximum 8).
    """
    if workers_config != 'auto':
        return max(1, int(workers_config))

    if available_ram_gb is None:
        available_ram_gb = _get_available_ram_gb()

    usable = available_ram_gb - RAM_BASE_GB
    workers = int(usable / RAM_PER_WORKER_GB)
    workers = max(1, min(workers, 8))
    log.info("Auto workers: %.1f GB available, allocating %d worker(s)",
             available_ram_gb, workers)
    return workers


def _get_available_ram_gb() -> float:
    """Get available RAM in GB from the system."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
        # Fallback: use total memory
        total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        return total / (1024**3)
    except Exception:
        # Conservative fallback
        return 4.0


def log_ram():
    """Log current RSS memory usage."""
    import resource
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss_kb / 1024  # Linux reports KB
    log.info("RAM usage: %.0f MB (peak RSS)", rss_mb)
