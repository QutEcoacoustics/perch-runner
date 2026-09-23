from pathlib import Path


def test_dockerfile_uses_shared_kagglehub_cache_location():
    dockerfile = Path("Dockerfile").read_text()

    assert "KAGGLEHUB_CACHE=/opt/perch-runner/.cache/kagglehub" in dockerfile
    assert "COPY --from=models /opt/perch-runner/.cache/kagglehub /opt/perch-runner/.cache/kagglehub" in dockerfile
    assert "COPY --from=models /root/.cache/kagglehub /root/.cache/kagglehub" not in dockerfile
