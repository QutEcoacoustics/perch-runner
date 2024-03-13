import subprocess
from pathlib import Path

def test_classify_script():

    # Run the embed helper script
    command = ["./scripts/classify.sh", "./tests/files/embeddings/100sec.parquet", "./tests/output/", "pw", "cr08"]
    print(f'running command: {" ".join(command)}')
    subprocess.run(command, check=True)

    # Specify the expected output files
    expected_output_files = ["./tests/output/100sec.csv"] 

    for file in expected_output_files:
        assert Path(file).exists()
