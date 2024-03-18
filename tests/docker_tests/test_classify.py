import subprocess
from pathlib import Path
import shutil

def test_classify_script():

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/embeddings/100sec.parquet", one)
    shutil.copy("tests/files/embeddings/100sec.parquet", two)

    # Run the embed helper script
    command = ["./scripts/classify.sh", "./tests/input/", "./tests/output/", "pw", "cr08"]
    print(f'running command: {" ".join(command)}')
    subprocess.run(command, check=True)

    # Specify the expected output files
    expected_output_files = ["./tests/output/files/one/100sec.csv", "./tests/output/files/two/100sec.csv"] 

    for file in expected_output_files:
        assert Path(file).exists()
