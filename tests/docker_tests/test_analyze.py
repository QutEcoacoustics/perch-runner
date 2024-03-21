import subprocess
from pathlib import Path
import shutil

scripts = [
    ["./scripts/analyze.sh"],
    ["pwsh", "-ExecutionPolicy", "Bypass", "-File",  "./scripts/analyze.ps1"]
]

index = 1

def test_analyze_script_classify():

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/embeddings/100sec.parquet", one)
    shutil.copy("tests/files/embeddings/100sec.parquet", two)

    # Run the embed helper script
    command = scripts[index] + ["classify", "./tests/input/", "./tests/output/", "pw"]
    print(f'running command: {" ".join(command)}')
    subprocess.run(command, check=True)

    # Specify the expected output files
    expected_output_files = ["./tests/output/files/one/100sec.csv", "./tests/output/files/two/100sec.csv"] 

    for file in expected_output_files:
        assert Path(file).exists()


def test_analysis_script_embed_file():

    # Run the embed helper script
    command = scripts[index] + ["generate", "./tests/files/audio/100sec.wav", "./tests/output/"]
    print(f'running command: {" ".join(command)}')
    subprocess.run(command, check=True)

    # Specify the expected output files
    expected_output_files = ["./tests/output/100sec.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()


def test_analysis_script_embed_folder():

    # Run the embed helper script
    command = scripts[index] + ["generate", "./tests/files/audio/", "./tests/output/"]
    print(f'running command: {" ".join(command)}')
    subprocess.run(command, check=True)

    # Specify the expected output files
    expected_output_files = ["./tests/output/100sec.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()