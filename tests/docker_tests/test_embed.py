import pytest
import subprocess
import os

def test_embed_script():

    # Run the embed helper script
    subprocess.run(["./scripts/embed.sh"], check=True)

    # Specify the expected output files
    expected_output_files = ["output_file1.txt", "output_file2.jpg", ...] 

    # Check if files exist in the output directory
    output_dir = "./tests/output"
    for file in expected_output_files:
        assert os.path.exists(os.path.join(output_dir, file))