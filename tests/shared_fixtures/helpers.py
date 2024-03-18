from pathlib import Path
import shutil

def copy_test_files(files):
    """
    Helper function to copy files from the 'files' to the 'input' testing directory
    """
    input_dir = Path("./tests/input")
    source_dir = Path("./tests/files")
    for file in files:
        source_path = source_dir / file
        dest_path = input_dir / file
        shutil.copyfile(source_path, dest_path)