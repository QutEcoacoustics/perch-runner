from pathlib import Path
from pytest_mock import mocker 
import pytest

from src.app import main
from src import batch
from ml_collections import ConfigDict

import inspect

def test_embed_command_file(mocker) -> None:

    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source', 'tests/files/audio/100sec.wav', '--output', 'tests/output/'])
    mocked_embed_file_and_save = mocker.patch('src.app.embed_file_and_save')
    mocked_embed_file_and_save.return_value = "hi there"
    main()
    mocked_embed_file_and_save.assert_called_once_with(Path('tests/files/audio/100sec.wav'), 'tests/output/', ConfigDict(**{}))


def test_embed_command_folder(mocker) -> None:

    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source', 'tests/files/audio', '--output', 'tests/output/'])
    mocked_embed_file_and_save = mocker.patch('src.app.embed_folder')
    mocked_embed_file_and_save.return_value = "hi there"
    main()
    mocked_embed_file_and_save.assert_called_once_with(Path('tests/files/audio'), 'tests/output/', ConfigDict(**{}))


def test_missing_source_file(mocker) -> None:
    # Use a definitely non-existing file path
    non_existing_file = 'tests/files/audio/definitely_nonexistent_file.wav'
    # Patch `sys.argv` to simulate command line input
    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source', non_existing_file, '--output', 'tests/output/'])

    # Expect the main function to raise SystemExit due to argparse error
    with pytest.raises(SystemExit) as e:
        main()
    assert str(e.value) == "2", "Expected SystemExit with exit code 2 for argparse error"


def test_missing_source_folder(mocker) -> None:
    # Use a definitely non-existing folder path
    non_existing_folder = 'tests/files/nonexistent_folder'
    # Patch `sys.argv` to simulate command line input
    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source', non_existing_folder, '--output', 'tests/output/'])

    # Expect the main function to raise SystemExit due to argparse error
    with pytest.raises(SystemExit) as e:
        main()
    assert str(e.value) == "2", "Expected SystemExit with exit code 2 for argparse error"



def test_batch_entrypoint_item_0_1():
    """Test batch processing from row 0 to 1 of the batch list"""

    batch.batch('generate', source_csv='tests/files/batch_lists/batch_embed.csv', start_row=0, end_row=1, config_file=None)

    assert Path('tests/output/100sec.parquet').exists()
    assert Path('tests/output/some_subfolder/200sec.parquet').exists()
    assert not Path('tests/output/some_subfolder/100sec_again.parquet').exists()


def test_batch_entrypoint_item_2():
    """Test batch processing from row 2 to 2 of the batch list"""

    batch.batch('generate', source_csv='tests/files/batch_lists/batch_embed.csv', start_row=2, end_row=2, config_file=None)

    assert not Path('tests/output/100sec.parquet').exists()
    assert not Path('tests/output/some_subfolder/200sec.parquet').exists()
    assert Path('tests/output/some_subfolder/100sec_again.parquet').exists()


def test_embed_command_empty_config(mocker) -> None:

    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source', 'tests/files/audio', '--output', 'tests/output/', '--config_file', 'tests/files/configs/empty.yml'])  
    mocked_embed_file_and_save = mocker.patch('src.app.embed_folder')
    mocked_embed_file_and_save.return_value = "hi there"
    main()
    mocked_embed_file_and_save.assert_called_once_with(Path('tests/files/audio'), 'tests/output/', ConfigDict(**{}))