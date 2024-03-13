from pathlib import Path
from pytest_mock import mocker 

from src.app import main
from src import batch

import inspect

def test_embed_command(mocker) -> None:

    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source_file', 'tests/files/100sec.wav', '--output_folder', 'tests/output/'])

    mocked_embed_file_and_save = mocker.patch('src.app.embed_file_and_save')

    mocked_embed_file_and_save.return_value = "hi there"
    
    main()
    
    mocked_embed_file_and_save.assert_called_once_with('tests/files/100sec.wav', 'tests/output/', None)



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
