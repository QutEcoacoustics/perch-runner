
from src.app import main
from pytest_mock import mocker 

import src.embed_audio_slim

import inspect

def test_embed_command(mocker) -> None:

    mocker.patch('sys.argv', ['some_file_name.py', 'generate', '--source_file', 'tests/files/100sec.wav', '--output_folder', 'tests/output/'])

    mocked_embed_file_and_save = mocker.patch('src.app.embed_file_and_save')

    mocked_embed_file_and_save.return_value = "hi there"
    
    main()
    
    mocked_embed_file_and_save.assert_called_once_with('tests/files/100sec.wav', 'tests/output/', None)



