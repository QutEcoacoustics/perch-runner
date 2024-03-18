
# import os
# import shutil
from pathlib import Path
import pytest

import pandas as pd

from src import inference_parquet
from ml_collections import config_dict
from src.config import load_config

valid_model_paths = ('pw', 
            'pw/trained_model_pw_02.keras', 
            '/models/pw',
            '/models/pw/trained_model_pw_02.keras',
            'tests/files/models/r1', 
            'tests/files/models/r1/r1.keras')

invalid_model_paths = ('nothing', '/app/models/nothing', 'tests/files/models/r1/nothing.keras')


def test_find_model_file():

    for path in valid_model_paths:
        model_file, labels_file = inference_parquet.find_model(path)
        assert model_file.exists()
        assert labels_file.exists()
    
def test_dont_find_model():

    for path in invalid_model_paths:
        model_file, labels_file = inference_parquet.find_model(path)
        assert model_file is None
        assert labels_file is None

def test_load_model():

    for path in valid_model_paths:
        classifier = inference_parquet.load_classifier(path)
        assert classifier.model is not None
        assert classifier.labels is not None

def test_load_missing_model():

    for path in invalid_model_paths:

        with pytest.raises(ValueError) as e_info:
            classifier = inference_parquet.load_classifier(path)
        
        assert str(e_info.value) == f"no keras file found at {path}"


def test_classify_folder():

    source_folder = "tests/files/embeddings"
    output_folder = "tests/output"
    classifier = "pw"

    inference_parquet.process_folder(source_folder, output_folder, 
                                         config_dict.create(classifier=classifier, skip_if_file_exists=True))

    expected_files = [Path(output_folder) / Path("100sec.csv"), 
                      Path(output_folder) / Path("200sec.csv")]
    
    assert expected_files[0].exists()
    assert expected_files[1].exists()


def test_classify_one_file():

    source_file = "tests/files/embeddings/100sec.parquet"
    output_folder = "tests/output"
    config_file = "pw.classify.yml"
    
    config = load_config(config_file)

    inference_parquet.classify_file_and_save(source_file, output_folder, config=config)

    expected_file = Path(output_folder) / Path("100sec.csv")
    
    assert expected_file.exists()



def test_classify_one_embeddings_file():

    classifier = "pw"
    results = inference_parquet.classify_embeddings_file("tests/files/embeddings/100sec.parquet", classifier)
    #TODO: check rows and num columns
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ['filename', 'offset_seconds', 'neg', 'pos']
    assert results.shape[0] == 20 # 100 seconds at 5 second intervals



# def test_classify_one_file_and_save():
    
#     source = "tests/files/100sec.wav.parquet"
#     destination = "tests/output/100sec_embeddings.parquet"
#     model = "pw"

#     inference_parquet.classify_file_and_save(source, destination, model)

#     assert os.path.exists(destination)
#     #TODO: check rows and num columns


# def test_classify_files():

#     # set up input folder

#     one = Path("tests/input/embeddings/one")
#     two = Path("tests/input/embeddings/two")
#     one.mkdir(parents=True, exist_ok=True)
#     two.mkdir(parents=True, exist_ok=True)
#     shutil.copy("tests/files/embeddings/100sec.wav.parquet", one)
#     shutil.copy("tests/files/embeddings/100sec.wav.parquet", two)

#     source_folder = "tests/input/embeddings"
#     output_folder = "tests/output"

#     inference_parquet.classify_files(source_folder, output_folder)

#     expected_files = [Path(output_folder) / Path("one/100sec.csv"), 
#                       Path(output_folder) / Path("two/100sec.csv")]

#     assert os.path.exists(expected_files[0])
#     assert os.path.exists(expected_files[1])
    
#     # now read the embeddings back and check the shape and source columns are as expected
#     df = pd.read_csv(expected_files[0])
#     # TODO: assert shape 
#     # assert df.shape == (20, 1283)
#     # assert df.source[0] == "one/100sec.wav"


# # for now we don't need to batch up classifications, because that's pretty fast if we have embeddings