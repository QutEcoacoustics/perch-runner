import os
from pathlib import Path
from src.transcode_embeddings.transcode_embeddings import transcode_from_parquet, get_parquet_file_list



def test_transcode_from_parquet():
  # Define the input file path
  input_folder = Path("./tests/files/embeddings")

  output_folder = Path("./tests/output/")

  parquet_files = get_parquet_file_list(input_folder)

  # Call the transcode_from_parquet function
  transcode_from_parquet(parquet_files, output_folder, num_files=256)

  # Assert that the output files exist by checking that 
  # there are 256 files in the output folder with filenames embeddings-[date]-%[file_num]-of-00256
  # where date is a timestamp and file_num is a number between 0 and 255 with leading zeros
  # by getting a list of files that match that pattern, and checking that the length of the list is 256
  output_files = [f for f in output_folder.rglob('embeddings-*-*-of-00256')]
  assert len(output_files) == 256
