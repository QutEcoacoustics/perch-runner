"""
Uses the tensorflow_hub pacakge to download and save the model during build
This is probably more reliable than wget or similar, since it seems to deal with the redirect to kaggle etc
"""

import argparse
import shutil
from pathlib import Path
import os
import tensorflow_hub as hub

parser = argparse.ArgumentParser()
parser.add_argument("--version", help="version of the model to download")
parser.add_argument("--destination", default="4", help="where to download to")
args = parser.parse_args()

PERCH_TF_HUB_URL = 'https://tfhub.dev/google/bird-vocalization-classifier'

model_url = f'{PERCH_TF_HUB_URL}/{args.version}'
# This model behaves exactly like the usual saved_model.
# model = hub.load(model_url)

model_path = hub.resolve(model_url)

destination = Path(args.destination) / Path(args.version)

shutil.move(model_path, destination)

if os.path.exists(destination / Path("saved_model.pb")):
  print(f"Model saved to {destination}")
else:
  raise Exception("Failed to save model")


