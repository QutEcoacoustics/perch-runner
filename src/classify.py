from perch_hoplite.zoo import model_configs
import numpy as np


def classify(config):

    model = model_configs.load_model_by_name(config['model_choice'])

    class_list = class_list = model.class_list


def classify_file(model, class_list, audio_file):