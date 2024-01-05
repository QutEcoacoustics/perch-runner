#!/usr/local/bin/python

"""
utils for parsing configuration from yml
"""

import yaml
from ml_collections import config_dict


def parse_config(yml_source):

    if yml_source is None:
        return None

    with open(yml_source, 'r') as f:
        config = yaml.safe_load(f)

    return config_dict.create(**config)


