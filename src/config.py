"""
Parsing config files. 
def load_config(yml_source):
Only one config file can be specified to the public function load_config.
This config file can inherit from other config files, if they have the key 'inherit' with the value of the path to that parent config
"""

import yaml
from pathlib import Path
from ml_collections import config_dict

config_locations = ['', './src/default_configs', Path.cwd()]

def resolve_source(source):

    for loc in config_locations:
        # print the contents of the directory loc
        # print(f'checking {loc}:  {", ".join([str(entry.name) for entry in Path(loc).iterdir()])}')
        joined = Path(loc) / Path(source)
        if joined.exists():
            return joined

    raise FileNotFoundError(f'could not find {source} in {config_locations}. cwd is {Path.cwd()}')


def parse_and_merge(yml_source, stack=None):
    """
    load a yml file, check if it is supposed to inherit from anything, and if so load and merge the parent
    this is done recursively. stack will keep track of everything loaded to ensure there is no circular reference
    """

    yml_source = resolve_source(yml_source)

    if stack is None:
        stack = []
    elif yml_source in stack:
        raise ValueError(f'circular reference detected: {yml_source} in {stack}')
    
    with open(yml_source, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            config = {}

    # if inherit is inside the config, load an merge the parent
    if 'inherit' in config:
        parent_path = config.pop('inherit')
        parent_config = parse_and_merge(parent_path, stack + [yml_source])
        config = {**parent_config, **config}

    return config


def load_config(yml_source):

    if yml_source is None:
        config = {}
    else:
        config = parse_and_merge(yml_source)

    print("loaded config:")
    print(config)

    return config_dict.create(**config)



        

