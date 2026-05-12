"""Parsing config files."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar

from src.version import MODELS

default_config_dir = "/mnt/config/"


valid_values = {
    "model_choice": list(MODELS.keys()),
    "embed": ["parquet","csv","hoplite"],
    "classify": ["parquet", "csv", "hoplite"],
    "embedding_table_format": ["serialized", "columns"]
}

# if embed or classify is set to True without a specified format
# use these formats as the default
default_embed_format = "parquet"
default_classify_format = "csv"

default_config = {
    "embed": False,
    "classify": False,
    "model_choice": "perch_v2",
    "source": "/mnt/input",
    "output": "/mnt/output",
    "embedding_table_format": "serialized",
    "file_glob": None,
    "dataset_name": "search_set",
    "workers": "auto",
    "log_level": "INFO",
    "hoplite_log_level": "WARNING",
    "tf_log_level": "WARNING",
    "log_file": None,
}




_FALSY_STRINGS = frozenset({"none", "false", "null", ""})
_TRUTHY_STRINGS = frozenset({"true"})


def normalize_bool_string(value):
    """Normalize a value that may be a bool, None, or a bool-like string.

    Returns True, False, or the original string (lowered) if it is not
    a boolean-like token.
    """
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in _FALSY_STRINGS:
            return False
        if lower in _TRUTHY_STRINGS:
            return True
        return value  # a real format string like "parquet"
    return value


@dataclass
class EmbeddingsFormat:
    filetype: str = "parquet"
    table_format: str = "serialized"

    valid_filetypes: ClassVar[list[str]] = ["parquet", "csv", "hoplite"]
    valid_table_formats: ClassVar[list[str]] = ["serialized", "columns"]

    def __init__(self, filetype: str, table_format: str):
        if filetype not in self.valid_filetypes:
            raise ValueError(f"Invalid filetype: {filetype}. Valid options are: {self.valid_filetypes}")
        if table_format not in self.valid_table_formats:
            raise ValueError(f"Invalid table format: {table_format}. Valid options are: {self.valid_table_formats}")
        self.filetype = filetype
        self.table_format = table_format


def validate_embed_config(embed_config_val, fallback_table_formats):
    """Parse embed config into a list of EmbeddingsFormat.

    Items with an explicit format (e.g. "parquet-columns") keep that format.
    Items without (e.g. "csv") get expanded across all fallback_table_formats.
    """

    embed_values = parse_list_values(embed_config_val)
    results = []
    for val in embed_values:
        parts = val.split("-")
        if len(parts) == 1:
            results.extend([EmbeddingsFormat(filetype=val, table_format=tf) for tf in fallback_table_formats])
        elif len(parts) == 2:
            filetype, table_format = parts
            results.append(EmbeddingsFormat(filetype=filetype, table_format=table_format))
        else:
            raise ValueError(f"Invalid embed config value: {val}. Must be filetype or in the format 'filetype-tableformat'")
    return results


def find_config():
    """
    looks for config.yml or config.json in the default config directory
    if both are present, Raises an error. If neither are present, returns None
    """

    default_filenames  = ["config.yml", "config.yaml", "config.json"]
    found_config = [Path(default_config_dir) / filename for filename in default_filenames if (Path(default_config_dir) / filename).exists()]
    if len(found_config) > 1:
        raise FileExistsError("Multiple config files are present in the default config directory.")
    elif len(found_config) == 1:
        return found_config[0]
    else:
        return None


def parse_list_values(values):

    # if it's a string
    if isinstance(values, str):
        values = set([fmt.strip().lower() for fmt in values.split(",")])

    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"Invalid type: {type(values)}. Must be a string or a list.")

    values = set([fmt.strip().lower() for fmt in list(values)])
    return values


def validate_value(config, key):
    """
    Validates that values in the config for a given key are in the allow-list of valid values.
    Allows multiple values to be specified as a comma-separated string, which will be split and stripped before validation.
    Normalizes values into a set of lowercase strings for easier downstream processing.
    """

    values = config[key]
    values = parse_list_values(values)
    allowed_values = set(valid_values[key])

    if any(fmt not in allowed_values for fmt in values):
        raise ValueError(f"Invalid {key} format: {values}. Valid options are: {allowed_values}")
    
    return values


def load_config(config_path=None, args=None):
    """
    attemps to load a config file, either from the specified path or from the default config directory. 
    """
    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Specified config file {config_file} does not exist.")
    else:
        config_file = find_config()


    if config_file is not None:
        # open and parse yaml or json config file
        if config_file.suffix in ['.yml', '.yaml']:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        elif config_file.suffix == '.json':
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    else:
        print("No config file found. Using default configuration.")
        config = {}

    # apply default config, giving precedence to the loaded config
    config = {**default_config, **config}

    # merge with command line args, giving precedence to command line args
    if args is not None:
        args_dict = vars(args)
        # remove config-path from args_dict since it is not a config parameter
        args_dict.pop('config_file', None)
        config = {**config, **{k: v for k, v in args_dict.items() if v is not None}}

    # allow only config keys that are in the default config
    for key in config.keys():
        if key not in default_config:
            raise ValueError(f"Invalid config key: {key}. Allowed keys are: {list(default_config.keys())}")


    # validate allow-lists
    for key in ["model_choice", "embedding_table_format"]:
        if key in config:
            config[key] = validate_value(config, key)

    # Normalize embed/classify: bool-like strings → True/False, True → default format
    config['embed'] = normalize_bool_string(config['embed'])
    config['classify'] = normalize_bool_string(config['classify'])

    if config['embed'] is True:
        config['embed'] = default_embed_format  
    if config['classify'] is True:
        config['classify'] = default_classify_format

    # Build structured embed list, or empty list if disabled
    if config['embed']:
        config['embed'] = validate_embed_config(config['embed'], fallback_table_formats=config['embedding_table_format'])
    else:
        config['embed'] = []

    config['classify'] = validate_value(config, 'classify') if config['classify'] else set()

    # Normalize file_glob: falsy strings → None (triggers auto-detection)
    glob_val = normalize_bool_string(config.get('file_glob'))
    config['file_glob'] = None if glob_val is False else glob_val

    # Normalize workers: 'auto' stays as string, numbers get converted
    workers_val = config.get('workers', 'auto')
    if isinstance(workers_val, str) and workers_val.strip().lower() == 'auto':
        config['workers'] = 'auto'
    else:
        try:
            config['workers'] = int(workers_val)
        except (ValueError, TypeError):
            config['workers'] = 'auto'

    # Normalize log levels: uppercase string
    for key in ('log_level', 'hoplite_log_level', 'tf_log_level'):
        val = config.get(key)
        if val is not None:
            config[key] = str(val).upper()

    # Normalize log_file: falsy → None
    log_file = config.get('log_file')
    if not log_file or (isinstance(log_file, str) and log_file.strip().lower() in ('none', 'false', '')):
        config['log_file'] = None

    # ensure source and output are Path objects and exist
    config['source'] = Path(config['source'])
    if not config['source'].exists():
        raise FileNotFoundError(f"Source path {config['source']} does not exist.")

    config['output'] = Path(config['output'])
    if not config['output'].exists():
        raise FileNotFoundError(f"Output path {config['output']} does not exist.")

    # Validate that at least one action is specified
    if not config['embed'] and not config['classify']:
        raise ValueError("At least one of --embed or --classify must be specified.")

    return config
  


        

