"""Load, merge, and normalize runner configuration.

This module combines defaults, config-file values, and CLI arguments into one
validated config dict used by the rest of the pipeline. It also normalizes
multi-value options such as embed formats, recognizer configs, and output-path
templating settings.
"""

import json
import re
import warnings
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar

from src.version import MODELS

from src.output_paths import resolve_template_paths
from src.recognizer_utils import (
    build_classifier_config_list,
    resolve_model_choice_for_recognizers,
    validate_recognizers,
)

default_config_dir = "/mnt/config/"


valid_values = {
    "model_choice": list(MODELS.keys()),
    "embed": ["parquet","csv"],
    "classify": ["parquet", "csv", "hoplite"],
    "embedding_table_format": ["serialized", "columns"],
    "embeddings_output_path_type": ["flat_basename", "nested_basename", "nested", "flat"],
    "classify_output_path_type": ["flat_basename", "nested_basename", "nested", "flat"],
    "output_path_type": ["flat_basename", "nested_basename", "nested", "flat"],
}


# if embed or classify is set to True without a specified format
# use these formats as the default
default_embed_format = "parquet"
default_classify_format = "csv"

default_config = {
    "embed": False,
    "classify": False,
    "save_db": False,
    "model_choice": "perch_v2",
    "source": "/mnt/input",
    "output": "/mnt/output",
    "embedding_table_format": "serialized",
    "file_glob": None,
    "dataset_name": "search_set",
    "workers": "auto",
    "db_path": "db",
    "log_level": "INFO",
    "hoplite_log_level": "WARNING",
    "tf_log_level": "WARNING",
    "log_file": None,
    "embeddings_output_path_template": None,
    "embeddings_output_path_type": None,
    "classify_output_path_template": None,
    "classify_output_path_type": None,
    "output_path_type": None,
    "recognizers": None,
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

    valid_filetypes: ClassVar[list[str]] = ["parquet", "csv"]
    valid_table_formats: ClassVar[list[str]] = ["serialized", "columns"]

    def __init__(self, filetype: str, table_format: str):
        if filetype not in self.valid_filetypes:
            raise ValueError(f"Invalid filetype: {filetype}. Valid options are: {self.valid_filetypes}")
        if table_format not in self.valid_table_formats:
            raise ValueError(f"Invalid table format: {table_format}. Valid options are: {self.valid_table_formats}")
        self.filetype = filetype
        self.table_format = table_format


def _json_safe_value(value):
    """Convert config values into JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, EmbeddingsFormat):
        return {
            "filetype": value.filetype,
            "table_format": value.table_format,
        }
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe_value(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_to_json(config: dict, *, sort_keys: bool = True) -> str:
    """Serialize a config dict to JSON with support for non-JSON config types."""
    return json.dumps(_json_safe_value(config), sort_keys=sort_keys)


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
        values = [fmt.strip() for fmt in values.split(",")]

    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"Invalid type: {type(values)}. Must be a string or a list.")


    values = [fmt.lower() for fmt in values]

    # deduplicate via set, then sort for deterministic ordering
    values = sorted(set([fmt.strip().lower() for fmt in list(values)]))
    return values


def validate_value(config, key):
    """
    Validates that values in the config for a given key are in the allow-list of valid values.
    Allows multiple values to be specified as a comma-separated string, which will be split and stripped before validation.
    Normalizes values into a sorted list of unique lowercase strings for deterministic downstream processing.
    """

    values = config[key]
    values = parse_list_values(values)
    allowed_values = set(valid_values[key])

    if any(fmt not in allowed_values for fmt in values):
        raise ValueError(f"Invalid {key} format: {values}. Valid options are: {allowed_values}")
    
    return values


def validate_single_value(value, key):
    """Validate a single allow-listed value (not comma-separated list)."""
    values = parse_list_values(value)
    if len(values) != 1:
        raise ValueError(f"{key} must be a single value, got: {values}")

    single = values[0]
    allowed_values = set(valid_values[key])
    if single not in allowed_values:
        raise ValueError(f"Invalid {key} value: {single}. Valid options are: {allowed_values}")

    return single


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
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")
    else:
        print("No config file found. Using default configuration.")
        config = {}
    file_config = dict(config)

    # merge with command line args, giving precedence to command line args
    args_dict = {}
    if args is not None:
        args_dict = vars(args)
        # remove config-path from args_dict since it is not a config parameter
        args_dict.pop('config_file', None)
    explicit_config = {**file_config, **{k: v for k, v in args_dict.items() if v is not None}}

    # apply defaults last: explicit values from config/CLI take precedence
    config = {**default_config, **explicit_config}

    provided_model_choice = explicit_config.get("model_choice")

    # allow only config keys that are in the default config
    for key in config.keys():
        if key not in default_config:
            raise ValueError(f"Invalid config key: {key}. Allowed keys are: {list(default_config.keys())}")


    # normalize optional templating values first
    template_val = normalize_bool_string(config.get("embeddings_output_path_template"))
    type_val = normalize_bool_string(config.get("embeddings_output_path_type"))
    classify_template_val = normalize_bool_string(config.get("classify_output_path_template"))
    classify_type_val = normalize_bool_string(config.get("classify_output_path_type"))
    output_path_type_val = normalize_bool_string(config.get("output_path_type"))
    config["embeddings_output_path_template"] = None if template_val is False else template_val
    config["embeddings_output_path_type"] = None if type_val is False else type_val
    config["classify_output_path_template"] = None if classify_template_val is False else classify_template_val
    config["classify_output_path_type"] = None if classify_type_val is False else classify_type_val
    config["output_path_type"] = None if output_path_type_val is False else output_path_type_val
    config["recognizers"] = build_classifier_config_list(
        config.get("recognizers"),
        config_dir=(config_file.parent if config_file is not None else Path(default_config_dir)),
    )

    if config["embeddings_output_path_template"] and config["embeddings_output_path_type"]:
        raise ValueError(
            "embeddings_output_path_template and embeddings_output_path_type are mutually exclusive"
        )

    if config["classify_output_path_template"] and config["classify_output_path_type"]:
        raise ValueError(
            "classify_output_path_template and classify_output_path_type are mutually exclusive"
        )

    if config.get("recognizers"):
        if provided_model_choice is not None:
            provided_model_choice = validate_single_value(
                provided_model_choice,
                "model_choice",
            )
        config["model_choice"] = resolve_model_choice_for_recognizers(
            provided_model_choice,
            config["recognizers"],
            MODELS,
        )
    elif "model_choice" in config:
        config["model_choice"] = validate_single_value(
            config["model_choice"],
            "model_choice",
        )

    if "embedding_table_format" in config:
        config["embedding_table_format"] = validate_value(config, "embedding_table_format")


    if config["output_path_type"] is not None:
        config["output_path_type"] = validate_single_value(config["output_path_type"], "output_path_type")

        # if output_path_type is specified, use that value for the specific output path types (embeddings/classify)
        # unless they are specified individually, in which case the individually specified value takes precedence
        if config["embeddings_output_path_type"] is None:
            config["embeddings_output_path_type"] = config["output_path_type"]
        if config["classify_output_path_type"] is None:
            config["classify_output_path_type"] = config["output_path_type"]

    if config["embeddings_output_path_type"] is not None:
        config["embeddings_output_path_type"] = validate_single_value(
            config["embeddings_output_path_type"],
            "embeddings_output_path_type",
        )

    if config["classify_output_path_type"] is not None:
        config["classify_output_path_type"] = validate_single_value(
            config["classify_output_path_type"],
            "classify_output_path_type",
        )

    resolve_template_paths(config)

    # Normalize embed/classify/save_db: bool-like strings → True/False, True → default format
    config['embed'] = normalize_bool_string(config['embed'])
    config['classify'] = normalize_bool_string(config['classify'])
    config['save_db'] = normalize_bool_string(config.get('save_db', False))

    if config['embed'] is True:
        config['embed'] = default_embed_format  
    if config['classify'] is True:
        config['classify'] = default_classify_format

    # Build structured embed list, or empty list if disabled
    if config['embed']:
        config['embed'] = validate_embed_config(config['embed'], fallback_table_formats=config['embedding_table_format'])
    else:
        config['embed'] = []

    # Validate that dual-format parquet export requires {embedding_table_format} token
    parquet_formats = [ef for ef in config['embed'] if ef.filetype == 'parquet']
    has_columns = any(ef.table_format == 'columns' for ef in parquet_formats)
    has_serialized = any(ef.table_format == 'serialized' for ef in parquet_formats)
    if has_columns and has_serialized and '{embedding_table_format}' not in config['embeddings_output_path_template']:
        raise ValueError(
            "Exporting both parquet table formats (columns and serialized) requires {embedding_table_format} token in the embeddings output path template"
        )

    config['classify'] = validate_value(config, 'classify') if config['classify'] else set()

    # Validate that at least one output action is specified
    if not config['embed'] and not config['classify'] and not config['save_db'] and not config['recognizers']:
        raise ValueError("At least one of --embed, --classify, or --save_db must be specified.")

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

    # Normalize db_path: relative paths are resolved under output.
    db_path_val = config.get('db_path') or default_config['db_path']
    db_path = Path(db_path_val)
    if db_path.is_absolute():
        config['db_path'] = db_path
    else:
        config['db_path'] = Path(config['output']) / db_path

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

    return config
  


        

