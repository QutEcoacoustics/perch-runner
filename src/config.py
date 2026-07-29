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

from src.output_paths import validate_and_resolve_template_config
from src.recognizer_utils import (
    build_classifier_config_list,
    resolve_model_choice_for_recognizers,
    validate_recognizers,
)
from src.sourcemap import get_sourcemap_preset_names
from src.sourcemap import SourcemapConfig

default_config_dir = "/mnt/config/"


# any config keys that can be validated against an allow-list of values. 
# other conifig keys may have more sophisticated validation logic. 
valid_values = {
    "model_choice": list(MODELS.keys()),
    "embed": [True, False],
    "embeddings_table_format": ["serialized", "columns"],
    "embeddings_table_filetype": ["parquet", "csv"],
    "embeddings_output_path_type": ["flat_filestem", "nested_filestem", "nested", "flat"],
    "classify": [True, False],
    "classify_filetype": ["parquet", "csv"], # todo: maybe add hoplite as a way to save result if perch team adds that feature
    "classify_output_path_type": ["flat_filestem", "nested_filestem", "nested", "flat"],
    "recognizer_results_filetype": ["parquet", "csv"],
    "recognizer_output_path_type": ["flat_filestem", "nested_filestem", "nested", "flat"],
    "output_path_type": ["flat_filestem", "nested_filestem", "nested", "flat"],
    "save_db": [True, False],
    "sourcemap_name": get_sourcemap_preset_names(),
}


# Here are all the config options, their default values, and a short description (which is used in the CLI help)
# False are actual values that mean "disabled"
# None means that a default will be applied, but it's more complicated than just a single default value here. 
all_config_options = {
    "source": ("/mnt/input", "path to the source audio folder"),
    "output": ("/mnt/output", "path to the output folder"),

    "model_choice": ("perch_v2", "model to use, e.g. perch_v2"),
    "save_db": (False, "save the hoplite database after processing. Use --save_db with no value to enable (default: false)"),
    "file_glob": (None, "glob pattern for audio files, e.g. '*/*', '*/*/*'. Auto-detected if not specified."),
    "dataset_name": ("search_set", "dataset name used in runner configuration"),
    "workers": ("auto", "number of worker threads for embedding, or 'auto' (default) to choose based on available RAM."),
    "db_path": ("db", "database output path. Relative paths are resolved under --output (default: db)"),
    "sourcemap_name": (None, "optional sourcemap preset name used for source remapping"),
    "file_metadata": (None, "optional JSON object/dict of template token values used for sourcemap rendering"),
    "sourcemap_template": (None, "optional sourcemap destination template, e.g. https://.../{audio_recording_id}/original"),
    "file_metadata_pattern": (None, "optional sourcemap pattern preset name or regex used to extract named tokens from filename"),

    "embed": (None, "enable embedding export (boolean flag). Use --embeddings_table_format and --embeddings_table_filetype to control output format."),
    "embeddings_table_format": ("serialized", "table format for embeddings, e.g. serialized, columns"),
    "embeddings_table_filetype": ("parquet", "file format for the embedding table"),
    "embeddings_output_path_template": (None, "custom output path template for embeddings files. Supported tokens: {parents}, {filestem}, {ext}, {embeddings_table_format}, {analysis}."),
    "embeddings_output_path_type": (None, "preset output path type for embeddings: flat_filestem, nested_filestem, nested, flat"),

    "recognizers": (None, "path to recognizers JSON file. The file may contain either a recognizers list/dict or an object with a top-level 'recognizers' key."),
    "recognizer_output_path_template": (None, "custom output path template for recognizer result files. Supported tokens: {classifier_name}, {parents}, {filestem}, {ext}, {analysis}."),
    "recognizer_output_path_type": (None, "preset output path type for recognizer results: flat_filestem, nested_filestem, nested, flat"),
    "recognizer_results_filetype": ("csv", "file format for recognizer results"),

    "classify": (False, "enable classify output (boolean flag). Use --classify_filetype to control output format."),
    "classify_filetype": ("csv", "file format for classification tables"),
    "classify_species_list": (None, "path to the species list for classification"),
    "classify_output_path_template": (None, "custom output path template for classification files"),
    "classify_output_path_type": (None, "preset output path type for classification files"),

    "output_path_type": (None, "preset output path type applied to both embeddings and recognizer results (overridden by more specific keys): flat_filestem, nested_filestem, nested, flat"),

    "log_level": ("INFO", "log level for perch-runner output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)"),
    "hoplite_log_level": ("WARNING", "log level for perch-hoplite / library output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)"),
    "tf_log_level": ("WARNING", "log level for TensorFlow C++ output: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)"),
    "log_file": (None, "path to a log file. Output is sent to both console and file."),
}

# Default values only (help text remains in all_config_options).
default_config = {k: v[0] for k, v in all_config_options.items()}



_FALSY_STRINGS = frozenset({"none", "false", "null", ""})
_TRUTHY_STRINGS = frozenset({"true"})


def _warn_and_drop_disabled_keys(explicit_config, *, mode_name, related_keys):
    """
    If there are embedding related keys present in the config, but the embed flag is explicitly set to False
    then we remove those keys. This is to prevent validating them even when they are not used. 
    For example, if --embeddings_table_format is specified in the config.yml, but then --embed is set to False
    by the commandline, we should not validate the embeddings_table_format. 
    This also applies to recognizers and classify.
    
    """
    present_keys = [key for key in related_keys if key in explicit_config and explicit_config.get(key) is not None]
    if not present_keys:
        return

    warnings.warn(
        f"{mode_name} is explicitly disabled; related settings will be ignored: "
        + ", ".join(present_keys),
        UserWarning,
    )

    for key in present_keys:
        explicit_config.pop(key, None)


def normalize_bool_string(explicit_config, key):
    """Normalize a value that may be a bool, None, or a bool-like string.
       updates the dict in place. Does not modify the dict if the key is not present
       or not bool-like
    """
    if key not in explicit_config:
        return
    value = explicit_config[key]
    if value is None or value is False:
        explicit_config[key] = False
    elif value is True:
        explicit_config[key] = True
    elif isinstance(value, str):
        lower = value.strip().lower()
        if lower in _FALSY_STRINGS:
            explicit_config[key] =  False
        if lower in _TRUTHY_STRINGS:
            explicit_config[key] =  True



def _json_safe_value(value):
    """Convert config values into JSON-serializable structures."""
    if isinstance(value, SourcemapConfig):
        return value.to_log_dict()
    if isinstance(value, Path):
        return str(value)
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


def validate_list_value(config, key):
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


def validate_single_value(config, key):
    """Validate a single allow-listed value (not comma-separated list)."""

    if key not in config:
        return
    
    value = config.get(key)
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{key} must be a single value, got: {value}")
    
    if key in valid_values:
        allowed_values = set(valid_values[key])
        if value not in allowed_values:
            raise ValueError(f"Invalid {key} value: {value}. Valid options are: {allowed_values}")


def validate_embedding_config(explicit_config):
    """ validate all the embedding-related config values, including embed, embeddings_table_format, embeddings_table_filetype, embeddings_output_path_template, and embeddings_output_path_type. """

    # Normalize embed/classify/save_db: bool-like strings -> True/False.
    normalize_bool_string(explicit_config, 'embed')

    # if these keys are specified it imples that embed should be true. e.g. setting the embeddings_table_format is enough
    embed_related_keys = [
        "embeddings_table_format",
        "embeddings_table_filetype",
        "embeddings_output_path_template",
        "embeddings_output_path_type",
    ]

    if any(explicit_config.get(key, False) for key in embed_related_keys):
        if explicit_config.get('embed') == False:
            _warn_and_drop_disabled_keys(
                explicit_config,
                mode_name="embed",
                related_keys=embed_related_keys,
            )
        else:
            explicit_config['embed'] = True

    validate_single_value(explicit_config, "embed")
    validate_single_value(explicit_config, "embeddings_table_format")
    validate_single_value(explicit_config, "embeddings_table_filetype")

    # embeddings_output_path stuff is validated elsewhere


def validate_recognizer_config(explicit_config, config_file):
    """ validate all the recognizer-related config values, including recognizers, recognizer_output_path_template, recognizer_output_path_type, and recognizer_results_filetype. """


    explicit_config["recognizers"] = build_classifier_config_list(
        explicit_config.get("recognizers"),
        config_dir=(config_file.parent if config_file is not None else Path(default_config_dir)),
    )

    validate_single_value(explicit_config, 'recognizer_results_filetype')

    recognizer_related_keys = [
        "recognizer_output_path_template",
        "recognizer_output_path_type",
        "recognizer_results_filetype",
    ]
    if any(explicit_config.get(key) for key in recognizer_related_keys):
        if not explicit_config.get("recognizers"):
            _warn_and_drop_disabled_keys(
                explicit_config,
                mode_name="recognizers",
                related_keys=recognizer_related_keys,
            )

    # validate that provided model choice and recognizers are compatible, and resolve the effective model choice to use
    validate_single_value(explicit_config, "model_choice")

    if explicit_config.get("recognizers"):
        provided_model_choice = explicit_config.get("model_choice")
        explicit_config["model_choice"] = resolve_model_choice_for_recognizers(
            provided_model_choice,
            explicit_config["recognizers"],
            MODELS,
        )
 


def validate_classify_config(explicit_config):
    """ validate all the classify-related config values, including classify and classify_filetype. """

    normalize_bool_string(explicit_config, 'classify')

    classify_related_keys = [
        "classify_filetype",
        "classify_species_list",
        "classify_output_path_template",
        "classify_output_path_type"
    ]

    if any(explicit_config.get(key, False) for key in classify_related_keys):
        if explicit_config.get('classify') == False:
            _warn_and_drop_disabled_keys(
                explicit_config,
                mode_name="classify",
                related_keys=classify_related_keys,
            )
        else:
            explicit_config['classify'] = True

    validate_single_value(explicit_config, "classify")

    validate_single_value(explicit_config, "classify_filetype")


def validate_sourcemap_config(explicit_config):
    """Validate and normalize sourcemap-related config values."""
    validate_single_value(explicit_config, "sourcemap_name")

    if "file_metadata" in explicit_config:
        file_metadata = explicit_config.get("file_metadata")
        if isinstance(file_metadata, str):
            try:
                file_metadata = json.loads(file_metadata)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid file_metadata JSON: {e}") from e

        if file_metadata is None:
            explicit_config["file_metadata"] = None
        elif not isinstance(file_metadata, dict):
            raise ValueError("file_metadata must be a dictionary or a JSON object string")
        else:
            explicit_config["file_metadata"] = file_metadata

    # wrap these related configs in a dataclass object
    explicit_config["sourcemap_config"] = SourcemapConfig.from_inputs(
        sourcemap_name=explicit_config.get("sourcemap_name"),
        file_metadata=explicit_config.get("file_metadata"),
        sourcemap_template=explicit_config.get("sourcemap_template"),
        file_metadata_pattern=explicit_config.get("file_metadata_pattern"),
    )


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

    # allow only config keys that are in the default config
    for key in explicit_config.keys():
        if key not in default_config:
            raise ValueError(f"Invalid config key: {key}. Allowed keys are: {list(default_config.keys())}")

    validate_embedding_config(explicit_config)
    validate_recognizer_config(explicit_config, config_file)
    validate_classify_config(explicit_config)
    validate_sourcemap_config(explicit_config)

    
    normalize_bool_string(explicit_config, 'save_db')
    validate_single_value(explicit_config,"save_db")


    # merge explicit config with defaults
    config = dict(default_config)
    for k, v in explicit_config.items():
        config[k] = v


    # Validate that at least one output action is specified
    if not config['embed'] and not config['classify'] and not config['save_db'] and not config['recognizers']:
        raise ValueError("At least one of --embed, --classify, --save_db or --recognizers must be specified.")


    validate_and_resolve_template_config(config)

    # file_glob indicates which audio files to process. 
    # if not specified (falsy), it will later autodetect the file_glob string
    normalize_bool_string(config, 'file_glob')
    config['file_glob'] = None if config['file_glob'] is False else config['file_glob']

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
    db_path_val = config['db_path']
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
  


        

