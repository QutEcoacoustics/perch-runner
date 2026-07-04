
"""Recognizer normalization and model-selection helpers."""

import json
import logging
from pathlib import Path


def _normalize_embedding_dims(embedding_dim):
    if isinstance(embedding_dim, (list, tuple, set)):
        dims = list(embedding_dim)
    else:
        dims = [embedding_dim]

    unique_dims = []
    for dim in dims:
        if dim not in unique_dims:
            unique_dims.append(dim)
    return unique_dims


def _single_classifier_embedding_dim(classifier_config_list):
    dims = _normalize_embedding_dims(classifier_config_list.embedding_dim)
    if len(dims) == 1:
        return dims[0]
    return None


def _model_choice_from_embedding_dim(embedding_dim, models):
    matches = [name for name, info in models.items() if info.get("embedding_dim") == embedding_dim]
    if len(matches) == 1:
        return matches[0]
    return None


def _derived_model_choice(classifier_config_list, models):
    if classifier_config_list.embedding_model_name is not None:
        return classifier_config_list.embedding_model_name

    embedding_dim = _single_classifier_embedding_dim(classifier_config_list)
    if embedding_dim is None:
        return None
    return _model_choice_from_embedding_dim(embedding_dim, models)


def resolve_model_choice_for_recognizers(provided_model_choice, classifier_config_list, models):
    """Resolve final model choice from provided value and recognizer metadata."""
    derived_model_choice = _derived_model_choice(classifier_config_list, models)
    classifier_embedding_dim = _single_classifier_embedding_dim(classifier_config_list)

    if provided_model_choice is not None:
        if derived_model_choice is not None:
            if provided_model_choice != derived_model_choice:
                raise ValueError(
                    f"model_choice {provided_model_choice!r} does not match recognizer embedding model {derived_model_choice!r}"
                )
            return provided_model_choice

        if classifier_embedding_dim is None:
            raise ValueError(
                "embedding model name not provided, and can't be determined from embedding dimension"
            )

        provided_dim = models[provided_model_choice]["embedding_dim"]
        if provided_dim != classifier_embedding_dim:
            raise ValueError(
                f"model_choice {provided_model_choice!r} has embedding_dim={provided_dim}, "
                f"but recognizers require embedding_dim={classifier_embedding_dim}"
            )
        return provided_model_choice

    if derived_model_choice is not None:
        logging.info(
            "Embedding model name not provided; inferred %r from recognizer embedding dimension %d",
            derived_model_choice,
            classifier_embedding_dim,
        )
        return derived_model_choice

    raise ValueError(
        "embedding model name not provided, and can't be determined from embedding dimension"
    )




def _load_recognizers_from_json_file(recognizers_path: Path):
    """Load recognizers payload from a JSON file.

    Accepts either:
    - a top-level list/dict of recognizer config(s), or
    - an object containing a `recognizers` key.
    """
    with open(recognizers_path, 'r') as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "recognizers" in payload:
            return payload["recognizers"]
        else:
            return [payload]
    elif isinstance(payload, list):
        return payload
    else:
        raise ValueError(f"Invalid recognizers payload: {payload}. Must be a dict or a list.")  


def validate_recognizers(recognizers, config_dir=None):
    """Normalize recognizers config into a list of JSON-like dict objects.

    `recognizers` may be:
    - None
    - a dict
    - a list of dicts
    - a string path to a JSON file containing recognizer config(s)
    - a string comma separated list of paths to different recognizer configs (not yet implemented)
    - a list of string paths to different recognizer configs (not yet implemented)
    """
    if recognizers is None:
        return []
        
    if isinstance(recognizers, str): 
        recognizers = [fmt.strip() for fmt in recognizers.split(",")]
    elif isinstance(recognizers, dict):
        recognizers = [recognizers]
    elif not isinstance(recognizers, list):
        raise ValueError("recognizers must be a list of JSON objects")

    # recognizers is now a list. Each item of a list is either
    # - a dict (JSON object)
    # - a string path to a JSON file containing recognizer config(s)

    parsed_recognizers = []

    for i, item in enumerate(recognizers):
        if isinstance(item, str):
            
            resolved = resolve_recognizer_path(item, config_dir=config_dir)

            try:
                parsed_recognizers.extend(_load_recognizers_from_json_file(resolved))
            except json.JSONDecodeError as e:
                raise ValueError(f"recognizers file is not valid JSON: {resolved}") from e
            
        elif isinstance(item, dict):
            parsed_recognizers.append(item)


    for i, recognizer in enumerate(parsed_recognizers):
        if not isinstance(recognizer, dict):
            raise ValueError(
                f"recognizers[{i}] must be a JSON object (dict), got {type(recognizer).__name__}"
            )

    return parsed_recognizers


def build_classifier_config_list(recognizers, config_dir=None):
    """Normalize recognizers and return a ClassifierConfigList when configured."""
    if recognizers is None or recognizers == []:
        return []

    # If caller already provided a normalized object, pass it through.
    if hasattr(recognizers, "configs") and hasattr(recognizers, "embedding_dim"):
        return recognizers

    from embeddings_classifier.app import ClassifierConfigList

    normalized = validate_recognizers(recognizers, config_dir=config_dir)
    if not normalized:
        return []
    return ClassifierConfigList.from_any(normalized)


def resolve_recognizer_path(path_to_recognizer_json, config_dir=None):

    def check_recognizers_path(path):
        if not path.exists():
            return None
        if not path.is_file():
            return None
        return path
    
    path_to_recognizer_json = Path(path_to_recognizer_json).expanduser()

    resolved = None
    attempted = []
    if path_to_recognizer_json.is_absolute():
        resolved = check_recognizers_path(path_to_recognizer_json)
        attempted.append(path_to_recognizer_json)
    else:
        if config_dir is not None:
            candidate = Path(config_dir) / path_to_recognizer_json
            resolved = check_recognizers_path(candidate)
            attempted.append(candidate)
        if not resolved:
            resolved = check_recognizers_path(path_to_recognizer_json)
            attempted.append(path_to_recognizer_json)
        
    if not resolved:
        attempted = ", ".join(str(p) for p in attempted)
        raise ValueError(
            "recognizers string must be an existing JSON file path; "
            f"tried: {attempted}"
        )
    return resolved  

