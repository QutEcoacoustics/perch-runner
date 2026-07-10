"""Validate and render templated output paths for embeddings and recognizers.

This module holds the default relative output templates, template-token
validation, template rendering, and path-safety checks used when writing both
embedding exports and recognizer result files.
"""

import json
import re
import warnings
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar

#DEFAULT_EMBEDDINGS_OUTPUT_PATH_TEMPLATE = "{parents}/{basename}/{analysis}{ext}"
#DEFAULT_RECOGNIZER_OUTPUT_PATH_TEMPLATE = "{classifier_name}/{parents}/{basename}/{analysis}{ext}"

DEFAULT_PATH_TYPE = "flat"

ALLOWED_OUTPUT_TEMPLATE_TOKENS = {
    "embeddings": frozenset({"parents", "basename", "ext", "embeddings_table_format", "analysis"}),
    "recognizers": frozenset({"classifier_name", "parents", "basename", "ext", "analysis"}),
}

# preset templates for output paths
OUTPUT_PATH_TYPE_TEMPLATES = {
    "nested_basename": "{parents}/{basename}{ext}",
    "flat_basename": "{basename}{ext}",
    "nested": "{parents}/{analysis}{ext}",
    "flat": "{analysis}{ext}",
}



_TEMPLATE_TOKEN_PATTERN = re.compile(r"\{([^{}]+)\}")



def validate_output_path_template(template, template_type):
    """Validate output path template token usage and basic path safety."""
    if not isinstance(template, str):
        raise ValueError("embeddings_output_path_template must be a string")
    
    allowed_tokens = ALLOWED_OUTPUT_TEMPLATE_TOKENS[template_type]

    candidate = template.strip()
    if not candidate:
        raise ValueError("embeddings_output_path_template cannot be empty")

    tokens = _TEMPLATE_TOKEN_PATTERN.findall(candidate)
    invalid_tokens = [t for t in tokens if t not in allowed_tokens]
    if invalid_tokens:
        raise ValueError(
            "Invalid token(s) in embeddings_output_path_template: "
            f"{invalid_tokens}. Allowed tokens are: {sorted(allowed_tokens)}"
        )

    normalized = candidate.replace("\\", "/")
    if normalized.startswith("/"):
        raise ValueError("embeddings_output_path_template must be relative (absolute paths are not allowed)")

    for part in Path(normalized).parts:
        if part == "..":
            raise ValueError("embeddings_output_path_template may not contain '..' path components")

    return candidate


def _ensure_relative_safe_path(path_obj):
    """Reject absolute / traversal paths for relative output paths."""
    if path_obj.is_absolute():
        raise ValueError("Output path must be relative")
    if any(part == ".." for part in path_obj.parts):
        raise ValueError("Output path may not contain '..' path components")


def render_output_relative_path(
        template,
        audio_file,
        analysis,
        ext,
        embeddings_table_format = None,
        recognizer_name = None
):
    """Render a relative output path from template tokens.
    Applies extension rules:
    - If {ext} is absent: append extension.
    - If rendered already ends with extension: keep it.
    - If rendered has a mismatching hardcoded extension: warn and append.
    """
    
    # bit of a hack to determine which template type we are rendering for, since the caller doesn't pass that in.
    # todo: why doesn't it pass it?
    template_type = "recognizers" if recognizer_name is not None else "embeddings"

    template = validate_output_path_template(template, template_type=template_type)

    audio_rel = Path(audio_file)

    _ensure_relative_safe_path(audio_rel)

    parents = "" if audio_rel.parent == Path(".") else audio_rel.parent.as_posix()
    basename = audio_rel.name

    if ext is not None:
        ext = ext if str(ext).startswith(".") else f".{ext}"

    def replace_val(token, value):
        token_placeholder = "{" + token + "}"
        if value is None:
            if token_placeholder in template:
                raise ValueError(f"Template contains token {token_placeholder} but no value is provided")
        else:
                return rendered.replace(token_placeholder, str(value))
        return rendered

    rendered = template

    rendered = replace_val("parents", parents)
    rendered = replace_val("basename", basename)
    rendered = replace_val("ext", ext)
    rendered = replace_val("embeddings_table_format", embeddings_table_format)
    rendered = replace_val("analysis", analysis)
    rendered = replace_val("classifier_name", recognizer_name)


    rendered = rendered.replace("\\", "/")
    while "//" in rendered:
        rendered = rendered.replace("//", "/")
    rendered = rendered.lstrip("/")

    current_suffix = Path(rendered).suffix

    if current_suffix != ext:
        # no ext placeholder and the template has a hardcoded exstension that does not match the output type. warn and append correct extension.
        # if the template doesn't have any extension, don't warn, just append the correct extension.
        if current_suffix and current_suffix != ext:
            warnings.warn(
                "Template contains a hardcoded extension that does not match the "
                f"output type ({current_suffix} vs {ext}); appending correct extension.",
                UserWarning,
            )
        rendered = f"{rendered}{ext}"

    rel_path = Path(rendered)
    _ensure_relative_safe_path(rel_path)
    return rel_path


def ensure_output_path_within_root(relative_path, output_root):
    """Ensure final output path remains inside output_root."""
    rel_path = Path(relative_path)
    _ensure_relative_safe_path(rel_path)

    output_root = Path(output_root).resolve()
    abs_path = (output_root / rel_path).resolve()
    abs_path.relative_to(output_root)
    return abs_path






def validate_and_resolve_template_config(config):
    """specified or default presets with specified or default templates."""


    # this will be used as the default where analysis type specific output path types are not provided
    output_path_type_val = config.get("output_path_type")

    def process_for_analysis_type(
            analysis_output_path_template, # e.g. embeddings_output_path_template or recognizer_output_path_template
            analysis_output_path_type, # e.g. embeddings_output_path_type or recognizer_output_path_type
            analysis_type, # e.g. embeddings or recognizers
            ):
        
        if config.get(analysis_output_path_template) is not None and config.get(analysis_output_path_type) is not None:
            # it doesn't make sense for the user to provide both a template string and a template type, 
            # since the types are just preset template strings
            raise ValueError(
                "Cannot specify both {} and {}".format(analysis_output_path_template, analysis_output_path_type)
            )
        
        # the general "output_path_type" is used as the default if provided, applying to all analysis types
        # this allows the user to set a single output path type for all analysis types
        if config["output_path_type"] is not None:
            default_path_type = config["output_path_type"]
        else:
            default_path_type = DEFAULT_PATH_TYPE

        if config.get(analysis_output_path_type) is None:
            config[analysis_output_path_type] = default_path_type

        if config.get(analysis_output_path_template) is None:
            config[analysis_output_path_template] = OUTPUT_PATH_TYPE_TEMPLATES[config[analysis_output_path_type]]

        # validates the template string
        validate_output_path_template(config[analysis_output_path_template], template_type=analysis_type) 


    # for each analysis type that uses templated output paths, resolve defaults and validate
    process_for_analysis_type("embeddings_output_path_template", "embeddings_output_path_type", "embeddings")
    process_for_analysis_type("recognizer_output_path_template", "recognizer_output_path_type", "recognizers")
 

