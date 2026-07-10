"""Build sourcemap functions from preset regex/template definitions.

Sourcemaps rewrite the exported `source` value (for embeddings and recognizer
tables) from the original recording path into a user-defined destination
string, typically a URL.

The public entrypoint is `build_sourcemap_from_preset(...)`.
"""

import re
from pathlib import Path
from typing import Any

# Matches workbench canonical recording filenames, e.g.
#   20210428T100000Z_Five-Rivers-Dry-A_909057.flac
#   20210428T100000+1000_Site-Name_12345.wav
# Named groups: timestamp, site_name, arid (audio recording id), extension.
_CANONICAL_FILENAME_PATTERN = re.compile(
    r'^(?P<timestamp>\d{8}T\d{6}(?:[+-]\d{4,6}|Z))_'  # Timestamp
    r'(?P<site_name>.+)_'                               # Site name (greedy, may contain underscores)
    r'(?P<arid>\d+)\.'									# Audio recording id
    r'(?P<extension>.+)$'                               # Extension (without leading dot)
)

sourcemap_presets = {
    "canonical_name_to_original_recording_url": {
        "pattern": _CANONICAL_FILENAME_PATTERN.pattern,
        "destination": "{domain}/audio_recordings/{arid}/original",
    },
    "original_recording_url": {
        "pattern": None,
        "destination": "{domain}/audio_recordings/{arid}/original",
    }
}


_TOKEN_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def compile_source_pattern(pattern: str) -> re.Pattern:
    """Compile a user-supplied regex, raising ValueError on invalid patterns."""
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid source_map_pattern: {e}") from e


def _validate_token_vals(token_vals: dict[str, Any] | None) -> dict[str, str]:
    """Validate and normalise a user-supplied token values dict.

    Keys must be non-empty identifiers (letters/digits/underscore, starting
    with a letter or underscore). Values are coerced to strings. None values
    are rejected because they would produce literal 'None' output.

    Returns a new dict with all values coerced to str, or an empty dict if
    token_vals is None.
    """
    if token_vals is None:
        return {}
    if not isinstance(token_vals, dict):
        raise ValueError("sourcemap_token_vals must be a dictionary")
    normalized: dict[str, str] = {}
    for key, value in token_vals.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sourcemap_token_vals keys must be non-empty strings")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid sourcemap token key: {key}")
        if value is None:
            raise ValueError(f"sourcemap_token_vals[{key}] may not be null")
        normalized[key] = str(value)
    return normalized


def _extract_template_tokens(template: str) -> set[str]:
    """Return the set of token names found in a destination template string.

    Tokens are bare identifiers wrapped in curly braces, e.g. ``{domain}``.
    Only names matching ``[A-Za-z_][A-Za-z0-9_]*`` are considered valid tokens;
    anything else is rejected later by ``_render_destination_template``.
    """
    return set(_TOKEN_PATTERN.findall(template))


def _render_destination_template(template: str, token_vals: dict[str, str]) -> str:
    """Substitute all ``{token}`` placeholders in template with resolved values.

    Raises ValueError if any token in the template has no matching entry in
    token_vals, or if any unrecognised brace syntax remains after substitution
    (which would indicate a malformed template rather than a missing token).
    """
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in token_vals:
            raise ValueError(f"Missing sourcemap token value for {{{token}}}")
        return token_vals[token]

    rendered = _TOKEN_PATTERN.sub(replace, template)
    if "{" in rendered or "}" in rendered:
        raise ValueError(
            "Invalid sourcemap destination template: unsupported brace syntax; "
            "use only simple {token_name} placeholders"
        )
    return rendered


def get_sourcemap_preset_names() -> list[str]:
    """Return a sorted list of available sourcemap preset names."""
    return sorted(sourcemap_presets.keys())


def build_sourcemap_from_preset(
    preset_name: str | None,
    token_vals: dict[str, Any] | None = None,
):
    """Build a callable sourcemap from a named preset and optional token overrides.

    Args:
        preset_name: Name of a preset from ``sourcemap_presets``. If None or
            empty, returns None (meaning no remapping is applied).
        token_vals: Optional dict of static token values merged into the preset
            destination template. Tokens extracted by the preset's regex take
            precedence for the current filename; static values fill the rest.

    Returns:
        A callable ``(filename: str) -> str`` that remaps source values, or
        None if no preset was specified.

    Raises:
        ValueError: if the preset name is unknown, a required token is missing,
            or token_vals contains invalid keys.
    """
    if not preset_name:
        return None

    preset = sourcemap_presets.get(preset_name)
    if preset is None:
        raise ValueError(
            "Unknown sourcemap_preset: "
            f"{preset_name}. Available presets: {get_sourcemap_preset_names()}"
        )

    destination = preset.get("destination")
    if not isinstance(destination, str) or not destination.strip():
        raise ValueError(f"sourcemap preset '{preset_name}' has an invalid destination template")

    pattern = None
    pattern_str = preset.get("pattern")
    if pattern_str:
        if not isinstance(pattern_str, str):
            raise ValueError(f"sourcemap preset '{preset_name}' has non-string pattern")
        pattern = compile_source_pattern(pattern_str)

    static_token_vals = _validate_token_vals(token_vals)
    template_tokens = _extract_template_tokens(destination)
    pattern_group_names = set(pattern.groupindex.keys()) if pattern is not None else set()

    missing_tokens = sorted(template_tokens - (set(static_token_vals.keys()) | pattern_group_names))
    if missing_tokens:
        raise ValueError(
            f"sourcemap preset '{preset_name}' is missing token values for: {missing_tokens}"
        )

    def mapper(filename: str) -> str:
        """Remap one source filename to a destination string.

        The preset pattern is matched against the basename of the path only,
        so directory components are ignored. If the pattern does not match,
        the original filename is returned unchanged (passthrough semantics).
        """
        resolved_tokens = dict(static_token_vals)

        if pattern is not None:
            basename = Path(filename).name
            match = pattern.search(basename)
            if match is None:
                return filename
            for key, value in match.groupdict().items():
                if value is not None:
                    resolved_tokens[key] = value

        return _render_destination_template(destination, resolved_tokens)

    return mapper


def apply_source_map(filename: str, pattern: re.Pattern, template: str) -> str:
    """Apply a regex pattern to a filename and fill a template with captured groups.

    Template placeholders are {0} for the full match, {1}, {2}, etc. for
    captured groups.  Only simple numeric references are supported — no
    attribute access, format specs, or expressions.

    Returns the original filename unchanged if the pattern does not match.
    """
    basename = Path(filename).name
    match = pattern.search(basename)
    if match is None:
        return filename

    result = template
    # Replace {0} with full match, {1}..{N} with captured groups.
    for i, value in enumerate([match.group(0)] + list(match.groups())):
        if value is not None:
            result = result.replace(f'{{{i}}}', value)

    return result


def create_sourcemap_function(pattern_str: str, template: str):
    """Create a sourcemap function from a regex pattern and template string."""
    pattern = compile_source_pattern(pattern_str)
    return lambda filename: apply_source_map(filename, pattern, template)
