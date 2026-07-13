"""Build sourcemap functions from preset regex/template definitions.

Sourcemaps rewrite the exported `source` value (for embeddings and recognizer
tables) from the original recording path into a user-defined destination
string, typically a URL.

The public entrypoint is `build_sourcemap(...)`.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    "canonical_to_baw_original": {
        "pattern": _CANONICAL_FILENAME_PATTERN.pattern,
        "template": "{domain}/audio_recordings/{arid}/original",
    },
    "canonical_to_ecosounds_original": {
        "pattern": _CANONICAL_FILENAME_PATTERN.pattern,
        "template": "https://api.ecosounds.org/audio_recordings/{arid}/original",
    },
    "canonical_to_a2o_original": {
        "pattern": _CANONICAL_FILENAME_PATTERN.pattern,
        "template": "https://api.acousticsobservatory.org/audio_recordings/{arid}/original",
    },
    "baw_original": {
        "pattern": None,
        "template": "{domain}/audio_recordings/{arid}/original",
    }, 
    "ecosounds_original": {
        "pattern": None,
        "template": "https://api.ecosounds.org/audio_recordings/{arid}/original",
    },
    "a2o_original": {
        "pattern": None,
        "template": "https://api.acousticsobservatory.org.au/audio_recordings/{arid}/original",
    },
}

_PATTERN_PRESETS = {
    "canonical_filename": _CANONICAL_FILENAME_PATTERN.pattern,
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
        raise ValueError("sourcemap_values must be a dictionary")
    normalized: dict[str, str] = {}
    for key, value in token_vals.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sourcemap_values keys must be non-empty strings")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid sourcemap token key: {key}")
        if value is None:
            raise ValueError(f"sourcemap_values[{key}] may not be null")
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


def get_sourcemap_pattern_preset_names() -> list[str]:
    """Return a sorted list of available sourcemap pattern preset names."""
    return sorted(_PATTERN_PRESETS.keys())


def _resolve_pattern_spec(pattern_spec: str | None) -> str | None:
    """Resolve a pattern spec into a concrete regex string.

    Accepts a named pattern preset, sourcemap preset name with a pattern,
    or a raw regex string.
    """
    if pattern_spec is None:
        return None

    if pattern_spec in _PATTERN_PRESETS:
        return _PATTERN_PRESETS[pattern_spec]

    if pattern_spec in sourcemap_presets:
        preset_pattern = sourcemap_presets[pattern_spec].get("pattern")
        if preset_pattern is None:
            raise ValueError(
                f"sourcemap_pattern preset '{pattern_spec}' does not define a pattern"
            )
        return str(preset_pattern)

    return pattern_spec


def _preset_template(preset: dict[str, Any], preset_name: str) -> str:
    """Fetch and validate a preset template."""
    template = preset.get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"sourcemap preset '{preset_name}' has an invalid destination template")
    return template


@dataclass(frozen=True)
class SourcemapConfig:
    """Resolved sourcemap configuration used to build per-file mappers."""

    sourcemap_values: dict[str, str]
    sourcemap_template: str
    sourcemap_pattern: str | None = None

    @classmethod
    def from_inputs(
        cls,
        sourcemap: str | None = None,
        sourcemap_values: dict[str, Any] | None = None,
        sourcemap_template: str | None = None,
        sourcemap_pattern: str | None = None,
    ) -> "SourcemapConfig | None":
        """Resolve preset/overrides and return a validated SourcemapConfig.

        Returns None when no sourcemap options are configured.
        """
        if not sourcemap and sourcemap_template is None and sourcemap_pattern is None and sourcemap_values is None:
            return None

        resolved_template: str | None = None
        resolved_pattern_spec: str | None = None

        if sourcemap:
            preset = sourcemap_presets.get(sourcemap)
            if preset is None:
                raise ValueError(
                    "Unknown sourcemap: "
                    f"{sourcemap}. Available presets: {get_sourcemap_preset_names()}"
                )
            resolved_template = _preset_template(preset, sourcemap)
            preset_pattern = preset.get("pattern")
            resolved_pattern_spec = str(preset_pattern) if preset_pattern is not None else None

        if sourcemap_template is not None:
            if not isinstance(sourcemap_template, str) or not sourcemap_template.strip():
                raise ValueError("sourcemap_template must be a non-empty string")
            resolved_template = sourcemap_template

        if sourcemap_pattern is not None:
            if not isinstance(sourcemap_pattern, str) or not sourcemap_pattern.strip():
                raise ValueError("sourcemap_pattern must be a non-empty string")
            resolved_pattern_spec = _resolve_pattern_spec(sourcemap_pattern)

        if resolved_template is None:
            raise ValueError("sourcemap_template is required unless the selected sourcemap preset provides one")

        pattern = None
        if resolved_pattern_spec:
            pattern = compile_source_pattern(resolved_pattern_spec)

        static_token_vals = _validate_token_vals(sourcemap_values)
        template_tokens = _extract_template_tokens(resolved_template)
        pattern_group_names = set(pattern.groupindex.keys()) if pattern is not None else set()

        missing_tokens = sorted(template_tokens - (set(static_token_vals.keys()) | pattern_group_names))
        if missing_tokens:
            raise ValueError(
                "sourcemap is missing token values for: "
                f"{missing_tokens}. Provide them via sourcemap_values and/or named pattern groups."
            )

        return cls(
            sourcemap_values=static_token_vals,
            sourcemap_template=resolved_template,
            sourcemap_pattern=resolved_pattern_spec,
        )

    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for logging/debug metadata."""
        return {
            "sourcemap_template": self.sourcemap_template,
            "sourcemap_pattern": self.sourcemap_pattern,
            "sourcemap_values": dict(self.sourcemap_values),
        }

    def build_mapper(self) -> Callable[[str], str]:
        """Build a per-file mapping function from this resolved config."""
        pattern = compile_source_pattern(self.sourcemap_pattern) if self.sourcemap_pattern else None

        def mapper(filename: str) -> str:
            resolved_tokens = dict(self.sourcemap_values)

            if pattern is not None:
                basename = Path(filename).name
                match = pattern.search(basename)
                if match is None:
                    return filename
                for key, value in match.groupdict().items():
                    if value is not None:
                        resolved_tokens[key] = value

            return _render_destination_template(self.sourcemap_template, resolved_tokens)

        return mapper


def build_sourcemap(sourcemap_config: SourcemapConfig | None) -> Callable[[str], str] | None:
    """Build a source-mapping callable from a resolved sourcemap config."""
    if sourcemap_config is None:
        return None
    return sourcemap_config.build_mapper()
