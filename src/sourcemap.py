"""Build sourcemap functions from preset regex/template definitions.

Sourcemaps rewrite the exported `source` value (for embeddings and recognizer
tables) from the original recording path into a user-defined destination
string, typically a URL.

The public entrypoint is `build_sourcemap(...)`.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Matches workbench canonical recording filenames, e.g.
#   20210428T100000Z_Five-Rivers-Dry-A_909057.flac
#   20210428T100000+1000_Site-Name_12345.wav
# Named groups: timestamp, site_name, audio_recording_id (audio recording id), extension.
_CANONICAL_FILENAME_PATTERN = re.compile(
    r'^(?P<timestamp>\d{8}T\d{6}(?:[+-]\d{4,6}|Z))_'  # Timestamp
    r'(?P<site_name>.+)_'                               # Site name (greedy, may contain underscores)
    r'(?P<audio_recording_id>\d+)\.'									# Audio recording id
    r'(?P<extension>.+)$'                               # Extension (without leading dot)
)

NAMED_SOURCEMAP_TEMPLATES = {
    "baw_original": "{domain}/audio_recordings/{audio_recording_id}/original", 
    "ecosounds_original": "https://api.ecosounds.org/audio_recordings/{audio_recording_id}/original",
    "a2o_original": "https://api.acousticsobservatory.org.au/audio_recordings/{audio_recording_id}/original",
}

# patterns for extracting metadata from filenames. 
NAMED_METADATA_PATTERNS = {
    "canonical_filename": _CANONICAL_FILENAME_PATTERN,
}

# extracting tokens in curly braces from templates, e.g. {audio_recording_id}
_TOKEN_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")



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
        raise ValueError("file_metadata must be a dictionary")
    normalized: dict[str, str] = {}
    for key, value in token_vals.items():
        if not isinstance(key, str) or not key:
            raise ValueError("file_metadata keys must be non-empty strings")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid sourcemap token key: {key}")
        if value is None:
            raise ValueError(f"file_metadata[{key}] may not be null")
        normalized[key] = str(value)
    return normalized


def _extract_template_tokens(template: str | None) -> set[str]:
    """Return the set of token names found in a destination template string.

    Tokens are bare identifiers wrapped in curly braces, e.g. ``{domain}``.
    Only names matching ``[A-Za-z_][A-Za-z0-9_]*`` are considered valid tokens;
    anything else is rejected later by ``_render_destination_template``.
    """
    if template is None:
        return set()
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
    return sorted(NAMED_SOURCEMAP_TEMPLATES.keys())


def get_file_metadata_pattern_preset_names() -> list[str]:
    """Return a sorted list of available sourcemap pattern preset names."""
    return sorted(NAMED_METADATA_PATTERNS.keys())


def _resolve_pattern_spec(pattern: str | re.Pattern[str]) -> re.Pattern[str]:
    """Resolve a pattern spec into a concrete regex string.

    Accepts a named pattern preset, a compiled regex or a raw regex string
    or a raw regex string.
    """

    # Handle strings (either a preset name or a raw regex string)
    if isinstance(pattern, str):
        if pattern in NAMED_METADATA_PATTERNS:
            return NAMED_METADATA_PATTERNS[pattern]
        # It's a raw regex string, so compile it
        try:
            return re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid source_map_pattern: {e}") from e

    # Handle already compiled patterns
    if isinstance(pattern, re.Pattern):
        return pattern

    # Catch invalid types passed from bad config files
    raise TypeError(
        f"Expected a string or compiled regex, but got {type(pattern).__name__}"
    )


@dataclass(frozen=True)
class SourcemapConfig:
    """Resolved sourcemap configuration used to build per-file mappers."""

    # a fixed dictionary of metadata name:value pairs to substitute into the destination template or extra columns
    file_metadata: dict[str, str]

    # a regex pattern string to extract additional metadata from the input filename
    file_metadata_pattern: re.Pattern[str] | None = None

    # a template to replace any 'source' column with string templated by file metadata (e.g. audio_recording_id)
    sourcemap_template: str | None = None
    


    @classmethod
    def from_inputs(
        cls,
        sourcemap_name: str | None = None,
        file_metadata: dict[str, Any] | None = None,
        sourcemap_template: str | None = None,
        file_metadata_pattern: str | re.Pattern[str] | None = None,
    ) -> "SourcemapConfig | None":
        """Resolve preset/overrides and return a validated SourcemapConfig.

        Returns None when no sourcemap options are configured.
        """
        if not sourcemap_name and sourcemap_template is None and file_metadata_pattern is None and file_metadata is None:
            return None


        if sourcemap_name and sourcemap_template:
            raise ValueError(
                "Cannot specify both sourcemap_name and sourcemap_template"
            )

        if sourcemap_name:
            sourcemap_template = NAMED_SOURCEMAP_TEMPLATES.get(sourcemap_name)
            if sourcemap_template is None:
                raise ValueError(
                    "Unknown sourcemap name: "
                    f"{sourcemap_name}. Available presets: {get_sourcemap_preset_names()}"
                )

        if sourcemap_template is not None:
            if not isinstance(sourcemap_template, str) or not sourcemap_template.strip():
                raise ValueError("sourcemap_template must be a non-empty string")

        if file_metadata_pattern is not None:
            file_metadata_pattern = _resolve_pattern_spec(file_metadata_pattern)

    
        file_metadata = _validate_token_vals(file_metadata)

        SourcemapConfig.validate_template(sourcemap_template, file_metadata_pattern, file_metadata)

        return cls(
            file_metadata=file_metadata,
            sourcemap_template=sourcemap_template,
            file_metadata_pattern=file_metadata_pattern,
        )


    @staticmethod
    def validate_template(sourcemap_template: str | None, pattern: re.Pattern[str] | None, static_metadata: dict[str, Any]) -> None:
        """Check that for the given template, all tokens can be resolved from either the static metadata or the pattern's named groups."""

        template_tokens = _extract_template_tokens(sourcemap_template)
        pattern_group_names = set(pattern.groupindex.keys()) if pattern is not None else set()

        missing_tokens = sorted(template_tokens - (set(static_metadata.keys()) | pattern_group_names))
        if missing_tokens:
            raise ValueError(
                "sourcemap is missing token values for: "
                f"{missing_tokens}. Provide them via file_metadata and/or named pattern groups."
            )


    def to_log_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for logging/debug metadata."""
        return {
            "sourcemap_template": self.sourcemap_template,
            "file_metadata_pattern": self.file_metadata_pattern,
            "file_metadata": dict(self.file_metadata),
        }

    def get_file_metadata(self, filename: str) -> dict[str, str]:
        """
        Return a dict of merged metadata values for the given filename, 
        combining static metadata and any extracted from the filename via the pattern.
        """
        resolved_tokens = dict(self.file_metadata)

        if self.file_metadata_pattern is not None:
            filestem = Path(filename).name
            match = self.file_metadata_pattern.search(filestem)
            if match is not None:
                for key, value in match.groupdict().items():
                    if value is not None:
                        resolved_tokens[key] = value

        return resolved_tokens

    def build_mapper(self) -> Callable[[str], str]:
        """Build a per-file mapping function from this resolved config."""

        if self.sourcemap_template is None:
            return lambda filename: filename  # identity function


        def mapper(filename: str) -> str:
            # copy of existing static metadata to which we will add any dynamically retrieved from the filename. 
            resolved_tokens = self.get_file_metadata(filename)
            result = _render_destination_template(self.sourcemap_template, resolved_tokens)

            if result is None:
                # fallback to identity if template rendering fails, probably due to missing token values.
                # this could happen if the pattern fails to match required token values and the static metadata does not provide them either.
                logging.warning(
                    "Sourcemap template rendering failed for filename %r; returning original filename",
                    filename,
                )
                return filename  

            return result

        return mapper

    def build_extra_columns_mapper(self, column_names: list[str]) -> Callable[[str], dict[str, Any]]:
        """Build a per-file extra columns mapping function from this resolved config."""

        def extra_columns_mapper(filename: str) -> dict[str, Any]:
            resolved_metadata = self.get_file_metadata(filename)
            return {key: resolved_metadata[key] for key in column_names if key in resolved_metadata}

        return extra_columns_mapper


def build_sourcemap(sourcemap_config: SourcemapConfig | None) -> Callable[[str], str] | None:
    """Build a source-mapping callable from a resolved sourcemap config."""
    if sourcemap_config is None:
        return lambda filename: filename  # identity function
    return sourcemap_config.build_mapper()


def build_extra_columns_map(sourcemap_config: SourcemapConfig | None, column_names: list[str]) -> Callable[[str], dict[str, Any]]:
    """Build a callable that returns extra columns values for a given filename based on the sourcemap config."""
    if sourcemap_config is None:
        return lambda filename: {}  # return an empty dict
    return sourcemap_config.build_extra_columns_mapper(column_names)