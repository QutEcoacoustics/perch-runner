import re
from pathlib import Path


def compile_source_pattern(pattern: str) -> re.Pattern:
    """Compile a user-supplied regex, raising ValueError on invalid patterns."""
    try:
        return re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid source_map_pattern: {e}") from e


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
