import re
import urllib
from pathlib import Path

FILE_ID_TO_UID_PATTERN = re.compile(r".*_(\d+).[^\.]+$")

default_baw_domain = "api.ecosounds.org"


def parse_canonical_filename(
    filename: str
) -> str:
  """Construct an baw audio URL."""
  # Extract the recording UID. Example:
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac' -> 909057
  # 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.wav' -> 909057

  # get the basename
  filename = Path(filename).name

  # split by underscore
  parts = filename.split("_")

  if len(parts) < 2:
    return None

  # the first part is the timestamp
  timestamp = parts[0]

  # the last part is the file id and extension
  arid, ext = tuple(parts[-1].split("."))

  # everything inbetween is the site name
  site = "_".join(parts[1:-1])

  return {
    "site": site,
    "timestamp": timestamp,
    "arid": arid,
    "ext": ext
  }


def original_recording_url(
    arid,
    baw_domain=default_baw_domain
):
  """Construct an baw audio URL."""
  return f"https://{baw_domain}/audio_recordings/{arid}/original"


def recording_url_from_filename(
    filename: str,
    baw_domain=default_baw_domain
):
  """
  Construct an baw audio URL.
  filename: str: the filename of the audio recording in the format timestamp_site_arid.ext
  baw_domain: str: the domain of the BAW API
  If the filename is not in the correct format, the original filename is returned.  
  """

  result = parse_canonical_filename(filename)
  if result is None:
    return filename
  else:
    return original_recording_url(result["arid"], baw_domain=baw_domain)

