from src import baw_utils

def test_parse_canonical_filename():
    """
    Tests various paths for the parse_canonical_filename function
    """

    filename = 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac'
    result = baw_utils.parse_canonical_filename(filename)
    assert result['arid'] == "909057"
    assert result['site'] == "Five-Rivers-Dry-A"
    assert result['timestamp'] == "20210428T100000+1000"
    assert result['ext'] == "flac"

    filename = '20210428T100000+1000_Five-Rivers-Dry-A_909057.flac'
    result = baw_utils.parse_canonical_filename(filename)
    assert result['arid'] == "909057"
    assert result['site'] == "Five-Rivers-Dry-A"
    assert result['timestamp'] == "20210428T100000+1000"
    assert result['ext'] == "flac"

    filename = 'folder_with_underscores/a_b/20210428T100000Z_Five-Rivers-Dry-A_909057.wav'
    result = baw_utils.parse_canonical_filename(filename)
    assert result['timestamp'] == "20210428T100000Z"
    assert result['ext'] == "wav"


    filename = '20210428T100000Z_Five_Rivers_Dry_A_12345.wav'
    result = baw_utils.parse_canonical_filename(filename)
    assert result['arid'] == "12345"
    assert result['site'] == "Five_Rivers_Dry_A"
    assert result['timestamp'] == "20210428T100000Z"
    assert result['ext'] == "wav"



def test_original_recording_url():
    """
    Tests various paths for the original_recording_url function
    """

    arid = "12345"
    result = baw_utils.original_recording_url(arid)
    assert result == "https://api.ecosounds.org/audio_recordings/12345/original"

    arid = "54321"
    result = baw_utils.original_recording_url(arid, "api.acousticobservatory.org")
    assert result == "https://api.acousticobservatory.org/audio_recordings/54321/original"


def test_recording_url_from_filename():
    """
    Tests various paths for the recording_url_from_filename function
    """

    filename = 'site_0277/20210428T100000+1000_Five-Rivers-Dry-A_909057.flac'
    result = baw_utils.recording_url_from_filename(filename)
    assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    filename = '20210428T100000+1000_Five-Rivers-Dry-A_909057.flac'
    result = baw_utils.recording_url_from_filename(filename)
    assert result == "https://api.ecosounds.org/audio_recordings/909057/original"

    filename = 'folder_with_underscores/a_b/20210428T100000Z_Five-Rivers-Dry-A_909057.wav'
    result = baw_utils.recording_url_from_filename(filename, "api.acousticobservatory.org")
    assert result == "https://api.acousticobservatory.org/audio_recordings/909057/original"

    filename = '20210428T100000Z_Five_Rivers_Dry_A_12345.wav'
    result = baw_utils.recording_url_from_filename(filename, "api.acousticobservatory.org")
    assert result == "https://api.acousticobservatory.org/audio_recordings/12345/original"