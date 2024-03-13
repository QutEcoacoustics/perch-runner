import pytest
from src.config import load_config , config_locations

# Assume test configs are in a subdirectory of the current test directory named 'test_configs'

TEST_CONFIGS_DIR = "tests/files/configs/"

# Fixture to modify locations
@pytest.fixture(autouse=True)
def modify_locations():
    """
    Automatically use this fixture in every test to modify the global
    locations variable to include only the temporary test directory.
    """
    # Store the original locations
    original_locations = config_locations.copy()
    
    # Update locations to only include a temporary directory for testing
    config_locations.clear()
    config_locations.append(TEST_CONFIGS_DIR)
    config_locations.append("")

    # This part runs after the test has finished
    yield

    # Reset locations to its original state
    config_locations.clear()
    config_locations.extend(original_locations)


def test_load_simple_config():
    config = load_config(f"{TEST_CONFIGS_DIR}base_config.yml")
    assert config.parameter1 == 'value1'
    assert config.parameter2 == 'value2'
    assert config.parameter3 == 'value3'

def test_load_extended_config():
    config = load_config(f"{TEST_CONFIGS_DIR}extend.yml")
    assert config.parameter1 == 'value1'
    assert config.parameter2 == 'overridden_value2'
    assert config.parameter3 == 'overridden_value3'
    assert config.parameter4 == 'value4'
    assert config.parameter5 == 'value5'


def test_load_extended_again_config():
    config = load_config(f"extend_again.yml")
    assert config.parameter1 == 'value1'
    assert config.parameter2 == 'overridden_value2'
    assert config.parameter3 == 'value3'
    assert config.parameter5 == 'value5'
    assert 'inherit' not in config


def test_circular_reference_detection():
    with pytest.raises(ValueError, match=r"circular reference detected"):
        load_config(f"{TEST_CONFIGS_DIR}circular_a.yml")


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_file.yml")