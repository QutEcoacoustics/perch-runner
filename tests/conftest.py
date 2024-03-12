# Note: I could only get the clear_output_files autouse fixture to work by defining in pytest_plugins, rather than by direct import. 
# pytest_plugins can only be used in the top level conftest.py file, not in any other nested conftest.py file.

pytest_plugins = [
  "tests.shared_fixtures.clear_output_files",
  "tests.shared_fixtures.helpers"
]



