import os
import subprocess
from pathlib import Path
import shutil
import pytest

scripts = [
    ["./scripts/analyze.sh"],
    ["pwsh", "-ExecutionPolicy", "Bypass", "-File",  "./scripts/analyze.ps1"]
]


#image = "pr10:latest"

def run_analyze_script(args, script_type=0):


    command = scripts[script_type] + args
    env_vars = {"PERCH_RUNNER_MOUNT_SRC": "true"}
    env = os.environ.copy()
    env.update(env_vars) 
    env_prefix = " ".join([f"{key}={value}" for key, value in env_vars.items()])
    print(f'running command: {env_prefix} {" ".join(command)}')
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    return result


def run_container(input_folder = None,
                  input_file = None,
                  output_folder = None,
                  output_file = None,
                  analysis = None,
                  config = None):
    """
    Launches a docker container with the given inputs outputs config analysis and recognizer
    """

    # perch runner can accept input folder or input file, 
    # and output folder or output file. 
    # we have a helper script which, for simplicity, only works with input folder and output folder
    # so, we want to test invokind perch runner with an explicit docker run command or via the helper script
    # but only for some tests via the helper script

 
    image = "qutecoacoustics/perchrunner:latest"

    mount = lambda source, dest: ["-v", f"{str(Path(source).absolute())}:{dest}"]

    mounts = []
     
    input_mount_container = f"/mnt/input"
    if input_folder is not None:
        mounts += mount(input_folder, input_mount_container)
        input_arg = input_mount_container
    elif input_file is not None:
        mounts += mount(str(Path(input_file).parent), input_mount_container)
        input_arg = input_mount_container + "/" + Path(input_file).name
    else:
        raise ValueError("input_folder or input_file must be provided")
    
    output_mount_container = f"/mnt/output"
    if output_folder is not None:
        mounts += mount(output_folder, output_mount_container)
        output_arg = input_mount_container
    elif output_file is not None:
        mounts += mount(str(Path(output_file).parent), output_mount_container)
        output_arg = output_mount_container + "/" + Path(output_file).name
    else:
        raise ValueError("input_folder or input_file must be provided")

    if config is None:
        config_arg = []
    else:
        mounts  += mount(Path(config).parent, "/mnt/config")
        config_arg = ["--config", "/mnt/config/" + Path(config).name]

    mounts += mount("./src", "/app/src")

    command=["python", "/app/src/app.py", analysis, "--source", input_arg, "--output", output_arg] + config_arg

    command = ["docker", "run", "--rm"] + mounts + [image] + command
    print(f'running command: {" ".join(command)}')
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result




parametrize_script_type = pytest.mark.parametrize("script_type", [0,1])

@parametrize_script_type
def test_analyze_script_classify(script_type):

    one = Path("tests/input/files/one")
    two = Path("tests/input/files/two")
    one.mkdir(parents=True, exist_ok=True)
    two.mkdir(parents=True, exist_ok=True)
    shutil.copy("tests/files/embeddings/100sec.parquet", one)
    shutil.copy("tests/files/embeddings/100sec.parquet", two)

    result = run_analyze_script(["classify", "./tests/input/", "./tests/output/", "mgw"], script_type)

    assert result.returncode == 0


    expected_output_files = ["./tests/output/files/one/100sec.csv", "./tests/output/files/two/100sec.csv"] 

    for file in expected_output_files:
        assert Path(file).exists()

@parametrize_script_type
def test_analysis_script_embed_file(script_type):

    run_analyze_script(["generate", "./tests/files/audio/100sec.wav", "./tests/output/"], script_type)
    expected_output_files = ["./tests/output/100sec.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()




@parametrize_script_type
def test_analysis_script_embed_flac_file(script_type):

    run_analyze_script(["generate", "./tests/files/audio/segment.flac", "./tests/output/"], script_type)
    expected_output_files = ["./tests/output/segment.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()

@parametrize_script_type
def test_analysis_script_embed_folder(script_type):

    run_analyze_script(["generate", "./tests/files/audio/", "./tests/output/"], script_type)
    expected_output_files = ["./tests/output/100sec.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()

# The following tests don't use the helper scripts, but instead use the docker run command directly

def test_container_embed_file_with_output_filename():
    # helper scripts don't support output files, only folders

    run_container(analysis="generate", input_file="./tests/files/audio/100sec.wav", output_file="./tests/output/myembeddings.parquet")
    expected_output_files = ["./tests/output/myembeddings.parquet"] 

    for file in expected_output_files:
        assert Path(file).exists()

    assert not Path("./tests/output/100sec.parquet").exists()


def test_analysis_script_embed_empty_folder():
    # helper scripts may have different behavior for empty folders

    result = run_container(analysis="generate", input_file="./tests/input/", output_file="./tests/output/")

    assert "no audio files found in /mnt/input" in result.stdout

    # Specify the expected output files
    expected_output_files = [] 

    for file in expected_output_files:
        assert Path(file).exists()
