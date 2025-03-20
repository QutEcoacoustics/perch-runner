param (
    [string]$analysis,
    [string]$source,
    [string]$output,
    [string]$recognizer,
    [string]$image = "qutecoacoustics/perchrunner:latest"
)

# Get mount_src from environment variable or default to false
$mount_src = if ($env:PERCH_RUNNER_MOUNT_SRC) { $env:PERCH_RUNNER_MOUNT_SRC } else { "false" }


# Required Parameter Validation
if (-not $analysis -or -not $source -or -not $output) {
    Write-Host "Error: Missing required parameters (--analysis, --source, --output)"
    exit 1
}

# Additional validation for 'classify' analysis
if ($analysis -eq "classify" -and -not $recognizer) {
    Write-Host "Error: Missing required parameter (--recognizer) for classify analysis"
    exit 1
}

if (-not (Test-Path $source)) {
    Write-Host "Error: Source path does not exist: $source"
    exit 1
}
elseif ((Test-Path $source -PathType Container) -and (-not (Get-ChildItem -Path $source))) {
    Write-Host "Error: Source directory is empty: $source"
    exit 1
}

# Initialize command with a default value
$config = ""

# Creating an associative array equivalent in PowerShell
$recognizer_configs = @{
    "pw" = "pw.classify.yml"
    "cgw" = "cgw.classify.yml"
    "mgw" = "mgw.classify.yml"
}

# Check if the recognizer argument has been provided
if ($recognizer -and $analysis -eq "classify") {
    # Check if the provided recognizer is supported and set the config variable
    if ($recognizer_configs.ContainsKey($recognizer)) {
        $config = $recognizer_configs[$recognizer]
        Write-Host "Using config file: $config"
        $config = "--config $config"
    }
    else {
        Write-Host "Recognizer $recognizer not supported"
        exit 1
    }
}

# Determine input container path
$input_container = "/mnt/input"
if (Test-Path $source -PathType Leaf) {
    $source_base_name = [System.IO.Path]::GetFileName($source)
    $input_container = "/mnt/input/$source_base_name"
}

$output_container = "/mnt/output"

$command = "python /app/src/app.py $analysis --source $input_container --output $output_container $config"

Write-Host "app command: $command"

# Convert to absolute paths
$absolute_host_source = (Resolve-Path $source).Path
$absolute_host_output = [System.IO.Path]::GetFullPath($output)

# Build mount options
$mount_options = "-v `"${absolute_host_source}:${input_container}`" -v `"${absolute_host_output}:${output_container}`""

# Add source directory mount if mount_src is true
Write-Host "mount_src: $mount_src"
if ($mount_src -eq "true") {
    $absolute_host_src = (Resolve-Path "./src").Path
    $mount_options = "$mount_options -v `"${absolute_host_src}:/app/src`""
}

$docker_command = "docker run --user appuser:appuser --rm $mount_options $image $command"

Write-Host "launching container with command: $docker_command"

# Use the mount_options variable in the docker run command
Invoke-Expression $docker_command