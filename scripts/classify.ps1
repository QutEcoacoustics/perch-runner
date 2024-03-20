# Argument Parsing
$source = $args[0]
$output = $args[1]
$recognizer = $args[2]
$image = $args[3]
if ($null -eq $image) { $image = "qutecoacoustics/perchrunner:latest" }

# Required Parameter Validation
if ([string]::IsNullOrWhiteSpace($source) -or [string]::IsNullOrWhiteSpace($output) -or [string]::IsNullOrWhiteSpace($recognizer)) {
    Write-Host "Error: Missing required parameters (--source, --output, --recognizer)"
    exit 1
}

# Source Path Checks
if (-not (Test-Path -Path $source)) {
    Write-Host "Error: Source path does not exist: $source"
    exit 1
}
elseif ((Test-Path -Path $source -PathType Container) -and ((Get-ChildItem -Path $source).Count -eq 0)) {
    Write-Host "Error: Source directory is empty: $source"
    exit 1
}

# Output Folder Check
if (-not (Test-Path -Path $output -PathType Container)) {
    Write-Host "Error: Output folder does not exist: $output"
    exit 1
}

# Recognizer Config Mapping
$recognizer_configs = @{
    "pw" = "pw.classify.yml"
    "cgw" = "cgw.classify.yml"
}

$config_file = $recognizer_configs[$recognizer]
if ($null -eq $config_file) {
    Write-Host "Recognizer $recognizer not supported"
    exit 1
}
else {
    Write-Host "Using config file: $config_file"
}

# Paths inside the container, to be mounted
$embeddings_container = "/mnt/embeddings"
$output_container = "/mnt/output"
$output_dir = Join-Path $output_container "search_results"

$command = "python /app/src/app.py classify --source $embeddings_container --output $output_container --config_file $config_file"

# Convert to absolute paths
$absolute_source = (Resolve-Path -Path $source).Path
$absolute_output = (Resolve-Path -Path $output).Path

Write-Host "launching container with command: $command"

# Launch Docker container
docker run --user appuser:appuser --rm `
-v "$absolute_source":$embeddings_container `
-v "$absolute_output":$output_container $image $command