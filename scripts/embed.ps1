# Argument Parsing
$source = $args[0]
$output = $args[1]
$image = $args[2]
if ($null -eq $image) { $image = "qutecoacoustics/perchrunner:latest" }

# Required Parameter Validation
if ([string]::IsNullOrWhiteSpace($source) -or [string]::IsNullOrWhiteSpace($output)) {
    Write-Host "Error: Missing required parameters (source, output)"
    exit 1
}

Write-Host (Get-Location)
Write-Host $source

# Source Path Checks
if (-not (Test-Path -Path $source)) {
    Write-Host "Error: Source audio folder does not exist: $source"
    exit 1
}

# Paths to things inside the container, to be mounted
$source_container = "/mnt/input"
$output_container = "/mnt/output"

$source_folder_host = Split-Path -Path $source -Parent
$source_basename = Split-Path -Path $source -Leaf

$command = "python /app/src/app.py generate --source $source_container/$source_basename --output $output_container"

Write-Host "launching container with command: $command"

# Convert to absolute paths
$absolute_source = (Resolve-Path -Path $source_folder_host).Path
$absolute_output = (Resolve-Path -Path $output).Path

$source_volume = "`"${absolute_source}:${source_container}`""
$output_volume = "`"${absolute_output}:${output_container}`""

$dockerCommand = "docker run --user appuser:appuser --rm -v $source_volume -v $output_volume $image $command"

Write-Host "Docker command: $dockerCommand" # For debugging

Invoke-Expression $dockerCommand