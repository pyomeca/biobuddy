param(
    [string]$EnvironmentName = "biobuddy"
)

$condaCommand = if ($env:CONDA_EXE) {
    $env:CONDA_EXE
} else {
    (Get-Command conda -ErrorAction SilentlyContinue).Source
}

if (-not $condaCommand) {
    throw "Conda was not found. Activate Miniconda/Anaconda or set CONDA_EXE before running this script."
}

& $condaCommand run -n $EnvironmentName python -m pytest -v --color=yes --cov-report=xml --cov=biobuddy tests
