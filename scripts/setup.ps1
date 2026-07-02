param(
    [switch]$InstallTorchCpu
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $ProjectRoot
Write-Host "Project root: knowledge-distillation-research (full path omitted)"

$BasePythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $BasePythonCommand) {
    Write-Error "Python was not found. Install Python 3.11-3.13, then rerun this script."
    exit 1
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "[1/4] Creating project virtual environment..."
    & python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create .venv."
        exit 1
    }
} else {
    Write-Host "[1/4] Existing project .venv found."
}

Write-Host "[2/4] Updating pip inside .venv..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to update pip inside .venv."
    exit 1
}

& $VenvPython -c "import torch, torchvision" 2>$null
$TorchReady = $LASTEXITCODE -eq 0

if (-not $TorchReady) {
    if (-not $InstallTorchCpu) {
        Write-Host ""
        Write-Host "PyTorch is not installed in .venv."
        Write-Host "No build was selected automatically because CPU/GPU compatibility is a user decision."
        Write-Host "For the approved CPU-only setup, rerun:"
        Write-Host "  .\scripts\setup.ps1 -InstallTorchCpu"
        Write-Host "For a GPU build, use https://pytorch.org/get-started/locally/ after checking the NVIDIA driver."
        exit 2
    }

    Write-Host "[3/4] Installing the explicitly selected CPU-only PyTorch build..."
    & $VenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CPU-only PyTorch installation failed."
        exit 1
    }
} else {
    Write-Host "[3/4] PyTorch and TorchVision are already available in .venv."
}

Write-Host "[4/4] Installing the project and development dependencies..."
& $VenvPython -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Project installation failed."
    exit 1
}

Write-Host ""
Write-Host "Setup completed."
Write-Host "Activate the environment in this terminal with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then verify it with:"
Write-Host "  .\scripts\verify_setup.ps1"
