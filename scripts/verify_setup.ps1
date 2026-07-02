$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Error ".venv was not found. Run .\scripts\setup.ps1 after choosing a PyTorch build."
    exit 1
}

Write-Host "[1/3] Environment check"
& $VenvPython "scripts\check_environment.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Environment check failed."
    exit 1
}

Write-Host ""
Write-Host "[2/3] Smoke test"
& $VenvPython "scripts\smoke_test.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed."
    exit 1
}

Write-Host ""
Write-Host "[3/3] pytest"
& $VenvPython -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Error "pytest failed."
    exit 1
}

Write-Host ""
Write-Host "All setup verification steps passed."
