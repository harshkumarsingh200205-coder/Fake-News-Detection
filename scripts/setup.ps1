$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$venvDir = Join-Path $backendDir ".venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"

Write-Host "Setting up backend virtual environment..."
if (-not (Test-Path $pythonExe)) {
    python -m venv $venvDir
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r (Join-Path $backendDir "requirements.txt")

if (-not (Test-Path (Join-Path $backendDir ".env"))) {
    Copy-Item (Join-Path $backendDir ".env.example") (Join-Path $backendDir ".env")
}

Write-Host "Installing frontend dependencies..."
Push-Location $frontendDir
npm install
if (-not (Test-Path (Join-Path $frontendDir ".env.local"))) {
    Copy-Item (Join-Path $frontendDir ".env.local.example") (Join-Path $frontendDir ".env.local")
}
Pop-Location

Write-Host "Setup complete."
Write-Host "Next: run '.\scripts\dev.ps1' to launch backend and frontend."
