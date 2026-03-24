$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$pythonExe = Join-Path $backendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Backend virtual environment not found. Run .\scripts\setup.ps1 first."
}

Push-Location $backendDir
& $pythonExe -m pip install -r requirements-dev.txt
& $pythonExe -m pytest tests --basetemp=.pytest-tmp -o cache_dir=.pytest-tmp/.pytest_cache
Pop-Location

Push-Location $frontendDir
npm run lint
npm run typecheck
npm run build
Pop-Location

Write-Host "All checks passed."
