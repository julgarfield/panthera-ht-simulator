$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12, including the Python launcher, then retry.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' tools/fetch_assets.py --verify
if ($LASTEXITCODE -ne 0) { throw 'Official asset verification failed.' }
Write-Host 'Setup complete. Run .\run.ps1 or .\.venv\Scripts\python.exe main.py'
