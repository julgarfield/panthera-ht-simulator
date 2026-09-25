$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Run setup.ps1 first to create the Python environment.'
}
& '.\.venv\Scripts\python.exe' main.py @args
exit $LASTEXITCODE
