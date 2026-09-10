[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$WslPython = 'python3'
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a ($Root -replace '\\','/')).Trim()
if (-not $WslRoot) { throw 'Could not resolve the MEDIROAD package path in WSL.' }

$ArgsList = @(
    '-d', 'Ubuntu-22.04', '--', 'env',
    'PYTHONDONTWRITEBYTECODE=1', 'MPLBACKEND=Agg',
    "PYTHONPATH=$WslRoot/src",
    'OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4',
    $WslPython,
    "$WslRoot/12_scripts/v6/run_model_v1_stage4_finalization.py",
    '--package-root', $WslRoot,
    '--config', "$WslRoot/configs/model_v1/stage4_finalization.yaml"
)

& wsl.exe @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "MEDIROAD Stage 4.1 finalization failed with exit code $LASTEXITCODE"
}
