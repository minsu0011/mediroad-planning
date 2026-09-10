[CmdletBinding()]
param(
    [ValidateSet('official','diagnostic')]
    [string]$Mode = 'official',

    [ValidateSet('cp-sat','greedy')]
    [string]$Solver = 'cp-sat',

    [switch]$Force,
    [switch]$SkipFrontier,
    [switch]$SkipNetworkValidation
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$WslPython = 'python3'
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a ($Root -replace '\\','/')).Trim()
$argsList = @(
    '-d', 'Ubuntu-22.04', '--', 'env',
    'PYTHONDONTWRITEBYTECODE=1', 'MPLBACKEND=Agg',
    "PYTHONPATH=$WslRoot/src",
    'OMP_NUM_THREADS=8', 'OPENBLAS_NUM_THREADS=8', 'MKL_NUM_THREADS=8',
    $WslPython, '-m', 'mediroad.stage4',
    '--package-root', $WslRoot,
    '--config', "$WslRoot/configs/model_v1/stage4.yaml",
    '--mode', $Mode,
    '--solver', $Solver
)
if ($Force) { $argsList += '--force' }
if ($SkipFrontier) { $argsList += '--skip-frontier' }
if ($SkipNetworkValidation) { $argsList += '--skip-network-validation' }

& wsl.exe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MEDIROAD Stage 4 failed with exit code $LASTEXITCODE"
}
