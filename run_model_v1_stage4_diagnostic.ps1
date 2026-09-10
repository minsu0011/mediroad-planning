$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$WslPython = 'python3'
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a ($Root -replace '\\','/')).Trim()
$ArgsList = @(
    '-d', 'Ubuntu-22.04', '--', 'env',
    'PYTHONDONTWRITEBYTECODE=1', 'MPLBACKEND=Agg',
    "PYTHONPATH=$WslRoot/src",
    $WslPython, '-m', 'mediroad.stage4',
    '--package-root', $WslRoot,
    '--config', "$WslRoot/configs/model_v1/stage4.yaml",
    '--mode', 'diagnostic', '--solver', 'greedy',
    '--skip-network-validation', '--skip-frontier'
)
& wsl.exe @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Diagnostic Stage 4 failed: $LASTEXITCODE" }
