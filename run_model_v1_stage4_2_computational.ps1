param(
    [string]$ProjectRoot = ".",
    [switch]$Diagnostic
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path -LiteralPath $ProjectRoot).Path
Set-Location $Root
$Config = if ($Diagnostic) { "configs/model_v1/stage4_2_diagnostic.yaml" } else { "configs/model_v1/stage4_2.yaml" }
$WslPython = "python3"
$WslRoot = (& wsl.exe -d Ubuntu-22.04 -- wslpath -a ($Root -replace '\\','/')).Trim()
if (-not $WslRoot) { throw "Could not resolve the MEDIROAD package path in WSL." }
$WslConfig = "$WslRoot/$($Config -replace '\\','/')"
& wsl.exe -d Ubuntu-22.04 -- env `
    PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONPATH="$WslRoot/src" `
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 `
    $WslPython "$WslRoot/12_scripts/v6/run_model_v1_stage4_2.py" `
    --project-root $WslRoot --config $WslConfig computational
if ($LASTEXITCODE -ne 0) { throw "Stage 4.2 computational run failed with exit code $LASTEXITCODE" }
