param(
    [string]$ProjectRoot = ".",
    [string]$FieldValidation = "",
    [string]$FieldEvidence = "",
    [string]$OperationalInputDir = ""
)
$ErrorActionPreference = "Stop"
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$wslDistro = "Ubuntu-22.04"
$python = "python3"
& wsl.exe -d $wslDistro -- test -x $python
if ($LASTEXITCODE -ne 0) { throw "WSL Python not found in ${wslDistro}: $python" }
$wslRoot = (& wsl.exe -d $wslDistro -- wslpath -a -u ($resolvedRoot -replace '\\','/')).Trim()
if ($LASTEXITCODE -ne 0 -or -not $wslRoot) { throw "Could not convert project root to a WSL path" }
$arguments = @(
    "env",
    "PYTHONPATH=$wslRoot/src",
    "PYTHONDONTWRITEBYTECODE=1",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1",
    "NUMEXPR_NUM_THREADS=1",
    "CUDA_VISIBLE_DEVICES=0",
    $python,
    "$wslRoot/scripts/run_stage4_operational_final.py",
    "--project-root", $wslRoot,
    "--config", "$wslRoot/configs/model_v1/stage4_operational_final.yaml"
)
if ($FieldValidation) {
    $resolvedField = (Resolve-Path -LiteralPath $FieldValidation).Path -replace '\\','/'
    $field = (& wsl.exe -d $wslDistro -- wslpath -a -u $resolvedField).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $field) { throw "Could not convert field-validation path" }
    $arguments += @("--field-validation", $field)
}
if ($FieldEvidence) {
    $resolvedEvidence = (Resolve-Path -LiteralPath $FieldEvidence).Path -replace '\\','/'
    $evidence = (& wsl.exe -d $wslDistro -- wslpath -a -u $resolvedEvidence).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $evidence) { throw "Could not convert field-evidence path" }
    $arguments += @("--field-evidence", $evidence)
}
if ($OperationalInputDir) {
    $resolvedOperational = (Resolve-Path -LiteralPath $OperationalInputDir).Path -replace '\\','/'
    $op = (& wsl.exe -d $wslDistro -- wslpath -a -u $resolvedOperational).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $op) { throw "Could not convert operational-input path" }
    $arguments += @("--operational-input-dir", $op)
}
& wsl.exe -d $wslDistro -- @arguments
if ($LASTEXITCODE -ne 0) { throw "Stage 4 Operational Final readiness run failed with exit code $LASTEXITCODE" }
