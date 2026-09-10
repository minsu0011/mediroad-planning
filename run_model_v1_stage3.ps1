param(
    [ValidateRange(1, 8)]
    [int]$Jobs = 8,
    [switch]$NoPromote,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WslPython = "python3"

if ($Python) {
    $Arguments = @((Join-Path $PackageRoot "run_model_v1_stage3.py"), "--jobs", "$Jobs")
    if (-not $NoPromote) { $Arguments += "--promote" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 failed with exit code $LASTEXITCODE" }
    exit 0
}

if (Get-Command wsl.exe -ErrorAction SilentlyContinue) {
    $WslRoot = (& wsl.exe wslpath -a ($PackageRoot -replace '\\','/')).Trim()
    $Arguments = @($WslPython, "$WslRoot/run_model_v1_stage3.py", "--jobs", "$Jobs")
    if (-not $NoPromote) { $Arguments += "--promote" }
    & wsl.exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 failed with exit code $LASTEXITCODE" }
    exit 0
}

$Arguments = @((Join-Path $PackageRoot "run_model_v1_stage3.py"), "--jobs", "$Jobs")
if (-not $NoPromote) { $Arguments += "--promote" }
& python @Arguments
if ($LASTEXITCODE -ne 0) { throw "Stage 3 failed with exit code $LASTEXITCODE" }
