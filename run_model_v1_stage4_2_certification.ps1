[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$Config = "configs/model_v1/stage4_2_certification.yaml",
    [string]$CondaEnv = "mediroad-stage4-2c",
    [switch]$UseWsl
)

$ErrorActionPreference = "Stop"

function Quote-BashLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    if ($Value.Contains("'")) {
        throw "WSL path or argument contains an unsupported single quote: $Value"
    }
    return "'$Value'"
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    $normalized = $Value.Replace('\', '/')
    if ($normalized -match '^([A-Za-z]):(/.*)$') {
        return "/mnt/$($Matches[1].ToLowerInvariant())$($Matches[2])"
    }
    $translated = (& wsl.exe wslpath -a -- "$normalized" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($translated)) {
        throw "Could not translate Windows path into WSL path: $Value`n$translated"
    }
    return $translated
}

$ResolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if ($UseWsl) {
    $WslRoot = Convert-ToWslPath $ResolvedRoot
    if ([System.IO.Path]::IsPathRooted($Config)) {
        $WslConfig = Convert-ToWslPath $Config
    }
    else {
        $WslConfig = $Config.Replace('\', '/')
    }

    $WslPython = $null
    $WslPrefixes = @(
        "$env:MEDIROAD_WSL_HOME/miniforge3/envs/$CondaEnv",
        "$env:MEDIROAD_WSL_HOME/miniconda3/envs/$CondaEnv",
        "$env:MEDIROAD_WSL_HOME/anaconda3/envs/$CondaEnv"
    )
    foreach ($prefix in $WslPrefixes) {
        $candidate = "$prefix/bin/python"
        & wsl.exe test -x $candidate
        if ($LASTEXITCODE -eq 0) {
            $WslPython = $candidate
            $WslPrefix = $prefix
            break
        }
    }
    if ([string]::IsNullOrWhiteSpace($WslPython)) {
        throw "Could not find WSL conda environment: $CondaEnv"
    }

    $WslEnv = @(
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=$WslRoot/src",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "VECLIB_MAXIMUM_THREADS=1",
        "BLIS_NUM_THREADS=1",
        "PYTHONHASHSEED=42"
    )
    $CudaTarget = "$WslPrefix/targets/x86_64-linux"
    & wsl.exe test -d $CudaTarget
    if ($LASTEXITCODE -eq 0) {
        $WslEnv += "CUDA_PATH=$CudaTarget"
        $WslEnv += "LD_LIBRARY_PATH=$CudaTarget/lib`:$WslPrefix/lib"
    }
    $WslArgs = $WslEnv + @(
        $WslPython,
        "-m", "mediroad.stage4_2_certification",
        "--project-root", $WslRoot,
        "--stage42-config", "configs/model_v1/stage4_2.yaml",
        "--certification-config", $WslConfig
    )
    & wsl.exe env @WslArgs
    exit $LASTEXITCODE
}

Set-Location -LiteralPath $ResolvedRoot
$env:PYTHONPATH = "$PWD\src"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:PYTHONHASHSEED = "42"

$EnvPrefix = (& conda run -n $CondaEnv --no-capture-output python -c "import sys; print(sys.prefix)" | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($EnvPrefix)) {
    throw "Could not resolve conda environment prefix: $CondaEnv"
}
$CudaTarget = Join-Path $EnvPrefix "targets\x86_64-linux"
if (Test-Path -LiteralPath $CudaTarget -PathType Container) {
    $env:CUDA_PATH = $CudaTarget
    $env:PATH = "$EnvPrefix\Library\bin;$env:PATH"
}

& conda run -n $CondaEnv --no-capture-output python -m mediroad.stage4_2_certification `
    --project-root "$PWD" `
    --stage42-config configs/model_v1/stage4_2.yaml `
    --certification-config $Config
exit $LASTEXITCODE
