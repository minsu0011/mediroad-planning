[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$Config = "configs/model_v1/stage4_2e_equity_tail_certification.yaml",
    [string]$CondaEnv = "mediroad-stage4-2c",
    [switch]$UseWsl,
    [switch]$NativeWindows,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if ($UseWsl -and $NativeWindows) { throw "Choose either -UseWsl or -NativeWindows, not both." }

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$Value)
    $Normalized = $Value.Replace('\', '/')
    if ($Normalized -match '^([A-Za-z]):(/.*)$') {
        return "/mnt/$($Matches[1].ToLowerInvariant())$($Matches[2])"
    }
    $Translated = (& wsl.exe wslpath -a -- "$Normalized" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Translated)) {
        throw "Could not translate Windows path into WSL path: $Value`n$Translated"
    }
    return $Translated
}

$ResolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

# The validated HiGHS 1.15.1 WSL environment is the default. BLAS helpers stay
# single-threaded so the two direct HiGHS solves can use all eight CPU threads.
$EffectiveUseWsl = -not $NativeWindows
if ($EffectiveUseWsl) {
    if ($CondaEnv -notmatch '^[A-Za-z0-9_.-]+$') { throw "Unsafe conda environment name: $CondaEnv" }
    $WslRoot = Convert-ToWslPath $ResolvedRoot
    if ([System.IO.Path]::IsPathRooted($Config)) {
        $WslConfig = Convert-ToWslPath $Config
    }
    else {
        $WslConfig = $Config.Replace('\', '/')
    }

    $WslPrefix = $null
    foreach ($Prefix in @(
        "$env:MEDIROAD_WSL_HOME/miniforge3/envs/$CondaEnv",
        "$env:MEDIROAD_WSL_HOME/miniconda3/envs/$CondaEnv",
        "$env:MEDIROAD_WSL_HOME/anaconda3/envs/$CondaEnv"
    )) {
        & wsl.exe test -x "$Prefix/bin/python"
        if ($LASTEXITCODE -eq 0) { $WslPrefix = $Prefix; break }
    }
    if ([string]::IsNullOrWhiteSpace($WslPrefix)) { throw "Validated WSL environment not found: $CondaEnv" }

    $WslArgs = @(
        "--cd", $WslRoot,
        "env",
        "PYTHONPATH=$WslRoot/src",
        "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1", "BLIS_NUM_THREADS=1",
        "PYTHONHASHSEED=42", "CUDA_VISIBLE_DEVICES=0",
        "CUDA_PATH=$WslPrefix/targets/x86_64-linux",
        "LD_LIBRARY_PATH=$WslPrefix/targets/x86_64-linux/lib:$WslPrefix/lib",
        "$WslPrefix/bin/python", "-m", "mediroad.stage4_2e_equity_tail_certification",
        "--project-root", $WslRoot,
        "--stage42-config", "configs/model_v1/stage4_2.yaml",
        "--stage42c-config", "configs/model_v1/stage4_2_certification.yaml",
        "--stage42d-config", "configs/model_v1/stage4_2d_equity_certification.yaml",
        "--stage42e-config", $WslConfig
    )
    if ($DryRun) { Write-Output ($WslArgs -join "`n"); exit 0 }
    & wsl.exe @WslArgs
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
$env:CUDA_VISIBLE_DEVICES = "0"

& conda run -n $CondaEnv --no-capture-output python -m mediroad.stage4_2e_equity_tail_certification `
    --project-root "$PWD" `
    --stage42-config configs/model_v1/stage4_2.yaml `
    --stage42c-config configs/model_v1/stage4_2_certification.yaml `
    --stage42d-config configs/model_v1/stage4_2d_equity_certification.yaml `
    --stage42e-config $Config
exit $LASTEXITCODE
