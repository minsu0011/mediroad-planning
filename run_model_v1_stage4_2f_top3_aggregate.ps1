[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$Config = "configs/model_v1/stage4_2f_top3_aggregate.yaml",
    [string]$CondaEnv = "mediroad-stage4-2c",
    [switch]$NativeWindows,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"

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
if (-not $NativeWindows) {
    if ($CondaEnv -notmatch '^[A-Za-z0-9_.-]+$') { throw "Unsafe conda environment name: $CondaEnv" }
    $WslRoot = Convert-ToWslPath $ResolvedRoot
    $WslConfig = if ([System.IO.Path]::IsPathRooted($Config)) {
        Convert-ToWslPath $Config
    } else {
        $Config.Replace('\', '/')
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
    $Arguments = @(
        "--cd", $WslRoot,
        "env", "PYTHONPATH=$WslRoot/src", "PYTHONDONTWRITEBYTECODE=1", "PYTHONHASHSEED=42",
        "$WslPrefix/bin/python", "-m", "mediroad.stage4_2f_top3_aggregate",
        "--project-root", $WslRoot,
        "--stage42-config", "configs/model_v1/stage4_2.yaml",
        "--stage42c-config", "configs/model_v1/stage4_2_certification.yaml",
        "--stage42d-config", "configs/model_v1/stage4_2d_equity_certification.yaml",
        "--stage42e-config", "configs/model_v1/stage4_2e_equity_tail_certification.yaml",
        "--stage42f-config", $WslConfig
    )
    if ($DryRun) { Write-Output ($Arguments -join "`n"); exit 0 }
    & wsl.exe @Arguments
    exit $LASTEXITCODE
}

Set-Location -LiteralPath $ResolvedRoot
$env:PYTHONPATH = "$PWD\src"
$Arguments = @(
    "run", "-n", $CondaEnv, "--no-capture-output", "python", "-m",
    "mediroad.stage4_2f_top3_aggregate",
    "--project-root", "$PWD",
    "--stage42-config", "configs/model_v1/stage4_2.yaml",
    "--stage42c-config", "configs/model_v1/stage4_2_certification.yaml",
    "--stage42d-config", "configs/model_v1/stage4_2d_equity_certification.yaml",
    "--stage42e-config", "configs/model_v1/stage4_2e_equity_tail_certification.yaml",
    "--stage42f-config", $Config
)
if ($DryRun) { Write-Output ($Arguments -join "`n"); exit 0 }
& conda @Arguments
exit $LASTEXITCODE
