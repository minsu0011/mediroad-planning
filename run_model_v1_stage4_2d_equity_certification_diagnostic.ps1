[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$CondaEnv = "mediroad-stage4-2c",
    [switch]$UseWsl,
    [switch]$NativeWindows
)
$Script = Join-Path $PSScriptRoot "run_model_v1_stage4_2d_equity_certification.ps1"
$RunnerArgs = @{
    ProjectRoot = $ProjectRoot
    CondaEnv = $CondaEnv
    Config = "configs/model_v1/stage4_2d_equity_certification_diagnostic.yaml"
}
if ($NativeWindows) { $RunnerArgs.NativeWindows = $true } else { $RunnerArgs.UseWsl = $true }
& $Script @RunnerArgs
exit $LASTEXITCODE
