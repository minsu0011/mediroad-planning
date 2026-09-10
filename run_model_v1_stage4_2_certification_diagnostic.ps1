[CmdletBinding()]
param(
    [string]$ProjectRoot = (Get-Location).Path,
    [string]$CondaEnv = "mediroad-stage4-2c",
    [switch]$UseWsl
)

$runner = Join-Path $PSScriptRoot "run_model_v1_stage4_2_certification.ps1"
& $runner `
    -ProjectRoot $ProjectRoot `
    -Config "configs/model_v1/stage4_2_certification_diagnostic.yaml" `
    -CondaEnv $CondaEnv `
    -UseWsl:$UseWsl
exit $LASTEXITCODE
