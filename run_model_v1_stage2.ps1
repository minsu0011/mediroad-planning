[CmdletBinding()]
param(
    [ValidateRange(1, 10000)]
    [int]$Bootstrap = 1000,
    [ValidateRange(1, 8)]
    [int]$Jobs = 8,
    [switch]$AllowGateFail
)

$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LinuxRoot = '.'
$Python = 'python3'
$RunLock = '/tmp/mediroad_model_v1_stage2.lock'

$Arguments = @(
    'run_model_v1_stage2.py',
    '--package-root', $LinuxRoot,
    '--bootstrap', $Bootstrap,
    '--jobs', $Jobs
)
if ($AllowGateFail) {
    $Arguments += '--allow-gate-fail'
}

Push-Location $PackageRoot
try {
    # An advisory OS lock prevents concurrent writers from mixing correlation,
    # ablation, report, and provenance generations in the same output tree.
    & wsl.exe flock --nonblock $RunLock $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MEDIROAD Stage 2 failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
