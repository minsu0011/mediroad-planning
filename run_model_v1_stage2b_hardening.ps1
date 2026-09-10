[CmdletBinding()]
param(
    [ValidateRange(1, 10000)]
    [int]$Bootstrap = 1000,
    [ValidateRange(1, 50000)]
    [int]$Permutation = 5000,
    [ValidateRange(1, 8)]
    [int]$Jobs = 8,
    [switch]$Diagnostic,
    [switch]$AllowGateFail
)

$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LinuxRoot = '.'
$Python = 'python3'
$RunLock = '/tmp/mediroad_model_v1_stage2b_hardening.lock'

$Arguments = @(
    'run_model_v1_stage2b_hardening.py',
    '--package-root', $LinuxRoot,
    '--bootstrap', $Bootstrap,
    '--permutations', $Permutation,
    '--jobs', $Jobs
)
if ($Diagnostic) {
    $Arguments += '--diagnostic'
}
if ($AllowGateFail) {
    $Arguments += '--allow-gate-fail'
}

Push-Location $PackageRoot
try {
    # A separate non-blocking lock preserves the frozen Stage 2 official tree
    # and prevents mixed-generation hardening outputs from concurrent writers.
    & wsl.exe flock --nonblock $RunLock $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "MEDIROAD Stage 2B hardening failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
