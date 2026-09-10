param(
  [string]$ProjectRoot = '.',
  [bool]$UseWsl = $true,
  [string]$WslPython = $env:MEDIROAD_WSL_PYTHON,
  [string]$CondaEnv = 'mediroad-stage4-2c'
)
$ErrorActionPreference = 'Stop'

if ($UseWsl) {
  $resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
  $normalizedRoot = $resolvedRoot.Replace('\', '/')
  if ($normalizedRoot -notmatch '^([A-Za-z]):(/.*)$') {
    throw "Stage4.2G WSL wrapper requires a Windows drive path: $resolvedRoot"
  }
  $wslRoot = "/mnt/$($Matches[1].ToLowerInvariant())$($Matches[2])"
  if ($WslPython -notmatch '^(.*)/bin/python$') {
    throw "WslPython must end with /bin/python: $WslPython"
  }
  $wslPrefix = $Matches[1]
  $wslArgs = @(
    '--cd', $wslRoot,
    'env',
    "PYTHONPATH=$wslRoot/src",
    'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
    'NUMEXPR_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1', 'BLIS_NUM_THREADS=1',
    'PYTHONHASHSEED=42', 'CUDA_VISIBLE_DEVICES=0',
    "CUDA_PATH=$wslPrefix/targets/x86_64-linux",
    "LD_LIBRARY_PATH=$wslPrefix/targets/x86_64-linux/lib`:$wslPrefix/lib",
    $WslPython, '-m', 'mediroad.stage4_2g_candidate_expansion',
    '--project-root', $wslRoot
  )
  & wsl.exe @wslArgs
  if ($LASTEXITCODE -ne 0) { throw "Stage4.2G WSL execution failed with exit code $LASTEXITCODE" }
} else {
  $resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
  Set-Location -LiteralPath $resolvedRoot
  $env:PYTHONPATH = "$resolvedRoot\src"
  $env:OMP_NUM_THREADS = '1'; $env:OPENBLAS_NUM_THREADS = '1'
  $env:MKL_NUM_THREADS = '1'; $env:NUMEXPR_NUM_THREADS = '1'
  $env:VECLIB_MAXIMUM_THREADS = '1'; $env:BLIS_NUM_THREADS = '1'
  $env:PYTHONHASHSEED = '42'; $env:CUDA_VISIBLE_DEVICES = '0'
  & conda run -n $CondaEnv --no-capture-output python -m mediroad.stage4_2g_candidate_expansion --project-root $resolvedRoot
  if ($LASTEXITCODE -ne 0) { throw "Stage4.2G Windows execution failed with exit code $LASTEXITCODE" }
}
