$ErrorActionPreference = "Stop"
$PackageRoot = Resolve-Path $PSScriptRoot
$WslDistribution = if ($env:MEDIROAD_WSL_DISTRIBUTION) { $env:MEDIROAD_WSL_DISTRIBUTION } else { "Ubuntu-22.04" }
$WslPython = if ($env:MEDIROAD_WSL_PYTHON) { $env:MEDIROAD_WSL_PYTHON } else { "python3" }
$WslRoot = "/mnt/" + $PackageRoot.Drive.Name.ToLowerInvariant() + "/" + $PackageRoot.Path.Substring(3).Replace('\', '/')
$WslEnvPrefix = $WslPython.Substring(0, $WslPython.LastIndexOf('/bin/python'))

& wsl.exe -d $WslDistribution -- env `
    "PYTHONPATH=$WslRoot/src" `
    "MPLBACKEND=Agg" `
    "PROJ_DATA=$WslEnvPrefix/share/proj" `
    "GDAL_DATA=$WslEnvPrefix/share/gdal" `
    "OMP_NUM_THREADS=8" `
    "OPENBLAS_NUM_THREADS=8" `
    "PYTHONHASHSEED=42" `
    $WslPython "$WslRoot/run_model_v1.py" --package-root $WslRoot --bootstrap 1000 --n-jobs 8
if ($LASTEXITCODE -ne 0) {
    throw "MODEL V1 Stage-1 failed with exit code $LASTEXITCODE"
}
