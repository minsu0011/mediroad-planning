param(
    [Parameter(Mandatory=$true)][string]$FieldValidation,
    [Parameter(Mandatory=$true)][string]$TeamBases,
    [Parameter(Mandatory=$true)][string]$TeamCapability,
    [Parameter(Mandatory=$true)][string]$Vehicles,
    [Parameter(Mandatory=$true)][string]$Calendar,
    [Parameter(Mandatory=$true)][string]$VenueCalendar,
    [Parameter(Mandatory=$true)][string]$Travel,
    [string]$ProjectRoot = "."
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:PYTHONPATH = "$PWD\src"
$env:MPLBACKEND = "Agg"
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
python 12_scripts/v6/run_model_v1_stage4_2.py `
    --project-root . `
    --config configs/model_v1/stage4_2.yaml `
    operational-final `
    --field-validation $FieldValidation `
    --team-bases $TeamBases `
    --team-capability $TeamCapability `
    --vehicles $Vehicles `
    --calendar $Calendar `
    --venue-calendar $VenueCalendar `
    --travel $Travel
if ($LASTEXITCODE -ne 0) { throw "Stage 4.2 operational final failed with exit code $LASTEXITCODE" }
