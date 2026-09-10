param(
    [Parameter(Mandatory=$true)][string]$FieldValidation,
    [string]$ProjectRoot = ".",
    [string]$TeamBases = "",
    [string]$TeamCapability = "",
    [string]$Vehicles = "",
    [string]$Calendar = "",
    [string]$VenueCalendar = "",
    [string]$Travel = ""
)
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:PYTHONPATH = "$PWD\src"
$env:MPLBACKEND = "Agg"
$ArgsList = @(
    "12_scripts/v6/run_model_v1_stage4_2.py",
    "--project-root", ".",
    "--config", "configs/model_v1/stage4_2.yaml",
    "audit-field",
    "--field-validation", $FieldValidation
)
$Operational = @($TeamBases, $TeamCapability, $Vehicles, $Calendar, $VenueCalendar, $Travel)
if (($Operational | Where-Object { $_ -ne "" }).Count -gt 0) {
    if (($Operational | Where-Object { $_ -eq "" }).Count -gt 0) { throw "All six operational input paths must be supplied together." }
    $ArgsList += @(
        "--team-bases", $TeamBases,
        "--team-capability", $TeamCapability,
        "--vehicles", $Vehicles,
        "--calendar", $Calendar,
        "--venue-calendar", $VenueCalendar,
        "--travel", $Travel
    )
}
python @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Stage 4.2 field audit failed with exit code $LASTEXITCODE" }
