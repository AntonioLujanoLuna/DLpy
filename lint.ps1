param(
    [switch]$Interactive = $true
)

# Stop on error
$ErrorActionPreference = "Stop"

function Write-Step {
    param([Parameter(Mandatory=$true)][string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Run-CodeChecks {
    param(
        [Parameter(Mandatory=$true)][string]$TargetFolder
    )

    Write-Step "Autoflake - Remove Unused Imports/Variables"
    autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "Autoflake failed on $TargetFolder"
    }

    Write-Step "ISort - Sort Imports"
    isort $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "isort failed on $TargetFolder"
    }

    Write-Step "Black - Reformat Code"
    black --target-version py39 $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "black failed on $TargetFolder"
    }

    Write-Step "Mypy - Type Checking"
    mypy $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "mypy found type errors in $TargetFolder"
    }

    Write-Step "Flake8 - Style & Lint Checks"
    flake8 $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "flake8 found style or lint issues in $TargetFolder"
    }
}

try {
    Write-Host "Starting code checks and formatting...`n" -ForegroundColor Blue

    Run-CodeChecks -TargetFolder "DLpy/"

    if ($Interactive) {
        $proceed = Read-Host "`nChecks complete. Press y to confirm (y/n)"
        if ($proceed -ne "y") {
            Write-Host "Stopping here. No further actions." -ForegroundColor Yellow
            exit 0
        }
    }

    Write-Host "`nAll done! The code is now formatted, linted, and checked." -ForegroundColor Green
}
catch {
    Write-Host "An error occurred: $($_.Exception.Message)" -ForegroundColor Red
}
