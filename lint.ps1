param(
    [switch]$Interactive = $true
)

# Stop on error
$ErrorActionPreference = "Stop"

function Write-Step {
    param([Parameter(Mandatory=$true)][string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Run-CodeFormatters {
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
    # Adjust the --target-version as appropriate for your Python code
    black --target-version py39 $TargetFolder
    if ($LASTEXITCODE -ne 0) {
        throw "black failed on $TargetFolder"
    }
}

try {
    Write-Host "Starting minimal code formatting...`n" -ForegroundColor Blue

    # 1) Basic Formatting on DLpy folder
    Run-CodeFormatters -TargetFolder "DLpy/"

    if ($Interactive) {
        $proceed = Read-Host "`nFormatting complete. Press y to confirm changes (y/n)"
        if ($proceed -ne "y") {
            Write-Host "Stopping here. No further actions." -ForegroundColor Yellow
            exit 0
        }
    }

    Write-Host "`nAll done! The code is now formatted and cleaned." -ForegroundColor Green
}
catch {
    Write-Host "An error occurred: $($_.Exception.Message)" -ForegroundColor Red
}
