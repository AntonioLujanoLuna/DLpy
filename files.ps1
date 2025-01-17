# Set the output file name
$OutputFile = "python_files_inventory.txt"

# Write a header to the file
"// Python Files Concatenated on $(Get-Date -Format 'MM/dd/yyyy HH:mm:ss')" | Set-Content $OutputFile
"// ----------------------------------------" | Add-Content $OutputFile
"" | Add-Content $OutputFile

# Function to process each Python file
function Process-File {
    param($FilePath)
    
    # Write file separator
    "" | Add-Content $OutputFile
    "// File: $FilePath" | Add-Content $OutputFile
    "// ----------------------------------------" | Add-Content $OutputFile
    
    # Copy file contents
    Get-Content $FilePath | Add-Content $OutputFile
}

# Find all Python files recursively, excluding venv directory
$PythonFiles = Get-ChildItem -Path . -Filter *.py -Recurse | 
    Where-Object { $_.FullName -notmatch 'venv' } |
    Sort-Object FullName

# Process each file
foreach ($File in $PythonFiles) {
    Process-File $File.FullName
}

# Print completion message
Write-Host "Inventory has been created in $OutputFile"
Write-Host "Found files:"
$PythonFiles | ForEach-Object { Write-Host $_.FullName }

# Add a count of files at the end
"" | Add-Content $OutputFile
"// ----------------------------------------" | Add-Content $OutputFile
"// Total Python files found: $($PythonFiles.Count)" | Add-Content $OutputFile