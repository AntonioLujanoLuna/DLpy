# Ensure the script runs on PowerShell 5+ for compatibility
$PSVersion = $PSVersionTable.PSVersion.Major
if ($PSVersion -lt 5) {
    Write-Error "This script requires PowerShell version 5 or higher."
    exit
}

# Set the output file name
$OutputFile = "python_files_inventory.txt"

# Initialize StreamWriter with UTF8 encoding
$stream = [System.IO.StreamWriter]::new($OutputFile, $false, [System.Text.Encoding]::UTF8)

try {
    # Write header
    $header = "// Python Files Concatenated on $(Get-Date -Format 'MM/dd/yyyy HH:mm:ss')`n// ----------------------------------------`n`n"
    $stream.WriteLine($header)

    # Find all Python files recursively, excluding the venv directory
    $PythonFiles = Get-ChildItem -Path . -Filter *.py -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\venv\\' } |
        Sort-Object FullName

    # Process each file
    foreach ($File in $PythonFiles) {
        try {
            # Write file separator and information
            $stream.WriteLine("")
            $stream.WriteLine("// File: $($File.FullName)")
            $stream.WriteLine("// ----------------------------------------")

            # Initialize line counter
            $LineNumber = 1

            # Open StreamReader
            $reader = [System.IO.File]::OpenText($File.FullName)
            try {
                while (!$reader.EndOfStream) {
                    $line = $reader.ReadLine()
                    
                    # Safely concatenate the line number and the line content
                    $formattedLineNumber = "{0:D4}: " -f $LineNumber
                    $stream.WriteLine($formattedLineNumber + $line)
                    
                    $LineNumber++
                }
            }
            finally {
                $reader.Close()
            }
        }
        catch {
            # Log the error in Spanish as per original error messages
            $stream.WriteLine("// Error processing $($File.FullName): $_")
        }
    }

    # Write summary
    $stream.WriteLine("")
    $stream.WriteLine("// ----------------------------------------")
    $stream.WriteLine("// Total Python files found: $($PythonFiles.Count)")
}
finally {
    # Ensure the stream is closed
    $stream.Close()
}

# Print completion message
Write-Host "Inventory has been created in $OutputFile"
Write-Host "Found files:"
$PythonFiles | ForEach-Object { Write-Host $_.FullName }
