# lint.ps1
$ErrorActionPreference = "Stop"

function Write-Step {
    param($Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

#
# This function fixes type annotation issues within a single Python file.
# Key points:
# - We import "NDArray" from "numpy.typing" (no brackets!)
# - We skip rewriting that line so it doesn't become NDArray[Any]
# - We then replace usage "np.NDArray" => "NDArray[Any]", "NDArray" => "NDArray[Any]" in other lines.
# - We also insert from typing import ... if we see e.g. List, Dict, Optional, etc.
#
function Fix-TypeAnnotations {
    param($file)

    $content = Get-Content $file -Raw
    $modified = $false

    # We'll insert these near the top if we see references to them
    $typing_imports = "from typing import Any, Optional, List, Tuple, Dict, Union, Callable"
    # Note: do not subscript with [Any] in the import line!
    $numpy_t_import = "from numpy.typing import NDArray"

    # 1) Check if we need the typing imports
    $typingNeeded = $false
    if ($content -match "(?<![\w\.])(Any|Optional|List|Dict|Union|Tuple|Callable|float|int|TypeVar|Generator)(?![\w\.])") {
        if (-not ($content -match [regex]::Escape($typing_imports))) {
            $typingNeeded = $true
        }
    }

    # 2) Check if we need "from numpy.typing import NDArray"
    $numpyNeeded = $false
    if ($content -match "(np\.NDArray|NDArray)") {
        if (-not ($content -match [regex]::Escape($numpy_t_import))) {
            $numpyNeeded = $true
        }
    }

    # 3) If we need either import, insert them after any existing import lines, or at the top
    if ($typingNeeded -or $numpyNeeded) {
        $lines = $content -split "`r?`n"
        $insertIndex = 0
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match "^(import|from)\s+") {
                $insertIndex = $i + 1
            } else {
                break
            }
        }

        if ($typingNeeded) {
            $lines = $lines[0..($insertIndex - 1)] + $typing_imports + $lines[$insertIndex..($lines.Count - 1)]
            $insertIndex++
        }
        if ($numpyNeeded) {
            $lines = $lines[0..($insertIndex - 1)] + $numpy_t_import + $lines[$insertIndex..($lines.Count - 1)]
        }

        $content = ($lines -join "`n")
        $modified = $true
    }

    # 4) Replace references to np.NDArray => NDArray[Any], NDArray => NDArray[Any]
    #    but skip lines that contain the actual import statement from numpy.typing.
    $lines2 = $content -split "`r?`n"
    for ($i = 0; $i -lt $lines2.Count; $i++) {
        $line = $lines2[$i]

        # If the line has "from numpy.typing import NDArray", skip rewriting it
        if ($line -match "^\s*from\s+numpy\.typing\s+import\s+NDArray") {
            continue
        }

        # Replace 'np.NDArray' with 'NDArray[Any]' if not already subscripted
        $regexNDArray1 = "np\.NDArray(?!\[)"
        if ($line -match $regexNDArray1) {
            $line = $line -replace $regexNDArray1, "NDArray[Any]"
            $modified = $true
        }

        # Replace 'NDArray' with 'NDArray[Any]' if not already subscripted
        $regexNDArray2 = "(?<!import\s+)NDArray(?!\[)"
        if ($line -match $regexNDArray2) {
            $line = $line -replace $regexNDArray2, "NDArray[Any]"
            $modified = $true
        }

        $lines2[$i] = $line
    }
    $content = $lines2 -join "`n"

    # 5) Fix missing type parameters for List, Dict, Tuple, Callable, etc.
    $genericFixes = @(
        @{ regex = [regex] "List\[(?!.*\])"; repl  = "List[Any" },
        @{ regex = [regex] "Dict\[(?!.*\])"; repl  = "Dict[Any, Any" },
        @{ regex = [regex] "Tuple\[(?!.*\])"; repl  = "Tuple[Any" },
        @{ regex = [regex] "Callable\[(?!.*\])"; repl  = "Callable[..., Any" },
        @{ regex = [regex] "(?<=:\s*)tuple\[(?!.*\])"; repl  = "Tuple[Any" },
        @{ regex = [regex] "(?<=:\s*)dict\[(?!.*\])";  repl  = "Dict[Any, Any" }
    )
    foreach ($gf in $genericFixes) {
        if ($content -match $gf.regex) {
            $content = $content -replace $gf.regex, $gf.repl
            $modified = $true
        }
    }

    # 6) Convert "def foo(x: int=None)" => "def foo(x: Optional[int]=None)"
    #    This is a naive single-line regex; can break on multi-line defs
    $paramOptionalRegex = '(?<name>\w+)\s*:\s*(?<type>[\w\[\], ]+)\s*=\s*None'
    if ($content -match $paramOptionalRegex) {
        $content = [regex]::Replace($content, $paramOptionalRegex, {
            param($match)
            $typeText = $match.Groups['type'].Value.Trim()
            if ($typeText -notmatch '^Optional\[.*\]$') {
                return "$($match.Groups['name'].Value): Optional[$typeText] = None"
            }
            else {
                return $match.Value
            }
        })
        $modified = $true
    }

    # 7) Fix naive forward(...) overrides if we see them
    $patterns = @(
        @{
            find = "def forward\(self, \*args: Any, \*\*kwargs: Any\) -> None"
            replace = "def forward(self, x: Tensor) -> Tensor"
        },
        @{
            find = "@staticmethod\s+def forward\(ctx: Context, \*args: Any, \*\*kwargs: Any\) -> Tensor"
            replace = "@staticmethod`n    def forward(ctx: Context, x: Tensor) -> Tensor"
        }
    )
    foreach ($pattern in $patterns) {
        if ($content -match $pattern.find) {
            $content = $content -replace $pattern.find, $pattern.replace
            $modified = $true
        }
    }

    if ($modified) {
        $content | Set-Content $file -NoNewline
        Write-Host "Fixed type annotations in $file" -ForegroundColor Green
    }
}

function Get-RefurbSuggestions {
    $output = refurb DLpy/ 2>&1
    $fixes = @()

    foreach ($line in $output) {
        if ($line -match '([^:]+):(\d+):(\d+) \[([^\]]+)\]: Replace `([^`]+)` with `([^`]+)`') {
            $fixes += @{
                file = $matches[1]
                line = $matches[2]
                column = $matches[3]
                code = $matches[4]
                old = $matches[5]
                new = $matches[6]
            }
        }
    }
    return $fixes
}

function Apply-RefurbFixes {
    # Extra manual fixes
    $manualFixes = @(
        @{
            file = "DLpy\core\autograd.py"
            old  = "len(node.in_edges) == 0"
            new  = "not node.in_edges"
        },
        @{
            file = "DLpy\core\module.py"
            old  = 'isinstance(x, (..., type(None)))'
            new  = 'x is None or isinstance(x, ...)'
        },
        @{
            file = "DLpy\core\module.py"
            old  = 'name in ["training"]'
            new  = 'name == "training"'
        },
        @{
            file = "DLpy\core\serialization.py"
            old  = 'open(path, "wb")'
            new  = 'path.open("wb")'
        },
        @{
            file = "DLpy\core\serialization.py"
            old  = 'open(path, "rb")'
            new  = 'path.open("rb")'
        },
        @{
            file = "DLpy\core\tensor.py"
            old  = 'axes if axes else None'
            new  = 'axes or None'
        }
    )

    Write-Host "Getting new Refurb suggestions..." -ForegroundColor Gray
    $autoFixes = Get-RefurbSuggestions
    $fixes = $manualFixes + $autoFixes

    $processedFiles = @{}

    foreach ($fix in $fixes) {
        if (-not $processedFiles.ContainsKey($fix.file)) {
            Write-Host "Fixing $($fix.file)..." -ForegroundColor Gray
            $processedFiles[$fix.file] = $true
        }

        $content = Get-Content $fix.file -Raw
        if ($content -match [regex]::Escape($fix.old)) {
            $newContent = $content.Replace($fix.old, $fix.new)
            $newContent | Set-Content $fix.file -NoNewline
            Write-Host "Applied fix: $($fix.old) -> $($fix.new)" -ForegroundColor Green
        }
    }

    Write-Host "`nProcessed $($processedFiles.Count) files with $($fixes.Count) total fixes" -ForegroundColor Yellow
}

try {
    Write-Host "Starting code analysis and formatting...`n" -ForegroundColor Blue
    
    #
    # Stage 1: Basic formatting
    #
    Write-Step "Stage 1: Basic Formatting"
    Write-Host "Removing unused imports and variables..." -ForegroundColor Gray
    autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r DLpy/
    if ($LASTEXITCODE -ne 0) { throw "Autoflake failed" }

    Write-Host "Sorting imports..." -ForegroundColor Gray
    isort DLpy/
    if ($LASTEXITCODE -ne 0) { throw "isort failed" }

    Write-Host "Formatting with black..." -ForegroundColor Gray
    black DLpy/
    if ($LASTEXITCODE -ne 0) { throw "black failed" }

    $proceed = Read-Host "`nStage 1 complete. Check autograd.py for no_grad export. Proceed? (y/n)"
    if ($proceed -ne "y") {
        Write-Host "Stopping after Stage 1." -ForegroundColor Yellow
        exit 0
    }

    #
    # Stage 3: Run single test
    #
    Write-Step "Stage 3: Testing single module"
    Write-Host "Running test on tensor.py..." -ForegroundColor Gray
    python -m pytest tests/test_tensor.py -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Test failed. Please fix the issues before proceeding." -ForegroundColor Red
        exit 1
    }

    $proceed = Read-Host "`nSingle test complete. Proceed with full test suite? (y/n)"
    if ($proceed -ne "y") {
        Write-Host "Stopping after single test." -ForegroundColor Yellow
        exit 0
    }

    #
    # Stage 4: MonkeyType
    #
    Write-Step "Stage 4: Running MonkeyType type collection"
    monkeytype run -m pytest tests/
    if ($LASTEXITCODE -ne 0) { throw "MonkeyType test run failed" }

    Write-Step "Applying collected types"
    $modules = @("core", "nn", "ops", "utils")
    foreach ($module in $modules) {
        Write-Host "Processing DLpy.$module" -ForegroundColor Gray
        monkeytype apply "DLpy.$module"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "No traces or monkeytype apply failed for $module (may be normal)."
        }
    }

    #
    # Apply refurb suggestions
    #
    Write-Step "Applying Refurb fixes"
    Apply-RefurbFixes

    Write-Host "Re-running formatters after fixes..." -ForegroundColor Gray
    isort DLpy/
    black DLpy/

    #
    # Type fixes with our function, then mypy checks
    #
    Write-Step "Running type fixes and checks"
    Get-ChildItem -Path "DLpy" -Recurse -Filter "*.py" | ForEach-Object {
        Fix-TypeAnnotations $_.FullName
    }

    Write-Host "`nRunning basic type checks..." -ForegroundColor Gray
    mypy --ignore-missing-imports DLpy/
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Basic type checks passed, running with stricter options..." -ForegroundColor Green
        mypy --strict-optional --warn-redundant-casts --warn-unused-ignores --warn-return-any DLpy/
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Stricter checks passed, running with full enforcement..." -ForegroundColor Green
            mypy --strict-optional --warn-redundant-casts --warn-unused-ignores --warn-return-any --disallow-untyped-defs DLpy/
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Type checking failed. Fix the above errors before proceeding." -ForegroundColor Yellow
        $proceed = Read-Host "Would you like to continue with the remaining checks? (y/n)"
        if ($proceed -ne "y") {
            throw "mypy failed"
        }
    }

    Write-Step "Running style checks with flake8"
    flake8 DLpy/
    if ($LASTEXITCODE -ne 0) { throw "flake8 failed" }

    Write-Host "`nCode analysis and formatting complete! ✨" -ForegroundColor Green
}
catch {
    Write-Host "`nError: $_" -ForegroundColor Red
    exit 1
}
