# Runs Atelier's own decision eval (tests/decisionsEval.test.ts) against a System One server on this computer.
#   powershell -File tools/atelier_eval.ps1 -Url http://127.0.0.1:8011 -Model atelier-nano -Out analysis/results/atelier-eval-nano.txt
# Vitest must run from an upper-case C:\ path on Windows (from c:\ every suite fails to load).
param(
  [Parameter(Mandatory = $true)][string]$Url,
  [Parameter(Mandatory = $true)][string]$Model,
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Atelier = "C:\Users\senne\github\atelier"
)
$outPath = [System.IO.Path]::GetFullPath($Out)
Set-Location $Atelier
$env:ELECTRON_RUN_AS_NODE = $null
$env:ATELIER_DECISIONS_EVAL = 'custom'
$env:ATELIER_DECISIONS_URL = $Url
$env:ATELIER_DECISIONS_MODEL = $Model
$env:VITE_CONFIG_NATIVE_IGNORE_WARNING = 'true'
npx vitest run tests/decisionsEval.test.ts --silent=false 2>&1 | Out-File -Encoding utf8 $outPath
Get-Content $outPath | Select-String -Pattern "rules alone|labeling|wrong:|Tests |FAIL" | ForEach-Object { $_.Line.Trim() }
