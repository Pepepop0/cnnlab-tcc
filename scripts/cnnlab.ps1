# Starts the cnnlab-tcc interfaces without typing paths, environment variables or JSON.
# Called by the launchers in scripts\windows (start_tui.bat, start_streamlit.bat, ...).
# ASCII only on purpose: Windows PowerShell 5.1 misreads UTF-8 files without a BOM.
param(
    [ValidateSet("tui", "view")] [string]$Interface = "tui",
    [string]$Mode = "fp32",
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
if (-not $Mode) { $Mode = "fp32" }

$Repo = Split-Path -Parent $PSScriptRoot
$Cli = Join-Path $Repo ".venv\Scripts\cnnlab-tcc.exe"
$RunsRoot = if ($env:CNNLAB_TCC_RUNS_ROOT) { $env:CNNLAB_TCC_RUNS_ROOT } else { Join-Path ([Environment]::GetFolderPath("MyDocuments")) "cnnlab-tcc-runs" }
$LastRunFile = Join-Path $RunsRoot "ultima-run.txt"
# Scheduled runs (queue tab and sleep mode) share one folder and one database.
$QueueDir = Join-Path $RunsRoot "fila"

function Invoke-Cli([string[]]$CliArgs) {
    if ($DryRun) { Write-Host ("[dry-run] cnnlab-tcc " + ($CliArgs -join " ")); return }
    & $Cli @CliArgs
}

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "[erro] $Message" -ForegroundColor Red
    exit 1
}

function Get-LastRun([switch]$NeedsDatabase) {
    if (-not (Test-Path $LastRunFile)) { Fail "Nenhuma run registrada ainda. Rode start_tui.bat primeiro." }
    $folder = (Get-Content $LastRunFile -Raw).Trim()
    if (-not (Test-Path (Join-Path $folder "spec.json"))) { Fail "A pasta da ultima run nao existe mais: $folder" }
    if ($NeedsDatabase -and -not (Test-Path (Join-Path $folder "db\tcc.sqlite3"))) {
        # The newest run was created but never prepared: fall back to the newest run that has a database.
        $withDb = Get-ChildItem $RunsRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { Test-Path (Join-Path $_.FullName "db\tcc.sqlite3") } |
            Sort-Object { (Get-Item (Join-Path $_.FullName "db\tcc.sqlite3")).LastWriteTime } -Descending |
            Select-Object -First 1
        if ($withDb) {
            Write-Host "[aviso] A ultima run ($folder) ainda nao foi preparada (tecle p na TUI). Abrindo a run mais recente com dados: $($withDb.FullName)" -ForegroundColor Yellow
            return $withDb.FullName
        }
        Fail "A ultima run ($folder) ainda nao tem banco. Abra com start_tui.bat continuar e tecle p."
    }
    return $folder
}

function New-Run([string]$Slice) {
    New-Item -ItemType Directory -Force $RunsRoot | Out-Null
    $stamp = Get-Date -Format "yyyy-MM-dd-HHmm"
    $label = if ($Slice -eq "fp16_block") { "fp16" } else { "fp32" }
    $name = "run-$stamp-$label"
    $folder = Join-Path $RunsRoot $name
    $suffix = 2
    while (Test-Path $folder) { $folder = Join-Path $RunsRoot "$name-$suffix"; $suffix++ }
    New-Item -ItemType Directory -Force $folder | Out-Null

    # FP16 draws heavier networks (up to 6 min each): one generation and a 15 min GPU cap.
    if ($Slice -eq "fp16_block") { $generations = 1; $budget = 900 } else { $generations = 2; $budget = 3600 }
    $spec = @"
{"method": "joint", "seed_block": 0, "search_seed": 1001, "train_seed": 2001, "calibration_seed": 0,
 "budget_gpu_s": $budget, "code_label": "$(Split-Path -Leaf $folder)", "author": "Pedro Antunes",
 "factory": "cnnlab.tcc.demo_wiring:demo_wiring",
 "factory_args": {"train_per_class": 100, "epochs": 5, "population": 4, "generations": $generations, "slice": "$Slice"}}
"@
    [System.IO.File]::WriteAllText((Join-Path $folder "spec.json"), $spec, (New-Object System.Text.UTF8Encoding($false)))
    [System.IO.File]::WriteAllText($LastRunFile, $folder, (New-Object System.Text.UTF8Encoding($false)))
    return $folder
}

if (-not (Test-Path $Cli)) { Fail "Ambiente nao encontrado em $Repo\.venv. Rode: pip install -e .[tcc-ui,tcc-trt]" }

# The worker is started by the TUI and inherits these variables.
$env:CNNLAB_TCC_CIFAR_ROOT = Join-Path $Repo "data"
Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

if ($Interface -eq "view") {
    if ($Mode -eq "fila") {
        $folder = $QueueDir
        if (-not (Test-Path (Join-Path $folder "db\tcc.sqlite3"))) { Fail "A fila ainda nao rodou nenhuma run. Use start_tui.bat fila e tecle Rodar fila." }
    } else {
        $folder = if ($Mode -and $Mode -ne "fp32") { $Mode } else { Get-LastRun -NeedsDatabase }
    }
    $db = Join-Path $folder "db\tcc.sqlite3"
    Write-Host "Pagina de leitura da run: $folder"
    Write-Host "Abre em http://localhost:8501 . Para fechar: Ctrl+C."
    Invoke-Cli @("view", "--db", $db)
    exit $LASTEXITCODE
}

if ($Mode -eq "fila") {
    New-Item -ItemType Directory -Force $QueueDir | Out-Null
    Write-Host "Fila de runs: $QueueDir"
    Write-Host "Aba Fila: planeje runs e o sono, depois Rodar fila. A fila continua se a TUI fechar."
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $dirty = & git -C $Repo status --porcelain --untracked-files=no 2>$null
        if ($LASTEXITCODE -eq 0 -and $dirty) {
            Write-Host "[aviso] Ha arquivos versionados modificados. As runs serao gravadas com a marca de codigo modificado (dirty=True)." -ForegroundColor Yellow
        }
    } catch { }
    $ErrorActionPreference = $previous
    Invoke-Cli @("tui", "--db", (Join-Path $QueueDir "db\tcc.sqlite3"), "--artifacts", (Join-Path $QueueDir "artifacts"), "--queue-dir", $QueueDir)
    exit $LASTEXITCODE
}

$readOnly = $false
switch ($Mode.ToLower()) {
    "fp32" { $folder = New-Run "fp32" }
    "fp16" { $folder = New-Run "fp16_block" }
    "continuar" { $folder = Get-LastRun }
    "leitura" { $folder = Get-LastRun -NeedsDatabase; $readOnly = $true }
    default {
        if (Test-Path (Join-Path $Mode "db\tcc.sqlite3")) { $folder = (Resolve-Path $Mode).Path; $readOnly = $true }
        else { Fail "Opcao desconhecida: $Mode. Use fp32, fp16, fila, continuar, leitura ou o caminho de uma pasta de run." }
    }
}

$env:CNNLAB_TCC_WORK_DIR = Join-Path $folder "work"
$db = Join-Path $folder "db\tcc.sqlite3"
$art = Join-Path $folder "artifacts"
$spec = Join-Path $folder "spec.json"

# Warning only: a missing git or a folder outside the repository must not stop the interface.
if (-not $readOnly) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $dirty = & git -C $Repo status --porcelain --untracked-files=no 2>$null
        if ($LASTEXITCODE -eq 0 -and $dirty) {
            Write-Host "[aviso] Ha arquivos versionados modificados. A run sera gravada com a marca de codigo modificado (dirty=True)." -ForegroundColor Yellow
        }
    } catch { }
    $ErrorActionPreference = $previous
}

Write-Host "Pasta da run: $folder"
if ($readOnly) {
    Write-Host "Modo somente leitura: nada sera alterado."
    Invoke-Cli @("tui", "--db", $db, "--artifacts", $art, "--read-only")
} elseif (Test-Path $db) {
    Write-Host "Tecle s sobre um job para iniciar, retomar ou continuar."
    Invoke-Cli @("tui", "--db", $db, "--artifacts", $art, "--spec", $spec)
} else {
    Write-Host "Tecle p para preparar e depois s para iniciar. Pagina: start_streamlit.bat em outra janela."
    Invoke-Cli @("tui", "--db", $db, "--artifacts", $art, "--spec", $spec)
}
exit $LASTEXITCODE
