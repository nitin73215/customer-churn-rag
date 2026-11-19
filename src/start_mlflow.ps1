[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 5000,

    [string]$ListenAddress = "127.0.0.1"
)

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDirectory

$mlflowExecutable = Join-Path $projectRoot ".venv\Scripts\mlflow.exe"
$mlflowDatabasePath = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot "mlops\mlflow.db")
)
$mlflowArtifactPath = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot "mlops\mlflow_artifacts")
)

if (-not (Test-Path -LiteralPath $mlflowExecutable -PathType Leaf)) {
    throw "MLflow was not found at '$mlflowExecutable'. Install MLflow in the project-root .venv first."
}

# Refuse to silently create a new, empty database at an unexpected path.
if (-not (Test-Path -LiteralPath $mlflowDatabasePath -PathType Leaf)) {
    throw "The expected tracking database does not exist: '$mlflowDatabasePath'."
}

if (-not (Test-Path -LiteralPath $mlflowArtifactPath -PathType Container)) {
    New-Item -ItemType Directory -Path $mlflowArtifactPath -Force | Out-Null
}

$listeners = @(
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
)

if ($listeners.Count -gt 0) {
    $serverRootPids = @()

    $listenerDetails = foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process `
            -Filter "ProcessId = $($listener.OwningProcess)" `
            -ErrorAction SilentlyContinue

        if ($null -ne $process) {
            $serverRoot = $process
            $parent = Get-CimInstance Win32_Process `
                -Filter "ProcessId = $($serverRoot.ParentProcessId)" `
                -ErrorAction SilentlyContinue

            while (
                $null -ne $parent -and
                $parent.CommandLine -match '(?i)(mlflow|uvicorn)' -and
                $parent.CommandLine -like "*$projectRoot*"
            ) {
                $serverRoot = $parent
                $parent = Get-CimInstance Win32_Process `
                    -Filter "ProcessId = $($serverRoot.ParentProcessId)" `
                    -ErrorAction SilentlyContinue
            }

            $serverRootPids += [int]$serverRoot.ProcessId

            "Listener PID $($process.ProcessId), server-tree root PID $($serverRoot.ProcessId): $($process.Name) $($process.CommandLine)"
        }
        else {
            "PID $($listener.OwningProcess): process details unavailable"
            $serverRootPids += [int]$listener.OwningProcess
        }
    }

    $details = $listenerDetails -join [Environment]::NewLine
    $stopCommands = @(
        $serverRootPids |
            Sort-Object -Unique |
            ForEach-Object { "taskkill.exe /PID $_ /T" }
    ) -join [Environment]::NewLine

    throw @"
Port $Port is already in use. No second MLflow server was started.
$details

If this is an old MLflow server, confirm the command line above and stop its
process tree with:
$stopCommands
Do not run the command if the process is unrelated to MLflow.
"@
}

$sqlitePath = $mlflowDatabasePath.Replace("\", "/")
$backendStoreUri = "sqlite:///$sqlitePath"
$defaultArtifactRoot = ([System.Uri]$mlflowArtifactPath).AbsoluteUri

Write-Host "Project root:        $projectRoot"
Write-Host "MLflow executable:   $mlflowExecutable"
Write-Host "Tracking database:   $mlflowDatabasePath"
Write-Host "Backend store URI:   $backendStoreUri"
Write-Host "Artifact root:       $mlflowArtifactPath"
Write-Host "MLflow UI:           http://${ListenAddress}:$Port"

Set-Location -LiteralPath $projectRoot

# A single worker avoids intermittent Windows socket/multiprocessing failures.
$previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
$PSNativeCommandUseErrorActionPreference = $false

& $mlflowExecutable server `
    --backend-store-uri $backendStoreUri `
    --default-artifact-root $defaultArtifactRoot `
    --host $ListenAddress `
    --port $Port `
    --workers 1 2>&1 | ForEach-Object { Write-Host $_ }

$mlflowExitCode = $LASTEXITCODE
$PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
exit $mlflowExitCode
