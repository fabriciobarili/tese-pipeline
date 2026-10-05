# =============================================================================
# Checa o progresso da simulação olhando os shards já gravados no GCS.
# Uso (PowerShell):
#   .\check_sim_progress.ps1
# =============================================================================
$ErrorActionPreference = 'Stop'

$GcsOut = 'gs://doutorado-501917-synthetic/driver_trips'

# resolve o gcloud (igual ao runner)
$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
    if (Test-Path $fallback) { $gcloud = $fallback }
    else { Write-Error 'gcloud nao encontrado.'; exit 1 }
}

Write-Host ">> listando shards em $GcsOut ..."
$lines = & $gcloud storage ls --recursive "$GcsOut/**" 2>$null
$parquet = $lines | Where-Object { $_ -match '\.parquet$' }

if (-not $parquet) {
    Write-Host "Nenhum shard ainda. A VM pode estar instalando dependencias / rodando os primeiros dias."
    exit 0
}

# agrupa por particao de mes (month=NN)
$byMonth = @{}
foreach ($p in $parquet) {
    if ($p -match 'month=(\d+)') {
        $m = [int]$matches[1]
        $byMonth[$m] = ($byMonth[$m] + 1)
    }
}

Write-Host ("shards .parquet gravados: {0}" -f $parquet.Count)
Write-Host "por mes:"
$byMonth.Keys | Sort-Object | ForEach-Object {
    Write-Host ("  mes {0,2}: {1} shards" -f $_, $byMonth[$_])
}
$maxMonth = ($byMonth.Keys | Measure-Object -Maximum).Maximum
Write-Host ("ultimo mes com dados: {0} de 12" -f $maxMonth)
