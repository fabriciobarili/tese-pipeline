# =============================================================================
# Roda o classificador de eventos (event_classifier) no dataset COMPLETO (267M)
# numa VM Compute Engine de alta memória, lendo direto do GCS e gravando os
# resultados de volta no GCS. Versão PowerShell (Windows), sem bash/WSL.
#
# Uso (PowerShell):
#   .\scripts\run_event_gce.ps1
#
# Pré-requisitos: Google Cloud SDK autenticado (gcloud auth login).
# A VM processa 267M linhas (load + SMOTE/undersample + LightGBM + SHAP) e
# copia reports/event_analysis para o GCS. Delete a VM ao final (comando no fim).
# =============================================================================
$ErrorActionPreference = 'Stop'

$Project     = 'doutorado-501917'
$Zone        = 'southamerica-east1-a'
$VmName      = 'event-clf'
$MachineType = 'e2-highmem-16'          # 16 vCPUs, 128 GB (267M em pandas exige RAM)
$RepoUrl     = 'https://github.com/fabriciobarili/tese-pipeline.git'
$Branch      = 'master'
$GcsData     = 'gs://doutorado-501917-synthetic/driver_trips'
$GcsOut      = 'gs://doutorado-501917-synthetic/event_analysis'

$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
    if (Test-Path $fallback) { $gcloud = $fallback }
    else { Write-Error 'gcloud nao encontrado.'; exit 1 }
}
Write-Host ">> usando gcloud: $gcloud"

$startup = @'
set -euo pipefail
apt-get update -y && apt-get install -y git python3-pip
cd /opt
git clone --branch __BRANCH__ __REPO_URL__ tese-pipeline || (cd tese-pipeline && git pull)
cd /opt/tese-pipeline
pip3 install -e . --break-system-packages
pip3 install gcsfs --break-system-packages

# aponta a fonte para o GCS (dataset completo) e usa mais dados por classe
python3 - <<'PY'
import yaml
p = "config/event_classifier.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg["data"]["source"] = "__GCS_DATA__"
cfg["resampling"]["target_per_class"] = 1000000   # 1M/classe no treino balanceado
cfg["model"]["eval_max_rows"] = 10000000           # teste subamostrado a 10M (tratável)
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True)
PY

python3 -m tese_pipeline.modeling.event_classifier

# copia resultados para o GCS
gsutil -m cp -r reports/event_analysis/* __GCS_OUT__/ 2>/dev/null || \
  gcloud storage cp --recursive reports/event_analysis __GCS_OUT__/
echo "EVENT_CLF CONCLUIDO"
'@

$startup = $startup.Replace('__BRANCH__', $Branch).
                    Replace('__REPO_URL__', $RepoUrl).
                    Replace('__GCS_DATA__', $GcsData).
                    Replace('__GCS_OUT__', $GcsOut)

$tmp = Join-Path $env:TEMP 'event_clf_startup.sh'
[System.IO.File]::WriteAllText($tmp, ($startup -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))

Write-Host ">> criando VM $VmName ($MachineType) em $Zone..."
& $gcloud compute instances create $VmName `
    --project=$Project --zone=$Zone `
    --machine-type=$MachineType `
    --image-family=debian-12 --image-project=debian-cloud `
    --boot-disk-size=100GB `
    --scopes=cloud-platform `
    --metadata-from-file=startup-script=$tmp

Write-Host ""
Write-Host ">> acompanhe:"
Write-Host "   & '$gcloud' compute instances get-serial-port-output $VmName --zone $Zone"
Write-Host ">> resultados em: $GcsOut/  (ao ver EVENT_CLF CONCLUIDO)"
Write-Host ">> delete a VM ao terminar:"
Write-Host "   & '$gcloud' compute instances delete $VmName --zone $Zone --quiet"
