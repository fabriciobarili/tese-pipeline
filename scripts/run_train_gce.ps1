# =============================================================================
# Roda o pipeline de TREINO de regressão (optimize -> train -> explain) numa VM
# Compute Engine, com a engenharia de features do 05_6 já embutida em make_xy.
# Lê data/synthetic/rides.parquet (via dvc pull) e copia reports/ e models/ para
# o GCS. Versão PowerShell (Windows), sem bash/WSL.
#
# Uso (PowerShell):
#   cd C:\Users\I742960\PROJETOS\DOUTORADO\TESE_PIPELINE
#   .\scripts\run_train_gce.ps1
#
# Pré-requisitos: Google Cloud SDK autenticado (gcloud auth login) e o artefato
# data/synthetic/rides.parquet versionado no remoto DVC (gs://). A VM é criada e
# roda o treino; delete-a ao final (comando impresso no fim).
#
# Pipeline leve (~50k linhas): e2-standard-4 basta. O custo é o Optuna — ajuste
# $NTrials/$TimeoutS abaixo (sobrescrevem o params.yaml dentro da VM).
# =============================================================================
$ErrorActionPreference = 'Stop'

# ---- parâmetros (ajuste conforme necessário) --------------------------------
$Project     = 'doutorado-501917'
$Zone        = 'southamerica-east1-a'
$VmName      = 'train-lgbm'
$MachineType = 'e2-standard-4'          # 4 vCPUs, 16 GB — suficiente p/ rides.parquet
$RepoUrl     = 'https://github.com/fabriciobarili/tese-pipeline.git'
$Branch      = 'feature_engineering'
$GcsOut      = 'gs://doutorado-501917-synthetic/train_run'
$NTrials     = 100                      # trials do Optuna (override do params.yaml)
$TimeoutS    = 3600                     # timeout do estudo Optuna, em segundos

# ---- resolve o executável do gcloud -----------------------------------------
$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
    if (Test-Path $fallback) { $gcloud = $fallback }
    else { Write-Error "gcloud nao encontrado. Instale o Google Cloud SDK ou ajuste o caminho."; exit 1 }
}
Write-Host ">> usando gcloud: $gcloud"

# ---- startup-script (bash) que roda DENTRO da VM Debian ---------------------
# Placeholders __X__ sao substituidos abaixo (here-string literal evita conflito
# com os '$' do bash/python internos).
$startup = @'
set -euo pipefail
apt-get update -y && apt-get install -y git python3-pip
cd /opt
git clone --branch __BRANCH__ __REPO_URL__ tese-pipeline || (cd tese-pipeline && git pull)
cd /opt/tese-pipeline
pip3 install -e . --break-system-packages
pip3 install gcsfs "dvc[gs]" --break-system-packages

# dataset de treino versionado no DVC (falha claro se nao estiver no remoto)
dvc pull data/synthetic/rides.parquet

# override do Optuna (n_trials / timeout) no params.yaml
python3 - <<'PY'
import yaml
p = "params.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg["model"]["optuna"]["n_trials"] = __N_TRIALS__
cfg["model"]["optuna"]["timeout_s"] = __TIMEOUT_S__
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True)
PY

# treino com a engenharia de features do 05_6 (optimize -> train -> explain)
python3 -m tese_pipeline.modeling.optimize
python3 -m tese_pipeline.modeling.train
python3 -m tese_pipeline.modeling.explain

# copia resultados para o GCS
gsutil -m cp -r reports models __GCS_OUT__/ 2>/dev/null || \
  gcloud storage cp --recursive reports models __GCS_OUT__/
echo "TRAIN_LGBM CONCLUIDO"
'@

$startup = $startup.Replace('__BRANCH__',    $Branch).
                    Replace('__REPO_URL__',  $RepoUrl).
                    Replace('__GCS_OUT__',   $GcsOut).
                    Replace('__N_TRIALS__',  [string]$NTrials).
                    Replace('__TIMEOUT_S__', [string]$TimeoutS)

# grava em arquivo temporario (LF, sem BOM) para passar via --metadata-from-file
$tmp = Join-Path $env:TEMP 'train_lgbm_startup.sh'
[System.IO.File]::WriteAllText($tmp, ($startup -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))

# ---- cria a VM --------------------------------------------------------------
Write-Host ">> criando VM $VmName ($MachineType) em $Zone..."
& $gcloud compute instances create $VmName `
    --project=$Project --zone=$Zone `
    --machine-type=$MachineType `
    --image-family=debian-12 --image-project=debian-cloud `
    --boot-disk-size=50GB `
    --scopes=cloud-platform `
    --metadata-from-file=startup-script=$tmp

Write-Host ""
Write-Host ">> acompanhe o progresso com:"
Write-Host "   & '$gcloud' compute instances get-serial-port-output $VmName --zone $Zone"
Write-Host ">> resultados em: $GcsOut/  (ao ver TRAIN_LGBM CONCLUIDO)"
Write-Host ">> delete a VM ao terminar:"
Write-Host "   & '$gcloud' compute instances delete $VmName --zone $Zone --quiet"
