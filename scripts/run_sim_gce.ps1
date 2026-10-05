# =============================================================================
# Executa a simulação de motoristas (simulate_drivers) numa VM do Compute Engine,
# gravando o dataset particionado direto no GCS. Versão PowerShell (Windows),
# equivalente a run_sim_gce.sh — não depende de bash/WSL.
#
# Uso (PowerShell):
#   cd C:\Users\I742960\PROJETOS\DOUTORADO\TESE_PIPELINE\scripts
#   .\run_sim_gce.ps1
#
# Pré-requisitos: Google Cloud SDK instalado e autenticado (gcloud auth login).
# A VM é criada e roda a simulação; delete-a ao final (comando impresso no fim).
# =============================================================================

$ErrorActionPreference = 'Stop'

# ---- parâmetros (ajuste conforme necessário) --------------------------------
$Project     = 'doutorado-501917'
$Zone        = 'southamerica-east1-a'
$VmName      = 'driver-sim'
$MachineType = 'c4-highcpu-16'          # 16 vCPUs (cabe na quota padrao de 32)
                                        # quota CPUS_ALL_REGIONS limita o total;
                                        # p/ -48/-192 peca aumento de quota.
$RepoUrl     = 'https://github.com/fabriciobarili/tese-pipeline.git'
$Branch      = 'master'
$GcsOut      = 'gs://doutorado-501917-synthetic/driver_trips'

# ---- resolve o executável do gcloud -----------------------------------------
$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd'
    if (Test-Path $fallback) { $gcloud = $fallback }
    else { Write-Error "gcloud nao encontrado. Instale o Google Cloud SDK ou ajuste o caminho."; exit 1 }
}
Write-Host ">> usando gcloud: $gcloud"

# ---- startup-script (bash) que roda DENTRO da VM Debian ---------------------
# Placeholders __X__ são substituídos abaixo (here-string literal evita conflito
# com os '$' do bash/python internos).
$startup = @'
set -euo pipefail
apt-get update -y && apt-get install -y git python3-pip
cd /opt
git clone --branch __BRANCH__ __REPO_URL__ tese-pipeline || (cd tese-pipeline && git pull)
cd /opt/tese-pipeline
pip3 install -e . --break-system-packages
pip3 install gcsfs "dvc[gs]" --break-system-packages
dvc pull ext/sul-261003_vias_h3r13.parquet data/raw/flights_anac.parquet \
         data/processed/thermal_comfort.parquet || true
python3 - <<'PY'
import yaml
p = "config/driver_simulation.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg.setdefault("runtime", {})
cfg["runtime"]["n_workers"] = 0                 # todos os vCPUs
cfg["runtime"]["output_dir"] = "__GCS_OUT__"
cfg["runtime"]["stream_per_day"] = True         # RAM constante
cfg["period"]["max_days"] = 365                 # ano completo
cfg["drivers"]["total_drivers"] = 30000         # frota completa
cfg["roads"]["max_road_cells"] = None           # malha completa
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True)
PY
python3 -m tese_pipeline.synthesis.simulate_drivers
echo "SIMULACAO CONCLUIDA"
'@

$startup = $startup.Replace('__BRANCH__',  $Branch).
                    Replace('__REPO_URL__', $RepoUrl).
                    Replace('__GCS_OUT__',  $GcsOut)

# grava em arquivo temporário (LF, sem BOM) para passar via --metadata-from-file
$tmp = Join-Path $env:TEMP 'driver_sim_startup.sh'
$startupLf = $startup -replace "`r`n", "`n"
[System.IO.File]::WriteAllText($tmp, $startupLf, (New-Object System.Text.UTF8Encoding($false)))

# ---- cria a VM --------------------------------------------------------------
Write-Host ">> criando VM $VmName ($MachineType) em $Zone..."
& $gcloud compute instances create $VmName `
    --project=$Project --zone=$Zone `
    --machine-type=$MachineType `
    --image-family=debian-12 --image-project=debian-cloud `
    --boot-disk-size=100GB `
    --scopes=storage-rw `
    --metadata-from-file=startup-script=$tmp

Write-Host ""
Write-Host ">> acompanhe o progresso com:"
Write-Host "   & '$gcloud' compute ssh $VmName --zone $Zone --command 'sudo journalctl -u google-startup-scripts -f'"
Write-Host ">> ao terminar, delete a VM com:"
Write-Host "   & '$gcloud' compute instances delete $VmName --zone $Zone --quiet"
