#!/usr/bin/env bash
# =============================================================================
# Executa a simulação de motoristas (simulate_drivers) em uma VM do Compute
# Engine com muitos vCPUs, gravando o dataset particionado direto no GCS.
#
# Uso:
#   bash scripts/run_sim_gce.sh
#
# Pré-requisitos: gcloud autenticado (gcloud auth login) e projeto definido.
# A VM é criada, roda a simulação e é DELETADA ao final (--delete no fim).
# Ajuste as variáveis abaixo conforme necessário.
# =============================================================================
set -euo pipefail

PROJECT="${PROJECT:-doutorado-501917}"
ZONE="${ZONE:-southamerica-east1-a}"
VM_NAME="${VM_NAME:-driver-sim}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-highcpu-16}"   # 16 vCPUs, família E2 (ampla disponibilidade)
                                                # C4/C3 vêm com quota 0 por região em southamerica-east1;
                                                # E2/N2 usam a quota geral de CPU. Alternativa: n2-highcpu-16
REPO_URL="${REPO_URL:-https://github.com/fabriciobarili/tese-pipeline.git}"
BRANCH="${BRANCH:-master}"
GCS_OUT="${GCS_OUT:-gs://doutorado-501917-synthetic/driver_trips}"

# startup-script roda dentro da VM no boot
read -r -d '' STARTUP <<EOF || true
set -euo pipefail
apt-get update -y && apt-get install -y git python3-pip
cd /opt
git clone --branch ${BRANCH} ${REPO_URL} tese-pipeline || (cd tese-pipeline && git pull)
cd /opt/tese-pipeline
pip3 install -e . --break-system-packages
pip3 install gcsfs --break-system-packages   # backend GCS do pyarrow

# puxa artefatos de dados versionados no DVC (malha, voos, conforto térmico)
pip3 install 'dvc[gs]' --break-system-packages
dvc pull ext/sul-261003_vias_h3r13.parquet data/raw/flights_anac.parquet \
         data/processed/thermal_comfort.parquet || true

# aponta a saída e o paralelismo via overrides no YAML
python3 - <<'PY'
import yaml, os, multiprocessing
p = "config/driver_simulation.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg.setdefault("runtime", {})
cfg["runtime"]["n_workers"] = 0                       # usa todos os vCPUs
cfg["runtime"]["output_dir"] = os.environ.get("GCS_OUT", "${GCS_OUT}")
cfg["runtime"]["stream_per_day"] = True               # RAM constante (300M+ linhas)
cfg["period"]["max_days"] = 365                       # ano completo
cfg["drivers"]["total_drivers"] = 30000               # frota completa
cfg["roads"]["max_road_cells"] = None                 # malha completa
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True)
PY

GCS_OUT="${GCS_OUT}" python3 -m tese_pipeline.synthesis.simulate_drivers
echo "SIMULACAO CONCLUIDA"
EOF

echo ">> criando VM ${VM_NAME} (${MACHINE_TYPE}) em ${ZONE}..."
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --scopes=storage-rw \
  --metadata=startup-script="${STARTUP}"

echo ">> acompanhe o progresso com:"
echo "   gcloud compute ssh ${VM_NAME} --zone ${ZONE} --command 'sudo journalctl -u google-startup-scripts -f'"
echo ">> ao terminar, delete a VM com:"
echo "   gcloud compute instances delete ${VM_NAME} --zone ${ZONE} --quiet"
