#!/usr/bin/env bash
# =============================================================================
# Executa o pipeline de TREINO de regressão (optimize -> train -> explain) numa
# VM do Compute Engine, com a engenharia de features do 05_6 já embutida em
# make_xy. Lê data/synthetic/rides.parquet (via dvc pull) e copia reports/ e
# models/ para o GCS.
#
# Uso:
#   bash scripts/run_train_gce.sh
#
# Pré-requisitos: gcloud autenticado (gcloud auth login) e o artefato
# data/synthetic/rides.parquet versionado no remoto DVC (gs://).
# Pipeline leve (~50k linhas): e2-standard-4 basta. Ajuste N_TRIALS/TIMEOUT_S
# (sobrescrevem o params.yaml dentro da VM) e as variáveis abaixo via env.
# A VM é criada e roda o treino; delete-a ao final (comando impresso no fim).
# =============================================================================
set -euo pipefail

PROJECT="${PROJECT:-doutorado-501917}"
ZONE="${ZONE:-southamerica-east1-a}"
VM_NAME="${VM_NAME:-train-lgbm}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"   # 4 vCPUs, 16 GB — suficiente p/ rides.parquet
REPO_URL="${REPO_URL:-https://github.com/fabriciobarili/tese-pipeline.git}"
BRANCH="${BRANCH:-feature_engineering}"
GCS_OUT="${GCS_OUT:-gs://doutorado-501917-synthetic/train_run}"
N_TRIALS="${N_TRIALS:-100}"                     # trials do Optuna (override do params.yaml)
TIMEOUT_S="${TIMEOUT_S:-3600}"                  # timeout do estudo Optuna, em segundos

# startup-script roda dentro da VM no boot. Heredoc NÃO-citado (<<EOF): as
# variáveis do host (${BRANCH}, ${GCS_OUT}, ${N_TRIALS}, ...) são expandidas aqui;
# o bloco python <<'PY' é citado, mas, por estar aninhado no EOF, também recebe os
# valores já substituídos pelo host.
read -r -d '' STARTUP <<EOF || true
set -euo pipefail
apt-get update -y && apt-get install -y git python3-pip
cd /opt
git clone --branch ${BRANCH} ${REPO_URL} tese-pipeline || (cd tese-pipeline && git pull)
cd /opt/tese-pipeline
pip3 install -e . --break-system-packages
pip3 install gcsfs 'dvc[gs]' --break-system-packages

# dataset de treino versionado no DVC (falha claro se não estiver no remoto)
dvc pull data/synthetic/rides.parquet

# override do Optuna (n_trials / timeout) no params.yaml
python3 - <<'PY'
import yaml
p = "params.yaml"
cfg = yaml.safe_load(open(p, encoding="utf-8"))
cfg["model"]["optuna"]["n_trials"] = ${N_TRIALS}
cfg["model"]["optuna"]["timeout_s"] = ${TIMEOUT_S}
yaml.safe_dump(cfg, open(p, "w", encoding="utf-8"), allow_unicode=True)
PY

# treino com a engenharia de features do 05_6 (optimize -> train -> explain)
python3 -m tese_pipeline.modeling.optimize
python3 -m tese_pipeline.modeling.train
python3 -m tese_pipeline.modeling.explain

# copia resultados para o GCS
gsutil -m cp -r reports models ${GCS_OUT}/ 2>/dev/null || \
  gcloud storage cp --recursive reports models ${GCS_OUT}/
echo "TRAIN_LGBM CONCLUIDO"
EOF

echo ">> criando VM ${VM_NAME} (${MACHINE_TYPE}) em ${ZONE}..."
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT}" --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --scopes=cloud-platform \
  --metadata=startup-script="${STARTUP}"

echo ">> acompanhe o progresso com:"
echo "   gcloud compute instances get-serial-port-output ${VM_NAME} --zone ${ZONE}"
echo ">> resultados em: ${GCS_OUT}/  (ao ver TRAIN_LGBM CONCLUIDO)"
echo ">> ao terminar, delete a VM com:"
echo "   gcloud compute instances delete ${VM_NAME} --zone ${ZONE} --quiet"
