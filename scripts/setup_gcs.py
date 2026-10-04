#!/usr/bin/env python
"""Cria e configura os buckets GCS do projeto DOUTORADO.

Projeto GCP: doutorado-501917 (nº: 337771254941)
Região: southamerica-east1 (São Paulo)

Uso:
    # Com Application Default Credentials (gcloud auth application-default login):
    python scripts/setup_gcs.py

    # Com service account:
    export GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json
    python scripts/setup_gcs.py

    # Para apenas listar os buckets (sem criar):
    python scripts/setup_gcs.py --list-only

Arquitetura de buckets:
    doutorado-501917-dvc        → remote DVC (hash-addressed, todos os artefatos da pipeline)
    doutorado-501917-ext        → dados geoespaciais externos (KMZ, OSM PBF, H3 parquet)
    doutorado-501917-raw        → dados brutos da ingestão (meteo, voos)
    doutorado-501917-processed  → dados processados / feature engineering
    doutorado-501917-synthetic  → corridas sintéticas
    doutorado-501917-models     → modelos treinados (LightGBM)
    doutorado-501917-reports    → métricas, figuras SHAP, logs auditáveis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass

from google.cloud import storage
from google.cloud.exceptions import Conflict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger(__name__)

PROJECT_ID = "doutorado-501917"
LOCATION = "southamerica-east1"     # São Paulo — dados dentro do Brasil

# Definição declarativa de todos os buckets
# uniform_iam: True força IAM uniforme (sem ACLs por objeto) — mais seguro
@dataclass
class BucketSpec:
    name: str
    purpose: str
    stage: str
    storage_class: str = "STANDARD"
    lifecycle_days: int | None = None   # mover para NEARLINE após N dias (None = sem lifecycle)
    versioning: bool = False


BUCKETS: list[BucketSpec] = [
    BucketSpec(
        name=f"{PROJECT_ID}-dvc",
        purpose="Remote DVC — cache hash-addressed de todos os artefatos da pipeline",
        stage="dvc-cache",
        versioning=True,                # permite recuperar versões anteriores de artefatos
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-ext",
        purpose="Dados geoespaciais externos: KMZ (aeroporto, UDHs), OSM PBF, H3 parquets",
        stage="external",
        lifecycle_days=365,             # mover para NEARLINE após 1 ano (raramente reprocessado)
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-raw",
        purpose="Dados brutos de ingestão: meteo.parquet (Open-Meteo) + flights.parquet (OpenSky)",
        stage="ingest",
        lifecycle_days=180,
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-processed",
        purpose="Dados processados e feature-engineered: features.parquet (join meteo×voos + H3)",
        stage="process",
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-synthetic",
        purpose="Corridas sintéticas: rides.parquet (50k corridas geradas pelo processo gerador)",
        stage="synthesize",
        versioning=True,                # cada re-run do gerador produz uma nova versão
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-models",
        purpose="Modelos treinados: lgbm.txt (LightGBM), best_params.json (Optuna)",
        stage="train",
        versioning=True,
    ),
    BucketSpec(
        name=f"{PROJECT_ID}-reports",
        purpose="Relatórios auditáveis: eval.json, summary.png (SHAP), logs de execução",
        stage="reports",
        lifecycle_days=365,
    ),
]


def create_bucket(client: storage.Client, spec: BucketSpec) -> storage.Bucket:
    """Cria um bucket com as configurações do BucketSpec."""
    bucket = storage.Bucket(client, name=spec.name)
    bucket.storage_class = spec.storage_class

    # labels para organização e faturamento
    bucket.labels = {
        "project": "doutorado",
        "project-id": PROJECT_ID,
        "stage": spec.stage,
        "managed-by": "scripts-setup-gcs",
    }

    try:
        bucket = client.create_bucket(
            bucket,
            project=PROJECT_ID,
            location=LOCATION,
        )
        LOG.info("Bucket criado: gs://%s (%s)", spec.name, LOCATION)
    except Conflict:
        LOG.warning("Bucket já existe: gs://%s (pulando criação)", spec.name)
        bucket = client.bucket(spec.name)

    # uniform bucket-level access (mais seguro)
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True

    # versionamento
    bucket.versioning_enabled = spec.versioning
    if spec.versioning:
        LOG.info("  + versionamento habilitado")

    # lifecycle: mover para NEARLINE após N dias (economiza ~40% de custo de armazenamento)
    if spec.lifecycle_days is not None:
        bucket.add_lifecycle_set_storage_class_rule(
            "NEARLINE",
            age=spec.lifecycle_days,
        )
        LOG.info("  + lifecycle: NEARLINE após %d dias", spec.lifecycle_days)

    bucket.patch()
    return bucket


def print_summary(buckets: list[BucketSpec]) -> None:
    LOG.info("\n%s", "=" * 70)
    LOG.info("BUCKETS CRIADOS/CONFIGURADOS NO PROJETO %s", PROJECT_ID)
    LOG.info("%s", "=" * 70)
    for b in buckets:
        LOG.info("gs://%-40s [%s]", b.name, b.stage)
        LOG.info("   Propósito: %s", b.purpose)
        if b.versioning:
            LOG.info("   Versionamento: HABILITADO")
        if b.lifecycle_days:
            LOG.info("   Lifecycle: NEARLINE após %d dias", b.lifecycle_days)
    LOG.info("%s", "=" * 70)
    LOG.info("\nPróximo passo — configure o remote DVC:")
    LOG.info("  python -m dvc remote add -d gcs gs://%s-dvc/cache", PROJECT_ID)
    LOG.info("  git add .dvc/config && git commit -m 'chore(dvc): configura remote GCS'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-only", action="store_true",
                    help="Apenas listar os buckets planejados, sem criar")
    ap.add_argument("--project", default=PROJECT_ID,
                    help=f"ID do projeto GCP (padrão: {PROJECT_ID})")
    args = ap.parse_args()

    if args.list_only:
        print_summary(BUCKETS)
        return

    try:
        client = storage.Client(project=args.project)
    except Exception as exc:
        LOG.error(
            "Falha ao autenticar no GCP: %s\n"
            "Execute: gcloud auth application-default login\n"
            "Ou defina: export GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json",
            exc,
        )
        sys.exit(1)

    for spec in BUCKETS:
        create_bucket(client, spec)

    print_summary(BUCKETS)

    # persiste um JSON de referência dos buckets criados
    ref = [
        {
            "bucket": f"gs://{b.name}",
            "stage": b.stage,
            "purpose": b.purpose,
            "versioning": b.versioning,
            "lifecycle_days": b.lifecycle_days,
        }
        for b in BUCKETS
    ]
    import pathlib
    pathlib.Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    pathlib.Path("reports/metrics/gcs_buckets.json").write_text(
        json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Referência dos buckets salva: reports/metrics/gcs_buckets.json")


if __name__ == "__main__":
    main()
