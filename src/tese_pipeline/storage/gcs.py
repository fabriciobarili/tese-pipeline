"""Estágio 5 — Acesso programático ao Google Cloud Storage.

Prefira SEMPRE o DVC (``dvc push``/``dvc pull``) para os dados da esteira.
Use este módulo apenas para artefatos auxiliares (ex.: publicar um relatório).
Credenciais via ``GOOGLE_APPLICATION_CREDENTIALS`` — nunca versionadas.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def upload(bucket: str, local: str | Path, remote_path: str) -> str:
    """Envia um arquivo local para ``gs://{bucket}/{remote_path}`` e retorna a URI."""
    from google.cloud import storage  # import tardio: dependência opcional em runtime

    client = storage.Client()
    blob = client.bucket(bucket).blob(remote_path)
    blob.upload_from_filename(str(local))
    uri = f"gs://{bucket}/{remote_path}"
    LOG.info("upload: %s", uri)
    return uri


def download(bucket: str, remote_path: str, local: str | Path) -> Path:
    """Baixa ``gs://{bucket}/{remote_path}`` para o caminho local."""
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(remote_path)
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    LOG.info("download: %s -> %s", remote_path, local)
    return Path(local)
