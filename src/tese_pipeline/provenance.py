"""Registro de proveniência dos artefatos da pipeline.

Cada artefato gravado em ``data/``/``models/`` acompanha um sidecar
``<arquivo>.meta.json`` com hash, SHA do git, timestamp e parâmetros usados —
base da auditabilidade exigida pela tese.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def git_sha() -> str:
    """SHA do commit atual; ``"unknown"`` se não houver repositório git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_sha256(path: str | Path) -> str:
    """Hash SHA-256 do conteúdo de um arquivo."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def write_provenance(output: str | Path, source: str, params: dict[str, Any]) -> Path:
    """Grava o sidecar de proveniência ao lado de ``output`` e retorna seu caminho."""
    output = Path(output)
    meta = {
        "artifact": str(output),
        "sha256": file_sha256(output),
        "git_sha": git_sha(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": source,
        "params": params,
    }
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path
