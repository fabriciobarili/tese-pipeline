"""Estágio 7 — Narração em linguagem natural (template determinístico).

Converte a predição + principais contribuições SHAP em texto fiel e cauteloso.
Linguagem associativa (nunca causal) e 100% reprodutível.
"""

from __future__ import annotations

import re


def narrar_predicao(
    pred: float, contribs: list[tuple[str, float]], unidade: str = "min", top: int = 3
) -> str:
    """Gera uma frase a partir da predição e das top contribuições SHAP.

    ``contribs``: lista de ``(feature, shap_value)`` ordenada por |shap| desc.
    """
    partes = []
    for feat, val in contribs[:top]:
        direcao = "aumentou" if val > 0 else "reduziu"
        partes.append(f"{feat} {direcao} a estimativa em ~{abs(val):.1f} {unidade}")
    corpo = "; ".join(partes) if partes else "sem fatores dominantes"
    return (
        f"Estimativa: {pred:.1f} {unidade}. "
        f"Principais fatores associados: {corpo}. "
        f"Valores são estimativas de um modelo sobre dados sintéticos."
    )


def validar_numeros(texto: str, permitidos: set[float], tol: float = 0.1) -> bool:
    """Confere que todo número no texto corresponde a um valor permitido (anti-alucinação)."""
    encontrados = [float(n) for n in re.findall(r"-?\d+\.?\d*", texto)]
    for n in encontrados:
        if not any(abs(n - a) <= tol for a in permitidos):
            return False
    return True
