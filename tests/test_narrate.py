"""Testes da narração (template determinístico e validação numérica)."""

from __future__ import annotations

from tese_pipeline.narration.narrate import narrar_predicao, validar_numeros


def test_narrativa_deterministica() -> None:
    contribs = [("is_rain", 3.2), ("hour", -1.1)]
    a = narrar_predicao(28.4, contribs)
    b = narrar_predicao(28.4, contribs)
    assert a == b  # mesmo input -> mesmo texto
    assert "28.4" in a


def test_validacao_numerica_rejeita_alucinacao() -> None:
    texto = "Estimativa: 28.4 min; fator 3.2 min."
    assert validar_numeros(texto, permitidos={28.4, 3.2})
    assert not validar_numeros("Estimativa: 99.9 min.", permitidos={28.4, 3.2})
