# ADR 0002 — Premissas do processo de síntese de corridas

- **Status:** proposto (preencher justificativas)
- **Data:** 2026-10-04

## Contexto

As corridas são **sintéticas**: o processo gerador é a contribuição metodológica e,
portanto, suas premissas precisam ser explícitas e defensáveis. Todas vivem em
`config/synthesis.yaml`.

## Decisão (premissas atuais — ajustar com base na literatura/dados reais)

| Premissa | Valor | Justificativa (preencher) |
|----------|-------|---------------------------|
| Corridas por chegada (Poisson) | média 3.0 | <ref/empírico> |
| Duração base | 25 ± 8 min | <ref/empírico> |
| Multiplicador de chuva | 1.20 | <ref: efeito de precipitação no tempo de viagem> |
| Multiplicador de pico (7,8,17,18h) | 1.35 | <ref: congestionamento em horário de pico> |
| Distância | 18 ± 7 km | <ref/empírico> |
| Ruído | σ = 2.0 min | aleatoriedade residual |

## Consequências

- O modelo (Estágio 6) deve recuperar essas relações; isso valida a esteira.
- Mudanças nas premissas devem gerar nova versão deste ADR e novo `dvc repro synthesize`.
- **Limitação:** achados sobre dados sintéticos refletem as premissas aqui assumidas,
  não observações do mundo real — declarar explicitamente na tese.
