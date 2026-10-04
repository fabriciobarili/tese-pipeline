# Descrição Detalhada da Esteira

Documento de referência da pipeline, legível por um avaliador externo. Para o passo a
passo de cada estágio, ver os módulos em `src/tese_pipeline/` e os arquivos de
referência da skill `tese-pipeline`.

## Objetivo científico

Investigar, por meio de corridas sintéticas calibradas por condições meteorológicas e
operação aeroportuária, os padrões que um modelo de árvore (LightGBM) aprende e que a
análise SHAP torna explicáveis — com narração em linguagem natural fiel aos dados.

## Fluxo de dados e artefatos

| Estágio | Entrada | Saída | Proveniência |
|---------|---------|-------|--------------|
| ingest_meteo | config/meteo.yaml | data/raw/meteo.parquet | `.meta.json` |
| ingest_flights | config/flights.yaml | data/raw/flights.parquet | `.meta.json` |
| process | meteo + flights | data/processed/features.parquet | `.meta.json` |
| synthesize | features + config/synthesis.yaml | data/synthetic/rides.parquet | `.meta.json` |
| optimize | rides | reports/metrics/best_params.json | optuna.db |
| train | rides + best_params | models/lgbm.txt + eval.json | model card |
| explain | lgbm + rides | reports/shap/ | — |

## Decisões-chave

Registradas como ADRs em `docs/decisions/`:
- 0001 — Estrutura e ferramentas (OpenSky, Open-Meteo, GCS+DVC, LightGBM+Optuna+SHAP).
- 0002 — Premissas do processo de síntese de corridas.

## Garantias metodológicas

- **Sem leakage:** split temporal em treino, CV (TimeSeriesSplit) e teste.
- **Determinismo:** seed única propagada; LightGBM/Optuna com seed explícita.
- **Validação do sintético:** o modelo deve recuperar as relações injetadas na síntese
  (ex.: chuva associada a maior duração) — ver `reports/` e o model card.
