# ADR 0001 — Estrutura e ferramentas da pipeline

- **Status:** aceito
- **Data:** 2026-10-04

## Contexto

A tese exige uma pipeline de ponta a ponta (meteorologia + voos → corridas sintéticas →
IA explicável → linguagem natural) que seja **auditável e reprodutível** por um
avaliador externo, hospedada no GitHub pessoal.

## Decisão

- **Linguagem:** Python ≥ 3.11, pacote em `src/tese_pipeline/`.
- **Dados meteorológicos:** Open-Meteo (API aberta; histórico via ERA5).
- **Dados de voos:** OpenSky Network (API aberta, orientada à pesquisa; OAuth2 client
  credentials).
- **Armazenamento:** Google Cloud Storage com versionamento por DVC (dados fora do Git).
- **Esteira:** declarativa em `dvc.yaml`; `dvc repro` reconstrói tudo.
- **Modelagem:** LightGBM, hiperparâmetros via Optuna, explicabilidade via SHAP.
- **Narração:** template determinístico (padrão) ou LLM com grounding e validação numérica.

## Consequências

- Reprodutibilidade forte: dados versionados por hash e esteira determinística.
- Repositório Git leve (apenas ponteiros DVC e código).
- Dependência de credenciais externas (OpenSky, GCP) geridas via variáveis de ambiente.
- Fontes abertas favorecem a publicação e a citação dos dados.
