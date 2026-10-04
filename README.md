# tese-pipeline

Pipeline reprodutível e auditável da tese: ingestão de dados **meteorológicos**
(Open-Meteo) e de **voos** (OpenSky Network), processamento e feature engineering,
geração de **corridas sintéticas**, armazenamento em nuvem (**Google Cloud Storage +
DVC**), modelagem **explicável** (LightGBM + Optuna + SHAP) e **narração em linguagem
natural**.

## Esteira

```
Open-Meteo ─┐
            ├─> processamento ─> corridas sintéticas ─> GCS/DVC ─> LightGBM ─> SHAP ─> narração
OpenSky ────┘        (features)                                   (+Optuna)
```

| # | Estágio | Comando (via DVC) |
|---|---------|-------------------|
| 1 | Ingestão meteorológica | `dvc repro ingest_meteo` |
| 2 | Ingestão de voos | `dvc repro ingest_flights` |
| 3 | Processamento & features | `dvc repro process` |
| 4 | Corridas sintéticas | `dvc repro synthesize` |
| 5 | Armazenamento (GCS+DVC) | `dvc push` |
| 6 | Modelagem explicável | `dvc repro optimize train explain` |
| 7 | Narração | `tese_pipeline.narration` |

## Como reproduzir

```bash
# 1. Ambiente
python -m venv .venv
source .venv/Scripts/activate        # Git Bash no Windows
pip install -e ".[dev]"

# 2. Credenciais (NUNCA versionadas)
export OPENMETEO_API_KEY=...                 # opcional, conforme endpoint
export OPENSKY_CLIENT_ID=...  OPENSKY_CLIENT_SECRET=...
export GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json

# 3. Dados versionados
dvc pull            # recupera os dados exatos do commit (após remote configurado)

# 4. Reconstruir a esteira
dvc repro           # reexecuta apenas o que mudou, de forma determinística

# 5. Testes
pytest
```

## Reprodutibilidade & auditoria

- Esteira declarativa em `dvc.yaml`; hashes em `dvc.lock`.
- Seeds centralizadas em `params.yaml`; nenhuma constante mágica no código.
- Dados versionados fora do Git (DVC + GCS); Git guarda só ponteiros.
- Proveniência por artefato (`*.meta.json`: sha256 + git SHA + timestamp).
- `docs/`: data cards, model cards e ADRs (decisões de arquitetura).
- Modelagem com split **temporal** (sem vazamento).

## Estrutura

- `src/tese_pipeline/` — código por estágio (ingestion, processing, synthesis, storage, modeling, narration)
- `config/` — configuração de cada estágio
- `params.yaml` — seeds e hiperparâmetros
- `data/` — dados versionados por DVC (não vão para o Git)
- `docs/` — documentação auditável
- `reports/` — métricas, figuras, SHAP, logs

## Licença

MIT — ver [LICENSE](LICENSE). Para citar, ver [CITATION.cff](CITATION.cff).
