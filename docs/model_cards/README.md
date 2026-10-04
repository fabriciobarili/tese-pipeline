# Model Cards

Um arquivo por modelo treinado. Modelo abaixo (copie para `lgbm.md`).

```markdown
# Model Card — LightGBM (<versão/data>)
- **Objetivo:** prever <target> (regressão)
- **Dados:** data/synthetic/rides.parquet (hash em `*.meta.json`)
- **Split:** temporal (treino no passado, teste no futuro); test_frac = <valor>
- **Hiperparâmetros vencedores (Optuna):** ver reports/metrics/best_params.json
- **Métricas (fora da amostra):** RMSE / MAE / R² vs baseline (reports/metrics/eval.json)
- **Explicabilidade (SHAP):** top features e achados (reports/shap/)
- **Validação da esteira:** o modelo recuperou as relações injetadas na síntese? (sim/não)
- **Limitações:** dados sintéticos; achados condicionados às premissas do ADR 0002
```
