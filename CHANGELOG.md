# Changelog

Histórico legível das evoluções da pipeline da tese. Versões seguem o padrão
`v0.MINOR.0` usado nas mensagens de commit.

## v0.21.0 — Validação causal via SHAP (recuperação do gerador)

### Adicionado
- **`modeling/validacao_causal.py`**: verifica se o classificador de eventos *recupera* a
  estrutura que o gerador sintético injetou. Quatro peças: ranking de importância SHAP por
  classe; *dependence plot* do driver espacial (`dist_to_airport_km`); **teste de intervenção**
  (relocar radialmente a origem para longe do SBPA, recomputando lat/lng/distância, e medir a
  queda de P(evento)); e um *score de concordância* contra expectativas pré-registradas. Saídas
  em `reports/causal_validation/` (JSONs + figuras).
- Resultado da amostra de validação: `dist_to_airport_km` é o SHAP de topo nas duas classes de
  evento; a intervenção derruba P(evento) com Spearman de −0,99 (voo) e −0,90 (clima) — o modelo
  recuperou o ancoramento espacial, não um atalho.

### Alterado
- `docs/artigo/sbbd-2026-draft.md`: **Seção 5.3** redigida com esses resultados + Figuras 2 e 3.

### Dependências
- `scipy` declarado explicitamente (antes, transitivo via scikit-learn/shap).

## v0.20.1 — Artigo: pipeline de ponta a ponta com predição conversacional

### Alterado
- `docs/artigo/sbbd-2026-draft.md`: a esteira descrita no artigo agora fecha o laço completo —
  da coleta/fabricação do dado à **predição conversacional** devolvida ao motorista. Inclui o
  estágio conversacional na Figura 1, a nova Seção 3.8 (modelos pareados + prior prospectivo +
  ficha SHAP + narração em linguagem natural, sem *train/serve skew*), uma 4ª contribuição na
  Introdução, e redige as Seções 5.4 (o que o motorista aprende) e 6 (conclusão).

## v0.20.0 — Narrador conversacional de demanda (motorista × modelos)

### Adicionado
- **Pacote `narrador`** (`src/tese_pipeline/narrador/`) — o ambiente de conversa que fecha a
  pipeline de ponta a ponta, respondendo ao motorista *\"vale a pena ir ao Salgado Filho às X?\"*.
  Porta de `03_ANALISE/05_5_Narrador_Natural` + a produção do `05_6`, sobre a base do pacote.
  - `features.py` — construção de features **compartilhada** entre treino e inferência (elimina
    *train/serve skew*): clima físico, codificação cíclica, prior prospectivo de atraso
    (climatologia + reputação por `dia_semana × hora`).
  - `treina_pareados.py` — treina e exporta os **dois modelos binários pareados** (clima/atraso)
    + `climatologia_atraso_sbpa.csv` + `reputacao_slot_sbpa.csv` (eventos de `driver_trips`,
    clima de `meteo.parquet`, voos de `flights_anac.parquet`).
  - `narrador.py` — classe `Narrador`: monta features em UTC, roda as duas dimensões, explica
    via **SHAP**, agrega padrão histórico de voos/reputação, pondera o **esforço** do motorista,
    varre +2h e **narra em linguagem natural com a API da Claude** (API pública por padrão;
    suporta gateway via `base_url`/`auth_token`).
- **Notebooks** `Work/06_modelos_pareados_demanda.ipynb` (treino dos pareados) e
  `Work/07_narrador_natural.ipynb` (a conversa com o motorista).
- Dependências `duckdb` e `anthropic`.

## v0.19.0 — Engenharia de features climáticas no treino (porta do 05_6)

### Adicionado
- **Utilitários de features** (`modeling/feateng_granularidade.py`): porta da lógica
  do notebook `05_6_FeatureEng_Individual_vs_Agrupado` — enriquecimento climático por
  TEMPO (médias móveis 3h/6h/24h, `delta_3h`, `lag`/`diff` de 1 janela, razão
  vento/temperatura) e codificação cíclica `sin/cos` de hora/dia/mês. Funções puras,
  todas backward-looking (sem vazamento do futuro).
- **Bloco `feature_eng` em `params.yaml`**: liga/desliga o enriquecimento e parametriza
  horizontes, colunas climáticas e variáveis cíclicas (sem constantes mágicas no código).
- **Runners de nuvem do treino** (`scripts/run_train_gce.ps1` e `.sh`): rodam
  `optimize → train → explain` numa VM Compute Engine (`e2-standard-4`), puxando
  `rides.parquet` via DVC e publicando `reports/` + `models/` em
  `gs://doutorado-501917-synthetic/train_run/`; `n_trials`/`timeout_s` do Optuna
  ajustáveis via override do `params.yaml`.

### Alterado
- **`modeling/dataset.make_xy`** passa a enriquecer as features (05_6) antes do one-hot,
  controlado pelo bloco `feature_eng`; `train.py`, `optimize.py` e `explain.py` repassam
  essa configuração — mantendo a MESMA base de dados (`rides.parquet`).

## v0.18.0 — Classificação de corridas sob evento de demanda (Alvo A)

### Adicionado
- **Classificador multiclasse** (`modeling/event_classifier.py`): prevê se uma
  corrida ocorre sob evento de demanda — `nenhum` / `voo_atrasado` /
  `estresse_termico` — com **LightGBM + SMOTE** e explicabilidade **SHAP**.
  Alvo derivado de `event_name`; nenhuma feature vaza o rótulo
  (`event_name`/`chained_ride` excluídos). Features espaço-temporais + contexto,
  incluindo `dist_to_airport_km` (origem → SBPA).
- **Rebalanceamento para dados grandes**: subamostra a classe majoritária e
  aplica SMOTE nas minoritárias até um alvo por classe; o conjunto de TESTE
  mantém a distribuição natural (avaliação honesta).
- **Leitura do dataset completo via GCS** (gcsfs/pyarrow) além de amostra local.
- **Runner de nuvem** (`scripts/run_event_gce.ps1`): roda o classificador no
  dataset completo (267 M) numa VM `e2-highmem-16` (128 GB), lendo do GCS e
  gravando os resultados em `gs://doutorado-501917-synthetic/event_analysis/`.
- Config `config/event_classifier.yaml` com fonte de dados, features,
  reamostragem, hiperparâmetros LightGBM e amostra SHAP.

## v0.17.1 — Execução na nuvem no Windows

### Adicionado
- **Script PowerShell** (`scripts/run_sim_gce.ps1`): versão nativa do runner de
  nuvem para Windows, sem depender de bash/WSL. Resolve o caminho do `gcloud`
  automaticamente (inclusive quando o SDK não está no PATH).
- **Monitor de progresso** (`scripts/check_sim_progress.ps1`): conta os shards
  `.parquet` já gravados no GCS, agrupados por mês, para acompanhar o run.

### Alterado
- Tipo de máquina padrão da VM: `e2-standard-16` (16 vCPUs, 64 GB). A
  `e2-highcpu-16` (16 GB) sofreu OOM no run completo — 16 workers sobre 4,9M de
  células H3 res 12 excedem 16 GB (pico ~27 GB). As famílias C4/C3 vêm com quota
  0 por região em `southamerica-east1`; a E2 usa a quota geral de CPU do projeto.

## v0.17.0 — Geração de dados sintéticos por simulação de motoristas

### Adicionado
- **Novo gerador de corridas baseado em agentes** (`simulate_drivers.py`): em vez
  de criar corridas isoladas por chegada de voo, agora simulamos **motoristas**
  circulando pela malha viária de Porto Alegre e recebendo chamados. É a
  adaptação, para a pipeline reprodutível, do notebook de simulação do Colab.
- **Demanda guiada por eventos reais**: voos atrasados no Salgado Filho (ANAC
  VRA) e estresse térmico (UTCI) aquecem a região do aeroporto. A demanda nos
  eventos de voo é **proporcional ao volume de passageiros** da janela de horário.
- **Nota do motorista (reputação)**: cada motorista tem nota entre 4,79 e 4,99
  (alta ≥4,95; média 4,85–4,94; regular <4,85), diluída num histórico de 10 a 20
  mil corridas anteriores. Cada corrida recebe avaliação de 3 a 5. Motoristas
  nota alta recebem corrida encadeada perto do destino (podem recusar).
- **Jornada e turnos realistas**: 25–35 corridas/dia por motorista, teto de 12h
  de trabalho, 85% iniciam o turno entre 4h e 14h e 15% em turno noturno (18h)
  que cruza a meia-noite — garantindo motoristas circulando na madrugada.
- **Retorno para casa**: após 20 corridas, os destinos do motorista convergem
  para o ponto da primeira corrida do dia.
- **Execução em escala**: modo multiprocessing (paralelo por motorista) e
  gravação em Parquet particionado por mês (compacto, pronto para LightGBM/SMOTE),
  com destino local ou no Google Cloud Storage (`gs://`). Modo streaming grava
  dia a dia para manter a memória constante no run completo (300M+ corridas).
- **Script de execução no Google Cloud** (`scripts/run_sim_gce.sh`): provisiona
  uma VM Compute Engine de muitos núcleos, roda a simulação completa (30 mil
  motoristas, ano inteiro) e grava direto no GCS.
- **Configuração declarativa** (`config/driver_simulation.yaml`): todas as
  premissas — frota, mobilidade, eventos, notas, jornada, execução — num só lugar.

### Observações
- O gerador anterior (`generate_rides.py`, corridas por chegada de voo) continua
  disponível; o novo módulo é um caminho alternativo de síntese, mais rico em
  comportamento de oferta/demanda.
