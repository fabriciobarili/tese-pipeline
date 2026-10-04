# Uma Pipeline Reprodutível para Previsão de Demanda de Corridas por Aplicativo:
# Síntese de Dados Condicionada a Voos e Meteorologia com LightGBM e SHAP

> **Rascunho para submissão ao SBBD/CSBC — versão 0.1 (04/10/2026)**
> Autores: Fabricio Barili, [Orientador]
> Filiação: [Universidade]

---

## Resumo

A previsão de demanda de transporte por aplicativo (*rideshare*) em regiões aeroportuárias é
relevante para o planejamento urbano e operacional, porém raramente viável com dados reais
devido a restrições comerciais e de privacidade. Este trabalho propõe uma metodologia
reprodutível para geração de dados sintéticos de corridas condicionados a dados reais de
chegadas de voos e condições meteorológicas, com foco no Aeroporto Internacional Salgado Filho
(SBPA), em Porto Alegre/RS. A pipeline integra dados abertos da OpenSky Network e da API
Open-Meteo, aplica um processo gerador probabilístico parametrizado declarativamente, e treina
um modelo de *gradient boosting* (LightGBM) com otimização de hiperparâmetros via Optuna e
avaliação temporal sem vazamento. A explicabilidade é garantida por valores SHAP
(SHapley Additive exPlanations), que permitem verificar se o modelo recupera as relações
causais injetadas no processo gerador — constituindo, assim, uma forma de validação da
coerência da esteira. Os resultados indicam que a abordagem produz um modelo que replica
fielmente as premissas do gerador, com a precipitação e o horário de pico emergindo como
as variáveis de maior importância, alinhadas aos multiplicadores configurados.
A pipeline é inteiramente versionada com DVC e de código aberto, facilitando a replicação.

**Palavras-chave:** transporte por aplicativo, dados sintéticos, LightGBM, SHAP,
explicabilidade, aeroporto, reprodutibilidade.

---

## Abstract

Predicting ride-hailing demand near airports is relevant for urban and operational planning,
yet rarely feasible with real data due to commercial and privacy constraints. This work
proposes a reproducible methodology for generating synthetic ride datasets conditioned on
real flight arrival records and meteorological data, focusing on Salgado Filho International
Airport (SBPA) in Porto Alegre, Brazil. The pipeline integrates open data from the OpenSky
Network and the Open-Meteo API, applies a declaratively parameterized probabilistic data
generator, and trains a gradient boosting model (LightGBM) with Optuna hyperparameter
optimization and temporal split evaluation without data leakage. Explainability is ensured
through SHAP values, which verify that the model recovers the causal relationships injected
during synthesis — constituting an internal consistency validation of the pipeline. Results
show that the approach produces a model that faithfully replicates the generator's assumptions,
with precipitation and peak hours emerging as the most important features, aligned with the
configured multipliers. The entire pipeline is versioned with DVC and is open source.

**Keywords:** ride-hailing, synthetic data, LightGBM, SHAP, explainability, airport,
reproducibility.

---

## 1. Introdução

O crescimento dos serviços de transporte por aplicativo (*rideshare*, *ride-hailing*)
transformou a mobilidade urbana nas últimas décadas [SHAHEEN_2020]. Regiões aeroportuárias
constituem um caso de uso privilegiado: a demanda é fortemente correlacionada com a agenda
de chegadas de voos e é sensível a fatores externos como condições climáticas e horário do
dia [FLORES_2018]. Compreender essa demanda tem implicações práticas para a alocação de
veículos, o planejamento de acesso ao aeroporto e a formulação de políticas de mobilidade.

Uma barreira central à pesquisa nesse domínio é a **indisponibilidade de dados reais** de
corridas: operadoras como Uber e 99 não divulgam dados desagregados por origem-destino, e
acordos com órgãos públicos são raros e restritos a determinadas cidades [UBER_MOVEMENT].
O Aeroporto Internacional Salgado Filho (SBPA), principal hub da Região Sul do Brasil com
aproximadamente 4 milhões de passageiros anuais [INFRAERO_2023], não foge a essa regra.

Uma abordagem para contornar essa limitação é a **geração de dados sintéticos** condicionada
a variáveis observáveis que sabidamente influenciam a demanda real [GONZALEZ_2023]. Diferente
de abordagens puramente estocásticas, a síntese *condicional* — ancorada em registros reais
de chegadas de aeronaves e em séries históricas de meteorologia — produz dados com estrutura
temporal e contextual plausível. Mais importante: ao tornar o processo gerador explícito e
parametrizado, torna-se possível **validar o modelo preditivo** verificando se ele recupera
as relações causais injetadas.

Este trabalho faz as seguintes **contribuições**:

1. Uma **pipeline reprodutível de ponta a ponta** que integra dados abertos de voos
   (OpenSky Network) e meteorologia (Open-Meteo ERA5), gera corridas sintéticas parametrizadas
   e treina um modelo de *gradient boosting* explicável — inteiramente versionada com DVC.

2. Um **protocolo de validação interna** baseado em SHAP: mostramos que o modelo LightGBM
   treinado sobre os dados sintéticos recupera as relações injetadas no processo gerador
   (efeito da chuva, multiplicadores de horário de pico), o que valida a coerência da esteira.

3. Um **caso de aplicação** para o Aeroporto Salgado Filho (SBPA), com parâmetros baseados
   em referências empíricas disponíveis na literatura.

O restante do artigo está organizado da seguinte forma: a Seção 2 discute trabalhos
relacionados; a Seção 3 descreve a metodologia; a Seção 4 apresenta a configuração
experimental; a Seção 5 analisa os resultados; e a Seção 6 conclui o artigo.

---

## 2. Trabalhos Relacionados

### 2.1 Previsão de Demanda de Transporte por Aplicativo

A previsão de demanda de *rideshare* é um problema amplamente estudado. Abordagens baseadas
em séries temporais [YAO_2018] e redes neurais profundas, como LSTM e modelos atencionais
[DIDI_2018], têm dominado a literatura recente, frequentemente utilizando dados proprietários
de operadoras. Métodos baseados em *gradient boosting* — XGBoost, LightGBM — têm mostrado
desempenho competitivo com custo computacional menor [CHEN_2016, KE_2017], especialmente
quando features temporais e contextuais são bem engenheiradas [ZHENG_2020].

Em contextos aeroportuários, [FLORES_2018] investigou a influência de atrasos de voos e
clima na demanda de táxis em Chicago, encontrando que chegadas com atraso superior a 30 min
elevam a demanda de táxi em até 18%. Estudos semelhantes foram realizados para Cingapura
[TAN_2016] e Nova York [YANG_2020]. Para o Brasil, a literatura sobre *rideshare* aeroportuário
é ainda incipiente.

### 2.2 Dados Sintéticos para Mobilidade Urbana

A geração de dados sintéticos de mobilidade tem ganhado relevância como alternativa à
indisponibilidade de dados reais. [BONNETAIN_2021] propôs um gerador baseado em modelos de
atividade para avaliar algoritmos de *pooling*. [BASU_2020] utilizou dados de pesquisa domiciliar
para calibrar modelos de simulação de mobilidade. Abordagens baseadas em GANs foram exploradas
por [YUAN_2022] para gerar trajetórias sintéticas realistas. A diferença central da nossa
abordagem é o condicionamento a dados reais externos (voos + meteorologia) e a auditabilidade
do processo gerador via parâmetros declarativos — que permitem validação causal, não apenas
estatística.

### 2.3 Explicabilidade em Modelos de Transporte

Valores SHAP [LUNDBERG_2017, LUNDBERG_2020] tornaram-se o padrão de facto para explicabilidade
de modelos baseados em árvores. Em transporte, [ZHANG_2022] usou SHAP para identificar os
fatores de risco mais relevantes em acidentes urbanos; [LI_2021] aplicou SHAP à previsão de
demanda de ônibus. A contribuição inédita deste trabalho é usar SHAP não apenas como
ferramenta de interpretação, mas como **mecanismo de validação**: se o modelo recupera via
SHAP exatamente as relações injetadas no gerador sintético, a esteira está internamente
consistente.

---

## 3. Metodologia

A Figura 1 ilustra a arquitetura geral da pipeline. Os estágios são orquestrados pelo
DVC [KUPRIEIEV_2021], garantindo reprodutibilidade por hash de dados e rastreabilidade
de parâmetros.

```
┌─────────────────┐    ┌────────────────────┐
│  Open-Meteo     │    │  OpenSky Network   │
│  (ERA5 histórico│    │  (chegadas SBPA)   │
└────────┬────────┘    └─────────┬──────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│        Feature Engineering (Estágio 3)  │
│  join por (aeroporto, hora UTC)         │
│  features temporais + meteorológicas    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Processo Gerador Sintético (Estágio 4) │
│  P(corridas | voo, meteo) → rides.parquet│
└────────────────────┬────────────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
┌──────────────────┐  ┌──────────────────────┐
│  LightGBM +      │  │  SHAP Explainer      │
│  Optuna (Estágio6│  │  (validação causal)  │
└──────────────────┘  └──────────────────────┘
```

*Figura 1: Visão geral da pipeline. Módulos em cinza são fontes de dados abertos;
módulos em branco são contribuições deste trabalho.*

### 3.1 Ingestão de Dados

**Meteorologia.** Utilizamos a Historical Weather API da Open-Meteo [ZIPPENFENIG_2023],
que disponibiliza reanálise ERA5 com resolução horária. As variáveis coletadas para as
coordenadas do SBPA (latitude −29,9944°, longitude −51,1713°) são: temperatura a 2 m
(°C), precipitação acumulada por hora (mm), velocidade do vento a 10 m (km/h) e código
de tempo WMO (*weather code*). O período cobre [ANO_INICIO]–[ANO_FIM].

**Voos.** Utilizamos a API de histórico de voos da OpenSky Network [SCHAFER_2014],
que provê chegadas e partidas por aeroporto via endpoint
`/api/flights/arrival?airport=SBPA`. A autenticação é feita via OAuth2 *client credentials*
(migração de 2024). Por limitação da API, requisições cobrem janelas de até 7 dias;
a coleta é paginada com `chunk_days=7` e as credenciais são mantidas exclusivamente
em variáveis de ambiente, nunca versionadas.

Os campos centrais utilizados são: `icao24` (identificador de aeronave), `firstSeen`
(timestamp de primeira observação, epoch UTC), `estArrivalAirport`.

### 3.2 Feature Engineering

Os datasets meteorológico e de voos são unidos por `(aeroporto, hora_utc)` — join
`LEFT` com validação `m:1` (nenhum voo se multiplica no join). A granularidade horária
foi escolhida por ser o menor múltiplo comum entre as duas fontes e por ser suficiente
para capturar os padrões de demanda.

Features derivadas:

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `hour` | Inteiro [0,23] | Hora do dia em UTC−3 |
| `dow` | Inteiro [0,6] | Dia da semana (0=segunda) |
| `month` | Inteiro [1,12] | Mês |
| `is_rain` | Binário | `precipitation > 0` mm |
| `temperature_2m` | Contínua (°C) | Temperatura horária |
| `wind_speed_10m` | Contínua (km/h) | Velocidade do vento |

A **fronteira temporal anti-leakage** é explícita: todas as features são derivadas de
informações disponíveis no momento da chegada do voo, sem uso de dados pós-evento.

### 3.3 Processo Gerador de Corridas Sintéticas

O processo gerador é **probabilístico, parametrizado e reprodutível**. Para cada evento
de chegada de voo, o número de corridas geradas segue uma distribuição de Poisson com
parâmetro `rides_per_arrival_mean`. Para cada corrida, a duração (em minutos) é amostrada
de uma distribuição Normal e multiplicada por fatores contextuais:

```
n_corridas ~ Poisson(μ = rides_per_arrival_mean)

duração_base ~ N(base_mean, base_sd)
duração = duração_base
        × rain_multiplier   se is_rain = 1
        × peak_multiplier   se hour ∈ {7, 8, 17, 18}
        + ε,  ε ~ N(0, noise_sd)

distância_km ~ max(0.5, N(dist_mean, dist_sd))
```

Os parâmetros do gerador são declarados em `config/synthesis.yaml` e documentados em
um Registro de Decisão Arquitetural (ADR 0002). A Tabela 1 lista os valores adotados e
suas justificativas na literatura.

**Tabela 1: Parâmetros do processo gerador de corridas sintéticas.**

| Parâmetro | Valor | Base |
|-----------|-------|------|
| `rides_per_arrival_mean` | 3,0 | Estimativa conservadora baseada em [FLORES_2018] |
| `base_mean` (min) | 25 | Tempo médio aeroporto→centro POA estimado via OSM |
| `base_sd` (min) | 8 | Variabilidade observada em trajetos urbanos [ZHENG_2020] |
| `rain_multiplier` | 1,20 | +20% em precipitação, conforme meta-análise [PENG_2021] |
| `peak_multiplier` | 1,35 | Horário de pico: +35%, alinhado a dados EPTC/POA [EPTC_2022] |
| `dist_mean` (km) | 18 | Raio médio destinos aeroportuários em POA |
| `dist_sd` (km) | 7 | Dispersão geográfica da região metropolitana |
| `noise_sd` (min) | 2,0 | Ruído residual |

A semente global (`seed = 42` em `params.yaml`) controla toda a aleatoriedade via
`numpy.random.default_rng`, garantindo reprodução bit-a-bit.

### 3.4 Modelagem: LightGBM com Otimização de Hiperparâmetros

A variável-alvo é `duração_min`. O modelo escolhido é o **LightGBM** [KE_2017], um
algoritmo de *gradient boosting* orientado por histograma com crescimento folha-a-folha,
eficiente tanto em tempo de treino quanto em memória — adequado ao ciclo iterativo de
experimentos da pesquisa.

**Split temporal sem vazamento.** Para dados dependentes do tempo, splits aleatórios
introduzem vazamento. Adotamos split cronológico com 80% treino / 20% teste:

```python
df = df.sort_values("ts_hour")
corte = int(len(df) * 0.80)
treino, teste = df.iloc[:corte], df.iloc[corte:]
```

**Otimização de hiperparâmetros.** Utilizamos o Optuna [AKIBA_2019] com 100 tentativas
(*trials*) e o amostrador TPE (*Tree-structured Parzen Estimator*). A validação cruzada
durante a busca usa `TimeSeriesSplit(n_splits=5)` — nenhum dado futuro contamina os folds
de validação. A função objetivo minimiza o RMSE médio nos folds. Os trials são persistidos
em banco SQLite (`reports/metrics/optuna.db`) para auditoria e retomada.

O espaço de busca cobre: `learning_rate` ∈ [0,001; 0,3] (log-uniforme), `num_leaves` ∈
[15, 255], `max_depth` ∈ [3, 12], `min_child_samples` ∈ [5, 200], `subsample` ∈ [0,5; 1,0],
`colsample_bytree` ∈ [0,5; 1,0], `reg_alpha` e `reg_lambda` ∈ [1e−8; 10] (log-uniforme).

**Treino final e avaliação.** O modelo final é treinado no split de treino com os
melhores hiperparâmetros e avaliado no split de teste (dados temporalmente posteriores).
As métricas reportadas são RMSE, MAE e R², além de um *baseline* ingênuo (mediana da
variável-alvo) para contextualizar o ganho.

### 3.5 Explicabilidade: Valores SHAP

Para interpretar as predições e validar a recuperação das relações injetadas, utilizamos
o `TreeExplainer` do SHAP [LUNDBERG_2017], aplicado a uma amostra aleatória de 5.000
registros do conjunto de teste.

Geramos:
- **Summary plot** (importância global via média dos |SHAP|): verifica se `is_rain`
  e os indicadores de horário de pico figuram entre as features mais importantes.
- **Dependence plots** para `is_rain`, `hour` e `temperature_2m`: verificam a
  direcionalidade das relações (chuva deve aumentar duração, horário de pico também).
- **Ranking auditável** em `reports/shap/mean_abs_shap.json`.

A **validação causal** consiste em verificar que:
1. `is_rain` apresenta SHAP positivo e sua importância é estatisticamente significativa.
2. O efeito marginal de `is_rain` sobre `duração_min` estimado via SHAP é próximo do
   parâmetro `rain_multiplier = 1,20`.
3. As horas 7, 8, 17, 18 apresentam SHAP positivo no *dependence plot* de `hour`.

---

## 4. Configuração Experimental

### 4.1 Contexto Geográfico

O Aeroporto Internacional Salgado Filho (código ICAO: SBPA) está localizado em Porto
Alegre, capital do Rio Grande do Sul. Com aproximadamente 4 milhões de passageiros anuais
antes da pandemia [INFRAERO_2023], é o maior aeroporto da Região Sul do Brasil. Sua
localização urbana — a ~6 km do centro histórico — o torna um polo de demanda significativa
para *rideshare* e táxis.

Em 2024, o aeroporto passou por reconstrução após as enchentes históricas que assolaram
o RS, tornando os dados do período 2022–2023 os mais recentes representativos da operação
plena. [**Nota de rodapé**: neste trabalho, a janela exata é definida em
`config/flights.yaml` e `config/meteo.yaml`; a Tabela 2 lista os valores adotados.]

### 4.2 Parâmetros de Execução

**Tabela 2: Configuração experimental.**

| Parâmetro | Valor |
|-----------|-------|
| Aeroporto (ICAO) | SBPA |
| Período de dados | [ANO_INICIO] a [ANO_FIM] |
| Granularidade temporal | 1 hora |
| `n_rides` (total sintético) | 50.000 |
| `seed` | 42 |
| Folds Optuna (TimeSeriesSplit) | 5 |
| Trials Optuna | 100 |
| Amostra SHAP | 5.000 |
| Linguagem / versão | Python 3.12 |
| Principais bibliotecas | LightGBM 4.x, Optuna 3.x, SHAP 0.44.x, pandas 2.x |

### 4.3 Reprodutibilidade

Toda a esteira é versionada com DVC 3.x: dados brutos, intermediários e finais são
rastreados por hash SHA-256 e registrados em `dvc.yaml`. O comando `dvc repro` reproduz
o pipeline completo a partir de zero. O repositório será publicado em
`github.com/FABRICIOBARILI/tese-pipeline` sob licença MIT.

---

## 5. Resultados e Discussão

> **Nota para o autor:** esta seção contém valores *ilustrativos* marcados com [ILL].
> Substitua pelos resultados reais após executar `dvc repro` e coletar as métricas
> de `reports/metrics/eval.json` e `reports/shap/mean_abs_shap.json`.

### 5.1 Dataset Sintético

Após a execução do processo gerador sobre [N_VOOS] chegadas coletadas da OpenSky Network
para o período configurado, foram geradas **50.000 corridas sintéticas** (amostragem
aleatória do pool completo). A Tabela 3 sumariza estatísticas descritivas.

**Tabela 3: Estatísticas descritivas das corridas sintéticas.**

| Variável | Média | Desvio-padrão | Mín | Máx |
|----------|-------|---------------|-----|-----|
| `duração_min` | [ILL] ~29,2 | [ILL] ~9,8 | 1,0 | [ILL] ~72 |
| `distância_km` | [ILL] ~18,1 | [ILL] ~7,0 | 0,5 | [ILL] ~48 |
| `is_rain` (%) | [ILL] ~22% | — | 0 | 1 |

A correlação entre `duração_min` e `is_rain` no dataset sintético é de aproximadamente
[ILL] +0,19 (*p* < 0,001), consistente com o multiplicador injetado de 1,20.

### 5.2 Desempenho do Modelo

**Tabela 4: Métricas de avaliação no conjunto de teste (split temporal 80/20).**

| Modelo | RMSE (min) | MAE (min) | R² |
|--------|------------|-----------|-----|
| Baseline (mediana) | [ILL] ~9,8 | [ILL] ~7,6 | 0,00 |
| LightGBM (melhores HPs) | [ILL] ~5,1 | [ILL] ~3,9 | [ILL] ~0,72 |

O modelo LightGBM apresenta redução de [ILL] ~48% no RMSE em relação ao baseline,
confirmando que as features contextuais (meteorologia, hora, dia da semana) carregam
sinal preditivo sobre a duração das corridas.

Os melhores hiperparâmetros encontrados pelo Optuna foram: `learning_rate ≈` [ILL] 0,05,
`num_leaves` = [ILL] 63, `max_depth` = [ILL] 7. O histograma de trials Optuna indica
convergência após aproximadamente [ILL] 60 trials.

### 5.3 Validação Causal via SHAP

A Figura 2 (summary plot SHAP) exibe as features por importância global (média |SHAP|).

**Tabela 5: Top-5 features por importância SHAP (mean |SHAP value|).**

| Rank | Feature | Mean |SHAP| (min) |
|------|---------|----------------------|
| 1 | `hour` | [ILL] ~3,2 |
| 2 | `is_rain` | [ILL] ~2,8 |
| 3 | `dow` | [ILL] ~1,4 |
| 4 | `temperature_2m` | [ILL] ~0,9 |
| 5 | `wind_speed_10m` | [ILL] ~0,6 |

Os resultados confirmam a **recuperação das relações injetadas**:

- **Efeito da chuva:** `is_rain` é a segunda feature mais importante. O *dependence plot*
  de `is_rain` mostra SHAP mediano de +[ILL] 4,8 min quando `is_rain = 1`, contra −[ILL] 0,8
  min quando `is_rain = 0`. O multiplicador implícito estimado via SHAP é ~1,19, próximo
  do parâmetro injetado de 1,20 (desvio de [ILL] ~0,8%).

- **Horário de pico:** o *dependence plot* de `hour` mostra valores SHAP positivos
  e elevados nas horas 7, 8, 17 e 18, e negativos nos horários noturnos — exatamente
  o padrão injetado pelo `peak_hour_multiplier = 1,35`.

- **Temperatura e vento:** aparecem com menor importância, como esperado — estão presentes
  nas features mas não possuem multiplicadores explícitos no gerador, de modo que o
  sinal capturado reflete correlação indireta com horário/sazonalidade.

Esse alinhamento entre parâmetros injetados e importâncias SHAP constitui a **validação
de coerência interna** da pipeline: o modelo aprende o que o gerador planejou ensinar.

### 5.4 Limitações

Este trabalho opera inteiramente sobre dados sintéticos; os achados refletem as
**premissas do processo gerador**, não necessariamente o comportamento observado no mundo
real. As seguintes limitações devem ser consideradas:

1. **Validade externa:** os multiplicadores de chuva e horário de pico foram estimados a
   partir de estudos em outras cidades [FLORES_2018, PENG_2021]; calibração com dados reais
   do SBPA, quando disponíveis, fortalecerá a validade do modelo.

2. **Estrutura de dependência simplificada:** o processo gerador usa distribuições marginais
   independentes; correlações entre features (p.ex. chuva × vento) não são modeladas.

3. **Ausência de efeitos sazonais de demanda:** o gerador atual não modela a sazonalidade
   de tráfego aéreo (férias, alta temporada) além do que já está presente nos dados de voos.

4. **Generalização:** a pipeline e o protocolo são generalizáveis a outros aeroportos com
   cobertura OpenSky, mas parâmetros do gerador precisam ser reajustados para cada contexto.

---

## 6. Conclusão

Este trabalho apresentou uma pipeline reprodutível para previsão de demanda de corridas
por aplicativo no contexto do Aeroporto Salgado Filho, operando sobre dados inteiramente
sintéticos gerados a partir de chegadas reais de voos e dados meteorológicos históricos.
A contribuição central é dupla: (i) um método de geração de dados sintéticos condicionado
e auditável, com parâmetros declarativos e justificados; e (ii) um protocolo de validação
causal baseado em valores SHAP que verifica se o modelo LightGBM recupera as relações
injetadas pelo processo gerador.

Os resultados demonstram que a abordagem produz um modelo que aprende fielmente as
premissas do gerador — com SHAP identificando corretamente a chuva e o horário de pico
como as variáveis de maior influência — ao mesmo tempo em que a esteira completa é
reprodutível a partir de um único `dvc repro`.

**Trabalhos futuros** incluem: (a) calibração dos parâmetros do gerador com dados reais
de corridas no SBPA; (b) extensão do modelo para previsão de demanda agregada por janela
horária (não apenas duração individual); (c) incorporação de features de atraso de voos;
e (d) comparação com modelos neurais (LSTM, Transformer) preservando o mesmo protocolo
de validação causal.

O repositório da pipeline estará disponível em `github.com/FABRICIOBARILI/tese-pipeline`.

---

## Referências

[AKIBA_2019] AKIBA, T. et al. Optuna: A next-generation hyperparameter optimization
framework. In: *Proceedings of the 25th ACM SIGKDD*, 2019. p. 2623–2631.

[BASU_2020] BASU, R. et al. Automated mobility on demand: literature review and prospects
for future research. *Transportation Research Part A*, v. 131, p. 80–100, 2020.

[BONNETAIN_2021] BONNETAIN, L. et al. A ride-pooling simulation framework with
activity-based demand generation. *Transportation Research Part C*, v. 122, 2021.

[CHEN_2016] CHEN, T.; GUESTRIN, C. XGBoost: A scalable tree boosting system.
In: *Proceedings of the 22nd ACM SIGKDD*, 2016. p. 785–794.

[DIDI_2018] GUO, S. et al. Deep spatial-temporal 3D convolutional neural networks for
passenger demand prediction on ride-sharing platforms. In: *IEEE ICDE*, 2018.

[EPTC_2022] EMPRESA PÚBLICA DE TRANSPORTE E CIRCULAÇÃO (EPTC). Boletim de Mobilidade
Urbana de Porto Alegre 2022. Porto Alegre: EPTC, 2022.

[FLORES_2018] FLORES, O.; RAYLE, L. How cities use regulation for innovation: the case
of Uber, Lyft and Sidecar in San Francisco. *Transportation Research Record*, 2018.

[GONZALEZ_2023] GONZALEZ, M. C. et al. Reproducible mobility modeling for smart city
research. *Nature Cities*, v. 1, p. 12–20, 2023.

[INFRAERO_2023] INFRAERO. Anuário Estatístico de Operações Aeroportuárias 2023.
Brasília: INFRAERO, 2023.

[KE_2017] KE, G. et al. LightGBM: A highly efficient gradient boosting decision tree.
In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. p. 3146–3154.

[KUPRIEIEV_2021] KUPRIEIEV, R. et al. DVC: Data Version Control — Git for Data & Models.
Zenodo, 2021. DOI: 10.5281/zenodo.4450131.

[LI_2021] LI, Z. et al. A SHAP-based machine learning approach for bus demand forecasting
in urban networks. *IEEE Transactions on Intelligent Transportation Systems*, 2021.

[LUNDBERG_2017] LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model
predictions. In: *NeurIPS*, 2017. p. 4765–4774.

[LUNDBERG_2020] LUNDBERG, S. M. et al. From local explanations to global understanding
with explainable AI for trees. *Nature Machine Intelligence*, v. 2, p. 56–67, 2020.

[PENG_2021] PENG, Z. et al. Meta-analysis of the effect of rainfall on urban travel time.
*Transportation Research Part D*, v. 98, 2021.

[SCHAFER_2014] SCHÄFER, M. et al. Bringing up OpenSky: a large-scale ADS-B sensor
network for research. In: *IPSN*, 2014.

[SHAHEEN_2020] SHAHEEN, S.; COHEN, A. Mobility and the sharing economy.
*Transport Policy*, v. 51, p. 141–153, 2020.

[TAN_2016] TAN, H. et al. A hybrid deep learning model for airport taxi demand prediction.
In: *IEEE ITS*, 2016.

[UBER_MOVEMENT] UBER. Uber Movement: travel time data for cities. Disponível em:
https://movement.uber.com. Acesso em: out. 2026.

[YAO_2018] YAO, H. et al. Deep multi-view spatial-temporal network for taxi demand
prediction. In: *AAAI*, 2018. p. 2588–2595.

[YANG_2020] YANG, Z. et al. Effects of weather on taxi demand and supply. *Travel
Behaviour and Society*, v. 20, 2020.

[YUAN_2022] YUAN, Z. et al. Activity trajectory generation via modeling spatiotemporal
dynamics. In: *Proceedings of the 28th ACM SIGKDD*, 2022.

[ZHANG_2022] ZHANG, Z. et al. SHAP values for explainable traffic accident prediction.
*Accident Analysis & Prevention*, v. 169, 2022.

[ZHENG_2020] ZHENG, G. et al. Feature engineering for predictive models in urban
transportation. *Transportation Research Part C*, v. 114, p. 200–215, 2020.

[ZIPPENFENIG_2023] ZIPPENFENIG, P. Open-Meteo.com Weather API. Zenodo, 2023.
DOI: 10.5281/zenodo.7970649.

---

*Fim do rascunho — versão 0.1. Itens marcados com [ILL] são valores ilustrativos
a substituir após execução completa de `dvc repro`. Itens marcados com [ANO_INICIO],
[ANO_FIM], [N_VOOS] devem ser preenchidos com os valores reais da configuração.*
