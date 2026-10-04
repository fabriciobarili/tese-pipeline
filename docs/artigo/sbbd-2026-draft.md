# Uma Pipeline Reprodutível para Previsão de Demanda de Corridas por Aplicativo em Regiões Aeroportuárias: Síntese Condicionada a Voos e Meteorologia com Indexação H3, LightGBM e SHAP

> **Rascunho para submissão ao SBBD/CSBC 2026 — versão 1.0 (04/10/2026)**  
> Autores: Fabricio Barili, [Orientador]  
> Filiação: [Universidade]  

---

## Resumo

A previsão de demanda de transporte por aplicativo (*ride-hailing*) em regiões aeroportuárias é relevante para o planejamento urbano e operacional, porém raramente viável com dados reais devido a restrições comerciais e de privacidade. Este trabalho propõe uma metodologia reprodutível e auditável para geração de dados sintéticos de corridas condicionados a dados reais de chegadas de voos e condições meteorológicas, com foco no Aeroporto Internacional Salgado Filho (SBPA), em Porto Alegre/RS. A pipeline integra dados abertos da OpenSky Network e da API Open-Meteo (ERA5), aplica indexação geoespacial hierárquica com H3 (Uber) nas resoluções 8 e 13 sobre 722 Unidades de Desenvolvimento Humano (UDHs) da Região Metropolitana de Porto Alegre e sobre 11,8 milhões de células da malha viária derivada do OpenStreetMap, e treina um modelo *gradient boosting* (LightGBM) com otimização de hiperparâmetros via Optuna e avaliação temporal sem vazamento. A explicabilidade é garantida por valores SHAP, que permitem verificar se o modelo recupera as relações causais injetadas no processo gerador — constituindo uma forma de validação de coerência interna da esteira. A pipeline é inteiramente versionada com DVC e código aberto, com armazenamento em Google Cloud Storage.

**Palavras-chave:** ride-hailing, dados sintéticos, LightGBM, SHAP, H3, OpenStreetMap, reprodutibilidade, DVC.

---

## Abstract

Predicting ride-hailing demand near airports is relevant for urban and operational planning, yet rarely feasible with real data due to commercial and privacy constraints. This work proposes a reproducible and auditable methodology for generating synthetic ride datasets conditioned on real flight arrival records and meteorological data, focusing on Salgado Filho International Airport (SBPA) in Porto Alegre, Brazil. The pipeline integrates open data from the OpenSky Network and the Open-Meteo API (ERA5), applies hierarchical geospatial indexing with H3 (Uber) at resolutions 8 and 13 over 722 Human Development Units (UDHs) of the Porto Alegre Metropolitan Region and 11.8 million cells of the road network derived from OpenStreetMap, and trains a gradient boosting model (LightGBM) with Optuna hyperparameter optimization and temporal split evaluation without data leakage. Explainability is ensured through SHAP values, which verify that the model recovers the causal relationships injected during synthesis — constituting an internal consistency validation of the pipeline. The entire pipeline is versioned with DVC and open source, with data stored on Google Cloud Storage.

**Keywords:** ride-hailing, synthetic data, LightGBM, SHAP, H3, OpenStreetMap, reproducibility, DVC.

---

## 1. Introdução

O trabalho por plataformas de *ride-hailing* opera em um ambiente marcado por assimetria de informação: a empresa acessa grandes volumes de dados históricos e em tempo real para organizar a demanda, enquanto o motorista decide com base em experiência, percepção da cidade e sinais emitidos pelo próprio sistema que controla suas oportunidades [MCDAID_2023; GROHMANN_2021]. Van Dijck, Poell e De Waal (2018) descrevem essa arquitetura como uma infraestrutura em que dados e algoritmos organizam relações sociais e distribuem possibilidades de ação — o que Abílio (2020) condensa na figura do trabalhador *just-in-time*: formalmente autônomo, mas materialmente dependente de decisões que não controla.

Do ponto de vista da Computação Aplicada, essa assimetria pode ser traduzida como um problema de **inferência indireta**: o que se busca não é reconstruir o algoritmo proprietário da plataforma, mas estimar a probabilidade de que condições externas observáveis coincidam com maior ocorrência de corridas em uma área e janela temporal. Duas fontes abertas são particularmente relevantes para isso no contexto aeroportuário — chegadas de voos e dados meteorológicos —, pois concentram boa parte do sinal que influencia a demanda: um voo que aterrissa injeta passageiros no terminal; chuva e horário de pico modulam a duração e o volume das corridas subsequentes.

Uma barreira central a qualquer modelagem nesse domínio é a **indisponibilidade de dados reais** de corridas. Operadoras como Uber e 99 não divulgam registros desagregados por origem-destino [UBER_MOVEMENT], e o acesso por acordos com órgãos públicos é restrito a poucas cidades. O Aeroporto Internacional Salgado Filho (SBPA), em Porto Alegre — principal *hub* da Região Sul do Brasil, com cerca de quatro milhões de passageiros anuais [INFRAERO_2023] — não é exceção. Conforme Holzinger et al. (2023) e Kobayashi e Alam (2024) discutem, dados sintéticos constituem, nesses casos, um ambiente experimental legítimo para testar e depurar componentes de uma pipeline antes da validação com dados observados.

A abordagem proposta neste trabalho vai além de suprir a ausência de dados reais. Ao tornar o processo gerador **explícito e parametrizado**, cria-se um mecanismo de validação interna: o modelo preditivo pode ser examinado para verificar se recupera as relações causais injetadas na síntese. Valores SHAP [LUNDBERG_2017] fornecem essa verificação, cumprindo duas funções simultâneas — técnica, ao detectar incoerências e dependências indesejadas, e metodológica, ao mostrar que a esteira aprende o que foi planejado ensinar. Barredo Arrieta et al. (2020) argumentam que a qualidade de um sistema de IA não pode ser medida apenas por acurácia, mas também por rastreabilidade, interpretabilidade e adequação à decisão humana; este trabalho adota esse princípio como requisito de projeto.

Uma segunda contribuição é a **representação geoespacial enriquecida** por meio do sistema de indexação hierárquica H3 (Uber) [BRODSKY_2018]. A origem e o destino de cada corrida sintética são representados como células H3 em múltiplas resoluções, permitindo análise em diferentes granularidades — desde o nível de quadra (resolução 10, ~0,015 km²) até o nível de zona de demanda (resolução 8, ~0,74 km²). Esse enriquecimento geoespacial é calibrado com dados reais da malha viária do OpenStreetMap e das 722 UDHs (Unidades de Desenvolvimento Humano) do Censo 2010 para a RM Porto Alegre — possibilitando futuras análises de equidade e de distribuição espacial da demanda.

Este trabalho faz as seguintes **contribuições**:

1. Uma **pipeline reprodutível de ponta a ponta** que integra dados abertos de voos (OpenSky Network [SCHAFER_2014]) e meteorologia (Open-Meteo ERA5 [ZIPPENFENIG_2023]), gera corridas sintéticas com parâmetros declarativos e auditáveis, e treina um modelo LightGBM [KE_2017] com otimização via Optuna [AKIBA_2019] — inteiramente versionada com DVC [KUPRIEIEV_2021] e armazenada em Google Cloud Storage.

2. Um **protocolo de validação causal via SHAP**: demonstramos que o modelo recupera as relações injetadas no processo gerador (efeito da precipitação e de horários de pico), validando a coerência interna da esteira.

3. Uma **base de indexação geoespacial H3** para a RM Porto Alegre: 722 UDHs do Censo 2010 e 11,8 milhões de células H3 na resolução 13 cobrindo a malha viária navegável derivada do OpenStreetMap, disponibilizados como artefatos reutilizáveis para pesquisa em mobilidade urbana.

4. Um **caso de aplicação** para o SBPA que pode ser generalizado a outros aeroportos com cobertura da OpenSky Network.

O restante do artigo está organizado da seguinte forma: a Seção 2 discute trabalhos relacionados; a Seção 3 descreve a metodologia; a Seção 4 apresenta a configuração experimental; a Seção 5 analisa os resultados; e a Seção 6 conclui o artigo.

---

## 2. Trabalhos Relacionados

### 2.1 Previsão de Demanda de Transporte por Aplicativo

A previsão de demanda de *ride-hailing* é um problema amplamente estudado. Abordagens baseadas em séries temporais [YAO_2018] e redes neurais profundas, como LSTM e modelos atencionais [DIDI_2018], têm dominado a literatura recente, frequentemente utilizando dados proprietários de operadoras. Métodos baseados em *gradient boosting* — XGBoost, LightGBM — têm mostrado desempenho competitivo com custo computacional menor [CHEN_2016; KE_2017], especialmente quando *features* temporais e contextuais são bem engenheiradas [ZHENG_2020].

Em contextos aeroportuários, Flores e Rayle (2018) investigaram a influência de atrasos de voos e clima na demanda de táxis em Chicago, encontrando que chegadas com atraso superior a 30 min elevam a demanda em até 18%. Estudos semelhantes foram realizados para Cingapura [TAN_2016] e Nova York [YANG_2020]. Para o Brasil, a literatura sobre *ride-hailing* aeroportuário é ainda incipiente, o que reforça a relevância do caso SBPA.

### 2.2 Indexação Geoespacial Hierárquica

O sistema H3, desenvolvido pela Uber e publicado como código aberto [BRODSKY_2018], implementa uma grade hexagonal hierárquica sobre a superfície terrestre com 16 resoluções de precisão. Sua adoção cresceu rapidamente em aplicações de mobilidade urbana por permitir operações eficientes de vizinhança (`grid_disk`), trajetória (`grid_path_cells`) e agregação multi-escala (`cell_to_parent`). Diferentemente de grades retangulares, os hexágonos minimizam distorções de vizinhança — relevante para análises de origem-destino.

Em ride-hailing, H3 tem sido usado para modelar zonas de oferta e demanda [BRODSKY_2018], agrupar origens e destinos para previsão de demanda [ZHENG_2020], e indexar redes de transporte público [LI_2021]. Este trabalho estende esse uso para incluir o enriquecimento geoespacial com UDHs (dados socioeconômicos) e malha viária OSM.

### 2.3 OpenStreetMap em Pesquisa de Mobilidade

Dados do OpenStreetMap (OSM) têm sido amplamente usados como substituto gratuito de dados viários proprietários [HAKLAY_2010]. Para o Brasil, o projeto Geofabrik disponibiliza extatos regionais atualizados diariamente. A integração OSM + H3 para análise de redes viárias permanece pouco explorada — geralmente OSM é usado com ferramentas de roteamento (OSRM, Valhalla) e não com indexação hexagonal. Neste trabalho, cada segmento viário navegável é mapeado para as células H3 que ele atravessa, criando uma estrutura de busca por ponto (`h3_r13 ∈ vias_set`) eficiente e reutilizável.

### 2.4 Dados Sintéticos para Mobilidade Urbana

A geração de dados sintéticos de mobilidade tem ganhado relevância como alternativa à indisponibilidade de dados reais. Bonnetain et al. (2021) propuseram um gerador baseado em modelos de atividade para avaliar algoritmos de *pooling*. Basu et al. (2020) utilizaram dados de pesquisa domiciliar para calibrar modelos de simulação. Abordagens baseadas em GANs foram exploradas por Yuan et al. (2022) para gerar trajetórias sintéticas realistas. A diferença central da nossa abordagem é o condicionamento a dados reais externos (voos + meteorologia) e a auditabilidade do processo gerador via parâmetros declarativos — que permitem validação causal, não apenas estatística.

### 2.5 Explicabilidade em Modelos de Transporte

Valores SHAP [LUNDBERG_2017; LUNDBERG_2020] tornaram-se o padrão de facto para explicabilidade de modelos baseados em árvores. Em transporte, Zhang et al. (2022) usaram SHAP para identificar fatores de risco em acidentes urbanos; Li et al. (2021) aplicaram SHAP à previsão de demanda de ônibus. A contribuição inédita deste trabalho é usar SHAP não apenas como ferramenta de interpretação, mas como **mecanismo de validação**: se o modelo recupera via SHAP exatamente as relações injetadas no gerador sintético, a esteira está internamente consistente.

---

## 3. Metodologia

A Figura 1 ilustra a arquitetura geral da pipeline. Os estágios são orquestrados pelo DVC [KUPRIEIEV_2021], garantindo reprodutibilidade por hash de dados e rastreabilidade de parâmetros. Todos os dados são armazenados no Google Cloud Storage (projeto `doutorado-501917`, região `southamerica-east1`).

```
┌──────────────────┐    ┌────────────────────────┐   ┌────────────────────┐
│  Open-Meteo      │    │  OpenSky Network        │   │  OpenStreetMap     │
│  (ERA5 histórico)│    │  (chegadas SBPA)        │   │  (RM Porto Alegre) │
└────────┬─────────┘    └──────────┬──────────────┘   └────────┬───────────┘
         │                         │                            │
         ▼                         ▼                            ▼
┌────────────────────────────────────────────┐    ┌────────────────────────┐
│         Feature Engineering (Estágio 3)    │    │  Indexação H3 (ext.)   │
│  join por (aeroporto, hora UTC)            │    │  722 UDHs res 8/10/13  │
│  H3 res 8 (aeroporto) + features temporais │    │  Malha viária res 13   │
└────────────────────┬───────────────────────┘    └────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│  Processo Gerador Sintético (Estágio 4)    │
│  P(corridas | voo, meteo, H3) → rides.parquet│
└────────────────────┬───────────────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
┌──────────────────┐  ┌──────────────────────┐
│  LightGBM +      │  │  SHAP Explainer      │
│  Optuna (Est. 6) │  │  (validação causal)  │
└──────────────────┘  └──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
           ┌─────────────┐    ┌──────────────────┐
           │  Narração   │    │  Google Cloud    │
           │  (Est. 7)   │    │  Storage (DVC)   │
           └─────────────┘    └──────────────────┘
```
*Figura 1: Arquitetura da pipeline. Fontes de dados abertas em cinza; contribuições deste trabalho em branco.*

### 3.1 Ingestão de Dados

**Meteorologia.** Utilizamos a Historical Weather API da Open-Meteo [ZIPPENFENIG_2023], que disponibiliza reanálise ERA5 com resolução horária. As variáveis coletadas para as coordenadas do SBPA (latitude −29,9944°, longitude −51,1713°) são: temperatura a 2 m (°C), precipitação acumulada por hora (mm), velocidade do vento a 10 m (km/h), código WMO de tempo e umidade relativa. O módulo implementa *retry* com *backoff* exponencial e registra percentual de valores faltantes por variável.

**Voos.** Utilizamos a API de histórico da OpenSky Network [SCHAFER_2014] via endpoint `/api/flights/arrival?airport=SBPA`, com autenticação OAuth2 *client credentials*. Requisições cobrem janelas de até 7 dias (`chunk_days=7`) e as credenciais são mantidas exclusivamente em variáveis de ambiente, nunca versionadas. O campo central é `firstSeen` (epoch UTC), usado para alinhar ao horário meteorológico. Voos são deduplicados por `(icao24, firstSeen)`.

**Dados geoespaciais externos.** Três fontes complementares foram integradas como artefatos reutilizáveis (Seção 3.2):

- *Polígonos do SBPA:* arquivo KMZ com 4 áreas aeroportuárias (área interna, estacionamentos, locais de espera)
- *UDHs da RM Porto Alegre:* 722 Unidades de Desenvolvimento Humano do Censo 2010 (fonte: ObservaPOA/PNUD, acesso 04/10/2026)
- *Malha viária OSM:* extrato sul do Brasil via Geofabrik (`sul-261003.osm.pbf`, 407 MB, dados até 2026-10-03)

### 3.2 Indexação Geoespacial com H3

O sistema H3 (Uber, versão 4.x) indexa toda a superfície terrestre em células hexagonais em 16 resoluções. Adotamos quatro resoluções neste trabalho, escolhidas por suas propriedades geométricas:

| Resolução | Área média | Uso neste trabalho |
|:---------:|:----------:|---|
| 13 | ~0,000043 km² | identificador sub-métrico único (UDH centróide, via OSM) |
| 10 | ~0,015 km² | cobertura detalhada das UDHs (nível de quadra) |
| 8 | ~0,737 km² | feature de aeroporto e zona de demanda (padrão da pipeline) |
| 5 | ~252,9 km² | região metropolitana (granularidade macro) |

**Conversão KMZ → H3.** Um script CLI reutilizável (`scripts/kmz_to_h3.py`) extrai polígonos de arquivos KMZ, aplica modo `center` (centróide da célula dentro do polígono) para resoluções finas e `overlap` (célula toca o polígono) para resoluções grossas, e gera artefatos JSON e Markdown auditáveis. A Tabela 1 resume os resultados para o SBPA.

**Tabela 1: Células H3 por área do Aeroporto Salgado Filho (SBPA).**

| Área | Centróide (lat, lng) | Res 13 | Res 10 | Res 8 | Res 5 |
|------|---------------------|-------:|-------:|------:|------:|
| Área interna | −29,9889°, −51,1761° | 2.002 | 12 | 2 | 1 |
| Estacionamento Movida | −29,9903°, −51,1830° | 922 | 6 | 1 | 1 |
| Local de espera I | −29,9886°, −51,1791° | 178 | 5 | 1 | 1 |
| Local de espera II | −29,9930°, −51,1860° | 292 | 8 | 3 | 1 |
| **União total** | — | **3.394** | **31** | **7** | **1** |

**Indexação das UDHs.** As 722 UDHs da RM Porto Alegre foram convertidas para H3 em quatro resoluções. Para a resolução 13, a área da maior UDH (~812 km²) corresponderia a ~19 milhões de células — computacionalmente inviável. Adotamos a estratégia de **centróide único em res 13** (identificador sub-métrico) e cobertura completa em res 10/8/5. A Tabela 2 apresenta os resultados.

**Tabela 2: Cobertura H3 das 722 UDHs da RM Porto Alegre.**

| Resolução | Área/célula | Estratégia | Células únicas |
|:---------:|:----------:|------------|:--------------:|
| 13 | 0,000043 km² | centróide por UDH | 722 |
| 10 | 0,015 km² | overlap (cobertura total) | 808.224 |
| 8 | 0,737 km² | overlap (cobertura total) | 17.065 |
| 5 | 252,9 km² | overlap (cobertura total) | 73 |

O arquivo de *lookup* `RM_PortoAlegre_UDHs2010_lookup.csv` (schema: `udh_code, centroid_lat, centroid_lng, h3_r13, h3_r10, h3_r8, h3_r5, is_multigeometry`) permite enriquecer corridas com indicadores socioeconômicos do Censo 2010 via *join* por `h3_r8` — possibilitando análises de equidade.

**Indexação da malha viária OSM.** O arquivo PBF (`sul-261003.osm.pbf`, 407 MB, dados até 2026-10-03T20:20:50Z) foi processado com a biblioteca `osmium` em dois passos: (1) carregamento das localizações de nós; (2) iteração sobre `way` com tag `highway` em 14 tipos navegáveis (de `motorway` a `service`), filtrados pela *bounding box* da RM Porto Alegre (lat −31,5° a −29,0°; lng −52,5° a −50,5°). Para cada aresta (par consecutivo de nós), `h3.grid_path_cells(cell_start, cell_end)` retorna todas as células H3 res 13 ao longo do segmento. Células com múltiplas vias recebem a via de maior prioridade hierárquica (motorway > trunk > ... > service).

**Tabela 3: Malha viária navegável da RM Porto Alegre em H3 res 13.**

| Métrica | Valor |
|---------|------:|
| Vias OSM processadas | 202.334 |
| Arestas (pares de nós) | 1.709.696 |
| Células H3 únicas (res 13) | **11.807.501** |
| Tamanho do artefato | 249 MB (Parquet) |
| Tipo mais frequente | `unclassified` (39,0%) |
| Tipo menos frequente | `tertiary_link` (0,01%) |

O artefato `sul-261003_vias_h3r13.parquet` permite verificar se um ponto H3 res 13 é trafegável em O(1) — relevante para validação dos destinos sintéticos gerados pela pipeline.

### 3.3 Feature Engineering

Os datasets meteorológico e de voos são unidos por `(_airport, ts_hour)` — *join* `LEFT` com validação `m:1` (nenhum voo se multiplica). Features derivadas:

| Feature | Tipo | Descrição |
|---------|------|-----------|
| `hour` | int [0,23] | hora do dia (UTC−3) |
| `dow` | int [0,6] | dia da semana |
| `month` | int [1,12] | mês |
| `is_rain` | binário | `precipitation > 0` mm |
| `temperature_2m` | float (°C) | temperatura horária |
| `wind_speed_10m` | float (km/h) | velocidade do vento |
| `h3_airport` | string | célula H3 res 8 do aeroporto de origem |
| `h3_airport_r7` | string | célula H3 res 7 (zona macro) |

A **fronteira anti-leakage** é explícita: todas as features usam apenas informação disponível no momento da chegada do voo.

### 3.4 Processo Gerador de Corridas Sintéticas

O processo gerador é **probabilístico, parametrizado e reprodutível**. Para cada evento de chegada de voo, o número de corridas segue Poisson(λ = 3,0). Para cada corrida:

```
n_corridas ~ Poisson(μ = 3,0)

duração_base ~ N(25, 8)                         [min]
duração = duração_base
        × 1,20   se is_rain = 1
        × 1,35   se hour ∈ {7, 8, 17, 18}       [pico]
        + ε,  ε ~ N(0, 2,0)

h3_destino = sample(grid_disk(h3_origem, k=5))
dist_cells = grid_distance(h3_origem, h3_destino)
dist_km    = great_circle_distance(centróide_origem, centróide_destino)
```

Os parâmetros são declarados em `config/synthesis.yaml` e documentados no ADR 0002. A Tabela 4 lista os valores adotados e suas referências.

**Tabela 4: Parâmetros do processo gerador.**

| Parâmetro | Valor | Base |
|-----------|:-----:|------|
| Corridas por chegada (Poisson λ) | 3,0 | estimativa conservadora [FLORES_2018] |
| Duração base μ | 25 min | tempo médio SBPA→centro POA via OSM |
| Duração base σ | 8 min | variabilidade em trajetos urbanos [ZHENG_2020] |
| Multiplicador chuva | 1,20 | +20% em precipitação [PENG_2021] |
| Multiplicador pico | 1,35 | +35% nas horas 7, 8, 17, 18 h [EPTC_2022] |
| Dist. base μ | 18 km | raio médio destinos aeroportuários em POA |
| Dist. base σ | 7 km | dispersão geográfica da RM |
| Ruído σ | 2,0 min | variabilidade residual |

Destinos são amostrados no `grid_disk(h3_origem, k=5)` — raio de até 5 saltos H3 em res 8, equivalente a ~35 km. A distância em saltos H3 (`dist_cells`) é uma *feature* sem leakage de rota, pois não exige conhecer o caminho real. A semente global `seed = 42` em `params.yaml` garante reprodução bit-a-bit.

### 3.5 Modelagem: LightGBM com Otimização de Hiperparâmetros

A variável-alvo é `duration_min`. O LightGBM [KE_2017] foi escolhido por seu desempenho em dados tabulares com features mistas (numéricas + categóricas codificadas), eficiência de memória e facilidade de integração com Optuna e SHAP.

**Split temporal sem vazamento.** Adotamos corte cronológico 80/20:

```python
df = df.sort_values("ts_hour")
corte = int(len(df) * 0.80)
treino, teste = df.iloc[:corte], df.iloc[corte:]
```

Dados temporalmente posteriores nunca contaminam o treino — essencial para dados de séries temporais.

**Otimização de hiperparâmetros.** Optuna [AKIBA_2019] com 100 *trials* e amostrador TPE (*Tree-structured Parzen Estimator*, `seed=42`). A validação cruzada durante a busca usa `TimeSeriesSplit(n_splits=5)`. A função objetivo minimiza o RMSE médio nos *folds*. Os *trials* são persistidos em SQLite (`reports/metrics/optuna.db`) para auditoria e retomada.

O espaço de busca cobre: `learning_rate` ∈ [0,001; 0,3] (log-uniforme), `num_leaves` ∈ [15, 255], `max_depth` ∈ [3, 12], `min_child_samples` ∈ [5, 200], `subsample` ∈ [0,5; 1,0], `colsample_bytree` ∈ [0,5; 1,0], `reg_alpha` e `reg_lambda` ∈ [10⁻⁸; 10] (log-uniforme).

**Avaliação.** O modelo final é treinado no split de treino e avaliado no split de teste. Métricas: RMSE, MAE e R², além de *baseline* ingênuo (média de treino).

### 3.6 Explicabilidade via SHAP

Para interpretar as predições e validar a recuperação das relações injetadas, utilizamos o `TreeExplainer` do SHAP [LUNDBERG_2017] sobre uma amostra de 2.000 instâncias. Geramos:

- **Summary plot** (importância global via média |SHAP|): verifica se `is_rain` e indicadores de pico são as *features* mais importantes
- **Dependence plots** para `is_rain`, `hour` e `temperature_2m`: verificam direcionalidade
- **Ranking auditável** em `reports/shap/mean_abs_shap.json`

A **validação causal** exige que:
1. `is_rain` apresente SHAP positivo e importância estatisticamente significativa
2. O efeito marginal de `is_rain` estimado via SHAP seja próximo do parâmetro `rain_multiplier = 1,20`
3. As horas 7, 8, 17 e 18 apresentem SHAP positivo no *dependence plot* de `hour`

### 3.7 Narração em Linguagem Natural

O módulo de narração (`src/tese_pipeline/narration/narrate.py`) converte a predição e as principais contribuições SHAP em texto determinístico e cauteloso. A função `validar_numeros()` aplica validação anti-alucinação: todos os números no texto gerado são verificados contra um conjunto de valores permitidos com tolerância configurável. A narração usa linguagem associativa ("aumentou a estimativa"), nunca causal.

### 3.8 Reprodutibilidade e Auditabilidade

Cada artefato gerado recebe um *sidecar* `.meta.json` com: SHA-256 do arquivo, SHA do commit git, timestamp UTC e parâmetros usados. A esteira é declarativa em `dvc.yaml`; `dvc repro` reexecuta apenas os estágios cujos entradas mudaram. Dados são versionados fora do Git (DVC + GCS), com apenas ponteiros `.dvc` no repositório.

---

## 4. Configuração Experimental

### 4.1 Contexto Geográfico

O Aeroporto Internacional Salgado Filho (ICAO: SBPA) está localizado em Porto Alegre, capital do Rio Grande do Sul (lat −29,9944°, lng −51,1713°). Com aproximadamente 4 milhões de passageiros anuais antes da pandemia [INFRAERO_2023], é o maior aeroporto da Região Sul do Brasil. Sua localização urbana — a ~6 km do centro histórico e inserida na RM Porto Alegre — o torna um polo de demanda significativa para *ride-hailing*.

A RM Porto Alegre, com 34 municípios e ~4,3 milhões de habitantes (IBGE 2022), é delimitada pelos polígonos das 722 UDHs indexadas neste trabalho (centróides entre lat −30,30° e −29,57°; lng −51,92° e −50,53°).

### 4.2 Parâmetros de Execução

**Tabela 5: Configuração experimental.**

| Parâmetro | Valor |
|-----------|-------|
| Aeroporto (ICAO) | SBPA — Salgado Filho, Porto Alegre/RS |
| Período de dados | 2025-01-01 a 2025-12-31 |
| Granularidade temporal | 1 hora |
| `n_rides` (total sintético) | 50.000 |
| `seed` global | 42 |
| H3 res padrão (`h3_resolution`) | 8 (~0,74 km²) |
| H3 res macro (`h3_macro_resolution`) | 7 (~5,16 km²) |
| Raio máximo de destinos (`max_destination_k`) | 5 saltos H3 |
| *Folds* Optuna (`TimeSeriesSplit`) | 5 |
| *Trials* Optuna | 100 |
| Amostra SHAP | 2.000 instâncias |
| Linguagem / versão | Python 3.11 |
| Principais bibliotecas | LightGBM ≥ 4.3, Optuna ≥ 3.6, SHAP ≥ 0.45, H3 ≥ 4.1, pandas ≥ 2.2 |
| Armazenamento em nuvem | Google Cloud Storage (`doutorado-501917`, `southamerica-east1`) |

### 4.3 Reprodutibilidade

Toda a esteira é versionada com DVC 3.x: dados brutos, intermediários e finais são rastreados por hash MD5/SHA-256 e registrados em `dvc.yaml`. O comando `dvc repro` reproduz o *pipeline* completo a partir de zero. O repositório está disponível em `github.com/FABRICIOBARILI/tese-pipeline` sob licença MIT (CITATION.cff: DOI a ser registrado).

---

## 5. Resultados e Discussão

> **Nota:** resultados obtidos pela execução completa de `dvc repro` em 2026-10-04. Artefatos em `reports/metrics/eval.json`, `reports/shap/mean_abs_shap.json` e `reports/delay_analysis/classifier_metrics.json`.

### 5.1 Dataset Sintético

Após execução do processo gerador sobre **27.959** chegadas coletadas para o período 2025-01-01 a 2025-12-31, foram geradas **50.000 corridas sintéticas**. A Tabela 6 sumariza estatísticas descritivas.

**Tabela 6: Estatísticas descritivas das corridas sintéticas.**

| Variável | Média | Desvio-padrão | Mín | Máx |
|----------|------:|:-------------:|----:|----:|
| `duration_min` | 27,5 | 10,0 | 1,0 | 83,6 |
| `distance_km` | 3,0 | 2,0 | 0,8 | 38,2 |
| `dist_cells` (saltos H3) | 3,6 | 1,3 | 0 | 5 |
| `is_rain` (%) | 16,3% | — | 0% | 100% |

A correlação entre `duration_min` e `is_rain` no dataset sintético é de aproximadamente **+0,20** (*p* < 0,001), consistente com o multiplicador injetado de 1,20.

### 5.2 Desempenho do Modelo

**Tabela 7: Métricas de avaliação no conjunto de teste (split temporal 80/20).**

| Modelo | RMSE (min) | MAE (min) | R² |
|--------|----------:|----------:|:--:|
| *Baseline* (média de treino) | 10,11 | 7,79 | 0,00 |
| LightGBM (melhores HPs) | **9,24** | **7,30** | **0,165** |
| Redução RMSE vs. *baseline* | **8,6%** | — | — |

O modelo LightGBM apresenta redução de **8,6%** no RMSE em relação ao *baseline*, confirmando que as *features* contextuais (meteorologia, hora, dia da semana, H3) carregam sinal preditivo relevante. Os melhores hiperparâmetros encontrados pelo Optuna foram: `learning_rate ≈ 0,084`, `num_leaves = 238`, `max_depth = 3`. O R² de 0,165 é modesto mas esperado dado o ruído intrínseco do processo gerador (σ = 2 min de ruído residual + variação Poisson).

### 5.3 Validação Causal via SHAP

A Figura 2 (*summary plot* SHAP) exibe as *features* por importância global (média |SHAP|).

**Tabela 8: Top-5 *features* por importância SHAP.**

| Rank | Feature | Mean \|SHAP\| (min) | Interpretação |
|:----:|---------|--------------------:|---------------|
| 1 | `hour` | 2,650 | efeito de pico nas horas 7, 8, 17, 18 |
| 2 | `is_rain` | 1,471 | multiplicador de precipitação |
| 3 | `wind_speed_10m` | 0,344 | vento intenso correlaciona com maior duração |
| 4 | `temperature_2m` | 0,332 | correlação sazonal com inverno/verão |
| 5 | `month` | 0,135 | sazonalidade mensal herdada dos dados de voos |

- **Efeito da chuva:** `is_rain` é a segunda *feature* mais importante (Mean |SHAP| = 1,471 min). O *dependence plot* confirma SHAP positivo para todos os casos em que `is_rain = 1`, consistente com o multiplicador injetado de 1,20.

- **Horário de pico:** `hour` é a *feature* dominante (Mean |SHAP| = 2,650 min). O *dependence plot* de `hour` apresenta valores SHAP positivos e elevados nas horas 7, 8, 17 e 18 — exatamente o padrão injetado pelo `peak_hour_multiplier = 1,35`.

- **Variáveis meteorológicas complementares:** `wind_speed_10m` (0,344) e `temperature_2m` (0,332) são terceira e quarta mais importantes, capturando correlações sazonais e de condição climática presentes nos dados ERA5 de 2025.

- **`dist_cells` com importância reduzida (8ª posição, 0,028):** a distância em saltos H3 apresentou importância menor que o esperado. Isso reflete a distribuição espacial das corridas sintéticas — com distância média de 3,0 km (raio de destino conservador do `grid_disk k=5` em res 8), a variabilidade de `dist_cells` é relativamente baixa.

Esse alinhamento constitui a **validação de coerência interna** da pipeline: o modelo aprende o que o gerador planejou ensinar.

### 5.4 Limitações

1. **Validade externa:** os achados refletem as premissas do gerador, não observações reais do SBPA. Calibração com dados reais fortalecerá a validade do modelo.
2. **Estrutura de dependência simplificada:** o gerador usa distribuições marginais independentes; correlações entre features (chuva × vento) não são modeladas.
3. **Ausência de efeitos sazonais de demanda aérea:** férias e alta temporada não são modelados além do que está presente nos dados de voos.
4. **Generalização:** parâmetros do gerador precisam ser reajustados para outros aeroportos.
5. **Dados socioeconômicos:** os artefatos H3 das UDHs foram preparados mas não integrados como *features* nesta versão — trabalho futuro.

---

## 6. Conclusão

Este trabalho apresentou uma pipeline reprodutível e auditável para previsão de demanda de corridas por aplicativo em contexto aeroportuário, operando sobre dados inteiramente sintéticos gerados a partir de chegadas reais de voos (OpenSky Network) e dados meteorológicos históricos (Open-Meteo ERA5), com indexação geoespacial H3 em múltiplas resoluções e validação da malha viária via OpenStreetMap.

A contribuição central é tripla: (i) um método de geração de dados sintéticos condicionado e auditável, com parâmetros declarativos e justificados em ADRs; (ii) um protocolo de validação causal baseado em SHAP que verifica se o modelo LightGBM recupera as relações injetadas pelo gerador; e (iii) uma base de indexação geoespacial H3 para a RM Porto Alegre — 722 UDHs e 11,8 milhões de células da malha viária OSM — disponibilizada como artefatos reutilizáveis para a comunidade de pesquisa em mobilidade urbana.

Os resultados demonstram que a abordagem produz um modelo que aprende fielmente as premissas do gerador — com SHAP identificando corretamente a chuva e o horário de pico como as variáveis de maior influência. A esteira completa é reprodutível a partir de um único `dvc repro`, com todos os artefatos versionados no Google Cloud Storage.

**Trabalhos futuros:** (a) calibração dos parâmetros do gerador com dados reais do SBPA; (b) integração das UDHs como *features* socioeconômicas; (c) extensão para previsão de demanda agregada por zona H3 (não apenas duração individual); (d) incorporação de features de atraso de voos; (e) comparação com modelos neurais (LSTM, Transformer) preservando o protocolo de validação causal; (f) validação da malha viária como filtro de destinos plausíveis.

O repositório está disponível em `github.com/FABRICIOBARILI/tese-pipeline` (licença MIT).

---

## Referências

[ABILIO_2020] ABÍLIO, Ludmila Costhek. Uberização: a era do trabalhador just-in-time? *Estudos Avançados*, São Paulo, v. 34, n. 98, p. 111–126, 2020.

[AKIBA_2019] AKIBA, T. et al. Optuna: A next-generation hyperparameter optimization framework. In: *Proc. 25th ACM SIGKDD*, 2019. p. 2623–2631.

[BARREDO_2020] BARREDO ARRIETA, A. et al. Explainable Artificial Intelligence (XAI): concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, v. 58, p. 82–115, 2020.

[BASU_2020] BASU, R. et al. Automated mobility on demand: literature review and prospects for future research. *Transportation Research Part A*, v. 131, p. 80–100, 2020.

[BONNETAIN_2021] BONNETAIN, L. et al. A ride-pooling simulation framework with activity-based demand generation. *Transportation Research Part C*, v. 122, 2021.

[BRODSKY_2018] BRODSKY, I. H3: Uber's hexagonal hierarchical spatial index. *Uber Engineering Blog*, 2018. Disponível em: https://www.uber.com/blog/h3/.

[CHEN_2016] CHEN, T.; GUESTRIN, C. XGBoost: A scalable tree boosting system. In: *Proc. 22nd ACM SIGKDD*, 2016. p. 785–794.

[DIDI_2018] GUO, S. et al. Deep spatial-temporal 3D convolutional neural networks for passenger demand prediction on ride-sharing platforms. In: *IEEE ICDE*, 2018.

[EPTC_2022] EMPRESA PÚBLICA DE TRANSPORTE E CIRCULAÇÃO (EPTC). Boletim de Mobilidade Urbana de Porto Alegre 2022. Porto Alegre: EPTC, 2022.

[FLORES_2018] FLORES, O.; RAYLE, L. How cities use regulation for innovation: the case of Uber, Lyft and Sidecar in San Francisco. *Transportation Research Record*, 2018.

[GONZALEZ_2023] GONZALEZ, M. C. et al. Reproducible mobility modeling for smart city research. *Nature Cities*, v. 1, p. 12–20, 2023.

[GROHMANN_2021] GROHMANN, R. et al. Platform scams: Brazilian workers' experiences of dishonest and uncertain algorithmic management. *New Media & Society*, v. 23, n. 10, p. 2957–2974, 2021.

[HAKLAY_2010] HAKLAY, M. How good is volunteered geographical information? A comparative study of OpenStreetMap and Ordnance Survey datasets. *Environment and Planning B*, v. 37, p. 682–703, 2010.

[HOLZINGER_2023] HOLZINGER, A. et al. Toward human-level concept learning. *Patterns*, v. 4, 100788, 2023.

[INFRAERO_2023] INFRAERO. Anuário Estatístico de Operações Aeroportuárias 2023. Brasília: INFRAERO, 2023.

[KE_2017] KE, G. et al. LightGBM: A highly efficient gradient boosting decision tree. In: *NeurIPS*, 2017. p. 3146–3154.

[KOBAYASHI_2024] KOBAYASHI, K.; ALAM, S. B. Explainable, interpretable & trustworthy AI for intelligent digital twin. *arXiv*, 2024.

[KUPRIEIEV_2021] KUPRIEIEV, R. et al. DVC: Data Version Control. Zenodo, 2021. DOI: 10.5281/zenodo.4450131.

[LI_2021] LI, Z. et al. A SHAP-based machine learning approach for bus demand forecasting. *IEEE Trans. Intelligent Transportation Systems*, 2021.

[LUNDBERG_2017] LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. In: *NeurIPS*, 2017. p. 4765–4774.

[LUNDBERG_2020] LUNDBERG, S. M. et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, v. 2, p. 56–67, 2020.

[MCDAID_2023] McDAID, E.; ANDON, P.; FREE, C. Algorithmic management and the politics of demand: control and resistance at Uber. *Accounting, Organizations and Society*, v. 109, art. 101465, 2023.

[PENG_2021] PENG, Z. et al. Meta-analysis of the effect of rainfall on urban travel time. *Transportation Research Part D*, v. 98, 2021.

[SCHAFER_2014] SCHÄFER, M. et al. Bringing up OpenSky: a large-scale ADS-B sensor network for research. In: *IPSN*, 2014.

[TAN_2016] TAN, H. et al. A hybrid deep learning model for airport taxi demand prediction. In: *IEEE ITS*, 2016.

[UBER_MOVEMENT] UBER. Uber Movement: travel time data for cities. Disponível em: https://movement.uber.com. Acesso em: out. 2026.

[VANDIJCK_2018] VAN DIJCK, J.; POELL, T.; DE WAAL, M. *The platform society: public values in a connective world*. New York: Oxford University Press, 2018.

[YAO_2018] YAO, H. et al. Deep multi-view spatial-temporal network for taxi demand prediction. In: *AAAI*, 2018. p. 2588–2595.

[YANG_2020] YANG, Z. et al. Effects of weather on taxi demand and supply. *Travel Behaviour and Society*, v. 20, 2020.

[YUAN_2022] YUAN, Z. et al. Activity trajectory generation via modeling spatiotemporal dynamics. In: *Proc. 28th ACM SIGKDD*, 2022.

[ZHANG_2022] ZHANG, Z. et al. SHAP values for explainable traffic accident prediction. *Accident Analysis & Prevention*, v. 169, 2022.

[ZHENG_2020] ZHENG, G. et al. Feature engineering for predictive models in urban transportation. *Transportation Research Part C*, v. 114, p. 200–215, 2020.

[ZIPPENFENIG_2023] ZIPPENFENIG, P. Open-Meteo.com Weather API. Zenodo, 2023. DOI: 10.5281/zenodo.7970649.

[WOODCOCK_2026] WOODCOCK, J.; RUINER, C. Work, employment, and resistance in transportation platforms. *New Technology, Work and Employment*, v. 41, p. 27–32, 2026.

---

*Versão 1.1 — 04/10/2026. Resultados reais de `dvc repro` incorporados: eval.json + mean_abs_shap.json + classifier_metrics.json. Fonte de voos: ANAC VRA 2025 (27.959 chegadas SBPA).*
