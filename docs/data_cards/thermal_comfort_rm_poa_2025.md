# Data Card — Conforto Térmico RM Porto Alegre 2025

## Identificação

| Campo | Valor |
|-------|-------|
| **Artefato** | `data/processed/thermal_comfort.parquet` |
| **Fonte primária** | Open-Meteo ERA5 — `data/raw/meteo_rm.parquet` |
| **Biblioteca de cálculo** | pythermalcomfort 4.6.0 |
| **Referência bibliográfica** | Tartarini & Schiavon (2020). *pythermalcomfort: A Python package for thermal comfort calculations.* SoftwareX, 12, 100578. |
| **Módulo** | `src/tese_pipeline/processing/thermal_comfort.py` |
| **Data de geração** | 2026-10-04 |

## Cobertura

| Dimensão | Valor |
|----------|-------|
| Período | 2025-01-01 a 2025-12-31 (8.760 h/localidade) |
| Localidades | 10 pontos da RM Porto Alegre |
| Granularidade | 1 hora |
| Total de linhas | 87.600 |
| Tamanho | 1,2 MB (Parquet) |

## Localidades

| Nome | Município | Latitude | Longitude | UTCI médio (°C) |
|------|-----------|:--------:|:---------:|:---------------:|
| POA_CENTRO | Porto Alegre — Centro | −30,0346 | −51,2177 | 18,50 |
| SBPA | Porto Alegre — Aeroporto | −29,9944 | −51,1713 | 18,50 |
| CANOAS | Canoas | −29,9178 | −51,1836 | 18,68 |
| NOVO_HAMBURGO | Novo Hamburgo | −29,6783 | −51,1303 | 18,99 |
| SAO_LEOPOLDO | São Leopoldo | −29,7668 | −51,1493 | 18,69 |
| GRAVATAI | Gravataí | −29,9411 | −50,9919 | 17,84 |
| ALVORADA | Alvorada | −29,9888 | −51,0823 | 18,07 |
| VIAM_AO | Viamão | −30,0818 | −51,0232 | 17,43 |
| ESTEIO | Esteio | −29,8496 | −51,1760 | 18,51 |
| GUAIBA | Guaíba | −30,1123 | −51,3238 | 17,92 |

## Índices Calculados

### UTCI — Universal Thermal Climate Index

| Parâmetro | Valor |
|-----------|-------|
| Norma de referência | EN ISO 15743 / Bröde et al. (2012) |
| Entradas | tdb (°C), Tmrt (°C), v (m/s), rh (%) |
| Estimativa de Tmrt | srad > 0: `Tmrt = tdb + 0,7×ln(srad+1) − 2`; srad = 0: `Tmrt = tdb − 2` |
| Intervalo 2025 | −12,7 °C a +39,5 °C |
| Média 2025 | 18,3 °C (zona de sem estresse térmico) |

**Distribuição das categorias UTCI (% horas-localidade):**

| Categoria | N horas | % |
|-----------|--------:|:-:|
| no_stress (9–26 °C) | 64.157 | 73,2% |
| moderate_heat (26–32 °C) | 11.284 | 12,9% |
| slight_cold (0–9 °C) | 8.703 | 9,9% |
| strong_heat (32–38 °C) | 2.583 | 2,9% |
| moderate_cold (−13–0 °C) | 806 | 0,9% |
| very_strong_heat (38–46 °C) | 67 | 0,1% |

### AT — Apparent Temperature (Steadman 1994)
- Intervalo: −2,9 °C a +40,7 °C | Média: 20,1 °C
- Entradas: tdb, rh, v (m/s)

### THI — Temperature Humidity Index (Thom, 1959)
- Intervalo: 35,2 a 85,5 | Média: 66,0
- THI > 80 = desconforto severo; THI > 90 = emergência

### DI — Discomfort Index (Thom, 1959)
- Intervalo: 1,8 a 29,8 | Média: 18,9
- DI > 24 = desconforto; DI > 27 = perigo

### HI — Heat Index (Rothfusz, 1990)
- Calculado apenas para tdb ≥ 27 °C e rh ≥ 40%
- Aplicável em 9.619 horas (11,0% do total)

### WCI — Wind Chill Index
- Calculado apenas para tdb < 10 °C
- Aplicável em 4.506 horas (5,1% do total)

## Principais Achados Climáticos (2025)

1. **73% das horas sem estresse térmico** — clima temperado/subtropical típico
2. **~16% com algum grau de estresse de calor** — relevante para demanda de ride-hailing em verão
3. **Pico de calor extremo (UTCI ≥ 38°C): 67 horas** — concentradas no verão (jan–mar)
4. **Gradiente norte–sul**: Novo Hamburgo e São Leopoldo (norte) ~1,5°C mais quentes que Viamão e Guaíba (sul/leste)
5. **Heat Index ativo em 11% das horas** — indicador de desconforto percebido em dias quentes e úmidos

## Uso na Pipeline

O `thermal_comfort.parquet` pode ser unido aos dados de corridas (ANAC VRA + sintético) via `h3_r8`:

```python
rides = pd.read_parquet("data/synthetic/rides.parquet")
thermal = pd.read_parquet("data/processed/thermal_comfort.parquet")

# join por célula H3 res-8 e hora local
rides_enriched = rides.merge(
    thermal[["h3_r8", "time", "utci", "utci_stress", "at", "di"]],
    left_on=["h3_airport", "ts_hour"],
    right_on=["h3_r8", "time"],
    how="left",
)
```

## Limitações

1. **Tmrt simplificada**: a estimativa de temperatura radiante média usa apenas `shortwave_radiation` (GHI), sem distinguir radiação direta e difusa. Para maior precisão em estudos de conforto externo, usar `pythermalcomfort.models.solar_gain()` com dados de Direct Normal Irradiance (DNI).
2. **Resolução espacial ERA5 (~9 km)**: os dados ERA5 têm resolução de ~9 km. Variações microclimáticas urbanas (ilha de calor, sombra de edifícios) não são capturadas.
3. **Sem dados de vestimenta/atividade**: UTCI assume pedestre em movimento (clo padrão, met ≈ 2,3); PMV/SET para ambientes internos requerem configurações distintas.
4. **Ponto único por município**: a RM POA tem gradientes topográficos significativos; uma grade densa ou altimetria corrigiria isso.
