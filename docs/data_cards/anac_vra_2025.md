# Data Card — ANAC VRA (Voo Regular Ativo) 2025

## Identificação

| Campo | Valor |
|-------|-------|
| **Nome do dataset** | Voo Regular Ativo (VRA) — ANAC |
| **Versão ingerida** | 2025 completo (Jan–Dez) |
| **Fonte primária** | ANAC Dados Abertos |
| **URL de acesso** | https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra |
| **Data de acesso** | 2026-10-10 |
| **Licença** | Dados Abertos do Governo Federal (Lei nº 12.527/2011) |
| **Atualização da fonte** | Mensal (CSVs separados por mês) |

## Schema Original → Normalizado

| Coluna original | Coluna normalizada | Tipo | Descrição |
|---|---|---|---|
| `ICAO Empresa Aérea` | `airline_icao` | string | Código ICAO da companhia aérea |
| `Número Voo` | `flight_number` | string | Número do voo |
| `Código Autorização (DI)` | `auth_code` | string | Código de autorização ANAC |
| `Código Tipo Linha` | `route_type` | string | N=nacional, I=internacional, C=cargueiro |
| `ICAO Aeródromo Origem` | `origin_icao` | string | ICAO do aeroporto de origem |
| `ICAO Aeródromo Destino` | `dest_icao` | string | ICAO do aeroporto de destino |
| `Partida Prevista` | `scheduled_dep` | datetime | Horário previsto de partida |
| `Partida Real` | `actual_dep` | datetime | Horário real de partida |
| `Chegada Prevista` | `scheduled_arr` | datetime | Horário previsto de chegada |
| `Chegada Real` | `actual_arr` | datetime | Horário real de chegada |
| `Situação Voo` | `flight_status` | string | REALIZADO / CANCELADO |
| `Código Justificativa` | `delay_code` | string | Código de justificativa do atraso |

## Colunas Derivadas (normalization)

| Coluna | Tipo | Fórmula |
|--------|------|---------|
| `delay_arr_min` | float | `(actual_arr − scheduled_arr)` em minutos |
| `delay_dep_min` | float | `(actual_dep − scheduled_dep)` em minutos |
| `is_delayed` | bool | `delay_arr_min > 15` (limiar IATA) |
| `arr_hour_local` | datetime (tz=America/Sao_Paulo) | `actual_arr` convertida para horário local, arredondada para hora cheia |
| `source_file` | string | nome do CSV de origem (rastreabilidade) |
| `source_month` | string (YYYY-MM) | mês extraído do nome do arquivo |

## Filtros Aplicados

- `dest_icao == "SBPA"` — apenas chegadas no Aeroporto Salgado Filho (Porto Alegre)
- `flight_status == "REALIZADO"` — exclui voos cancelados

## Volume Estimado

| Mês | Chegadas SBPA (estimado) |
|-----|-------------------------|
| Jan/2025 | ~1.943 |
| Fev–Dez/2025 | ~1.800–2.100/mês |
| **Total 2025** | **~22.000–24.000 voos** |

## Formato dos Arquivos Originais

- **Formato:** CSV (`;` como separador)
- **Encoding:** UTF-8 com BOM (`utf-8-sig`)
- **Estrutura:**
  - Linha 1: metadado (`Atualizado em: YYYY-MM-DD`) — ignorada na leitura
  - Linha 2: cabeçalho
  - Linhas 3+: dados

## Limitações e Avisos

1. **Sem dados de passageiros:** o VRA registra operações de aeronaves, não manifesto de passageiros. O número de passageiros por chegada deve ser estimado (p. ex., pela capacidade típica da aeronave × fator de ocupação médio histórico).
2. **Horários em UTC implícito:** os campos de data/hora no VRA não têm fuso explícito. A ANAC documenta como horário local (America/Sao_Paulo), porém a normalização converte via UTC para garantir consistência no join com dados meteorológicos Open-Meteo (UTC).
3. **Granularidade de operação:** o VRA registra uma linha por operação de aeronave, não por passageiro. Para modelagem de demanda de corridas, assume-se uma distribuição Poisson de corridas por chegada (parametrizada em `config/synthesis.yaml`).
4. **Cobertura:** apenas voos regulares (não inclui aviação geral ou voos charter não registrados na ANAC).

## Referência Bibliográfica

AGÊNCIA NACIONAL DE AVIAÇÃO CIVIL (ANAC). **Voo Regular Ativo (VRA) — 2025**. Brasília: ANAC, 2025. Dados Abertos. Disponível em: https://www.gov.br/anac/pt-br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/voo-regular-ativo-vra. Acesso em: 10 out. 2026.
