# Data Cards

Um arquivo por dataset. Copie o modelo abaixo para cada novo card
(`meteo.md`, `flights.md`, `features.md`, `synthetic_rides.md`).

```markdown
# Data Card — <nome>
- **Fonte:** Open-Meteo / OpenSky Network / síntese
- **Período coletado:** AAAA-MM-DD a AAAA-MM-DD
- **Cobertura geográfica:** bbox / aeroportos (ICAO)
- **Variáveis:** lista e unidades (SI)
- **Licença / termos de uso:** link
- **Processamento aplicado:** resumo + script responsável
- **Fronteira temporal (anti-leakage):** o que é conhecido no instante do evento
- **Limitações conhecidas:** lacunas, vieses, % de missing
- **Hash (sha256):** do arquivo versionado (ver `*.meta.json`)
```
