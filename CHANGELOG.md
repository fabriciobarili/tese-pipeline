# Changelog

Histórico legível das evoluções da pipeline da tese. Versões seguem o padrão
`v0.MINOR.0` usado nas mensagens de commit.

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
