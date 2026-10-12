"""Ambiente de conversa motorista × modelos de demanda (porta de 05_5 + 05_6-produção).

Dois produtos encadeados, ambos sobre a base de dados do pacote:
  - ``treina_pareados`` — treina e exporta os dois modelos binários pareados (clima/atraso)
    + prior prospectivo de atraso (climatologia + reputação). Equivale à seção de PRODUÇÃO do
    notebook ``05_6_FeatureEng_Individual_vs_Agrupado``.
  - ``narrador`` — responde ao motorista ("vale a pena ir ao SBPA às X?") carregando esses
    modelos, montando features em UTC consistentes com o treino, explicando via SHAP e narrando
    com a API da Claude. Equivale ao ``05_5_Narrador_Natural``.

O módulo ``features`` concentra a construção de features usada pelos DOIS lados — garantindo
consistência treino×inferência (sem train/serve skew), que no notebook original era mantida
manualmente em dois lugares.
"""
