# Baldur SFC Analyzer

Aplicação desktop para visualização, tratamento e análise de cromatogramas obtidos por SFC-FID.

A ferramenta permite carregar amostras e padrões, detectar picos cromatográficos, calcular áreas e agrupar resultados conforme os métodos ASTM D5186 e ASTM D6550.

## Versão

- Aplicação: `0.0.1`
- Python recomendado: `3.11`
- Plataforma principal: Windows

> A versão exata do Python utilizada no desenvolvimento não está registrada no projeto. A recomendação de Python 3.10 ou superior é baseada na sintaxe e nos recursos utilizados pelo código.

## Funcionalidades

- Leitura de arquivos `.asc`, `.dat`, `.dat.asc`, `.txt` e `.csv`
- Visualização gráfica dos cromatogramas
- Correção linear de baseline
- Suavização do sinal
- Detecção automática de picos
- Ajuste de picos por modelos Gaussiano e EMG
- Deconvolução de picos sobrepostos
- Inclusão, edição e remoção manual de picos
- Carregamento de cromatograma padrão
- Referenciamento de compostos por tempo de retenção
- Cálculo de áreas por regiões analíticas
- Suporte aos métodos:
  - ASTM D5186
  - ASTM D6550
- Classificação de amostras Diesel e não Diesel
- Salvamento das configurações e picos detectados
- Exportação dos resultados para arquivos tabulares