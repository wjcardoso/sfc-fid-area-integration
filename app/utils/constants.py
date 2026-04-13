"""Constantes compartilhadas da aplicação SFC."""

SETTINGS_ORG = "Lacem"
SETTINGS_APP = "SFCViewer"
FILTRO_ARQUIVOS = "Dados (*.asc *.dat *.dat.asc *.txt *.csv);;Todos os arquivos (*)"
ALTURA_MINIMA_PICO = 10000
JANELA_SUAVIZACAO = 7
DIVISOR_DISTANCIA_MINIMA = 80
MIN_PROMINENCIA_PICO = 0.0
LARGURA_MINIMA_PICO = 3
FATOR_RUIDO = 4.0
ALTURA_RELATIVA_PICO = 0.98
FATOR_CURVATURA = 1.50
MODELO_AJUSTE_PICO = "Nenhum"
MODELOS_AJUSTE_PICO = ["Nenhum", "Gaussiano", "EMG"]
METODOS_ANALISE_ASTM = ["ASTM D5186", "ASTM D6550"]
METODO_ANALISE_ASTM_PADRAO = METODOS_ANALISE_ASTM[0]
ENSAIOS_DISPONIVEIS = ["E037", "E01943"]
ENSAIO_PADRAO = ENSAIOS_DISPONIVEIS[0]

COLUNAS_AMOSTRA = [
    "Pico",
    "Tempo (min)",
    "Altura",
    "Início",
    "Fim",
    "Largura",
    "Área",
]
COLUNAS_PADRAO = ["Composto", "Tempo (min)", "Início", "Fim"]
COLUNAS_REGIOES = [
    "Grupos",
    "Início (min)",
    "Fim (min)",
    "Área Total",
]

REGIOES_ANALITICAS_D5186_DIESEL = [
    ("Não-aromaticos", 4.58, 5.87),
    ("Mono-aromaticos", 5.87, 9.85),
    ("Di-aromaticos", 9.85, 16.49),
    ("Tri-aromaticos+", 16.49, 25.08),
]

REGIOES_ANALITICAS_D5186_NAO_DIESEL = [
    ("Não-aromaticos", 4.58, 5.87),
    ("Mono-aromaticos", 5.87, 9.85),
    ("Poli-aromaticos", 9.85, 25.08),
]

REGIOES_ANALITICAS_D6550_DIESEL = [
    ("Saturados", 4.58, 5.87),
    ("Mono-aromaticos", 5.87, 9.85),
    ("Di-aromaticos", 9.85, 16.49),
    ("Tri-aromaticos+", 16.49, 25.08),
    ("Saturados pesados", 25.08, 25.17),
    ("Olefinas", 27.29, 32.88),
]

REGIOES_ANALITICAS_D6550_NAO_DIESEL = [
    ("Saturados", 4.58, 5.87),
    ("Mono-aromaticos", 5.87, 9.85),
    ("Poli-aromaticos", 9.85, 25.08),
    ("Saturados pesados", 25.08, 25.17),
    ("Olefinas", 27.29, 32.88),
]

REGIOES_ANALITICAS_POR_METODO = {
    "ASTM D5186": {
        True: REGIOES_ANALITICAS_D5186_DIESEL,
        False: REGIOES_ANALITICAS_D5186_NAO_DIESEL,
    },
    "ASTM D6550": {
        True: REGIOES_ANALITICAS_D6550_DIESEL,
        False: REGIOES_ANALITICAS_D6550_NAO_DIESEL,
    },
}
REGIOES_ANALITICAS = REGIOES_ANALITICAS_POR_METODO[METODO_ANALISE_ASTM_PADRAO][False]

CORES_REGIOES = {
    "Não-aromaticos": "#66BB6A",
    "Saturados": "#66BB6A",
    "Mono-aromaticos": "#42A5F5",
    "Di-aromaticos": "#FFCA28",
    "Tri-aromaticos+": "#EF5350",
    "Poli-aromaticos": "#EF5350",
    "Saturados pesados": "#8D6E63",
    "Olefinas": "#AB47BC",
}
