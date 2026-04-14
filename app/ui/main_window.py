"""Janela principal da aplicação desktop de visualização SFC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from app.controllers.main_controller import MainController
from app.models.amostra import Amostra
from app.ui.canvas import CromatogramaCanvas
from app.utils import constants
from app.utils.table_utils import configurar_tabela, preencher_tabela, texto_para_float


class JanelaPrincipal(QMainWindow):
    """Janela principal da aplicação de análise de cromatogramas SFC."""

    SETTINGS_ORG = constants.SETTINGS_ORG
    SETTINGS_APP = constants.SETTINGS_APP
    FILTRO_ARQUIVOS = constants.FILTRO_ARQUIVOS
    ALTURA_MINIMA_PICO = constants.ALTURA_MINIMA_PICO
    JANELA_SUAVIZACAO = constants.JANELA_SUAVIZACAO
    DIVISOR_DISTANCIA_MINIMA = constants.DIVISOR_DISTANCIA_MINIMA
    MIN_PROMINENCIA_PICO = constants.MIN_PROMINENCIA_PICO
    LARGURA_MINIMA_PICO = constants.LARGURA_MINIMA_PICO
    FATOR_RUIDO = constants.FATOR_RUIDO
    ALTURA_RELATIVA_PICO = constants.ALTURA_RELATIVA_PICO
    FATOR_CURVATURA = constants.FATOR_CURVATURA
    MODELO_AJUSTE_PICO = constants.MODELO_AJUSTE_PICO
    MODELOS_AJUSTE_PICO = constants.MODELOS_AJUSTE_PICO
    METODOS_ANALISE_ASTM = constants.METODOS_ANALISE_ASTM
    METODO_ANALISE_ASTM_PADRAO = constants.METODO_ANALISE_ASTM_PADRAO
    COLUNAS_AMOSTRA = constants.COLUNAS_AMOSTRA
    COLUNAS_PADRAO = constants.COLUNAS_PADRAO
    COLUNAS_REGIOES = constants.COLUNAS_REGIOES
    REGIOES_ANALITICAS = constants.REGIOES_ANALITICAS
    REGIOES_ANALITICAS_POR_METODO = constants.REGIOES_ANALITICAS_POR_METODO
    CORES_REGIOES = constants.CORES_REGIOES
    COLUNA_PERCENTUAL_AREA = "% Área"
    COLUNAS_AMOSTRA_EXIBICAO = [*COLUNAS_AMOSTRA, COLUNA_PERCENTUAL_AREA]
    COLUNAS_PADRAO_EXIBICAO = [*COLUNAS_PADRAO, "Área", COLUNA_PERCENTUAL_AREA]
    COLUNAS_REGIOES_EXIBICAO = [*COLUNAS_REGIOES, COLUNA_PERCENTUAL_AREA]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Baldur 0.0.1 - CEMEP - SFC Analyzer")
        _logo = Path(__file__).resolve().parent.parent.parent / "logo.png"
        self.setWindowIcon(QIcon(str(_logo)))
        self.resize(1200, 800)

        self.settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._carregar_opcoes_salvas()
        self.amostras: dict[str, Amostra] = {}
        self.padrao: Amostra | None = None
        self._atualizando_tabelas = False
        self._limites_originais: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None
        self._tempo_cursor_grafico: float | None = None
        self._tempo_primeiro_clique: float | None = None
        self._chave_amostra_selecao_manual: str | None = None
        self._linha_tempo_cursor = None
        self._linha_tempo_fixo = None
        self._faixa_selecao_manual = None
        self.controller = MainController(self.settings)
        self.referencias_padrao = self.controller.load_standard_references(
            self.COLUNAS_PADRAO
        )

        self._criar_menu()
        self._criar_interface()
        self._atualizar_tabela_padrao()
        self._atualizar_tabela_regioes(self._criar_dataframe_picos_vazio())
        self._restaurar_padrao_salvo()
        self._mostrar_placeholder()
        QTimer.singleShot(0, self._reaplicar_layout_inicial)

    def _reaplicar_layout_inicial(self) -> None:
        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _criar_dataframe_picos_vazio(self) -> pd.DataFrame:
        return self.controller.create_empty_peaks(self.COLUNAS_AMOSTRA)

    def _criar_dataframe_padrao_vazio(self) -> pd.DataFrame:
        return self.controller.create_empty_standard(self.COLUNAS_AMOSTRA)

    def _criar_referencias_padrao_iniciais(self) -> pd.DataFrame:
        return self.controller.default_standard_references(self.COLUNAS_PADRAO)

    def _normalizar_metodo_analise(self, metodo: str | None) -> str:
        valor = str(metodo or "").strip()
        return (
            valor
            if valor in self.METODOS_ANALISE_ASTM
            else self.METODO_ANALISE_ASTM_PADRAO
        )

    def _chave_setting_metadado_amostra(self, chave: str, campo: str) -> str:
        chave_picos = self.controller.sample_settings_key(chave)
        prefixo = (
            chave_picos[: -len("/picos")]
            if chave_picos.endswith("/picos")
            else chave_picos
        )
        return f"{prefixo}/{campo}"

    def _chave_setting_metodo_amostra(self, chave: str) -> str:
        return self._chave_setting_metadado_amostra(chave, "metodo_analise")

    @staticmethod
    def _inferir_eh_diesel_do_caminho(caminho: Path) -> bool:
        return any("DIESEL" in str(parte).upper() for parte in caminho.parts)

    def _carregar_metodo_amostra(self, chave: str) -> str:
        valor = self.settings.value(
            self._chave_setting_metodo_amostra(chave),
            self.METODO_ANALISE_ASTM_PADRAO,
        )
        return self._normalizar_metodo_analise(valor)

    def _carregar_diesel_amostra(
        self,
        chave: str,
        caminho: Path | None = None,
    ) -> bool:
        padrao = self._inferir_eh_diesel_do_caminho(caminho) if caminho else False
        return self._ler_bool_setting(
            self._chave_setting_metadado_amostra(chave, "eh_diesel"),
            padrao,
        )

    def _salvar_metodo_amostra(self, chave: str, metodo: str) -> None:
        self.settings.setValue(
            self._chave_setting_metodo_amostra(chave),
            self._normalizar_metodo_analise(metodo),
        )
        self.settings.sync()

    def _salvar_diesel_amostra(self, chave: str, eh_diesel: bool) -> None:
        self.settings.setValue(
            self._chave_setting_metadado_amostra(chave, "eh_diesel"),
            bool(eh_diesel),
        )
        self.settings.sync()

    def _salvar_metadados_amostra(self, chave: str, amostra: Amostra) -> None:
        self._salvar_metodo_amostra(chave, amostra.metodo_analise)
        self._salvar_diesel_amostra(chave, amostra.eh_diesel)

    def _chave_setting_grupos_amostra(self, chave: str) -> str:
        return self._chave_setting_metadado_amostra(chave, "grupos_vinculados")

    def _chave_setting_grupos_zerados_amostra(self, chave: str) -> str:
        return self._chave_setting_metadado_amostra(chave, "grupos_zerados")

    def _carregar_vinculos_grupos_amostra(
        self,
        chave: str | None,
    ) -> dict[str, dict[str, float | int]]:
        if not chave:
            return {}

        bruto = self.settings.value(self._chave_setting_grupos_amostra(chave), "")
        if not bruto:
            return {}

        try:
            vinculos = json.loads(str(bruto))
        except Exception:
            return {}

        return vinculos if isinstance(vinculos, dict) else {}

    def _salvar_vinculos_grupos_amostra(
        self,
        chave: str,
        vinculos: dict[str, dict[str, float | int]],
    ) -> None:
        self.settings.setValue(
            self._chave_setting_grupos_amostra(chave),
            json.dumps(vinculos, ensure_ascii=False),
        )
        self.settings.sync()

    def _carregar_grupos_zerados_amostra(self, chave: str | None) -> set[str]:
        if not chave:
            return set()

        bruto = self.settings.value(
            self._chave_setting_grupos_zerados_amostra(chave), ""
        )
        if not bruto:
            return set()

        try:
            grupos = json.loads(str(bruto))
        except Exception:
            return set()

        if not isinstance(grupos, list):
            return set()

        return {str(grupo).strip() for grupo in grupos if str(grupo).strip()}

    def _salvar_grupos_zerados_amostra(self, chave: str, grupos: set[str]) -> None:
        dados = sorted({str(grupo).strip() for grupo in grupos if str(grupo).strip()})
        self.settings.setValue(
            self._chave_setting_grupos_zerados_amostra(chave),
            json.dumps(dados, ensure_ascii=False),
        )
        self.settings.sync()

    @staticmethod
    def _resolver_pico_vinculado(
        picos: pd.DataFrame,
        vinculo: object,
        tolerancia_tempo: float = 0.35,
    ) -> pd.Series | None:
        if picos.empty or not isinstance(vinculo, dict):
            return None

        tempos = pd.to_numeric(picos["Tempo (min)"], errors="coerce")
        numeros = pd.to_numeric(picos["Pico"], errors="coerce")
        tempo_ref = pd.to_numeric(
            pd.Series([vinculo.get("tempo")]), errors="coerce"
        ).iloc[0]
        pico_ref = pd.to_numeric(
            pd.Series([vinculo.get("pico")]), errors="coerce"
        ).iloc[0]

        if pd.notna(pico_ref):
            mascara = numeros == int(float(pico_ref))
            if mascara.any():
                return picos.loc[mascara].iloc[0]

        if pd.notna(tempo_ref) and tempos.notna().any():
            diferencas = tempos.sub(float(tempo_ref)).abs()
            indice = diferencas.idxmin()
            if pd.notna(diferencas.loc[indice]) and (
                float(diferencas.loc[indice]) <= tolerancia_tempo
            ):
                return picos.loc[indice]

        return None

    def _limpar_vinculos_grupos_invalidos(
        self,
        chave: str,
        picos: pd.DataFrame,
    ) -> None:
        vinculos = self._carregar_vinculos_grupos_amostra(chave)
        if not vinculos:
            return

        tempos = pd.to_numeric(picos.get("Tempo (min)"), errors="coerce")
        vinculos_validos: dict[str, dict[str, float | int]] = {}
        for grupo, vinculo in vinculos.items():
            if not isinstance(vinculo, dict):
                continue

            tempo_ref = pd.to_numeric(
                pd.Series([vinculo.get("tempo")]), errors="coerce"
            ).iloc[0]
            if pd.notna(tempo_ref) and tempos is not None and tempos.notna().any():
                diferencas = tempos.sub(float(tempo_ref)).abs()
                indice = diferencas.idxmin()
                if (
                    pd.notna(diferencas.loc[indice])
                    and float(diferencas.loc[indice]) <= 0.35
                ):
                    vinculos_validos[grupo] = vinculo
                    continue

            if self._resolver_pico_vinculado(picos, vinculo) is not None:
                vinculos_validos[grupo] = vinculo

        if vinculos_validos != vinculos:
            self._salvar_vinculos_grupos_amostra(chave, vinculos_validos)

    @staticmethod
    def _descrever_pico(pico: pd.Series) -> str:
        numero = int(float(pico.get("Pico", 0)))
        tempo = float(pico.get("Tempo (min)", 0.0))
        area = float(pico.get("Área", 0.0))
        return f"Pico {numero} · {tempo:.4f} min · área {area:.4f}"

    @staticmethod
    def _para_float_seguro(valor: object) -> float | None:
        numero = pd.to_numeric(pd.Series([valor]), errors="coerce").iloc[0]
        return float(numero) if pd.notna(numero) else None

    def _obter_referencia_padrao_composto(
        self,
        composto: str,
    ) -> dict[str, float | None] | None:
        if self.referencias_padrao.empty or "Composto" not in self.referencias_padrao:
            return None

        mascara = (
            self.referencias_padrao["Composto"].astype(str).str.strip().str.casefold()
            == str(composto).strip().casefold()
        )
        if not mascara.any():
            return None

        linha = self.referencias_padrao.loc[mascara].iloc[0]
        return {
            "Tempo (min)": self._para_float_seguro(linha.get("Tempo (min)")),
            "Início": self._para_float_seguro(linha.get("Início")),
            "Fim": self._para_float_seguro(linha.get("Fim")),
        }

    def _calcular_vale_entre_compostos_padrao(
        self,
        composto_esquerda: str,
        composto_direita: str,
    ) -> float | None:
        referencia_esquerda = self._obter_referencia_padrao_composto(composto_esquerda)
        referencia_direita = self._obter_referencia_padrao_composto(composto_direita)
        if referencia_esquerda is None or referencia_direita is None:
            return None

        tempo_esquerda = referencia_esquerda.get("Tempo (min)")
        tempo_direita = referencia_direita.get("Tempo (min)")
        if (
            self.padrao is not None
            and not self.padrao.dados.empty
            and tempo_esquerda is not None
            and tempo_direita is not None
            and tempo_direita > tempo_esquerda
        ):
            janela = self.padrao.dados.loc[
                (self.padrao.dados["Tempo"] >= float(tempo_esquerda))
                & (self.padrao.dados["Tempo"] <= float(tempo_direita))
            ]
            if not janela.empty:
                indice_vale = janela["Intensidade"].astype(float).idxmin()
                tempo_vale = self._para_float_seguro(janela.loc[indice_vale, "Tempo"])
                if tempo_vale is not None:
                    return tempo_vale

        fim_esquerda = referencia_esquerda.get("Fim")
        inicio_direita = referencia_direita.get("Início")
        if fim_esquerda is not None and inicio_direita is not None:
            return (float(fim_esquerda) + float(inicio_direita)) / 2.0
        if tempo_esquerda is not None and tempo_direita is not None:
            return (float(tempo_esquerda) + float(tempo_direita)) / 2.0
        return fim_esquerda if fim_esquerda is not None else inicio_direita

    @staticmethod
    def _selecionar_pico_representativo(
        picos: pd.DataFrame,
        *,
        centro: float | None = None,
        janela: float | None = None,
        tempo_min: float | None = None,
        tempo_max: float | None = None,
        preferir_ultimo: bool = False,
    ) -> pd.Series | None:
        if picos.empty or "Tempo (min)" not in picos.columns:
            return None

        candidatos = picos.copy()
        candidatos["__tempo__"] = pd.to_numeric(
            candidatos["Tempo (min)"], errors="coerce"
        )
        candidatos["__inicio__"] = pd.to_numeric(candidatos["Início"], errors="coerce")
        candidatos["__fim__"] = pd.to_numeric(candidatos["Fim"], errors="coerce")
        candidatos["__area__"] = pd.to_numeric(
            candidatos.get("Área"), errors="coerce"
        ).fillna(0.0)
        candidatos = candidatos[candidatos["__tempo__"].notna()].copy()

        if tempo_min is not None:
            candidatos = candidatos[candidatos["__tempo__"] >= float(tempo_min)]
        if tempo_max is not None:
            candidatos = candidatos[candidatos["__tempo__"] <= float(tempo_max)]

        if centro is not None and janela is not None:
            dentro_janela = candidatos.loc[
                (candidatos["__tempo__"] - float(centro)).abs() <= float(janela)
            ].copy()
            if not dentro_janela.empty:
                candidatos = dentro_janela

        if candidatos.empty:
            return None

        if preferir_ultimo:
            return candidatos.sort_values(
                ["__tempo__", "__fim__", "__area__"],
                ascending=[False, False, False],
            ).iloc[0]

        if centro is not None:
            candidatos["__desvio__"] = (candidatos["__tempo__"] - float(centro)).abs()
            return candidatos.sort_values(
                ["__desvio__", "__area__", "__fim__"],
                ascending=[True, False, False],
            ).iloc[0]

        return candidatos.sort_values(
            ["__area__", "__tempo__"],
            ascending=[False, True],
        ).iloc[0]

    @staticmethod
    def _estimar_fim_faixa_aromatica(
        picos: pd.DataFrame,
        inicio_busca: float,
        limite_superior: float | None = None,
    ) -> float | None:
        if picos.empty:
            return None

        candidatos = picos.copy()
        candidatos["__tempo__"] = pd.to_numeric(
            candidatos["Tempo (min)"], errors="coerce"
        )
        candidatos["__inicio__"] = pd.to_numeric(candidatos["Início"], errors="coerce")
        candidatos["__fim__"] = pd.to_numeric(candidatos["Fim"], errors="coerce")
        candidatos = candidatos[candidatos["__fim__"].notna()].copy()
        candidatos = candidatos[candidatos["__fim__"] >= float(inicio_busca)].copy()

        if limite_superior is not None:
            limite = float(limite_superior)
            candidatos = candidatos.loc[
                (candidatos["__inicio__"].fillna(candidatos["__tempo__"]) < limite)
                & (candidatos["__tempo__"].fillna(limite) <= limite)
            ].copy()

        if candidatos.empty:
            return None

        fim = pd.to_numeric(candidatos["__fim__"], errors="coerce").max()
        return float(fim) if pd.notna(fim) else None

    @staticmethod
    def _normalizar_intervalos_regioes(
        regioes: list[tuple[str, float, float]],
    ) -> list[tuple[str, float, float]]:
        intervalos_normalizados: list[tuple[str, float, float]] = []
        fim_anterior: float | None = None

        for nome, inicio, fim in regioes:
            inicio_float = float(inicio)
            fim_float = float(fim)
            if fim_anterior is not None:
                inicio_float = max(inicio_float, fim_anterior)
            if fim_float < inicio_float:
                fim_float = inicio_float
            intervalos_normalizados.append((nome, inicio_float, fim_float))
            fim_anterior = fim_float

        return intervalos_normalizados

    def _obter_regioes_analiticas(
        self,
        metodo: str | None = None,
        eh_diesel: bool = False,
        *,
        picos: pd.DataFrame | None = None,
        dados: pd.DataFrame | None = None,
    ) -> list[tuple[str, float, float]]:
        metodo_normalizado = self._normalizar_metodo_analise(metodo)
        regioes_por_tipo = self.REGIOES_ANALITICAS_POR_METODO.get(metodo_normalizado)

        if isinstance(regioes_por_tipo, dict):
            regioes_base = list(
                regioes_por_tipo.get(
                    bool(eh_diesel),
                    regioes_por_tipo.get(False, self.REGIOES_ANALITICAS),
                )
            )
        else:
            regioes_base = list(regioes_por_tipo or self.REGIOES_ANALITICAS)

        if not regioes_base:
            return []

        mapa_regioes = {
            str(nome): (float(inicio), float(fim)) for nome, inicio, fim in regioes_base
        }

        def resolver_intervalo(
            nome: str,
            inicio: float | None,
            fim: float | None,
        ) -> tuple[float, float]:
            inicio_base, fim_base = mapa_regioes.get(str(nome), (0.0, 0.0))
            inicio_final = (
                float(inicio)
                if inicio is not None and pd.notna(inicio)
                else float(inicio_base)
            )
            fim_final = (
                float(fim) if fim is not None and pd.notna(fim) else float(fim_base)
            )
            return inicio_final, fim_final

        picos_validos = (
            picos.copy()
            if picos is not None and not picos.empty and "Tempo (min)" in picos.columns
            else self._criar_dataframe_picos_vazio()
        )
        if picos_validos.empty and dados is not None and not dados.empty:
            try:
                picos_validos = self._detectar_picos_amostra(dados)
            except Exception:
                picos_validos = self._criar_dataframe_picos_vazio()

        if not picos_validos.empty:
            picos_validos["Tempo (min)"] = pd.to_numeric(
                picos_validos["Tempo (min)"], errors="coerce"
            )
            picos_validos["Início"] = pd.to_numeric(
                picos_validos["Início"], errors="coerce"
            )
            picos_validos["Fim"] = pd.to_numeric(picos_validos["Fim"], errors="coerce")
            picos_validos = picos_validos.sort_values("Tempo (min)").reset_index(
                drop=True
            )

        inicio_amostra = None
        if not picos_validos.empty and picos_validos["Início"].notna().any():
            inicio_amostra = float(picos_validos["Início"].dropna().min())

        tempo_final_corrida = None
        if dados is not None and not dados.empty and "Tempo" in dados.columns:
            tempo_final = pd.to_numeric(dados["Tempo"], errors="coerce").max()
            if pd.notna(tempo_final):
                tempo_final_corrida = float(tempo_final)
        if (
            tempo_final_corrida is None
            and not picos_validos.empty
            and picos_validos["Fim"].notna().any()
        ):
            tempo_final_corrida = float(picos_validos["Fim"].dropna().max())

        referencia_tolueno = self._obter_referencia_padrao_composto("Tolueno") or {}
        referencia_naftaleno = self._obter_referencia_padrao_composto("Naftaleno") or {}
        referencia_dibenzotiofeno = (
            self._obter_referencia_padrao_composto("Dibenzotiofeno") or {}
        )

        limite_hexadecano_tolueno = self._calcular_vale_entre_compostos_padrao(
            "Hexadecano",
            "Tolueno",
        )
        inicio_tolueno = referencia_tolueno.get("Início") or referencia_tolueno.get(
            "Tempo (min)"
        )
        tempo_naftaleno = referencia_naftaleno.get("Tempo (min)")
        inicio_naftaleno = referencia_naftaleno.get("Início") or tempo_naftaleno
        fim_dibenzotiofeno = referencia_dibenzotiofeno.get(
            "Fim"
        ) or referencia_dibenzotiofeno.get("Tempo (min)")

        pico_saturados_pesados = self._selecionar_pico_representativo(
            picos_validos,
            centro=25.0,
            janela=1.8,
        )
        inicio_saturados_pesados = self._para_float_seguro(
            pico_saturados_pesados.get("Início")
            if pico_saturados_pesados is not None
            else None
        )
        fim_saturados_pesados = self._para_float_seguro(
            pico_saturados_pesados.get("Fim")
            if pico_saturados_pesados is not None
            else None
        )

        limite_inferior_olefinas = max(
            26.0,
            float(fim_saturados_pesados) if fim_saturados_pesados is not None else 26.0,
        )
        pico_olefinas = self._selecionar_pico_representativo(
            picos_validos,
            centro=27.0,
            janela=4.0,
            tempo_min=limite_inferior_olefinas,
            preferir_ultimo=True,
        )
        inicio_olefinas = self._para_float_seguro(
            pico_olefinas.get("Início") if pico_olefinas is not None else None
        )

        limite_superior_aromaticos = None
        for candidato in (inicio_saturados_pesados, inicio_olefinas):
            if candidato is None:
                continue
            limite_superior_aromaticos = (
                float(candidato)
                if limite_superior_aromaticos is None
                else min(limite_superior_aromaticos, float(candidato))
            )

        marcadores_aromaticos = [
            valor
            for valor in (fim_dibenzotiofeno, tempo_naftaleno, inicio_tolueno)
            if valor is not None
        ]
        inicio_busca_final = (
            max(float(valor) for valor in marcadores_aromaticos)
            if marcadores_aromaticos
            else None
        )

        fim_aromaticos = (
            self._estimar_fim_faixa_aromatica(
                picos_validos,
                float(inicio_busca_final),
                limite_superior_aromaticos,
            )
            if inicio_busca_final is not None
            else None
        )
        if fim_aromaticos is not None and limite_superior_aromaticos is not None:
            fim_aromaticos = min(
                float(fim_aromaticos), float(limite_superior_aromaticos)
            )

        if metodo_normalizado == "ASTM D6550":
            if eh_diesel:
                inicio_saturados, fim_saturados = resolver_intervalo(
                    "Saturados",
                    inicio_amostra,
                    limite_hexadecano_tolueno,
                )
                inicio_mono, fim_mono = resolver_intervalo(
                    "Mono-aromaticos",
                    inicio_tolueno,
                    inicio_naftaleno,
                )
                inicio_di, fim_di = resolver_intervalo(
                    "Di-aromaticos",
                    inicio_naftaleno,
                    fim_dibenzotiofeno,
                )
                inicio_tri, fim_tri = resolver_intervalo(
                    "Tri-aromaticos+",
                    fim_dibenzotiofeno,
                    fim_aromaticos,
                )
                inicio_pesados, fim_pesados = resolver_intervalo(
                    "Saturados pesados",
                    inicio_saturados_pesados,
                    fim_saturados_pesados,
                )
                inicio_olef, fim_olef = resolver_intervalo(
                    "Olefinas",
                    inicio_olefinas,
                    tempo_final_corrida,
                )
                return self._normalizar_intervalos_regioes(
                    [
                        ("Saturados", inicio_saturados, fim_saturados),
                        ("Mono-aromaticos", inicio_mono, fim_mono),
                        ("Di-aromaticos", inicio_di, fim_di),
                        ("Tri-aromaticos+", inicio_tri, fim_tri),
                        ("Saturados pesados", inicio_pesados, fim_pesados),
                        ("Olefinas", inicio_olef, fim_olef),
                    ]
                )

            inicio_saturados, fim_saturados = resolver_intervalo(
                "Saturados",
                inicio_amostra,
                limite_hexadecano_tolueno,
            )
            inicio_mono, fim_mono = resolver_intervalo(
                "Mono-aromaticos",
                inicio_tolueno,
                tempo_naftaleno,
            )
            inicio_poli, fim_poli = resolver_intervalo(
                "Poli-aromaticos",
                tempo_naftaleno,
                fim_aromaticos,
            )
            inicio_pesados, fim_pesados = resolver_intervalo(
                "Saturados pesados",
                inicio_saturados_pesados,
                fim_saturados_pesados,
            )
            inicio_olef, fim_olef = resolver_intervalo(
                "Olefinas",
                inicio_olefinas,
                tempo_final_corrida,
            )
            return self._normalizar_intervalos_regioes(
                [
                    ("Saturados", inicio_saturados, fim_saturados),
                    ("Mono-aromaticos", inicio_mono, fim_mono),
                    ("Poli-aromaticos", inicio_poli, fim_poli),
                    ("Saturados pesados", inicio_pesados, fim_pesados),
                    ("Olefinas", inicio_olef, fim_olef),
                ]
            )

        if eh_diesel:
            inicio_nao_aromaticos, fim_nao_aromaticos = resolver_intervalo(
                "Não-aromaticos",
                inicio_amostra,
                limite_hexadecano_tolueno,
            )
            inicio_mono, fim_mono = resolver_intervalo(
                "Mono-aromaticos",
                inicio_tolueno,
                inicio_naftaleno,
            )
            inicio_di, fim_di = resolver_intervalo(
                "Di-aromaticos",
                inicio_naftaleno,
                fim_dibenzotiofeno,
            )
            inicio_tri, fim_tri = resolver_intervalo(
                "Tri-aromaticos+",
                fim_dibenzotiofeno,
                fim_aromaticos,
            )
            return self._normalizar_intervalos_regioes(
                [
                    ("Não-aromaticos", inicio_nao_aromaticos, fim_nao_aromaticos),
                    ("Mono-aromaticos", inicio_mono, fim_mono),
                    ("Di-aromaticos", inicio_di, fim_di),
                    ("Tri-aromaticos+", inicio_tri, fim_tri),
                ]
            )

        inicio_nao_aromaticos, fim_nao_aromaticos = resolver_intervalo(
            "Não-aromaticos",
            inicio_amostra,
            limite_hexadecano_tolueno,
        )
        inicio_mono, fim_mono = resolver_intervalo(
            "Mono-aromaticos",
            inicio_tolueno,
            tempo_naftaleno,
        )
        inicio_poli, fim_poli = resolver_intervalo(
            "Poli-aromaticos",
            tempo_naftaleno,
            fim_aromaticos,
        )
        return self._normalizar_intervalos_regioes(
            [
                ("Não-aromaticos", inicio_nao_aromaticos, fim_nao_aromaticos),
                ("Mono-aromaticos", inicio_mono, fim_mono),
                ("Poli-aromaticos", inicio_poli, fim_poli),
            ]
        )

    def _obter_metodo_amostra(self, amostra: Amostra | None) -> str:
        if amostra is None:
            return self.METODO_ANALISE_ASTM_PADRAO
        return self._normalizar_metodo_analise(amostra.metodo_analise)

    def _obter_tipo_amostra(self, amostra: Amostra | None) -> str:
        return "Diesel" if amostra is not None and amostra.eh_diesel else "Não diesel"

    def _perguntar_metadados_amostra(
        self,
        caminho: Path,
        chave: str,
        quantidade_arquivos: int = 1,
    ) -> tuple[str, bool] | None:
        dialogo = QDialog(self)
        dialogo.setWindowTitle(
            "Informações das amostras"
            if quantidade_arquivos > 1
            else "Informações da amostra"
        )
        layout = QVBoxLayout(dialogo)

        if quantidade_arquivos > 1:
            texto_descricao = (
                f"{quantidade_arquivos} arquivo(s) selecionado(s)\n"
                "Defina ASTM e tipo de amostra para todos os arquivos desta carga."
            )
        else:
            texto_descricao = (
                f"Arquivo: {caminho.name}\n"
                "Defina ASTM e tipo de amostra para esta carga."
            )

        descricao = QLabel(texto_descricao)
        descricao.setWordWrap(True)
        layout.addWidget(descricao)

        formulario = QFormLayout()

        campo_astm = QComboBox(dialogo)
        campo_astm.addItems(self.METODOS_ANALISE_ASTM)
        campo_astm.setCurrentText(self._carregar_metodo_amostra(chave))
        campo_astm.setToolTip(
            "Selecione o método ASTM:\n"
            "D5186: Aromáticos (mono e poli)\n"
            "D6550: Olefinas\n"
        )
        formulario.addRow("ASTM:", campo_astm)

        campo_diesel = QCheckBox("Amostra de diesel", dialogo)
        campo_diesel.setChecked(self._carregar_diesel_amostra(chave, caminho))
        formulario.addRow("Tipo:", campo_diesel)

        layout.addLayout(formulario)

        aviso = QLabel(
            "Essas informações aparecem como tags na lista de amostras. "
            "Depois, a ASTM pode ser alterada clicando no nome do arquivo."
        )
        if quantidade_arquivos > 1:
            aviso.setText(
                "Essas informações serão aplicadas a todos os arquivos carregados agora. "
                "Depois, a ASTM pode ser alterada clicando no nome do arquivo."
            )
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: #555;")
        layout.addWidget(aviso)

        botoes = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialogo,
        )
        botoes.accepted.connect(dialogo.accept)
        botoes.rejected.connect(dialogo.reject)
        layout.addWidget(botoes)

        if dialogo.exec() != QDialog.Accepted:
            return None

        return (
            self._normalizar_metodo_analise(campo_astm.currentText()),
            bool(campo_diesel.isChecked()),
        )

    def _ler_bool_setting(self, chave: str, padrao: bool) -> bool:
        valor = self.settings.value(chave, padrao)
        if isinstance(valor, bool):
            return valor
        return str(valor).strip().lower() not in {"0", "false", "no", "off", ""}

    def _executar_com_aviso_espera(self, acao, mensagem: str = "Espere...") -> None:
        aviso = QProgressDialog(mensagem, "", 0, 0, self)
        aviso.setWindowTitle("Processando")
        aviso.setWindowModality(Qt.ApplicationModal)
        aviso.setCancelButton(None)
        aviso.setMinimumDuration(0)
        aviso.setAutoClose(False)
        aviso.setAutoReset(False)
        aviso.show()
        QApplication.processEvents()
        try:
            acao()
        finally:
            aviso.close()
            aviso.deleteLater()
            QApplication.processEvents()

    def _conectar_botao_com_aviso_espera(
        self,
        botao: QPushButton,
        acao,
        mensagem: str = "Espere...",
    ) -> None:
        botao.clicked.connect(
            lambda _checked=False: self._executar_com_aviso_espera(acao, mensagem)
        )

    def _visibilidade_habilitada(self, atributo: str, padrao: bool = True) -> bool:
        controle = getattr(self, atributo, None)
        return controle.isChecked() if controle is not None else padrao

    def _carregar_opcoes_salvas(self) -> None:
        """Restaura do `QSettings` os parâmetros exibidos no menu Options."""
        self.ALTURA_MINIMA_PICO = float(
            self.settings.value("options/altura_minima_pico", self.ALTURA_MINIMA_PICO)
        )
        self.JANELA_SUAVIZACAO = int(
            self.settings.value("options/janela_suavizacao", self.JANELA_SUAVIZACAO)
        )
        self.DIVISOR_DISTANCIA_MINIMA = int(
            self.settings.value(
                "options/divisor_distancia_minima", self.DIVISOR_DISTANCIA_MINIMA
            )
        )
        self.MIN_PROMINENCIA_PICO = float(
            self.settings.value(
                "options/min_prominencia_pico", self.MIN_PROMINENCIA_PICO
            )
        )
        self.LARGURA_MINIMA_PICO = int(
            self.settings.value("options/largura_minima_pico", self.LARGURA_MINIMA_PICO)
        )
        self.FATOR_RUIDO = float(
            self.settings.value("options/fator_ruido", self.FATOR_RUIDO)
        )
        self.ALTURA_RELATIVA_PICO = min(
            max(
                float(
                    self.settings.value(
                        "options/altura_relativa_pico", self.ALTURA_RELATIVA_PICO
                    )
                ),
                0.50,
            ),
            0.995,
        )
        self.FATOR_CURVATURA = max(
            float(self.settings.value("options/fator_curvatura", self.FATOR_CURVATURA)),
            0.10,
        )
        modelo = str(
            self.settings.value("options/modelo_ajuste_pico", self.MODELO_AJUSTE_PICO)
        )
        self.MODELO_AJUSTE_PICO = (
            modelo if modelo in self.MODELOS_AJUSTE_PICO else self.MODELO_AJUSTE_PICO
        )

        self.mostrar_amostra = self._ler_bool_setting("plot/mostrar_amostra", True)
        self.mostrar_padrao = self._ler_bool_setting("plot/mostrar_padrao", True)
        self.mostrar_picos = self._ler_bool_setting("plot/mostrar_picos", True)
        self.mostrar_grupos = self._ler_bool_setting("plot/mostrar_grupos", True)
        self.mostrar_ajuste = self._ler_bool_setting("plot/mostrar_ajuste", True)

        if self.JANELA_SUAVIZACAO % 2 == 0:
            self.JANELA_SUAVIZACAO += 1

    def _prominencia_configurada(self) -> float | None:
        return (
            None
            if float(self.MIN_PROMINENCIA_PICO) <= 0
            else float(self.MIN_PROMINENCIA_PICO)
        )

    def _detectar_picos_amostra(
        self, dados: pd.DataFrame, chave: str | None = None
    ) -> pd.DataFrame:
        picos = self.controller.detect_sample_peaks(
            dados,
            self.COLUNAS_AMOSTRA,
            self.ALTURA_MINIMA_PICO,
            self.JANELA_SUAVIZACAO,
            self.DIVISOR_DISTANCIA_MINIMA,
            min_prominence=self._prominencia_configurada(),
            min_width=self.LARGURA_MINIMA_PICO,
            noise_factor=self.FATOR_RUIDO,
            rel_height=self.ALTURA_RELATIVA_PICO,
            curvature_factor=self.FATOR_CURVATURA,
        )
        return self.controller.apply_saved_peaks(chave, picos) if chave else picos

    def _detectar_picos_padrao(self, dados: pd.DataFrame) -> pd.DataFrame:
        return self.controller.detect_standard_peaks(
            dados,
            self._criar_referencias_padrao_iniciais(),
            self.COLUNAS_AMOSTRA,
            self.ALTURA_MINIMA_PICO,
            self.JANELA_SUAVIZACAO,
            self.DIVISOR_DISTANCIA_MINIMA,
            min_prominence=self._prominencia_configurada(),
            min_width=self.LARGURA_MINIMA_PICO,
            noise_factor=self.FATOR_RUIDO,
            rel_height=self.ALTURA_RELATIVA_PICO,
            curvature_factor=self.FATOR_CURVATURA,
        )

    def _obter_amostra_atual(self) -> tuple[str | None, Amostra | None]:
        item_atual = self.lista_arquivos.currentItem()
        if item_atual is None:
            return None, None

        chave = item_atual.data(Qt.UserRole)
        return chave, self.amostras.get(chave)

    def _obter_intervalo_visual_atual(self) -> tuple[float, float] | None:
        """Retorna o intervalo X atualmente visível no gráfico, se houver."""
        if not self.canvas.ax.has_data():
            return None

        x_min, x_max = self.canvas.ax.get_xlim()
        return float(min(x_min, x_max)), float(max(x_min, x_max))

    def _ferramenta_navegacao_ativa(self) -> bool:
        return bool(str(getattr(self.toolbar_grafico, "mode", "") or "").strip())

    def _limpar_guias_selecao_manual(self) -> None:
        for nome in (
            "_linha_tempo_cursor",
            "_linha_tempo_fixo",
            "_faixa_selecao_manual",
        ):
            artista = getattr(self, nome, None)
            if artista is not None:
                try:
                    artista.remove()
                except Exception:
                    pass
            setattr(self, nome, None)

    def _atualizar_guias_selecao_manual(self) -> None:
        self._limpar_guias_selecao_manual()

        if not hasattr(self, "canvas"):
            return
        if self._tempo_primeiro_clique is None and self._tempo_cursor_grafico is None:
            return

        eixo = self.canvas.ax
        if self._tempo_primeiro_clique is not None:
            self._linha_tempo_fixo = eixo.axvline(
                float(self._tempo_primeiro_clique),
                color="#c62828",
                linestyle="--",
                linewidth=1.15,
                alpha=0.95,
                zorder=5,
            )

        if self._tempo_cursor_grafico is not None:
            self._linha_tempo_cursor = eixo.axvline(
                float(self._tempo_cursor_grafico),
                color="#d62728",
                linestyle=":",
                linewidth=1.10,
                alpha=0.95,
                zorder=5,
            )

        if (
            self._tempo_primeiro_clique is not None
            and self._tempo_cursor_grafico is not None
            and abs(self._tempo_primeiro_clique - self._tempo_cursor_grafico) > 1e-12
        ):
            inicio, fim = sorted(
                [
                    float(self._tempo_primeiro_clique),
                    float(self._tempo_cursor_grafico),
                ]
            )
            self._faixa_selecao_manual = eixo.axvspan(
                inicio,
                fim,
                color="#d62728",
                alpha=0.08,
                zorder=1,
            )

    def _cancelar_selecao_manual(self, *, redesenhar: bool = True) -> None:
        self._tempo_primeiro_clique = None
        self._chave_amostra_selecao_manual = None
        self._tempo_cursor_grafico = None
        self._limpar_guias_selecao_manual()
        if redesenhar and hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _ao_mover_mouse_no_grafico(self, event) -> None:
        if self._ferramenta_navegacao_ativa():
            if self._tempo_cursor_grafico is not None:
                self._tempo_cursor_grafico = None
                self._atualizar_guias_selecao_manual()
                self.canvas.draw_idle()
            return

        if event.inaxes != self.canvas.ax or getattr(event, "xdata", None) is None:
            if self._tempo_cursor_grafico is not None:
                self._tempo_cursor_grafico = None
                self._atualizar_guias_selecao_manual()
                self.canvas.draw_idle()
            return

        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None or amostra.dados.empty:
            return

        tempo_min = float(amostra.dados["Tempo"].min())
        tempo_max = float(amostra.dados["Tempo"].max())
        self._tempo_cursor_grafico = min(max(float(event.xdata), tempo_min), tempo_max)
        self._atualizar_guias_selecao_manual()
        self.canvas.draw_idle()

        if self._tempo_primeiro_clique is None:
            self.statusBar().showMessage(
                f"Tempo do cursor: {self._tempo_cursor_grafico:.4f} min | "
                "Clique esquerdo para marcar o início do pico manual."
            )
        else:
            inicio, fim = sorted(
                [float(self._tempo_primeiro_clique), float(self._tempo_cursor_grafico)]
            )
            self.statusBar().showMessage(
                f"Intervalo manual: {inicio:.4f} a {fim:.4f} min | "
                "Clique esquerdo novamente para integrar."
            )

    def _ao_sair_do_grafico(self, _event) -> None:
        if self._tempo_cursor_grafico is None:
            return

        self._tempo_cursor_grafico = None
        self._atualizar_guias_selecao_manual()
        self.canvas.draw_idle()

    def _ao_clicar_linha_tabela_amostras(self, linha: int, _coluna: int) -> None:
        """Ao clicar numa linha da tabela, aplica zoom no intervalo do pico."""
        if self._atualizando_tabelas:
            return

        _, amostra = self._obter_amostra_atual()
        if amostra is None or amostra.picos.empty:
            return
        if linha < 0 or linha >= len(amostra.picos):
            return

        self._zoom_para_pico(amostra, amostra.picos.iloc[linha])

    def _abrir_menu_contexto_tabela_amostras(self, pos) -> None:
        """Mostra ações úteis da tabela Amostras, incluindo adição e remoção manual."""
        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None:
            return

        indice = self.tabela_amostras.indexAt(pos)
        if indice.isValid():
            self.tabela_amostras.selectRow(indice.row())

        menu = QMenu(self)
        acao_adicionar = menu.addAction("Adicionar pico...")
        acao_remover = None
        acao_zoom = None
        acao_remover_todos = None
        if not amostra.picos.empty:
            if indice.isValid():
                acao_zoom = menu.addAction("Zoom no pico selecionado")
                acao_remover = menu.addAction("Remover pico selecionado")
            menu.addSeparator()
            acao_remover_todos = menu.addAction("Remover todos os picos")

        selecionada = menu.exec(self.tabela_amostras.viewport().mapToGlobal(pos))
        if selecionada == acao_adicionar:
            self._adicionar_pico_manual()
        elif acao_zoom is not None and selecionada == acao_zoom:
            self._ao_clicar_linha_tabela_amostras(indice.row(), indice.column())
        elif acao_remover is not None and selecionada == acao_remover:
            self._remover_pico_selecionado()
        elif acao_remover_todos is not None and selecionada == acao_remover_todos:
            self._remover_todos_picos_amostra()

    def _adicionar_pico_manual(self) -> None:
        """Solicita início/fim e cria um novo pico manualmente na amostra atual."""
        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None:
            QMessageBox.information(
                self,
                "Adicionar pico",
                "Selecione uma amostra antes de adicionar um pico manual.",
            )
            return

        tempo_min = float(amostra.dados["Tempo"].min())
        tempo_max = float(amostra.dados["Tempo"].max())
        intervalo_visual = self._obter_intervalo_visual_atual()
        if intervalo_visual is None:
            inicio_padrao = tempo_min
            fim_padrao = min(tempo_min + 0.25, tempo_max)
        else:
            inicio_padrao = min(max(intervalo_visual[0], tempo_min), tempo_max)
            fim_padrao = min(max(intervalo_visual[1], tempo_min), tempo_max)
            if fim_padrao - inicio_padrao > 1.0:
                centro = (inicio_padrao + fim_padrao) / 2.0
                inicio_padrao = max(tempo_min, centro - 0.15)
                fim_padrao = min(tempo_max, centro + 0.15)

        dialogo = QDialog(self)
        dialogo.setWindowTitle("Adicionar pico manual")
        layout = QVBoxLayout(dialogo)
        formulario = QFormLayout()

        campo_inicio = QDoubleSpinBox(dialogo)
        campo_inicio.setDecimals(4)
        campo_inicio.setRange(tempo_min, tempo_max)
        campo_inicio.setValue(inicio_padrao)

        campo_fim = QDoubleSpinBox(dialogo)
        campo_fim.setDecimals(4)
        campo_fim.setRange(tempo_min, tempo_max)
        campo_fim.setValue(max(fim_padrao, inicio_padrao + 0.01))

        formulario.addRow("Início:", campo_inicio)
        formulario.addRow("Fim:", campo_fim)
        layout.addLayout(formulario)

        botoes = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialogo,
        )
        botoes.accepted.connect(dialogo.accept)
        botoes.rejected.connect(dialogo.reject)
        layout.addWidget(botoes)

        if dialogo.exec() != QDialog.Accepted:
            return

        try:
            self._inserir_pico_manual(
                chave,
                amostra,
                float(campo_inicio.value()),
                float(campo_fim.value()),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Adicionar pico", str(exc))

    def _inserir_pico_manual(
        self,
        chave: str,
        amostra: Amostra,
        inicio: float,
        fim: float,
    ) -> None:
        """Insere o pico calculado na tabela e no gráfico, com renumeração."""
        novo_pico = self.controller.create_peak_by_interval(amostra.dados, inicio, fim)
        picos_atualizados = pd.concat(
            [amostra.picos, pd.DataFrame([novo_pico])],
            ignore_index=True,
        )
        picos_atualizados = picos_atualizados.sort_values("Tempo (min)").reset_index(
            drop=True
        )
        picos_atualizados["Pico"] = range(1, len(picos_atualizados) + 1)
        amostra.picos = picos_atualizados[self.COLUNAS_AMOSTRA]

        self.controller.save_sample_peaks(chave, amostra.picos)
        self._atualizar_tabela(amostra.picos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)

        linha_nova = int(
            amostra.picos["Tempo (min)"]
            .sub(float(novo_pico["Tempo (min)"]))
            .abs()
            .idxmin()
        )
        self.tabela_amostras.selectRow(linha_nova)
        self._zoom_para_pico(amostra, amostra.picos.iloc[linha_nova])
        self.statusBar().showMessage("Pico manual adicionado.")

    def _remover_pico_selecionado(self) -> None:
        """Remove manualmente o pico selecionado na tabela de amostras."""
        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None or amostra.picos.empty:
            return

        linha = self.tabela_amostras.currentRow()
        if linha < 0 or linha >= len(amostra.picos):
            QMessageBox.information(
                self,
                "Remover pico",
                "Selecione uma linha válida na tabela Amostras.",
            )
            return

        pico = amostra.picos.iloc[linha]
        resposta = QMessageBox.question(
            self,
            "Remover pico",
            (
                f"Deseja remover o pico {int(float(pico['Pico']))} "
                f"({float(pico['Tempo (min)']):.4f} min)?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resposta != QMessageBox.Yes:
            return

        picos_atualizados = amostra.picos.drop(amostra.picos.index[linha]).reset_index(
            drop=True
        )
        if not picos_atualizados.empty:
            picos_atualizados["Pico"] = range(1, len(picos_atualizados) + 1)
            amostra.picos = picos_atualizados[self.COLUNAS_AMOSTRA]
        else:
            amostra.picos = self._criar_dataframe_picos_vazio()

        self._limpar_vinculos_grupos_invalidos(chave, amostra.picos)
        self.controller.save_sample_peaks(chave, amostra.picos)
        self._atualizar_tabela(amostra.picos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)

        if not amostra.picos.empty:
            linha_destino = min(linha, len(amostra.picos) - 1)
            self.tabela_amostras.selectRow(linha_destino)
            self._zoom_para_pico(amostra, amostra.picos.iloc[linha_destino])
        self.statusBar().showMessage("Pico removido.")

    def _remover_todos_picos_amostra(self) -> None:
        """Remove de uma vez todos os picos da amostra atualmente selecionada."""
        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None or amostra.picos.empty:
            return

        quantidade = len(amostra.picos)
        resposta = QMessageBox.question(
            self,
            "Remover todos os picos",
            (
                f"Deseja remover todos os {quantidade} picos da amostra "
                f"{amostra.caminho.name}?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resposta != QMessageBox.Yes:
            return

        amostra.picos = self._criar_dataframe_picos_vazio()
        self._limpar_vinculos_grupos_invalidos(chave, amostra.picos)
        self.controller.save_sample_peaks(chave, amostra.picos)
        self._atualizar_tabela(amostra.picos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)
        self.statusBar().showMessage("Todos os picos da amostra foram removidos.")

    def _abrir_menu_contexto_tabela_regioes(self, pos) -> None:
        """Permite vincular manualmente cada grupo a um pico da amostra."""
        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None:
            return

        indice = self.tabela_regioes.indexAt(pos)
        if not indice.isValid():
            return

        self.tabela_regioes.selectRow(indice.row())
        item_grupo = self.tabela_regioes.item(indice.row(), 0)
        if item_grupo is None:
            return

        grupo = item_grupo.text().strip()
        if not grupo or grupo == "Nenhum grupo calculado":
            return

        menu = QMenu(self)
        submenu_vincular = menu.addMenu(f"Vincular '{grupo}' a pico")
        acoes_picos: dict[QAction, dict[str, float | int]] = {}

        if amostra.picos.empty:
            acao_sem_picos = submenu_vincular.addAction("Nenhum pico disponível")
            acao_sem_picos.setEnabled(False)
        else:
            for _, pico in amostra.picos.iterrows():
                acao = submenu_vincular.addAction(self._descrever_pico(pico))
                acoes_picos[acao] = {
                    "pico": int(float(pico.get("Pico", 0))),
                    "tempo": float(pico.get("Tempo (min)", 0.0)),
                }

        vinculos = self._carregar_vinculos_grupos_amostra(chave)
        pico_vinculado = self._resolver_pico_vinculado(
            amostra.picos,
            vinculos.get(grupo),
        )

        acao_zoom = None
        if pico_vinculado is not None:
            menu.addSeparator()
            acao_zoom = menu.addAction("Zoom no pico vinculado")

        menu.addSeparator()
        acao_limpar = menu.addAction("Usar cálculo automático")
        acao_limpar.setEnabled(grupo in vinculos)

        grupos_zerados = self._carregar_grupos_zerados_amostra(chave)
        menu.addSeparator()
        acao_zerar = menu.addAction("Zerar área do grupo")
        acao_restaurar_area = menu.addAction("Restaurar área calculada")
        acao_restaurar_area.setEnabled(grupo in grupos_zerados)

        selecionada = menu.exec(self.tabela_regioes.viewport().mapToGlobal(pos))
        if selecionada in acoes_picos:
            self._vincular_grupo_a_pico(chave, amostra, grupo, acoes_picos[selecionada])
        elif acao_zoom is not None and selecionada == acao_zoom:
            self._zoom_para_pico(amostra, pico_vinculado)
        elif selecionada == acao_limpar:
            self._limpar_vinculo_grupo(chave, amostra, grupo)
        elif selecionada == acao_zerar:
            self._zerar_area_grupo(chave, amostra, grupo)
        elif selecionada == acao_restaurar_area:
            self._restaurar_area_grupo(chave, amostra, grupo)

    def _zerar_area_grupo(
        self,
        chave: str,
        amostra: Amostra,
        grupo: str,
    ) -> None:
        grupos_zerados = self._carregar_grupos_zerados_amostra(chave)
        if grupo in grupos_zerados:
            return

        grupos_zerados.add(grupo)
        self._salvar_grupos_zerados_amostra(chave, grupos_zerados)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self.statusBar().showMessage(f"Área do grupo {grupo} zerada.")

    def _restaurar_area_grupo(
        self,
        chave: str,
        amostra: Amostra,
        grupo: str,
    ) -> None:
        grupos_zerados = self._carregar_grupos_zerados_amostra(chave)
        if grupo not in grupos_zerados:
            return

        grupos_zerados.discard(grupo)
        self._salvar_grupos_zerados_amostra(chave, grupos_zerados)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self.statusBar().showMessage(f"Área calculada do grupo {grupo} restaurada.")

    def _vincular_grupo_a_pico(
        self,
        chave: str,
        amostra: Amostra,
        grupo: str,
        pico_info: dict[str, float | int],
    ) -> None:
        vinculos = self._carregar_vinculos_grupos_amostra(chave)
        vinculos[grupo] = {
            "pico": int(float(pico_info["pico"])),
            "tempo": float(pico_info["tempo"]),
        }
        self._salvar_vinculos_grupos_amostra(chave, vinculos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)

        pico_vinculado = self._resolver_pico_vinculado(amostra.picos, vinculos[grupo])
        if pico_vinculado is not None:
            self._zoom_para_pico(amostra, pico_vinculado)
            self.statusBar().showMessage(
                f"Grupo {grupo} vinculado a {self._descrever_pico(pico_vinculado)}."
            )

    def _limpar_vinculo_grupo(
        self,
        chave: str,
        amostra: Amostra,
        grupo: str,
    ) -> None:
        vinculos = self._carregar_vinculos_grupos_amostra(chave)
        if grupo not in vinculos:
            return

        vinculos.pop(grupo, None)
        self._salvar_vinculos_grupos_amostra(chave, vinculos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)
        self.statusBar().showMessage(f"Grupo {grupo} voltou para o cálculo automático.")

    def _zoom_para_pico(self, amostra: Amostra, pico: pd.Series) -> None:
        """Centraliza a visualização no pico usando ±1 min no eixo X."""
        tempo_pico = float(pico.get("Tempo (min)", 0.0))
        tempo_pico_min = float(pico.get("Início", 0.0))
        tempo_pico_max = float(pico.get("Fim", 0.0))
        tempo_min = float(amostra.dados["Tempo"].min())
        tempo_max = float(amostra.dados["Tempo"].max())

        margem_tempo = 1.0
        x_min = tempo_pico_min - margem_tempo
        x_max = tempo_pico_max + margem_tempo
        largura_janela = 2.0 * margem_tempo

        if x_min < tempo_min:
            x_max = min(tempo_max, x_max + (tempo_min - x_min))
            x_min = tempo_min
        if x_max > tempo_max:
            x_min = max(tempo_min, x_min - (x_max - tempo_max))
            x_max = tempo_max
        if x_max - x_min < min(largura_janela, tempo_max - tempo_min):
            x_min = max(tempo_min, tempo_pico - margem_tempo)
            x_max = min(tempo_max, tempo_pico + margem_tempo)

        altura_pico = max(float(pico.get("Altura", 0.0)), 1e-9)
        y_max = altura_pico * 1.12

        self.canvas.ax.set_xlim(x_min, x_max)
        self.canvas.ax.set_ylim(self._calcular_y_min_fixo(y_max), y_max)
        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _adicionar_acao_menu(self, menu, texto: str, callback) -> None:
        acao = QAction(texto, self)
        acao.triggered.connect(callback)
        menu.addAction(acao)

    @staticmethod
    def _criar_tag_amostra(
        texto: str,
        cor_fundo: str,
        cor_texto: str = "white",
    ) -> QLabel:
        tag = QLabel(texto)
        tag.setStyleSheet(
            f"background-color: {cor_fundo}; color: {cor_texto}; "
            "border-radius: 9px; padding: 2px 8px; font-size: 10px; font-weight: 600;"
        )
        return tag

    def _criar_widget_item_amostra(
        self,
        item: QListWidgetItem,
        amostra: Amostra,
    ) -> QWidget:
        container = QWidget(self.lista_arquivos)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        rotulo_nome = QLabel(amostra.caminho.name)
        rotulo_nome.setToolTip(
            f"{amostra.caminho}\nClique no nome para alterar a ASTM desta amostra."
        )
        rotulo_nome.setStyleSheet("font-size: 10px; font-weight: 600; color: white;")
        rotulo_nome.setCursor(Qt.PointingHandCursor)
        layout.addWidget(rotulo_nome)

        linha_tags = QHBoxLayout()
        linha_tags.setContentsMargins(0, 0, 0, 0)
        linha_tags.setSpacing(6)

        metodo = self._obter_metodo_amostra(amostra)
        cor_metodo = "#1565C0" if metodo == "ASTM D5186" else "#EF6C00"
        cor_tipo = "#6D4C41" if amostra.eh_diesel else "#2E7D32"

        linha_tags.addWidget(self._criar_tag_amostra(metodo, cor_metodo))
        linha_tags.addWidget(
            self._criar_tag_amostra(self._obter_tipo_amostra(amostra), cor_tipo)
        )
        linha_tags.addStretch(1)
        layout.addLayout(linha_tags)

        container.mousePressEvent = lambda _event, item=item: (
            self.lista_arquivos.setCurrentItem(item)
        )
        rotulo_nome.mousePressEvent = (
            lambda event, item=item, chave=item.data(Qt.UserRole): (
                self._ao_clicar_nome_amostra_lista(
                    event,
                    item,
                    chave,
                )
            )
        )

        item.setSizeHint(container.sizeHint())
        return container

    def _ao_clicar_nome_amostra_lista(
        self,
        event,
        item: QListWidgetItem,
        chave: str,
    ) -> None:
        self.lista_arquivos.setCurrentItem(item)

        menu = QMenu(self)
        metodo_atual = self._obter_metodo_amostra(self.amostras.get(chave))
        for metodo in self.METODOS_ANALISE_ASTM:
            acao = menu.addAction(metodo)
            acao.setCheckable(True)
            acao.setChecked(metodo == metodo_atual)

        posicao = (
            event.globalPosition().toPoint()
            if hasattr(event, "globalPosition")
            else self.lista_arquivos.mapToGlobal(
                self.lista_arquivos.visualItemRect(item).center()
            )
        )
        selecionada = menu.exec(posicao)
        if selecionada is not None:
            self._ao_alterar_metodo_amostra_lista(item, chave, selecionada.text())

    def _ao_alterar_metodo_amostra_lista(
        self,
        item: QListWidgetItem,
        chave: str,
        metodo: str,
    ) -> None:
        self.lista_arquivos.setCurrentItem(item)
        amostra = self.amostras.get(chave)
        if amostra is None:
            return

        metodo_normalizado = self._normalizar_metodo_analise(metodo)
        if amostra.metodo_analise != metodo_normalizado:
            amostra.metodo_analise = metodo_normalizado
            self._salvar_metodo_amostra(chave, metodo_normalizado)
            self.lista_arquivos.setItemWidget(
                item,
                self._criar_widget_item_amostra(item, amostra),
            )

        self._ao_selecionar_amostra(item, preservar_visualizacao=True)
        self.statusBar().showMessage(
            f"{amostra.caminho.name}: análise definida como {metodo_normalizado}."
        )

    def _criar_menu(self) -> None:
        menu_arquivos = self.menuBar().addMenu("ARQUIVOS")
        self._adicionar_acao_menu(
            menu_arquivos, "CARREGAR AMOSTRAS", self.abrir_arquivos
        )
        self._adicionar_acao_menu(menu_arquivos, "LIMPAR LISTA", self.limpar_arquivos)
        menu_arquivos.addSeparator()
        self._adicionar_acao_menu(
            menu_arquivos, "CARREGAR PADRÃO", self.carregar_padrao
        )
        menu_arquivos.addSeparator()
        self._adicionar_acao_menu(menu_arquivos, "SALVAR", self.salvar_dados)
        menu_arquivos.addSeparator()

        self._adicionar_acao_menu(menu_arquivos, "SAIR", self.close)

        menu_opcoes = self.menuBar().addMenu("OPÇÕES")
        self._adicionar_acao_menu(
            menu_opcoes,
            "ENCONTRAR PICOS",
            self._encontrar_picos_amostras,
        )
        self._adicionar_acao_menu(
            menu_opcoes,
            "ENCONTRAR GRUPOS",
            self._encontrar_grupos_amostras,
        )
        self._adicionar_acao_menu(
            menu_opcoes,
            "PARÂMETROS DE DETECÇÃO",
            self.abrir_dialogo_opcoes,
        )
        self._adicionar_acao_menu(
            menu_opcoes,
            "RESETAR ZOOM DO GRÁFICO",
            self._restaurar_zoom_original,
        )

    def _encontrar_picos_amostras(self) -> None:
        if not self.amostras:
            QMessageBox.information(
                self,
                "Encontrar picos",
                "Carregue ao menos uma amostra antes de procurar picos.",
            )
            return

        chave_atual, amostra_atual = self._obter_amostra_atual()
        if chave_atual is None or amostra_atual is None:
            QMessageBox.information(
                self,
                "Encontrar picos",
                "Selecione a amostra que deseja processar.",
            )
            return

        amostra_atual.picos = self._detectar_picos_amostra(
            amostra_atual.dados,
            chave_atual,
        )
        self._limpar_vinculos_grupos_invalidos(chave_atual, amostra_atual.picos)
        self.controller.save_sample_peaks(chave_atual, amostra_atual.picos)

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(item_atual, preservar_visualizacao=True)
        else:
            self._mostrar_placeholder()

        self.statusBar().showMessage(
            f"Encontrados {len(amostra_atual.picos)} pico(s) em {amostra_atual.caminho.name}."
        )

    def _encontrar_grupos_amostras(self) -> None:
        if not self.amostras:
            QMessageBox.information(
                self,
                "Encontrar grupos",
                "Carregue ao menos uma amostra antes de recalcular os grupos.",
            )
            return

        chave_atual, amostra_atual = self._obter_amostra_atual()
        if chave_atual is None or amostra_atual is None:
            QMessageBox.information(
                self,
                "Encontrar grupos",
                "Selecione a amostra que deseja processar.",
            )
            return

        vinculos = self._carregar_vinculos_grupos_amostra(chave_atual)
        grupos_redefinidos = len(vinculos)
        if vinculos:
            self._salvar_vinculos_grupos_amostra(chave_atual, {})
        self._limpar_vinculos_grupos_invalidos(chave_atual, amostra_atual.picos)

        amostra_atual.picos = self._integrar_grupos_como_picos(
            chave_atual, amostra_atual
        )
        self.controller.save_sample_peaks(chave_atual, amostra_atual.picos)

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(item_atual, preservar_visualizacao=True)
        else:
            self._mostrar_placeholder()

        self.statusBar().showMessage(
            (
                f"Grupos recalculados automaticamente para {amostra_atual.caminho.name}. "
                f"{grupos_redefinidos} vínculo(s) manual(is) removido(s)."
            )
        )

    def _integrar_grupos_como_picos(
        self,
        chave: str,
        amostra: Amostra,
    ) -> pd.DataFrame:
        metodo = self._obter_metodo_amostra(amostra)
        df_regioes = self._calcular_areas_por_regiao(
            amostra.picos,
            metodo,
            amostra.eh_diesel,
        )

        if df_regioes.empty:
            return self._criar_dataframe_picos_vazio()

        picos_integrados: list[dict[str, float | int]] = []
        for indice, (_, regiao) in enumerate(df_regioes.iterrows(), start=1):
            inicio = float(regiao["Início (min)"])
            fim = float(regiao["Fim (min)"])
            area = float(regiao["Área Total"])

            try:
                pico_info = self.controller.create_peak_by_interval(
                    amostra.dados,
                    inicio,
                    fim,
                )
            except ValueError:
                pico_info = {}

            linha = {col: 0.0 for col in self.COLUNAS_AMOSTRA}
            linha.update(
                {k: v for k, v in pico_info.items() if k in self.COLUNAS_AMOSTRA}
            )
            linha["Pico"] = indice
            linha["Início"] = inicio
            linha["Fim"] = fim
            linha["Área"] = area  # use the same integration as the GRUPOS table
            picos_integrados.append(linha)

        if not picos_integrados:
            return self._criar_dataframe_picos_vazio()

        return pd.DataFrame(picos_integrados)[self.COLUNAS_AMOSTRA]

    def _ao_alterar_visibilidade_grafico(self, _marcado: bool) -> None:
        self.mostrar_amostra = self.check_mostrar_amostra.isChecked()
        self.mostrar_padrao = self.check_mostrar_padrao.isChecked()
        self.mostrar_picos = self.check_mostrar_picos.isChecked()
        self.mostrar_grupos = self.check_mostrar_grupos.isChecked()
        self.mostrar_ajuste = self.check_mostrar_ajuste.isChecked()

        self.settings.setValue("plot/mostrar_amostra", self.mostrar_amostra)
        self.settings.setValue("plot/mostrar_padrao", self.mostrar_padrao)
        self.settings.setValue("plot/mostrar_picos", self.mostrar_picos)
        self.settings.setValue("plot/mostrar_grupos", self.mostrar_grupos)
        self.settings.setValue("plot/mostrar_ajuste", self.mostrar_ajuste)
        self.settings.sync()

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(item_atual, preservar_visualizacao=True)
        else:
            self._mostrar_placeholder()

    def abrir_dialogo_opcoes(self) -> None:
        """Exibe no menu Opções os parâmetros de detecção e ajuste de modelo."""
        dialogo = QDialog(self)
        dialogo.setWindowTitle("Opções - Parâmetros")
        layout = QVBoxLayout(dialogo)
        formulario = QFormLayout()

        campo_altura = QDoubleSpinBox(dialogo)
        campo_altura.setDecimals(1)
        campo_altura.setRange(0.0, 1_000_000_000.0)
        campo_altura.setValue(float(self.ALTURA_MINIMA_PICO))

        campo_janela = QSpinBox(dialogo)
        campo_janela.setRange(5, 999)
        campo_janela.setSingleStep(2)
        campo_janela.setValue(int(self.JANELA_SUAVIZACAO))

        campo_divisor = QSpinBox(dialogo)
        campo_divisor.setRange(1, 500)
        campo_divisor.setValue(int(self.DIVISOR_DISTANCIA_MINIMA))

        campo_prominencia = QDoubleSpinBox(dialogo)
        campo_prominencia.setDecimals(1)
        campo_prominencia.setRange(0.0, 1_000_000_000.0)
        campo_prominencia.setSpecialValueText("Automático")
        campo_prominencia.setValue(float(self.MIN_PROMINENCIA_PICO))

        campo_largura = QSpinBox(dialogo)
        campo_largura.setRange(1, 500)
        campo_largura.setValue(int(self.LARGURA_MINIMA_PICO))

        campo_ruido = QDoubleSpinBox(dialogo)
        campo_ruido.setDecimals(2)
        campo_ruido.setRange(0.5, 20.0)
        campo_ruido.setSingleStep(0.25)
        campo_ruido.setValue(float(self.FATOR_RUIDO))

        campo_altura_relativa = QDoubleSpinBox(dialogo)
        campo_altura_relativa.setDecimals(3)
        campo_altura_relativa.setRange(0.500, 0.995)
        campo_altura_relativa.setSingleStep(0.01)
        campo_altura_relativa.setValue(float(self.ALTURA_RELATIVA_PICO))

        campo_curvatura = QDoubleSpinBox(dialogo)
        campo_curvatura.setDecimals(2)
        campo_curvatura.setRange(0.10, 10.00)
        campo_curvatura.setSingleStep(0.10)
        campo_curvatura.setValue(float(self.FATOR_CURVATURA))

        campo_modelo = QComboBox(dialogo)
        campo_modelo.addItems(self.MODELOS_AJUSTE_PICO)
        campo_modelo.setCurrentText(self.MODELO_AJUSTE_PICO)

        formulario.addRow("Altura mínima:", campo_altura)
        formulario.addRow("Janela Savitzky-Golay:", campo_janela)
        formulario.addRow("Divisor distância mínima:", campo_divisor)
        formulario.addRow("Proeminência mínima:", campo_prominencia)
        formulario.addRow("Largura mínima:", campo_largura)
        formulario.addRow("Fator de ruído:", campo_ruido)
        formulario.addRow("Altura relativa:", campo_altura_relativa)
        formulario.addRow("Fator de curvatura:", campo_curvatura)
        formulario.addRow("Ajuste de pico:", campo_modelo)
        layout.addLayout(formulario)

        aviso = QLabel(
            "Altura relativa e curvatura ajustam os limites integrados do pico. "
            "Gaussiano/EMG são apenas sobreposições interpretativas no gráfico."
        )
        aviso.setWordWrap(True)
        aviso.setStyleSheet("color: #555;")
        layout.addWidget(aviso)

        botoes = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dialogo,
        )
        botao_resetar = botoes.addButton("Resetar opções", QDialogButtonBox.ResetRole)

        def resetar_campos() -> None:
            campo_altura.setValue(float(constants.ALTURA_MINIMA_PICO))
            campo_janela.setValue(int(constants.JANELA_SUAVIZACAO))
            campo_divisor.setValue(int(constants.DIVISOR_DISTANCIA_MINIMA))
            campo_prominencia.setValue(float(constants.MIN_PROMINENCIA_PICO))
            campo_largura.setValue(int(constants.LARGURA_MINIMA_PICO))
            campo_ruido.setValue(float(constants.FATOR_RUIDO))
            campo_altura_relativa.setValue(float(constants.ALTURA_RELATIVA_PICO))
            campo_curvatura.setValue(float(constants.FATOR_CURVATURA))
            campo_modelo.setCurrentText(constants.MODELO_AJUSTE_PICO)

        self._conectar_botao_com_aviso_espera(botao_resetar, resetar_campos)
        botoes.accepted.connect(dialogo.accept)
        botoes.rejected.connect(dialogo.reject)
        layout.addWidget(botoes)

        if dialogo.exec() != QDialog.Accepted:
            return

        self.ALTURA_MINIMA_PICO = float(campo_altura.value())
        self.JANELA_SUAVIZACAO = int(campo_janela.value())
        if self.JANELA_SUAVIZACAO % 2 == 0:
            self.JANELA_SUAVIZACAO += 1
        self.DIVISOR_DISTANCIA_MINIMA = int(campo_divisor.value())
        self.MIN_PROMINENCIA_PICO = float(campo_prominencia.value())
        self.LARGURA_MINIMA_PICO = int(campo_largura.value())
        self.FATOR_RUIDO = float(campo_ruido.value())
        self.ALTURA_RELATIVA_PICO = float(campo_altura_relativa.value())
        self.FATOR_CURVATURA = float(campo_curvatura.value())
        self.MODELO_AJUSTE_PICO = str(campo_modelo.currentText())

        self.settings.setValue("options/altura_minima_pico", self.ALTURA_MINIMA_PICO)
        self.settings.setValue("options/janela_suavizacao", self.JANELA_SUAVIZACAO)
        self.settings.setValue(
            "options/divisor_distancia_minima", self.DIVISOR_DISTANCIA_MINIMA
        )
        self.settings.setValue(
            "options/min_prominencia_pico", self.MIN_PROMINENCIA_PICO
        )
        self.settings.setValue("options/largura_minima_pico", self.LARGURA_MINIMA_PICO)
        self.settings.setValue("options/fator_ruido", self.FATOR_RUIDO)
        self.settings.setValue(
            "options/altura_relativa_pico", self.ALTURA_RELATIVA_PICO
        )
        self.settings.setValue("options/fator_curvatura", self.FATOR_CURVATURA)
        self.settings.setValue("options/modelo_ajuste_pico", self.MODELO_AJUSTE_PICO)
        self.settings.sync()
        self._reprocessar_amostras_carregadas(preservar_visualizacao=True)
        self.statusBar().showMessage("Parâmetros atualizados em Opções.")

    def _criar_interface(self) -> None:
        widget_central = QWidget()
        self.setCentralWidget(widget_central)

        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(10, 10, 10, 10)

        splitter_horizontal = QSplitter(Qt.Horizontal)
        layout_principal.addWidget(splitter_horizontal)

        painel_lateral = QFrame()
        painel_lateral.setFrameShape(QFrame.StyledPanel)
        painel_lateral.setMinimumWidth(300)
        painel_lateral.setMaximumWidth(300)
        layout_lateral = QVBoxLayout(painel_lateral)

        titulo_lista = QLabel("ARQUIVOS CARREGADOS")
        titulo_lista.setStyleSheet("font-size: 12px; font-weight: 600;")
        layout_lateral.addWidget(titulo_lista)

        self.lista_arquivos = QListWidget()
        self.lista_arquivos.setFont(QFont("Arial", 8))
        self.lista_arquivos.currentItemChanged.connect(self._ao_selecionar_amostra)
        layout_lateral.addWidget(self.lista_arquivos, 1)

        self.label_resumo = QLabel("NENHUM ARQUIVO CARREGADO.")
        self.label_resumo.setWordWrap(True)
        self.label_resumo.setStyleSheet("color: #555;")
        layout_lateral.addWidget(self.label_resumo)

        botao_carregar = QPushButton("ADICIONAR ARQUIVOS")
        self._conectar_botao_com_aviso_espera(botao_carregar, self.abrir_arquivos)
        layout_lateral.addWidget(botao_carregar)

        splitter_horizontal.addWidget(painel_lateral)

        painel_direito = QWidget()
        layout_direito = QVBoxLayout(painel_direito)
        layout_direito.setContentsMargins(0, 0, 0, 0)

        splitter_vertical = QSplitter(Qt.Vertical)
        layout_direito.addWidget(splitter_vertical)

        frame_grafico = QFrame()
        frame_grafico.setFrameShape(QFrame.StyledPanel)
        layout_grafico = QVBoxLayout(frame_grafico)
        layout_grafico.addWidget(QLabel("CROMATOGRAMA DA AMOSTRA SELECIONADA"))

        self.canvas = CromatogramaCanvas()
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect("scroll_event", self._ao_rolar_mouse_no_grafico)
        self.canvas.mpl_connect("motion_notify_event", self._ao_mover_mouse_no_grafico)
        self.canvas.mpl_connect("figure_leave_event", self._ao_sair_do_grafico)
        self.canvas.mpl_connect("button_press_event", self._ao_clicar_no_grafico)

        self.toolbar_grafico = NavigationToolbar(self.canvas, self)
        layout_grafico.addWidget(self.toolbar_grafico)

        barra_visibilidade = QHBoxLayout()
        barra_visibilidade.addWidget(QLabel("Mostrar:"))

        self.check_mostrar_amostra = QCheckBox("Amostra")
        self.check_mostrar_amostra.setChecked(self.mostrar_amostra)
        barra_visibilidade.addWidget(self.check_mostrar_amostra)

        self.check_mostrar_padrao = QCheckBox("Padrão")
        self.check_mostrar_padrao.setChecked(self.mostrar_padrao)
        barra_visibilidade.addWidget(self.check_mostrar_padrao)

        self.check_mostrar_picos = QCheckBox("Picos")
        self.check_mostrar_picos.setChecked(self.mostrar_picos)
        barra_visibilidade.addWidget(self.check_mostrar_picos)

        self.check_mostrar_grupos = QCheckBox("Grupos")
        self.check_mostrar_grupos.setChecked(self.mostrar_grupos)
        barra_visibilidade.addWidget(self.check_mostrar_grupos)

        self.check_mostrar_ajuste = QCheckBox("Ajuste")
        self.check_mostrar_ajuste.setChecked(self.mostrar_ajuste)
        barra_visibilidade.addWidget(self.check_mostrar_ajuste)
        barra_visibilidade.addStretch(1)

        for controle in (
            self.check_mostrar_amostra,
            self.check_mostrar_padrao,
            self.check_mostrar_picos,
            self.check_mostrar_grupos,
            self.check_mostrar_ajuste,
        ):
            controle.toggled.connect(self._ao_alterar_visibilidade_grafico)

        layout_grafico.addLayout(barra_visibilidade)
        layout_grafico.addWidget(
            QLabel(
                "Rolagem: zoom no eixo X | Shift + rolagem: zoom no eixo Y | Botão direito: resetar zoom"
            )
        )
        layout_grafico.addWidget(self.canvas)

        frame_tabela = QFrame()
        frame_tabela.setFrameShape(QFrame.StyledPanel)
        layout_tabela = QVBoxLayout(frame_tabela)
        layout_tabela.addWidget(QLabel("INTEGRAÇÃO E PICOS"))

        barra_acoes_tabela = QHBoxLayout()
        botao_adicionar_pico = QPushButton("ADICIONAR PICO")
        self._conectar_botao_com_aviso_espera(
            botao_adicionar_pico,
            self._adicionar_pico_manual,
        )
        barra_acoes_tabela.addWidget(botao_adicionar_pico)

        botao_remover_pico = QPushButton("REMOVER PICO")
        self._conectar_botao_com_aviso_espera(
            botao_remover_pico,
            self._remover_pico_selecionado,
        )
        barra_acoes_tabela.addWidget(botao_remover_pico)

        botao_remover_todos = QPushButton("REMOVER TODOS")
        self._conectar_botao_com_aviso_espera(
            botao_remover_todos,
            self._remover_todos_picos_amostra,
        )
        barra_acoes_tabela.addWidget(botao_remover_todos)

        barra_acoes_tabela.addStretch(1)
        layout_tabela.addLayout(barra_acoes_tabela)

        self.abas_tabelas = QTabWidget()

        self.tabela_padrao = QTableWidget()
        configurar_tabela(self.tabela_padrao, self.COLUNAS_PADRAO_EXIBICAO)
        self.tabela_padrao.itemChanged.connect(self._ao_editar_tabela_padrao)
        self.abas_tabelas.addTab(self.tabela_padrao, "PADRÃO")

        self.tabela_amostras = QTableWidget()
        configurar_tabela(self.tabela_amostras, self.COLUNAS_AMOSTRA_EXIBICAO)
        self.tabela_amostras.itemChanged.connect(self._ao_editar_tabela_amostras)
        self.tabela_amostras.cellClicked.connect(self._ao_clicar_linha_tabela_amostras)
        self.tabela_amostras.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabela_amostras.customContextMenuRequested.connect(
            self._abrir_menu_contexto_tabela_amostras
        )
        self.abas_tabelas.addTab(self.tabela_amostras, "AMOSTRAS")

        self.tabela_regioes = QTableWidget()
        configurar_tabela(self.tabela_regioes, self.COLUNAS_REGIOES_EXIBICAO)
        self.tabela_regioes.setToolTip(
            "Clique com o botão direito em uma linha para vincular o grupo a um pico da amostra e zerar/restaurar a área."
        )
        self.tabela_regioes.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabela_regioes.customContextMenuRequested.connect(
            self._abrir_menu_contexto_tabela_regioes
        )
        self.abas_tabelas.addTab(self.tabela_regioes, "GRUPOS")

        layout_tabela.addWidget(self.abas_tabelas)

        splitter_vertical.addWidget(frame_grafico)
        splitter_vertical.addWidget(frame_tabela)
        splitter_vertical.setSizes([420, 300])

        splitter_horizontal.addWidget(painel_direito)
        splitter_horizontal.setSizes([280, 980])

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Pronto.")

    def abrir_arquivos(self) -> None:
        """Abre o seletor de arquivos e adiciona novas amostras à interface."""
        ultima_pasta = str(self.settings.value("ui/ultima_pasta", ""))
        caminhos, _ = QFileDialog.getOpenFileNames(
            self,
            "Selecionar cromatogramas",
            ultima_pasta,
            self.FILTRO_ARQUIVOS,
        )

        if not caminhos:
            return

        caminhos_path = [Path(caminho) for caminho in caminhos]
        self.settings.setValue("ui/ultima_pasta", str(caminhos_path[0].parent))

        metadados_lote = self._perguntar_metadados_amostra(
            caminhos_path[0],
            str(caminhos_path[0].resolve()),
            quantidade_arquivos=len(caminhos_path),
        )
        if metadados_lote is None:
            return

        adicionados = 0
        erros: list[str] = []
        for caminho in caminhos_path:
            try:
                resultado = self._carregar_amostra(
                    caminho,
                    perguntar_metadados=False,
                    metadados=metadados_lote,
                )
                if resultado:
                    adicionados += 1
            except Exception as exc:
                erros.append(f"- {caminho.name}: {exc}")

        if self.lista_arquivos.count() and self.lista_arquivos.currentRow() < 0:
            self.lista_arquivos.setCurrentRow(0)

        self.statusBar().showMessage(f"{adicionados} arquivo(s) carregado(s).")
        if erros:
            QMessageBox.warning(
                self, "Falha ao carregar alguns arquivos", "\n".join(erros)
            )

    def limpar_arquivos(self) -> None:
        """Remove todas as amostras carregadas e restaura o estado inicial."""
        self.amostras.clear()
        self.lista_arquivos.clear()
        self._mostrar_placeholder()
        self.statusBar().showMessage("Lista de arquivos limpa.")

    def salvar_dados(self) -> None:
        """Salva a tabela de grupos de cada amostra em '<nome>_baldur.dat' na mesma pasta."""
        if not self.amostras:
            QMessageBox.information(self, "Salvar", "Nenhuma amostra carregada.")
            return

        salvos = 0
        erros: list[str] = []
        for chave, amostra in self.amostras.items():
            try:
                metodo = self._obter_metodo_amostra(amostra)
                regioes = self._obter_regioes_analiticas(
                    metodo,
                    amostra.eh_diesel,
                    picos=amostra.picos,
                    dados=amostra.dados,
                )
                resumo = self.controller.calculate_region_areas(
                    amostra.picos,
                    regioes,
                    self.COLUNAS_REGIOES,
                    dados=amostra.dados,
                )

                vinculos = self._carregar_vinculos_grupos_amostra(chave)
                if vinculos and not amostra.picos.empty and not resumo.empty:
                    for indice, linha in resumo.iterrows():
                        grupo = str(linha.get("Grupos", "")).strip()
                        pico_vinculado = self._resolver_pico_vinculado(
                            amostra.picos, vinculos.get(grupo)
                        )
                        if pico_vinculado is None:
                            continue
                        for campo, coluna in [
                            ("Início", "Início (min)"),
                            ("Fim", "Fim (min)"),
                            ("Área", "Área Total"),
                        ]:
                            valor = pd.to_numeric(
                                pd.Series([pico_vinculado.get(campo)]), errors="coerce"
                            ).iloc[0]
                            if pd.notna(valor):
                                resumo.at[indice, coluna] = float(valor)

                grupos_zerados = self._carregar_grupos_zerados_amostra(chave)
                if grupos_zerados:
                    for indice, linha in resumo.iterrows():
                        if str(linha.get("Grupos", "")).strip() in grupos_zerados:
                            resumo.at[indice, "Área Total"] = 0.0

                df = self._adicionar_percentual_area(resumo, "Área Total")
                caminho_saida = amostra.caminho.parent / (
                    amostra.caminho.stem + "_baldur.dat"
                )
                df[self.COLUNAS_REGIOES_EXIBICAO].to_csv(
                    caminho_saida, sep="\t", index=False, float_format="%.4f"
                )
                salvos += 1
            except Exception as exc:
                erros.append(f"- {amostra.caminho.name}: {exc}")

        mensagem = f"{salvos} arquivo(s) salvo(s)."
        if erros:
            mensagem += "\n\nErros:\n" + "\n".join(erros)
            QMessageBox.warning(self, "Salvar", mensagem)
        else:
            QMessageBox.information(self, "Salvar", mensagem)

    def carregar_padrao(self) -> None:
        """Carrega o cromatograma de referência utilizado na comparação."""
        ultima_pasta = str(self.settings.value("ui/ultima_pasta", ""))
        caminho, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar cromatograma do padrão",
            ultima_pasta,
            self.FILTRO_ARQUIVOS,
        )

        if not caminho:
            return

        try:
            self._carregar_padrao_de_arquivo(Path(caminho))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Falha ao carregar padrão",
                f"Não foi possível carregar o padrão.\n\n{exc}",
            )
            return

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(item_atual)
        else:
            self._mostrar_placeholder()

    def _carregar_padrao_de_arquivo(self, caminho: Path) -> None:
        dados = self.controller.load_chromatogram(caminho)
        picos_padrao = self._detectar_picos_padrao(dados)
        picos_plot = (
            picos_padrao[self.COLUNAS_AMOSTRA].copy()
            if not picos_padrao.empty
            else self._criar_dataframe_picos_vazio()
        )

        self.padrao = Amostra(caminho=caminho, dados=dados, picos=picos_plot)
        self._atualizar_referencias_com_picos_do_padrao(picos_padrao)

        self.settings.setValue("ui/ultima_pasta", str(caminho.parent))
        self.settings.setValue("padrao/arquivo", str(caminho))
        self.settings.sync()
        self.statusBar().showMessage(
            f"Padrão carregado e tempos atualizados: {caminho.name}"
        )

    def _restaurar_padrao_salvo(self) -> None:
        caminho_salvo = str(self.settings.value("padrao/arquivo", "")).strip()
        if not caminho_salvo:
            return

        caminho = Path(caminho_salvo)
        if not caminho.exists():
            return

        try:
            self._carregar_padrao_de_arquivo(caminho)
        except Exception:
            self.padrao = None

    def _atualizar_referencias_com_picos_do_padrao(
        self, picos_padrao: pd.DataFrame
    ) -> None:
        self.referencias_padrao = self._criar_referencias_padrao_iniciais()
        self.referencias_padrao["Área"] = 0.0
        self.referencias_padrao[self.COLUNA_PERCENTUAL_AREA] = 0.0

        if picos_padrao.empty or "Composto" not in picos_padrao.columns:
            self.controller.save_standard_references(self.referencias_padrao)
            self._atualizar_tabela_padrao()
            return

        areas_padrao = pd.to_numeric(picos_padrao.get("Área"), errors="coerce").fillna(
            0.0
        )
        total_area_padrao = float(areas_padrao.sum())

        for _, pico in picos_padrao.iterrows():
            composto = str(pico.get("Composto", "")).strip()
            if not composto:
                continue

            mascara = self.referencias_padrao["Composto"] == composto
            if not mascara.any():
                continue

            self.referencias_padrao.loc[mascara, "Tempo (min)"] = float(
                pico["Tempo (min)"]
            )
            self.referencias_padrao.loc[mascara, "Início"] = float(pico["Início"])
            self.referencias_padrao.loc[mascara, "Fim"] = float(pico["Fim"])
            area_valor = pd.to_numeric(pico.get("Área"), errors="coerce")
            area_pico = float(area_valor) if pd.notna(area_valor) else 0.0
            self.referencias_padrao.loc[mascara, "Área"] = area_pico
            self.referencias_padrao.loc[mascara, self.COLUNA_PERCENTUAL_AREA] = (
                area_pico / total_area_padrao * 100.0 if total_area_padrao > 0 else 0.0
            )

        self.controller.save_standard_references(self.referencias_padrao)
        self._atualizar_tabela_padrao()

    def _carregar_amostra(
        self,
        caminho: Path,
        *,
        perguntar_metadados: bool = True,
        metadados: tuple[str, bool] | None = None,
    ) -> bool | None:
        chave = str(caminho.resolve())
        if chave in self.amostras:
            return False

        dados = self.controller.load_chromatogram(caminho)
        picos = self._detectar_picos_amostra(dados, chave)

        if metadados is not None:
            metodo_analise, eh_diesel = metadados
        elif perguntar_metadados:
            metadados = self._perguntar_metadados_amostra(caminho, chave)
            if metadados is None:
                return None
            metodo_analise, eh_diesel = metadados
        else:
            metodo_analise = self._carregar_metodo_amostra(chave)
            eh_diesel = self._carregar_diesel_amostra(chave, caminho)

        amostra = Amostra(
            caminho=caminho,
            dados=dados,
            picos=picos,
            metodo_analise=metodo_analise,
            eh_diesel=eh_diesel,
        )
        self.amostras[chave] = amostra
        self._salvar_metadados_amostra(chave, amostra)
        self._limpar_vinculos_grupos_invalidos(chave, amostra.picos)

        item = QListWidgetItem()
        item.setToolTip(str(caminho))
        item.setData(Qt.UserRole, chave)
        self.lista_arquivos.addItem(item)
        self.lista_arquivos.setItemWidget(
            item, self._criar_widget_item_amostra(item, amostra)
        )
        self.label_resumo.setText(
            f"{self.lista_arquivos.count()} arquivo(s) carregado(s)."
        )
        return True

    def _ao_selecionar_amostra(
        self,
        item_atual: QListWidgetItem | None,
        _item_anterior: QListWidgetItem | None = None,
        *,
        preservar_visualizacao: bool = False,
    ) -> None:
        if item_atual is None:
            self._mostrar_placeholder()
            return

        chave = item_atual.data(Qt.UserRole)
        amostra = self.amostras.get(chave)
        if amostra is None:
            self._mostrar_placeholder()
            return

        if self._chave_amostra_selecao_manual not in {None, chave}:
            self._cancelar_selecao_manual(redesenhar=False)

        self._atualizar_grafico(
            amostra,
            preservar_visualizacao=preservar_visualizacao,
        )
        metodo_analise = self._obter_metodo_amostra(amostra)

        self._atualizar_tabela(amostra.picos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            metodo_analise,
            amostra.eh_diesel,
        )
        self.abas_tabelas.setCurrentWidget(self.tabela_amostras)

        tipo_amostra = self._obter_tipo_amostra(amostra)
        resumo_padrao = (
            f"Padrão: {self.padrao.caminho.name}"
            if self.padrao is not None
            else "Padrão: não carregado"
        )
        self.label_resumo.setText(
            f"Amostra: {amostra.caminho.name}\n"
            f"Método: {metodo_analise} | {tipo_amostra}\n"
            f"{resumo_padrao}"
        )
        self.statusBar().showMessage(
            f"Visualizando: {amostra.caminho.name} | {metodo_analise} | {tipo_amostra}"
        )

    def _atualizar_grafico(
        self,
        amostra: Amostra,
        preservar_visualizacao: bool = False,
    ) -> None:
        limites_atuais = (
            self._capturar_limites_visuais() if preservar_visualizacao else None
        )

        self.canvas.ax.clear()
        self._limpar_guias_selecao_manual()
        self.canvas.ax.set_axis_on()

        mostrar_amostra = self._visibilidade_habilitada(
            "check_mostrar_amostra", self.mostrar_amostra
        )
        mostrar_padrao = self._visibilidade_habilitada(
            "check_mostrar_padrao", self.mostrar_padrao
        )
        mostrar_picos = self._visibilidade_habilitada(
            "check_mostrar_picos", self.mostrar_picos
        )
        mostrar_grupos = self._visibilidade_habilitada(
            "check_mostrar_grupos", self.mostrar_grupos
        )
        mostrar_ajuste = self._visibilidade_habilitada(
            "check_mostrar_ajuste", self.mostrar_ajuste
        )

        metodo_analise = self._obter_metodo_amostra(amostra)

        if mostrar_grupos:
            dados_grupos = None
            picos_grupos = None
            chave_grupos = None
            if mostrar_amostra:
                dados_grupos = amostra.dados
                picos_grupos = amostra.picos
                chave_grupos, _ = self._obter_amostra_atual()
            elif mostrar_padrao and self.padrao is not None:
                dados_grupos = self.padrao.dados
            self._desenhar_faixas_grupos(
                dados_grupos,
                metodo_analise,
                amostra.eh_diesel,
                picos=picos_grupos,
                chave_amostra=chave_grupos,
            )

        if mostrar_amostra:
            self.canvas.ax.plot(
                amostra.dados["Tempo"],
                amostra.dados["Intensidade"],
                color="#1f77b4",
                linewidth=1.2,
                label="Amostra",
            )

        if mostrar_padrao and self.padrao is not None:
            self.canvas.ax.plot(
                self.padrao.dados["Tempo"],
                self.padrao.dados["Intensidade"],
                color="#ff7f0e",
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
                label=f"Padrão: {self.padrao.caminho.name}",
            )
            self._desenhar_guias_padrao()

        if mostrar_picos and not amostra.picos.empty:
            cores_picos = ["#90CAF9", "#81C784"]
            for indice, (_, pico) in enumerate(amostra.picos.iterrows()):
                inicio = float(pico["Início"])
                fim = float(pico["Fim"])
                tempo_pico = float(pico["Tempo (min)"])
                altura_pico = float(pico["Altura"])
                numero_pico = str(int(float(pico["Pico"])))
                cor_faixa = cores_picos[indice % len(cores_picos)]

                self.canvas.ax.axvspan(inicio, fim, color=cor_faixa, alpha=0.18)
                self.canvas.ax.annotate(
                    numero_pico,
                    xy=(tempo_pico, altura_pico),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white",
                    bbox={
                        "boxstyle": "circle,pad=0.25",
                        "fc": "#d62728",
                        "ec": "white",
                        "lw": 0.8,
                    },
                    zorder=4,
                )

        if mostrar_ajuste:
            self._desenhar_ajustes_modelo(amostra)

        self.canvas.ax.set_title(amostra.caminho.name)
        self.canvas.ax.set_xlabel("Tempo (min)")
        self.canvas.ax.set_ylabel("Intensidade")
        self.canvas.ax.grid(alpha=0.25)
        self.canvas.figure.tight_layout(pad=0.5)

        handles, labels = self.canvas.ax.get_legend_handles_labels()
        if handles:
            self.canvas.ax.legend(loc="upper right", fontsize=8)

        if limites_atuais is None:
            if mostrar_amostra or (mostrar_padrao and self.padrao is not None):
                self._aplicar_limites_y_fixos()
            else:
                tempo_min = float(amostra.dados["Tempo"].min())
                tempo_max = float(amostra.dados["Tempo"].max())
                y_max = max(float(amostra.dados["Intensidade"].max()), 1.0)
                self.canvas.ax.set_xlim(tempo_min, tempo_max)
                self.canvas.ax.set_ylim(self._calcular_y_min_fixo(y_max), y_max)
            self._guardar_limites_originais()
        else:
            (x_min, x_max), (y_min, y_max) = limites_atuais
            self.canvas.ax.set_xlim(x_min, x_max)
            self.canvas.ax.set_ylim(y_min, y_max)

        self._atualizar_guias_selecao_manual()
        self.canvas.draw_idle()

    def _capturar_limites_visuais(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if not self.canvas.ax.has_data():
            return None
        return tuple(self.canvas.ax.get_xlim()), tuple(self.canvas.ax.get_ylim())

    def _desenhar_ajustes_modelo(self, amostra: Amostra) -> None:
        if amostra.picos.empty or self.MODELO_AJUSTE_PICO == "Nenhum":
            return

        ajustes = self.controller.fit_peak_models(
            amostra.dados,
            amostra.picos,
            self.MODELO_AJUSTE_PICO,
        )
        for indice, ajuste in enumerate(ajustes):
            self.canvas.ax.plot(
                ajuste["x_fit"],
                ajuste["y_fit"],
                color="#2ca02c",
                linewidth=1.15,
                alpha=0.95,
                label=self.MODELO_AJUSTE_PICO if indice == 0 else None,
            )

    def _obter_intervalos_plot_grupos(
        self,
        metodo: str | None = None,
        eh_diesel: bool = False,
        picos: pd.DataFrame | None = None,
        dados: pd.DataFrame | None = None,
        chave_amostra: str | None = None,
    ) -> list[tuple[str, float, float]]:
        intervalos = [
            (nome, float(inicio), float(fim))
            for nome, inicio, fim in self._obter_regioes_analiticas(
                metodo,
                eh_diesel,
                picos=picos,
                dados=dados,
            )
        ]
        if chave_amostra is None or picos is None or picos.empty:
            return intervalos

        vinculos = self._carregar_vinculos_grupos_amostra(chave_amostra)
        if not vinculos:
            return intervalos

        intervalos_atualizados: list[tuple[str, float, float]] = []
        for nome, inicio, fim in intervalos:
            pico_vinculado = self._resolver_pico_vinculado(picos, vinculos.get(nome))
            if pico_vinculado is None:
                intervalos_atualizados.append((nome, inicio, fim))
                continue

            inicio_vinculado = pd.to_numeric(
                pd.Series([pico_vinculado.get("Início")]), errors="coerce"
            ).iloc[0]
            fim_vinculado = pd.to_numeric(
                pd.Series([pico_vinculado.get("Fim")]), errors="coerce"
            ).iloc[0]
            if (
                pd.notna(inicio_vinculado)
                and pd.notna(fim_vinculado)
                and float(fim_vinculado) >= float(inicio_vinculado)
            ):
                intervalos_atualizados.append(
                    (nome, float(inicio_vinculado), float(fim_vinculado))
                )
            else:
                intervalos_atualizados.append((nome, inicio, fim))

        return intervalos_atualizados

    def _desenhar_faixas_grupos(
        self,
        dados: pd.DataFrame | None = None,
        metodo: str | None = None,
        eh_diesel: bool = False,
        *,
        picos: pd.DataFrame | None = None,
        chave_amostra: str | None = None,
    ) -> None:
        if dados is None or dados.empty:
            return

        tempo = dados["Tempo"].astype(float)
        intensidade = dados["Intensidade"].astype(float).clip(lower=0.0)
        intervalos_grupos = self._obter_intervalos_plot_grupos(
            metodo,
            eh_diesel,
            picos=picos,
            dados=dados,
            chave_amostra=chave_amostra,
        )

        for nome, inicio, fim in intervalos_grupos:
            cor = self.CORES_REGIOES.get(nome, "#BDBDBD")
            mascara = (tempo >= inicio) & (tempo <= fim)
            if not mascara.any():
                continue

            self.canvas.ax.fill_between(
                tempo,
                0.0,
                intensidade,
                where=mascara,
                interpolate=True,
                color=cor,
                alpha=0.22,
                zorder=0,
            )

            altura_rotulo = float(intensidade[mascara].max())
            self.canvas.ax.text(
                (inicio + fim) / 2,
                max(altura_rotulo * 0.92, 0.0),
                nome,
                fontsize=7,
                ha="center",
                va="top",
                color=cor,
                bbox={"fc": "white", "ec": "none", "alpha": 0.60, "pad": 0.2},
                clip_on=True,
                zorder=2,
            )

    def _desenhar_guias_padrao(self) -> None:
        if self.referencias_padrao.empty:
            return

        transform = self.canvas.ax.get_xaxis_transform()
        for _, row in self.referencias_padrao.iterrows():
            tempo_ref = float(row["Tempo (min)"])
            inicio = float(row["Início"])
            fim = float(row["Fim"])
            composto = str(row["Composto"])

            self.canvas.ax.axvspan(inicio, fim, color="#ff7f0e", alpha=0.05)
            self.canvas.ax.axvline(
                tempo_ref,
                color="#ff7f0e",
                linestyle=":",
                linewidth=1.0,
                alpha=0.8,
            )
            self.canvas.ax.text(
                tempo_ref,
                0.98,
                composto,
                transform=transform,
                rotation=90,
                fontsize=8,
                ha="center",
                va="top",
                color="#7a4a00",
                clip_on=True,
            )

    def _ao_rolar_mouse_no_grafico(self, event) -> None:
        if event.inaxes != self.canvas.ax:
            return

        if getattr(event, "button", None) not in {"up", "down"}:
            return

        fator = 0.90 if event.button == "up" else 1 / 0.90
        tecla = str(getattr(event, "key", "") or "").lower()
        if "shift" in tecla:
            self._zoom_eixo_y(self.canvas.ax, fator)
        else:
            self._zoom_eixo_x(self.canvas.ax, event.xdata, fator)

        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _ao_clicar_no_grafico(self, event) -> None:
        if event.inaxes != self.canvas.ax:
            return

        botao = getattr(event, "button", None)
        if botao == 3:
            if self._tempo_primeiro_clique is not None:
                self._cancelar_selecao_manual(redesenhar=True)
                self.statusBar().showMessage("Seleção manual cancelada.")
            else:
                self._restaurar_zoom_original()
            return

        if botao != 1 or getattr(event, "xdata", None) is None:
            return
        if self._ferramenta_navegacao_ativa():
            return

        chave, amostra = self._obter_amostra_atual()
        if chave is None or amostra is None or amostra.dados.empty:
            return

        tempo_min = float(amostra.dados["Tempo"].min())
        tempo_max = float(amostra.dados["Tempo"].max())
        tempo_selecionado = min(max(float(event.xdata), tempo_min), tempo_max)
        self._tempo_cursor_grafico = tempo_selecionado

        if self._tempo_primeiro_clique is None:
            self._tempo_primeiro_clique = tempo_selecionado
            self._chave_amostra_selecao_manual = chave
            self._atualizar_guias_selecao_manual()
            self.canvas.draw_idle()
            self.statusBar().showMessage(
                f"Início do pico manual marcado em {tempo_selecionado:.4f} min. "
                "Mova o mouse e clique novamente para finalizar."
            )
            return

        if self._chave_amostra_selecao_manual not in {None, chave}:
            self._cancelar_selecao_manual(redesenhar=False)
            return

        inicio, fim = sorted([float(self._tempo_primeiro_clique), tempo_selecionado])
        if abs(fim - inicio) <= 1e-9:
            self.statusBar().showMessage(
                "Selecione um intervalo maior para integrar o pico manual."
            )
            return

        self._cancelar_selecao_manual(redesenhar=False)
        try:
            self._inserir_pico_manual(chave, amostra, inicio, fim)
            self.statusBar().showMessage(
                f"Pico manual integrado entre {inicio:.4f} e {fim:.4f} min."
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Adicionar pico", str(exc))

    def _guardar_limites_originais(self) -> None:
        self._limites_originais = (
            tuple(self.canvas.ax.get_xlim()),
            tuple(self.canvas.ax.get_ylim()),
        )

    @staticmethod
    def _calcular_y_min_fixo(y_max: float) -> float:
        return -0.05 * y_max

    def _restaurar_zoom_original(self) -> None:
        if self._limites_originais is None:
            return

        (x_min, x_max), (_, y_max) = self._limites_originais
        y_min = self._calcular_y_min_fixo(y_max)
        self.canvas.ax.set_xlim(x_min, x_max)
        self.canvas.ax.set_ylim(y_min, y_max)
        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _aplicar_limites_y_fixos(self, y_max: float | None = None) -> None:
        self.canvas.ax.relim()
        self.canvas.ax.autoscale_view(scalex=False, scaley=True)

        if y_max is None:
            _, y_max = self.canvas.ax.get_ylim()

        y_max = max(float(y_max), 1e-9)
        self.canvas.ax.set_ylim(self._calcular_y_min_fixo(y_max), y_max)

    def _zoom_eixo_x(self, eixo, x_centro: float | None, fator: float) -> None:
        eixo.set_autoscale_on(False)

        x_min, x_max = eixo.get_xlim()
        if x_centro is None:
            x_centro = (x_min + x_max) / 2

        novo_min = x_centro - (x_centro - x_min) * fator
        novo_max = x_centro + (x_max - x_centro) * fator

        limite_min, limite_max = self._obter_limites_x_dados()
        largura_total = limite_max - limite_min
        largura_nova = novo_max - novo_min
        largura_min = max(largura_total * 0.002, 1e-6)

        if largura_total > 0:
            if largura_nova <= largura_min:
                metade = largura_min / 2
                novo_min = x_centro - metade
                novo_max = x_centro + metade

            if largura_nova >= largura_total:
                novo_min, novo_max = limite_min, limite_max
            else:
                if novo_min < limite_min:
                    novo_max += limite_min - novo_min
                    novo_min = limite_min
                if novo_max > limite_max:
                    novo_min -= novo_max - limite_max
                    novo_max = limite_max

        eixo.set_xlim(novo_min, novo_max)

    def _zoom_eixo_y(self, eixo, fator: float) -> None:
        eixo.set_autoscale_on(False)
        _, y_max_atual = eixo.get_ylim()
        novo_y_max = max(float(y_max_atual) * fator, 1e-9)
        eixo.set_ylim(self._calcular_y_min_fixo(novo_y_max), novo_y_max)

    def _obter_limites_x_dados(self) -> tuple[float, float]:
        limites: list[tuple[float, float]] = []

        _, amostra_atual = self._obter_amostra_atual()
        if amostra_atual is not None and not amostra_atual.dados.empty:
            limites.append(
                (
                    float(amostra_atual.dados["Tempo"].min()),
                    float(amostra_atual.dados["Tempo"].max()),
                )
            )

        if self.padrao is not None and not self.padrao.dados.empty:
            limites.append(
                (
                    float(self.padrao.dados["Tempo"].min()),
                    float(self.padrao.dados["Tempo"].max()),
                )
            )

        if not limites:
            return self.canvas.ax.get_xlim()

        return min(limite[0] for limite in limites), max(
            limite[1] for limite in limites
        )

    def _reprocessar_amostras_carregadas(
        self,
        preservar_visualizacao: bool = True,
    ) -> None:
        for chave, amostra in self.amostras.items():
            if not amostra.picos.empty:
                amostra.picos = self._detectar_picos_amostra(amostra.dados, chave)
            self._limpar_vinculos_grupos_invalidos(chave, amostra.picos)

        if self.padrao is not None:
            picos_padrao = self._detectar_picos_padrao(self.padrao.dados)
            picos_plot = (
                picos_padrao[self.COLUNAS_AMOSTRA].copy()
                if not picos_padrao.empty
                else self._criar_dataframe_picos_vazio()
            )
            self.padrao = Amostra(
                caminho=self.padrao.caminho,
                dados=self.padrao.dados,
                picos=picos_plot,
            )
            self._atualizar_referencias_com_picos_do_padrao(picos_padrao)

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(
                item_atual,
                preservar_visualizacao=preservar_visualizacao,
            )
        else:
            self._mostrar_placeholder()

    def _atualizar_tabela(self, picos: pd.DataFrame) -> None:
        self._preencher_tabela(
            tabela=self.tabela_amostras,
            df=self._adicionar_percentual_area(picos, "Área"),
            colunas=self.COLUNAS_AMOSTRA_EXIBICAO,
            colunas_editaveis={"Tempo (min)", "Início", "Fim"},
            mensagem_vazia="Nenhum pico carregado. Use Opções > ENCONTRAR PICOS.",
        )

    def _adicionar_percentual_area(
        self,
        df: pd.DataFrame,
        coluna_area: str,
    ) -> pd.DataFrame:
        resultado = df.copy()
        resultado[self.COLUNA_PERCENTUAL_AREA] = 0.0

        if resultado.empty or coluna_area not in resultado.columns:
            return resultado

        areas = pd.to_numeric(resultado[coluna_area], errors="coerce").fillna(0.0)
        total_areas = float(areas.sum())
        if total_areas > 0:
            resultado[self.COLUNA_PERCENTUAL_AREA] = areas / total_areas * 100.0

        return resultado

    def _recalcular_areas_padrao_integradas(self) -> None:
        if self.referencias_padrao.empty:
            return

        if "Área" not in self.referencias_padrao.columns:
            self.referencias_padrao["Área"] = 0.0
        if self.COLUNA_PERCENTUAL_AREA not in self.referencias_padrao.columns:
            self.referencias_padrao[self.COLUNA_PERCENTUAL_AREA] = 0.0

        self.referencias_padrao["Área"] = 0.0
        self.referencias_padrao[self.COLUNA_PERCENTUAL_AREA] = 0.0

        if self.padrao is None or self.padrao.dados.empty:
            return

        areas_calculadas: list[float] = []
        for indice, linha in self.referencias_padrao.iterrows():
            inicio = pd.to_numeric(
                pd.Series([linha.get("Início")]), errors="coerce"
            ).iloc[0]
            fim = pd.to_numeric(pd.Series([linha.get("Fim")]), errors="coerce").iloc[0]

            if pd.isna(inicio) or pd.isna(fim):
                areas_calculadas.append(0.0)
                continue

            try:
                pico_integrado = self.controller.create_peak_by_interval(
                    self.padrao.dados,
                    float(inicio),
                    float(fim),
                )
            except ValueError:
                areas_calculadas.append(0.0)
                continue

            area = pd.to_numeric(
                pd.Series([pico_integrado.get("Área")]), errors="coerce"
            ).iloc[0]
            area_float = float(area) if pd.notna(area) else 0.0
            self.referencias_padrao.at[indice, "Área"] = area_float
            areas_calculadas.append(area_float)

        total_areas = float(sum(areas_calculadas))
        if total_areas > 0:
            self.referencias_padrao[self.COLUNA_PERCENTUAL_AREA] = (
                pd.Series(areas_calculadas, index=self.referencias_padrao.index)
                / total_areas
                * 100.0
            )

    def _calcular_areas_por_regiao(
        self,
        picos: pd.DataFrame,
        metodo: str | None = None,
        eh_diesel: bool = False,
    ) -> pd.DataFrame:
        _, amostra_atual = self._obter_amostra_atual()
        dados_amostra = amostra_atual.dados if amostra_atual is not None else None

        resumo = self.controller.calculate_region_areas(
            picos,
            self._obter_regioes_analiticas(
                metodo,
                eh_diesel,
                picos=picos,
                dados=dados_amostra,
            ),
            self.COLUNAS_REGIOES,
            dados=dados_amostra,
        )

        chave_atual, _ = self._obter_amostra_atual()
        if chave_atual is None or picos.empty or resumo.empty:
            return self._adicionar_percentual_area(resumo, "Área Total")

        vinculos = self._carregar_vinculos_grupos_amostra(chave_atual)
        if not vinculos:
            return self._adicionar_percentual_area(resumo, "Área Total")

        for indice, linha in resumo.iterrows():
            grupo = str(linha.get("Grupos", "")).strip()
            pico_vinculado = self._resolver_pico_vinculado(
                picos,
                vinculos.get(grupo),
            )
            if pico_vinculado is None:
                continue

            inicio = pd.to_numeric(
                pd.Series([pico_vinculado.get("Início")]), errors="coerce"
            ).iloc[0]
            fim = pd.to_numeric(
                pd.Series([pico_vinculado.get("Fim")]), errors="coerce"
            ).iloc[0]
            area = pd.to_numeric(
                pd.Series([pico_vinculado.get("Área")]), errors="coerce"
            ).iloc[0]

            if pd.notna(inicio):
                resumo.at[indice, "Início (min)"] = float(inicio)
            if pd.notna(fim):
                resumo.at[indice, "Fim (min)"] = float(fim)
            if pd.notna(area):
                resumo.at[indice, "Área Total"] = float(area)

        grupos_zerados = self._carregar_grupos_zerados_amostra(chave_atual)
        if grupos_zerados:
            for indice, linha in resumo.iterrows():
                grupo = str(linha.get("Grupos", "")).strip()
                if grupo in grupos_zerados:
                    resumo.at[indice, "Área Total"] = 0.0

        return self._adicionar_percentual_area(resumo, "Área Total")

    def _atualizar_tabela_regioes(
        self,
        picos: pd.DataFrame,
        metodo: str | None = None,
        eh_diesel: bool = False,
    ) -> None:
        self._preencher_tabela(
            tabela=self.tabela_regioes,
            df=self._calcular_areas_por_regiao(picos, metodo, eh_diesel),
            colunas=self.COLUNAS_REGIOES_EXIBICAO,
            colunas_editaveis=set(),
            mensagem_vazia="Nenhum grupo calculado",
        )
        self._aplicar_cores_tabela_regioes()

    def _aplicar_cores_tabela_regioes(self) -> None:
        for linha in range(self.tabela_regioes.rowCount()):
            item_grupo = self.tabela_regioes.item(linha, 0)
            if item_grupo is None:
                continue

            cor_hex = self.CORES_REGIOES.get(item_grupo.text().strip())
            if not cor_hex:
                continue

            cor = QColor(cor_hex)
            cor.setAlpha(55)
            for coluna in range(self.tabela_regioes.columnCount()):
                item = self.tabela_regioes.item(linha, coluna)
                if item is not None:
                    item.setBackground(cor)

    def _atualizar_tabela_padrao(self) -> None:
        self._recalcular_areas_padrao_integradas()
        self._preencher_tabela(
            tabela=self.tabela_padrao,
            df=self._adicionar_percentual_area(self.referencias_padrao, "Área"),
            colunas=self.COLUNAS_PADRAO_EXIBICAO,
            colunas_editaveis={"Tempo (min)", "Início", "Fim"},
            mensagem_vazia="Nenhuma referência disponível",
        )

    def _preencher_tabela(
        self,
        tabela: QTableWidget,
        df: pd.DataFrame,
        colunas: list[str],
        colunas_editaveis: set[str],
        mensagem_vazia: str,
    ) -> None:
        self._atualizando_tabelas = True
        tabela.blockSignals(True)
        try:
            preencher_tabela(
                tabela,
                df,
                colunas,
                colunas_editaveis,
                mensagem_vazia,
            )
        finally:
            tabela.blockSignals(False)
            self._atualizando_tabelas = False

    def _ao_editar_tabela_padrao(self, item: QTableWidgetItem) -> None:
        if self._atualizando_tabelas or item.column() not in {1, 2, 3}:
            return

        valor = texto_para_float(item.text())
        if valor is None:
            self.statusBar().showMessage("Valor inválido no padrão. Use número.")
            self._atualizar_tabela_padrao()
            return

        coluna = self.COLUNAS_PADRAO[item.column()]
        self.referencias_padrao.at[item.row(), coluna] = valor
        self._ajustar_intervalo_padrao(item.row())
        self._recalcular_areas_padrao_integradas()
        self.controller.save_standard_references(self.referencias_padrao)
        self._atualizar_tabela_padrao()

        item_atual = self.lista_arquivos.currentItem()
        if item_atual is not None:
            self._ao_selecionar_amostra(item_atual, preservar_visualizacao=True)
        else:
            self._mostrar_placeholder()

        self.statusBar().showMessage("Referência do padrão salva.")

    def _ajustar_intervalo_padrao(self, linha: int) -> None:
        tempo = float(self.referencias_padrao.at[linha, "Tempo (min)"])
        inicio = float(self.referencias_padrao.at[linha, "Início"])
        fim = float(self.referencias_padrao.at[linha, "Fim"])

        if inicio > fim:
            inicio, fim = fim, inicio
        if tempo < inicio:
            inicio = tempo
        if tempo > fim:
            fim = tempo

        self.referencias_padrao.at[linha, "Tempo (min)"] = tempo
        self.referencias_padrao.at[linha, "Início"] = inicio
        self.referencias_padrao.at[linha, "Fim"] = fim

    def _ao_editar_tabela_amostras(self, item: QTableWidgetItem) -> None:
        if self._atualizando_tabelas or item.column() not in {1, 3, 4}:
            return

        chave, amostra = self._obter_amostra_atual()
        if (
            chave is None
            or amostra is None
            or amostra.picos.empty
            or item.row() >= len(amostra.picos)
        ):
            return

        valor = texto_para_float(item.text())
        if valor is None:
            self.statusBar().showMessage("Valor inválido na amostra. Use número.")
            self._atualizar_tabela(amostra.picos)
            return

        coluna = self.COLUNAS_AMOSTRA[item.column()]
        indice = amostra.picos.index[item.row()]
        amostra.picos.at[indice, coluna] = valor

        atualizacao = self.controller.recalc_edited_peak(
            amostra.dados,
            amostra.picos.loc[indice],
        )
        for nome_coluna, valor_atualizado in atualizacao.items():
            amostra.picos.at[indice, nome_coluna] = valor_atualizado

        self.controller.save_sample_peaks(chave, amostra.picos)
        self._atualizar_tabela(amostra.picos)
        self._atualizar_tabela_regioes(
            amostra.picos,
            self._obter_metodo_amostra(amostra),
            amostra.eh_diesel,
        )
        self._atualizar_grafico(amostra, preservar_visualizacao=True)
        self.statusBar().showMessage("Edição da amostra salva.")

    def _mostrar_placeholder(self) -> None:
        self._cancelar_selecao_manual(redesenhar=False)
        self.canvas.ax.clear()

        mostrar_padrao = self._visibilidade_habilitada(
            "check_mostrar_padrao", self.mostrar_padrao
        )
        mostrar_grupos = self._visibilidade_habilitada(
            "check_mostrar_grupos", self.mostrar_grupos
        )

        _, amostra_atual = self._obter_amostra_atual()
        metodo_analise = self._obter_metodo_amostra(amostra_atual)
        eh_diesel = amostra_atual.eh_diesel if amostra_atual is not None else False

        if mostrar_grupos and self.padrao is not None and mostrar_padrao:
            self.canvas.ax.set_axis_on()
            self._desenhar_faixas_grupos(self.padrao.dados, metodo_analise, eh_diesel)

        if self.padrao is not None and mostrar_padrao:
            self.canvas.ax.set_axis_on()
            self.canvas.ax.plot(
                self.padrao.dados["Tempo"],
                self.padrao.dados["Intensidade"],
                color="#ff7f0e",
                linewidth=1.2,
                linestyle="--",
                label=f"Padrão: {self.padrao.caminho.name}",
            )
            self._desenhar_guias_padrao()
            self.canvas.ax.set_title("Padrão carregado")
            self.canvas.ax.set_xlabel("Tempo (min)")
            self.canvas.ax.set_ylabel("Intensidade")
            self.canvas.ax.grid(alpha=0.25)
            handles, labels = self.canvas.ax.get_legend_handles_labels()
            if handles:
                self.canvas.ax.legend(loc="upper right")
            self._aplicar_limites_y_fixos()
            self._guardar_limites_originais()
            self.label_resumo.setText(
                f"Padrão carregado: {self.padrao.caminho.name}\n"
                "Carregue uma amostra para comparar com o padrão."
            )
        else:
            self.canvas.ax.text(
                0.5,
                0.5,
                "Carregue um arquivo no menu Files para visualizar o cromatograma.",
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
            )
            if not mostrar_grupos:
                self.canvas.ax.set_axis_off()
            self.label_resumo.setText("Nenhum arquivo carregado.")

        self.canvas.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()
        vazio = self._criar_dataframe_picos_vazio()
        self._atualizar_tabela(vazio)
        self._atualizar_tabela_regioes(vazio)


def main() -> int:
    """Inicializa a aplicação Qt e exibe a janela principal."""
    app = QApplication.instance() or QApplication(sys.argv)

    janela = JanelaPrincipal()
    janela.show()
    janela.raise_()
    janela.activateWindow()

    return app.exec()
