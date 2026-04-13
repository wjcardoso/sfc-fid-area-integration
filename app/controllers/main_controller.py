"""Controller layer that orchestrates UI events and processing services."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QSettings

from app.services import processing_service


class MainController:
    """Thin controller used by the main window to avoid direct service coupling."""

    def __init__(self, settings: QSettings) -> None:
        self.settings = settings

    def load_standard_references(self, colunas_padrao: list[str]) -> pd.DataFrame:
        return processing_service.carregar_referencias_padrao(
            self.settings, colunas_padrao
        )

    def save_standard_references(self, referencias_padrao: pd.DataFrame) -> None:
        processing_service.salvar_referencias_padrao(self.settings, referencias_padrao)

    def sample_settings_key(self, chave: str) -> str:
        return processing_service.chave_settings_amostra(chave)

    def load_chromatogram(self, caminho) -> pd.DataFrame:
        return processing_service.ler_cromatograma(caminho)

    def create_empty_peaks(self, colunas_amostra: list[str]) -> pd.DataFrame:
        return processing_service.criar_dataframe_picos_vazio(colunas_amostra)

    def create_empty_standard(self, colunas_amostra: list[str]) -> pd.DataFrame:
        return processing_service.criar_dataframe_padrao_vazio(colunas_amostra)

    def default_standard_references(self, colunas_padrao: list[str]) -> pd.DataFrame:
        return processing_service.criar_referencias_padrao_iniciais(colunas_padrao)

    def detect_sample_peaks(
        self,
        dados: pd.DataFrame,
        colunas_amostra: list[str],
        altura_minima_pico: float,
        janela_suavizacao: int,
        divisor_distancia_minima: int,
        *,
        min_prominence: float | None,
        min_width: int,
        noise_factor: float,
        rel_height: float,
        curvature_factor: float,
    ) -> pd.DataFrame:
        return processing_service.detectar_picos_dataframe(
            dados,
            colunas_amostra,
            altura_minima_pico,
            janela_suavizacao,
            divisor_distancia_minima,
            min_prominence=min_prominence,
            min_width=min_width,
            noise_factor=noise_factor,
            rel_height=rel_height,
            curvature_factor=curvature_factor,
        )

    def detect_standard_peaks(
        self,
        dados: pd.DataFrame,
        referencias_base: pd.DataFrame,
        colunas_amostra: list[str],
        altura_minima_pico: float,
        janela_suavizacao: int,
        divisor_distancia_minima: int,
        *,
        min_prominence: float | None,
        min_width: int,
        noise_factor: float,
        rel_height: float,
        curvature_factor: float,
    ) -> pd.DataFrame:
        return processing_service.detectar_picos_padrao_referenciados(
            dados,
            referencias_base,
            colunas_amostra,
            altura_minima_pico,
            janela_suavizacao,
            divisor_distancia_minima,
            min_prominence=min_prominence,
            min_width=min_width,
            noise_factor=noise_factor,
            rel_height=rel_height,
            curvature_factor=curvature_factor,
        )

    def apply_saved_peaks(self, chave: str, picos: pd.DataFrame) -> pd.DataFrame:
        return processing_service.aplicar_picos_salvos(self.settings, chave, picos)

    def save_sample_peaks(self, chave: str, picos: pd.DataFrame) -> None:
        processing_service.salvar_picos_amostra(self.settings, chave, picos)

    def create_peak_by_interval(
        self,
        dados: pd.DataFrame,
        inicio: float,
        fim: float,
    ) -> dict[str, float]:
        return processing_service.criar_pico_por_intervalo(dados, inicio, fim)

    def recalc_edited_peak(
        self,
        dados: pd.DataFrame,
        registro: pd.Series,
    ) -> dict[str, float]:
        return processing_service.recalcular_pico_editado(dados, registro)

    def fit_peak_models(
        self,
        dados: pd.DataFrame,
        picos: pd.DataFrame,
        modelo: str,
    ) -> list[dict[str, object]]:
        return processing_service.ajustar_modelos_picos(dados, picos, modelo)

    def calculate_region_areas(
        self,
        picos: pd.DataFrame,
        regioes_analiticas: list[tuple[str, float, float]],
        colunas_regioes: list[str],
        dados: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return processing_service.calcular_areas_por_regiao(
            picos,
            regioes_analiticas,
            colunas_regioes,
            dados=dados,
        )
