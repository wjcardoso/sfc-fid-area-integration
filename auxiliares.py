"""Funções utilitárias para leitura e processamento de cromatogramas SFC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths, savgol_filter

TIME_STEP_MINUTES = 0.016667 / 10
Y_AXIS_HEADER = "Y Axis Multiplier:"


def _ensure_odd_window(window_length: int, signal_length: int) -> int:
    """Garante um tamanho de janela ímpar e compatível com Savitzky-Golay."""
    if signal_length <= 2:
        return max(signal_length, 1)

    window = max(3, int(window_length))
    if window % 2 == 0:
        window += 1

    if window > signal_length:
        window = signal_length if signal_length % 2 == 1 else signal_length - 1

    return max(window, 3)


@dataclass(slots=True)
class PeakDetectionConfig:
    """Parâmetros ajustáveis para a detecção de picos cromatográficos."""

    min_height: float
    min_distance: int
    smoothing_window: int = 7
    smoothing_polyorder: int = 3
    min_width: int = 3
    min_prominence: float | None = None
    rel_height: float = 0.98
    noise_factor: float = 4.0
    curvature_factor: float = 1.5

    @classmethod
    def from_legacy(
        cls,
        signal_length: int,
        min_height: float,
        smoothing_window: int,
        distance_divisor: int,
    ) -> "PeakDetectionConfig":
        """Converte os parâmetros legados em uma configuração mais robusta.

        Mantém a lógica original de altura mínima, suavização e distância mínima,
        porém evita que a separação entre picos cresça demais em cromatogramas
        longos, o que escondia ombros e picos parcialmente sobrepostos.
        """
        divisor = max(int(distance_divisor), 1)
        min_distance = max(4, signal_length // max(divisor * 8, 1))
        smoothing_window = _ensure_odd_window(max(smoothing_window, 5), signal_length)
        min_width = max(3, smoothing_window // 2)
        return cls(
            min_height=float(min_height),
            min_distance=min_distance,
            smoothing_window=smoothing_window,
            min_width=min_width,
        )


def ler_arquivo_dat(caminho: str) -> pd.DataFrame:
    """Lê um cromatograma textual e retorna as colunas `Tempo` e `Intensidade`."""
    linhas_intensidade: list[str] = []
    cabecalho_encontrado = False

    with open(caminho, encoding="utf-8", errors="replace") as arquivo:
        for linha in arquivo:
            conteudo = linha.rstrip("\n")

            if not cabecalho_encontrado:
                if Y_AXIS_HEADER in conteudo:
                    cabecalho_encontrado = True
                continue

            if conteudo.strip():
                linhas_intensidade.append(conteudo.strip())

    if not cabecalho_encontrado:
        raise ValueError("Cabeçalho 'Y Axis Multiplier' não encontrado no arquivo.")

    df = (
        pd.DataFrame(
            {"Intensidade": pd.to_numeric(linhas_intensidade, errors="coerce")}
        )
        .dropna(subset=["Intensidade"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Nenhum dado numérico válido encontrado no cromatograma.")

    df["Tempo"] = np.arange(1, len(df) + 1, dtype=float) * TIME_STEP_MINUTES
    return df


def corrigir_baseline_linear(
    tempo: np.ndarray | pd.Series,
    intensidade: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove uma linha de base linear estimada entre o início e o fim do sinal."""
    tempo_array = np.asarray(tempo, dtype=float)
    intensidade_array = np.asarray(intensidade, dtype=float)
    total_pontos = len(tempo_array)

    if total_pontos < 2:
        baseline = np.zeros(total_pontos, dtype=float)
        return np.maximum(intensidade_array, 0), baseline

    delta_tempo = tempo_array[-1] - tempo_array[0]
    if abs(delta_tempo) < 1e-12:
        baseline = np.full(
            total_pontos,
            float(np.nanmin(intensidade_array)),
            dtype=float,
        )
        return np.maximum(intensidade_array - baseline, 0), baseline

    intensidade_inicial = np.mean(intensidade_array[: min(3, total_pontos)])
    intensidade_final = np.mean(intensidade_array[max(0, total_pontos - 3) :])

    inclinacao = (intensidade_final - intensidade_inicial) / delta_tempo
    intercepto = intensidade_inicial - inclinacao * tempo_array[0]

    baseline = inclinacao * tempo_array + intercepto
    intensidade_corrigida = np.maximum(intensidade_array - baseline, 0)
    return intensidade_corrigida, baseline


def suavizar_sinal(
    y: np.ndarray | pd.Series,
    window_length: int = 7,
    polyorder: int = 3,
) -> np.ndarray:
    """Suaviza o sinal preservando largura e formato dos picos.

    Savitzky-Golay é preferido sobre média móvel por reduzir ruído sem atenuar
    excessivamente ápices estreitos ou deslocar o tempo de retenção.
    """
    sinal = np.asarray(y, dtype=float)
    if len(sinal) < 5:
        return sinal.copy()

    janela = _ensure_odd_window(window_length, len(sinal))
    ordem = min(max(1, int(polyorder)), janela - 1)

    try:
        suavizado = savgol_filter(
            sinal,
            window_length=janela,
            polyorder=ordem,
            mode="interp",
        )
    except Exception:
        suavizado = (
            pd.Series(sinal)
            .rolling(window=janela, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    return np.maximum(suavizado, 0.0)


def estimar_nivel_ruido(
    y: np.ndarray | pd.Series,
    y_suavizado: np.ndarray | None = None,
) -> float:
    """Estima o ruído pela MAD dos resíduos, robusta a picos intensos."""
    sinal = np.asarray(y, dtype=float)
    suavizado = sinal if y_suavizado is None else np.asarray(y_suavizado, dtype=float)
    residuo = sinal - suavizado
    mad = np.median(np.abs(residuo - np.median(residuo)))

    if mad <= 0:
        delta = np.diff(suavizado, prepend=suavizado[0])
        mad = np.median(np.abs(delta - np.median(delta)))

    return float(1.4826 * mad)


def _resolver_baseline_local(
    sinal: np.ndarray,
    centro_idx: int,
    *,
    baseline_thresh: float | None = None,
    left_candidate: int | None = None,
    right_candidate: int | None = None,
    left_base_idx: int | None = None,
    right_base_idx: int | None = None,
) -> tuple[float, float, int, int]:
    """Resolve a baseline local e a faixa-base do pico sem depender de percentil global."""
    total_pontos = len(sinal)
    centro_idx = int(np.clip(centro_idx, 0, max(total_pontos - 1, 0)))

    limite_esquerdo = 0
    if left_base_idx is not None:
        limite_esquerdo = int(np.clip(left_base_idx, 0, centro_idx))
    elif left_candidate is not None:
        limite_esquerdo = int(np.clip(left_candidate, 0, centro_idx))

    limite_direito = max(total_pontos - 1, 0)
    if right_base_idx is not None:
        limite_direito = int(np.clip(right_base_idx, centro_idx, total_pontos - 1))
    elif right_candidate is not None:
        limite_direito = int(np.clip(right_candidate, centro_idx, total_pontos - 1))

    referencias_baseline: list[float] = []
    if left_base_idx is not None:
        referencias_baseline.append(
            float(sinal[int(np.clip(left_base_idx, 0, centro_idx))])
        )
    elif left_candidate is not None:
        referencias_baseline.append(
            float(sinal[int(np.clip(left_candidate, 0, centro_idx))])
        )

    if right_base_idx is not None:
        referencias_baseline.append(
            float(sinal[int(np.clip(right_base_idx, centro_idx, total_pontos - 1))])
        )
    elif right_candidate is not None:
        referencias_baseline.append(
            float(sinal[int(np.clip(right_candidate, centro_idx, total_pontos - 1))])
        )

    if baseline_thresh is not None:
        referencias_baseline.append(float(baseline_thresh))

    if not referencias_baseline:
        referencias_baseline.extend(
            [
                float(sinal[limite_esquerdo]),
                float(sinal[limite_direito]),
            ]
        )

    baseline_local = float(min(referencias_baseline))
    altura_pico = max(float(sinal[centro_idx]) - baseline_local, 1e-12)
    return baseline_local, altura_pico, limite_esquerdo, limite_direito


def get_peak_limits(
    y: np.ndarray | pd.Series,
    centro_idx: int,
    baseline_thresh: float | None = None,
    tol_rise: float = 0.1,
    trend_window: int = 5,
    *,
    derivative: np.ndarray | None = None,
    curvature: np.ndarray | None = None,
    left_candidate: int | None = None,
    right_candidate: int | None = None,
    left_valley: int | None = None,
    right_valley: int | None = None,
    left_base_idx: int | None = None,
    right_base_idx: int | None = None,
    slope_tolerance: float | None = None,
    min_height_ratio: float = 0.01,
    curvature_factor: float = 1.5,
) -> tuple[int, int]:
    """Determina início e fim com baseline local, derivadas e altura relativa corrigida."""

    sinal = np.asarray(y, dtype=float)
    total_pontos = len(sinal)

    if total_pontos == 0:
        return 0, 0

    centro_idx = int(np.clip(centro_idx, 0, total_pontos - 1))

    derivada = (
        np.gradient(sinal)
        if derivative is None
        else np.asarray(derivative, dtype=float)
    )
    curvatura = (
        np.gradient(derivada)
        if curvature is None
        else np.asarray(curvature, dtype=float)
    )

    (
        baseline_local,
        altura_pico,
        limite_esquerdo,
        limite_direito,
    ) = _resolver_baseline_local(
        sinal,
        centro_idx,
        baseline_thresh=baseline_thresh,
        left_candidate=left_candidate,
        right_candidate=right_candidate,
        left_base_idx=left_base_idx,
        right_base_idx=right_base_idx,
    )

    if left_valley is not None:
        limite_esquerdo = max(limite_esquerdo, int(np.clip(left_valley, 0, centro_idx)))
    if right_valley is not None:
        limite_direito = min(
            limite_direito,
            int(np.clip(right_valley, centro_idx, total_pontos - 1)),
        )

    if limite_esquerdo >= centro_idx:
        limite_esquerdo = max(0, centro_idx - 1)
    if limite_direito <= centro_idx:
        limite_direito = min(total_pontos - 1, centro_idx + 1)

    derivada_local = np.abs(derivada[limite_esquerdo : limite_direito + 1])
    curvatura_local = np.abs(curvatura[limite_esquerdo : limite_direito + 1])

    slope_tol = max(
        float(slope_tolerance) if slope_tolerance is not None else 0.0,
        float(np.quantile(derivada_local, 0.30) * 1.5) if derivada_local.size else 0.0,
        1e-12,
    )
    curvature_tol = max(
        float(np.quantile(curvatura_local, 0.30) * max(float(curvature_factor), 0.1))
        if curvatura_local.size
        else 0.0,
        1e-12,
    )
    margem_base = max(altura_pico * max(float(min_height_ratio), 0.005), 1e-12)

    def altura_relativa(idx: int) -> float:
        altura_abs = max(float(sinal[idx]) - baseline_local, 0.0)
        return altura_abs / altura_pico

    def perto_da_base(idx: int) -> bool:
        return max(float(sinal[idx]) - baseline_local, 0.0) <= margem_base

    def is_flat(idx: int) -> bool:
        w = max(1, trend_window)
        inicio = max(limite_esquerdo, idx - w)
        fim = min(limite_direito + 1, idx + w + 1)
        trecho = derivada[inicio:fim]
        if len(trecho) < 3:
            return False
        return bool(np.all(np.abs(trecho) < slope_tol))

    def has_inflection(idx: int) -> bool:
        w = max(1, trend_window // 2)
        inicio = max(limite_esquerdo, idx - w)
        fim = min(limite_direito + 1, idx + w + 1)
        trecho = curvatura[inicio:fim]
        if len(trecho) < 3:
            return False
        relevantes = trecho[np.abs(trecho) > curvature_tol * 0.35]
        if len(relevantes) < 2:
            return False
        sinais = np.sign(relevantes)
        return bool(np.any(sinais[:-1] != sinais[1:]))

    def slope_stabilized(idx: int, lado: str) -> bool:
        w = max(2, trend_window // 2)
        if lado == "left":
            inicio = max(limite_esquerdo, idx - w)
            fim = min(centro_idx + 1, idx + 1)
            trecho = derivada[inicio:fim]
            if len(trecho) == 0:
                return False
            return np.count_nonzero(trecho < -slope_tol) <= max(1, len(trecho) // 4)

        inicio = max(centro_idx, idx)
        fim = min(limite_direito + 1, idx + w + 1)
        trecho = derivada[inicio:fim]
        if len(trecho) == 0:
            return False
        return np.count_nonzero(trecho > slope_tol) <= max(1, len(trecho) // 4)

    def find_derivative_anchor(lado: str) -> int | None:
        if lado == "left":
            if centro_idx - limite_esquerdo < 3:
                return None
            idx_slope = limite_esquerdo + int(
                np.argmax(derivada[limite_esquerdo : centro_idx + 1])
            )
            for idx in range(idx_slope, limite_esquerdo - 1, -1):
                rel = altura_relativa(idx)
                if slope_stabilized(idx, "left") and (
                    has_inflection(idx)
                    or abs(derivada[idx]) <= slope_tol * 1.15
                    or perto_da_base(idx)
                ):
                    if rel < max(min_height_ratio * 6.0, 0.025):
                        return idx
            return None

        if limite_direito - centro_idx < 3:
            return None
        idx_slope = centro_idx + int(
            np.argmin(derivada[centro_idx : limite_direito + 1])
        )
        for idx in range(idx_slope, limite_direito + 1):
            rel = altura_relativa(idx)
            if slope_stabilized(idx, "right") and (
                has_inflection(idx)
                or abs(derivada[idx]) <= slope_tol * 1.15
                or perto_da_base(idx)
            ):
                if rel < max(min_height_ratio * 6.0, 0.025):
                    return idx
        return None

    ancora_esquerda = find_derivative_anchor("left")
    ancora_direita = find_derivative_anchor("right")

    inicio_base = centro_idx if left_candidate is None else int(left_candidate)
    if ancora_esquerda is not None:
        inicio_base = min(inicio_base, ancora_esquerda)
    inicio_idx = int(np.clip(inicio_base, limite_esquerdo, centro_idx))

    while inicio_idx > limite_esquerdo:
        rel = altura_relativa(inicio_idx)
        curvatura_suave = abs(curvatura[inicio_idx]) <= curvature_tol
        derivada_estavel = slope_stabilized(inicio_idx, "left")
        inflexao_local = has_inflection(inicio_idx)

        condicao_parada = (
            rel < min_height_ratio
            and derivada_estavel
            and (is_flat(inicio_idx) or curvatura_suave or inflexao_local)
        ) or (
            perto_da_base(inicio_idx)
            and (
                derivada_estavel
                or is_flat(inicio_idx)
                or curvatura_suave
                or inflexao_local
            )
        )

        if condicao_parada:
            break

        if inicio_idx - trend_window >= limite_esquerdo:
            trecho = derivada[inicio_idx - trend_window : inicio_idx + 1]
            if (
                np.count_nonzero(trecho < -slope_tol) <= 1
                and rel < 0.0075
                and (inflexao_local or derivada_estavel)
            ):
                break

        inicio_idx -= 1

    fim_base = centro_idx if right_candidate is None else int(right_candidate)
    if ancora_direita is not None:
        fim_base = max(fim_base, ancora_direita)
    fim_idx = int(np.clip(fim_base, centro_idx, limite_direito))

    while fim_idx < limite_direito:
        rel = altura_relativa(fim_idx)
        curvatura_suave = abs(curvatura[fim_idx]) <= curvature_tol
        derivada_estavel = slope_stabilized(fim_idx, "right")
        inflexao_local = has_inflection(fim_idx)

        condicao_parada = (
            rel < min_height_ratio
            and derivada_estavel
            and (is_flat(fim_idx) or curvatura_suave or inflexao_local)
        ) or (
            perto_da_base(fim_idx)
            and (
                derivada_estavel
                or is_flat(fim_idx)
                or curvatura_suave
                or inflexao_local
            )
        )

        if condicao_parada:
            break

        if fim_idx + trend_window <= limite_direito:
            trecho = derivada[fim_idx : fim_idx + trend_window + 1]
            if (
                np.count_nonzero(trecho > slope_tol) <= 1
                and rel < 0.0075
                and (inflexao_local or derivada_estavel)
            ):
                break

        fim_idx += 1

    return inicio_idx, fim_idx


def detectar_picos(
    y: np.ndarray | pd.Series,
    height: float,
    distance: int,
    *,
    prominence: float | None = None,
    width: int | None = None,
    window_length: int = 7,
    polyorder: int = 3,
    rel_height: float = 0.98,
    noise_factor: float = 4.0,
    return_properties: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Detecta picos usando ruído, proeminência e largura como filtros principais."""
    sinal = np.asarray(y, dtype=float)
    if sinal.size == 0:
        if return_properties:
            return np.array([], dtype=int), {"signal_smooth": sinal, "noise": 0.0}
        return np.array([], dtype=int)

    sinal_suavizado = suavizar_sinal(
        sinal,
        window_length=window_length,
        polyorder=polyorder,
    )
    ruido = estimar_nivel_ruido(sinal, sinal_suavizado)
    amplitude_ref = max(float(np.nanmax(sinal_suavizado)), 1.0)

    altura_minima = max(float(height), ruido * max(2.5, noise_factor * 0.75))
    proeminencia_minima = (
        float(prominence)
        if prominence is not None
        else max(
            altura_minima * 0.25,
            ruido * noise_factor,
            amplitude_ref * 0.0025,
        )
    )
    largura_minima = max(
        2,
        int(width if width is not None else max(3, window_length // 2)),
    )
    distancia_minima = max(2, int(distance))

    indices, propriedades = find_peaks(
        sinal_suavizado,
        height=altura_minima,
        distance=distancia_minima,
        prominence=proeminencia_minima,
        width=largura_minima,
    )

    if len(indices) > 0:
        prominencias, bases_esquerda, bases_direita = peak_prominences(
            sinal_suavizado,
            indices,
        )
        larguras, alturas_largura, left_ips, right_ips = peak_widths(
            sinal_suavizado,
            indices,
            rel_height=min(max(rel_height, 0.50), 0.995),
            prominence_data=(prominencias, bases_esquerda, bases_direita),
        )
        propriedades.update(
            {
                "prominences": prominencias,
                "left_bases": bases_esquerda,
                "right_bases": bases_direita,
                "widths": larguras,
                "width_heights": alturas_largura,
                "left_ips": left_ips,
                "right_ips": right_ips,
            }
        )
    else:
        propriedades.update(
            {
                "prominences": np.array([], dtype=float),
                "left_bases": np.array([], dtype=int),
                "right_bases": np.array([], dtype=int),
                "widths": np.array([], dtype=float),
                "width_heights": np.array([], dtype=float),
                "left_ips": np.array([], dtype=float),
                "right_ips": np.array([], dtype=float),
            }
        )

    propriedades.update(
        {
            "signal_smooth": sinal_suavizado,
            "noise": float(ruido),
            "min_height": float(altura_minima),
            "min_prominence": float(proeminencia_minima),
        }
    )

    if return_properties:
        return indices, propriedades
    return indices


__all__ = [
    "PeakDetectionConfig",
    "ler_arquivo_dat",
    "corrigir_baseline_linear",
    "suavizar_sinal",
    "estimar_nivel_ruido",
    "get_peak_limits",
    "detectar_picos",
]
