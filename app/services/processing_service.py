"""Funções de processamento e persistência para cromatogramas SFC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.special import erfc

from auxiliares import (
    PeakDetectionConfig,
    corrigir_baseline_linear,
    detectar_picos,
    get_peak_limits,
    ler_arquivo_dat,
)

TIME_STEP_MINUTES = 0.016667 / 10


def criar_dataframe_picos_vazio(colunas_amostra: list[str]) -> pd.DataFrame:
    """Cria um `DataFrame` vazio com o schema esperado para picos."""
    return pd.DataFrame(columns=colunas_amostra)


def criar_dataframe_padrao_vazio(colunas_amostra: list[str]) -> pd.DataFrame:
    """Cria um `DataFrame` vazio com o schema esperado para o padrão."""
    return pd.DataFrame(columns=["Composto", *colunas_amostra])


def ler_cromatograma(caminho: Path) -> pd.DataFrame:
    """Lê e normaliza um cromatograma a partir de arquivo textual ou tabular."""
    try:
        return normalizar_dataframe(ler_arquivo_dat(str(caminho)))
    except Exception:
        pass

    bruto = pd.read_csv(
        caminho,
        sep=None,
        engine="python",
        header=None,
        comment="#",
        skip_blank_lines=True,
    )

    if bruto.empty:
        raise ValueError("arquivo sem dados utilizáveis")

    colunas_numericas: list[pd.Series] = []
    for coluna in bruto.columns:
        serie = pd.to_numeric(bruto[coluna], errors="coerce")
        if serie.notna().sum() > 1:
            colunas_numericas.append(serie.dropna().reset_index(drop=True))

    if not colunas_numericas:
        raise ValueError("não foi possível identificar colunas numéricas")

    if len(colunas_numericas) == 1:
        intensidade = colunas_numericas[0].reset_index(drop=True)
        tempo = np.arange(1, len(intensidade) + 1, dtype=float) * TIME_STEP_MINUTES
    else:
        combinado = pd.DataFrame(
            {"Tempo": colunas_numericas[0], "Intensidade": colunas_numericas[1]}
        ).dropna()
        tempo = combinado["Tempo"].reset_index(drop=True)
        intensidade = combinado["Intensidade"].reset_index(drop=True)

    return normalizar_dataframe(
        pd.DataFrame({"Tempo": tempo, "Intensidade": intensidade})
    )


def normalizar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza colunas numéricas e aplica correção linear de baseline."""
    base = df.copy()
    base["Tempo"] = pd.to_numeric(base["Tempo"], errors="coerce")
    base["Intensidade"] = pd.to_numeric(base["Intensidade"], errors="coerce")
    base = base.dropna(subset=["Tempo", "Intensidade"]).sort_values("Tempo")
    base = base.reset_index(drop=True)

    if base.empty:
        raise ValueError("nenhum ponto válido encontrado")

    base["Intensidade_Original"] = base["Intensidade"].astype(float)
    intensidade_corrigida, baseline = corrigir_baseline_linear(
        base["Tempo"].to_numpy(dtype=float),
        base["Intensidade_Original"].to_numpy(dtype=float),
    )
    base["Baseline"] = baseline
    base["Intensidade"] = intensidade_corrigida
    return base


def calcular_area_pico_local(
    tempo: np.ndarray,
    intensidade: np.ndarray,
    inicio_idx: int,
    fim_idx: int,
) -> float:
    """Integra o pico por regra do trapézio no sinal corrigido e não-negativo.

    O sinal recebido já passou por correção de baseline em `normalizar_dataframe`,
    então evitar uma segunda subtração local deixa a integração mais estável e
    monotônica: ao ampliar o intervalo, a área não diminui artificialmente.
    """
    if fim_idx <= inicio_idx:
        return 0.0

    x = tempo[inicio_idx : fim_idx + 1]
    y = intensidade[inicio_idx : fim_idx + 1]
    if len(x) < 2:
        return 0.0

    return float(trapezoid(np.maximum(y, 0.0), x))


def _modelo_gaussiano(
    x: np.ndarray,
    amplitude: float,
    centro: float,
    sigma: float,
    baseline: float,
) -> np.ndarray:
    sigma = max(abs(float(sigma)), 1e-9)
    return baseline + amplitude * np.exp(-0.5 * ((x - centro) / sigma) ** 2)


def _modelo_duas_gaussianas(
    x: np.ndarray,
    amplitude1: float,
    centro1: float,
    sigma1: float,
    amplitude2: float,
    centro2: float,
    sigma2: float,
    baseline: float,
) -> np.ndarray:
    """Modelo gaussiano duplo simples para deconvolução leve de ombros."""
    return (
        _modelo_gaussiano(x, amplitude1, centro1, sigma1, 0.0)
        + _modelo_gaussiano(x, amplitude2, centro2, sigma2, 0.0)
        + baseline
    )


def _recalcular_propriedades_detectadas(
    sinal_suavizado: np.ndarray,
    indices_picos: np.ndarray,
    rel_height: float,
    *,
    noise: float,
    min_height: float,
    min_prominence: float,
) -> dict[str, np.ndarray | float]:
    """Recalcula larguras, proeminências e bases após um refinamento local."""
    if len(indices_picos) == 0:
        return {
            "prominences": np.array([], dtype=float),
            "left_bases": np.array([], dtype=int),
            "right_bases": np.array([], dtype=int),
            "widths": np.array([], dtype=float),
            "width_heights": np.array([], dtype=float),
            "left_ips": np.array([], dtype=float),
            "right_ips": np.array([], dtype=float),
            "signal_smooth": sinal_suavizado,
            "noise": float(noise),
            "min_height": float(min_height),
            "min_prominence": float(min_prominence),
        }

    prominencias, bases_esquerda, bases_direita = peak_prominences(
        sinal_suavizado,
        indices_picos,
    )
    larguras, alturas_largura, left_ips, right_ips = peak_widths(
        sinal_suavizado,
        indices_picos,
        rel_height=min(max(rel_height, 0.50), 0.995),
        prominence_data=(prominencias, bases_esquerda, bases_direita),
    )
    return {
        "prominences": prominencias,
        "left_bases": bases_esquerda,
        "right_bases": bases_direita,
        "widths": larguras,
        "width_heights": alturas_largura,
        "left_ips": left_ips,
        "right_ips": right_ips,
        "signal_smooth": sinal_suavizado,
        "noise": float(noise),
        "min_height": float(min_height),
        "min_prominence": float(min_prominence),
    }


def _localizar_apice_local(
    sinal: np.ndarray,
    idx_central: int,
    *,
    inicio: int,
    fim: int,
    margem: int,
) -> int:
    """Reposiciona o ápice no máximo local mais próximo do centro estimado."""
    local_inicio = max(int(inicio), int(idx_central) - int(margem))
    local_fim = min(int(fim) + 1, int(idx_central) + int(margem) + 1)
    if local_fim <= local_inicio:
        return int(np.clip(idx_central, inicio, fim))
    return int(local_inicio + np.argmax(sinal[local_inicio:local_fim]))


def _ajustar_duas_gaussianas_leves(
    x: np.ndarray,
    y: np.ndarray,
    centros_iniciais: tuple[float, float] | None = None,
) -> dict[str, np.ndarray | float] | None:
    """Ajusta duas gaussianas em uma janela curta para separar picos sobrepostos.

    O ajuste é conservador: só é aceito quando melhora materialmente o erro em
    relação a uma única gaussiana e quando ambas as componentes têm amplitude e
    separação suficientes, evitando supersegmentação por ruído.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 9 or len(y) != len(x) or float(np.ptp(x)) <= 0:
        return None

    baseline0 = max(float(min(y[0], y[-1])), 0.0)
    amplitude0 = max(float(np.max(y) - baseline0), 1e-9)
    largura_segmento = max(float(x[-1] - x[0]), TIME_STEP_MINUTES * 6)
    sigma0 = max(largura_segmento / 12.0, TIME_STEP_MINUTES * 2)

    if centros_iniciais is None:
        picos_locais, _ = find_peaks(
            y,
            prominence=max(amplitude0 * 0.06, 1e-9),
            distance=max(2, len(y) // 8),
        )
        if len(picos_locais) >= 2:
            melhores = picos_locais[np.argsort(y[picos_locais])[-2:]]
            centros0 = sorted(float(x[idx]) for idx in melhores)
        else:
            centro_principal = float(x[int(np.argmax(y))])
            delta = max(largura_segmento * 0.10, TIME_STEP_MINUTES * 4)
            centros0 = [
                float(np.clip(centro_principal - delta, x[0], x[-1])),
                float(np.clip(centro_principal + delta, x[0], x[-1])),
            ]
    else:
        centro1, centro2 = sorted(float(valor) for valor in centros_iniciais)
        if abs(centro2 - centro1) < TIME_STEP_MINUTES * 2:
            delta = max(largura_segmento * 0.08, TIME_STEP_MINUTES * 4)
            centro1 = float(np.clip(centro1 - delta, x[0], x[-1]))
            centro2 = float(np.clip(centro2 + delta, x[0], x[-1]))
        centros0 = [centro1, centro2]

    try:
        parametros_duplos, _ = curve_fit(
            _modelo_duas_gaussianas,
            x,
            y,
            p0=[
                amplitude0 * 0.65,
                centros0[0],
                sigma0,
                amplitude0 * 0.45,
                centros0[1],
                sigma0,
                baseline0,
            ],
            bounds=(
                [
                    0.0,
                    float(x[0]),
                    TIME_STEP_MINUTES * 0.75,
                    0.0,
                    float(x[0]),
                    TIME_STEP_MINUTES * 0.75,
                    0.0,
                ],
                [
                    np.inf,
                    float(x[-1]),
                    largura_segmento,
                    np.inf,
                    float(x[-1]),
                    largura_segmento,
                    max(float(np.max(y)), baseline0 + amplitude0),
                ],
            ),
            maxfev=12000,
        )
    except Exception:
        return None

    if parametros_duplos[1] > parametros_duplos[4]:
        parametros_duplos = np.array(
            [
                parametros_duplos[3],
                parametros_duplos[4],
                parametros_duplos[5],
                parametros_duplos[0],
                parametros_duplos[1],
                parametros_duplos[2],
                parametros_duplos[6],
            ],
            dtype=float,
        )

    y_duplo = _modelo_duas_gaussianas(x, *parametros_duplos)
    componente1 = _modelo_gaussiano(
        x,
        parametros_duplos[0],
        parametros_duplos[1],
        parametros_duplos[2],
        0.0,
    )
    componente2 = _modelo_gaussiano(
        x,
        parametros_duplos[3],
        parametros_duplos[4],
        parametros_duplos[5],
        0.0,
    )

    try:
        parametros_simples, _ = curve_fit(
            _modelo_gaussiano,
            x,
            y,
            p0=[amplitude0, float(x[np.argmax(y)]), sigma0, baseline0],
            bounds=(
                [0.0, float(x[0]), TIME_STEP_MINUTES * 0.75, 0.0],
                [
                    np.inf,
                    float(x[-1]),
                    largura_segmento,
                    max(float(np.max(y)), baseline0 + amplitude0),
                ],
            ),
            maxfev=8000,
        )
        y_simples = _modelo_gaussiano(x, *parametros_simples)
        erro_simples = float(np.sum((y - y_simples) ** 2))
    except Exception:
        erro_simples = float(np.sum((y - np.mean(y)) ** 2))

    erro_duplo = float(np.sum((y - y_duplo) ** 2))
    melhoria = 1.0 - erro_duplo / max(erro_simples, 1e-12)
    separacao = abs(float(parametros_duplos[4] - parametros_duplos[1]))
    relacao_amplitudes = min(
        float(parametros_duplos[0]), float(parametros_duplos[3])
    ) / max(
        max(float(parametros_duplos[0]), float(parametros_duplos[3])),
        1e-12,
    )

    if (
        not np.isfinite(melhoria)
        or melhoria < 0.12
        or separacao < max(largura_segmento * 0.08, TIME_STEP_MINUTES * 6)
        or relacao_amplitudes < 0.16
    ):
        return None

    mascara_entre = (x >= parametros_duplos[1]) & (x <= parametros_duplos[4])
    if np.count_nonzero(mascara_entre) >= 2:
        diferenca = np.abs(componente1[mascara_entre] - componente2[mascara_entre])
        idx_rel = int(np.argmin(diferenca))
        vale_x = float(x[mascara_entre][idx_rel])
    else:
        vale_x = float((parametros_duplos[1] + parametros_duplos[4]) / 2.0)

    return {
        "centros": np.array([parametros_duplos[1], parametros_duplos[4]], dtype=float),
        "sigmas": np.array([parametros_duplos[2], parametros_duplos[5]], dtype=float),
        "amplitudes": np.array(
            [parametros_duplos[0], parametros_duplos[3]], dtype=float
        ),
        "vale_x": vale_x,
        "melhoria": float(melhoria),
    }


def _expandir_ombros_por_deconvolucao(
    tempo: np.ndarray,
    sinal_suavizado: np.ndarray,
    indices_picos: np.ndarray,
    propriedades: dict[str, np.ndarray | float],
    config: PeakDetectionConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Tenta recuperar ombros reais que passaram como um único pico largo."""
    if len(indices_picos) == 0:
        return indices_picos, propriedades

    larguras = np.asarray(
        propriedades.get("widths", np.zeros(len(indices_picos))), dtype=float
    ).copy()
    alturas_largura = np.asarray(
        propriedades.get("width_heights", np.zeros(len(indices_picos))), dtype=float
    ).copy()
    prominencias = np.asarray(
        propriedades.get("prominences", np.zeros(len(indices_picos))), dtype=float
    ).copy()
    left_ips = np.asarray(
        propriedades.get("left_ips", indices_picos), dtype=float
    ).copy()
    right_ips = np.asarray(
        propriedades.get("right_ips", indices_picos), dtype=float
    ).copy()
    bases_esquerda = np.asarray(
        propriedades.get("left_bases", np.zeros(len(indices_picos))), dtype=int
    ).copy()
    bases_direita = np.asarray(
        propriedades.get("right_bases", np.zeros(len(indices_picos))), dtype=int
    ).copy()
    mediana_largura = (
        float(np.median(larguras[larguras > 0]))
        if np.any(larguras > 0)
        else float(max(config.min_width, 3))
    )
    passo_tempo = (
        float(np.median(np.diff(tempo))) if len(tempo) > 1 else TIME_STEP_MINUTES
    )

    indices_refinados = [int(idx) for idx in indices_picos]
    extras_info: list[dict[str, float | int]] = []

    for pos, idx_pico in enumerate(indices_picos):
        idx_pico = int(idx_pico)
        if pos > 0:
            largura_vizinha = (
                float(larguras[pos - 1]) if pos - 1 < len(larguras) else mediana_largura
            )
            if idx_pico - int(indices_picos[pos - 1]) <= max(
                4,
                int((float(larguras[pos]) + largura_vizinha) * 0.75),
            ):
                continue
        if pos < len(indices_picos) - 1:
            largura_vizinha = (
                float(larguras[pos + 1]) if pos + 1 < len(larguras) else mediana_largura
            )
            if int(indices_picos[pos + 1]) - idx_pico <= max(
                4,
                int((float(larguras[pos]) + largura_vizinha) * 0.75),
            ):
                continue

        esquerda = max(
            0,
            int(np.floor(left_ips[pos]))
            if pos < len(left_ips)
            else idx_pico - config.smoothing_window * 2,
        )
        direita = min(
            len(sinal_suavizado) - 1,
            int(np.ceil(right_ips[pos]))
            if pos < len(right_ips)
            else idx_pico + config.smoothing_window * 2,
        )
        if direita - esquerda < max(config.smoothing_window * 2, 8):
            continue

        semi_largura_esq = max(idx_pico - left_ips[pos], 1e-6)
        semi_largura_dir = max(right_ips[pos] - idx_pico, 1e-6)
        assimetria = semi_largura_dir / semi_largura_esq
        largura_suspeita = (
            config.smoothing_window * 4
            if len(indices_picos) == 1
            else max(int(mediana_largura * 1.6), config.smoothing_window * 3)
        )
        suspeito = (direita - esquerda) >= largura_suspeita and (
            len(indices_picos) == 1 or assimetria > 1.25 or assimetria < 0.80
        )
        if not suspeito:
            continue

        janela_deconv = int(
            max(
                np.ceil(larguras[pos] * 1.35) if pos < len(larguras) else 0,
                config.smoothing_window * 2,
                8,
            )
        )
        esquerda_fit = max(esquerda, idx_pico - janela_deconv)
        direita_fit = min(direita, idx_pico + janela_deconv)

        ajuste = _ajustar_duas_gaussianas_leves(
            tempo[esquerda_fit : direita_fit + 1],
            sinal_suavizado[esquerda_fit : direita_fit + 1],
        )
        if ajuste is None or float(ajuste["melhoria"]) < 0.16:
            continue

        centros = np.asarray(ajuste["centros"], dtype=float)
        sigmas = np.asarray(ajuste["sigmas"], dtype=float)
        amplitudes = np.asarray(ajuste["amplitudes"], dtype=float)
        componente_principal = int(np.argmin(np.abs(centros - float(tempo[idx_pico]))))
        componente_extra = 1 - componente_principal

        def estimar_ips(
            centro: float, sigma: float
        ) -> tuple[float, float, int, int, float]:
            idx_centro = int(np.argmin(np.abs(tempo - centro)))
            sigma_pts = max(float(sigma) / max(passo_tempo, 1e-12), 1.0)
            left_ip = max(esquerda, idx_centro - 1.9 * sigma_pts)
            right_ip = min(direita, idx_centro + 1.9 * sigma_pts)
            left_base = int(max(esquerda, np.floor(idx_centro - 3.0 * sigma_pts)))
            right_base = int(min(direita, np.ceil(idx_centro + 3.0 * sigma_pts)))
            largura = max(float(right_ip - left_ip), 2.0)
            return left_ip, right_ip, left_base, right_base, largura

        centro_principal = float(centros[componente_principal])
        idx_principal = int(np.argmin(np.abs(tempo - centro_principal)))
        (
            left_ip_principal,
            right_ip_principal,
            left_base_principal,
            right_base_principal,
            largura_principal,
        ) = estimar_ips(centro_principal, float(sigmas[componente_principal]))

        indices_refinados[pos] = idx_principal
        left_ips[pos] = left_ip_principal
        right_ips[pos] = right_ip_principal
        bases_esquerda[pos] = left_base_principal
        bases_direita[pos] = right_base_principal
        larguras[pos] = largura_principal
        prominencias[pos] = max(
            float(prominencias[pos]),
            float(amplitudes[componente_principal]) * 0.75,
        )
        alturas_largura[pos] = max(
            float(np.interp(centro_principal, tempo, sinal_suavizado))
            - float(amplitudes[componente_principal]) * 0.5,
            0.0,
        )

        centro_extra = float(centros[componente_extra])
        idx_extra = int(np.argmin(np.abs(tempo - centro_extra)))
        if any(
            abs(idx_extra - existente) <= max(3, config.min_width)
            for existente in indices_refinados
        ) or any(
            abs(idx_extra - int(info["indice"])) <= max(3, config.min_width)
            for info in extras_info
        ):
            continue

        (
            left_ip_extra,
            right_ip_extra,
            left_base_extra,
            right_base_extra,
            largura_extra,
        ) = estimar_ips(centro_extra, float(sigmas[componente_extra]))
        altura_local = max(
            float(np.interp(centro_extra, tempo, sinal_suavizado)),
            float(amplitudes[componente_extra]),
        )
        limiar_local = max(
            float(np.max(sinal_suavizado[esquerda_fit : direita_fit + 1])) * 0.10,
            float(propriedades.get("min_prominence", 0.0)) * 1.05,
            float(propriedades.get("noise", 0.0)) * 3.0,
        )
        if altura_local < limiar_local:
            continue

        extras_info.append(
            {
                "indice": idx_extra,
                "prominence": max(float(amplitudes[componente_extra]) * 0.75, 1e-9),
                "left_base": left_base_extra,
                "right_base": right_base_extra,
                "left_ip": left_ip_extra,
                "right_ip": right_ip_extra,
                "width": largura_extra,
                "width_height": max(
                    altura_local - float(amplitudes[componente_extra]) * 0.5,
                    0.0,
                ),
            }
        )

    houve_reordenacao = any(
        int(atual) != int(original)
        for atual, original in zip(indices_refinados, indices_picos)
    )
    if not extras_info and not houve_reordenacao:
        return indices_picos, propriedades

    indices_arr = np.asarray(
        indices_refinados + [int(info["indice"]) for info in extras_info],
        dtype=int,
    )
    prominencias_arr = np.concatenate(
        [
            prominencias,
            np.asarray(
                [float(info["prominence"]) for info in extras_info], dtype=float
            ),
        ]
    )
    left_bases_arr = np.concatenate(
        [
            bases_esquerda,
            np.asarray([int(info["left_base"]) for info in extras_info], dtype=int),
        ]
    )
    right_bases_arr = np.concatenate(
        [
            bases_direita,
            np.asarray([int(info["right_base"]) for info in extras_info], dtype=int),
        ]
    )
    widths_arr = np.concatenate(
        [
            larguras,
            np.asarray([float(info["width"]) for info in extras_info], dtype=float),
        ]
    )
    width_heights_arr = np.concatenate(
        [
            alturas_largura,
            np.asarray(
                [float(info["width_height"]) for info in extras_info], dtype=float
            ),
        ]
    )
    left_ips_arr = np.concatenate(
        [
            left_ips,
            np.asarray([float(info["left_ip"]) for info in extras_info], dtype=float),
        ]
    )
    right_ips_arr = np.concatenate(
        [
            right_ips,
            np.asarray([float(info["right_ip"]) for info in extras_info], dtype=float),
        ]
    )

    ordem = np.argsort(indices_arr)
    indices_arr = indices_arr[ordem]
    propriedades_refinadas = {
        "prominences": prominencias_arr[ordem],
        "left_bases": left_bases_arr[ordem],
        "right_bases": right_bases_arr[ordem],
        "widths": widths_arr[ordem],
        "width_heights": width_heights_arr[ordem],
        "left_ips": left_ips_arr[ordem],
        "right_ips": right_ips_arr[ordem],
        "signal_smooth": sinal_suavizado,
        "noise": float(propriedades.get("noise", 0.0)),
        "min_height": float(propriedades.get("min_height", config.min_height)),
        "min_prominence": float(propriedades.get("min_prominence", 0.0)),
    }
    return indices_arr, propriedades_refinadas


def _ajustar_limites_sobrepostos_por_deconvolucao(
    tempo: np.ndarray,
    sinal_suavizado: np.ndarray,
    indices_picos: np.ndarray,
    candidatos_esquerda: np.ndarray,
    candidatos_direita: np.ndarray,
    vales_esquerda: list[int | None],
    vales_direita: list[int | None],
) -> tuple[np.ndarray, np.ndarray, list[int | None], list[int | None]]:
    """Usa o cruzamento entre gaussianas para separar melhor picos vizinhos."""
    for pos in range(len(indices_picos) - 1):
        idx_esq = int(indices_picos[pos])
        idx_dir = int(indices_picos[pos + 1])
        esquerda = max(0, min(int(candidatos_esquerda[pos]), idx_esq))
        direita = min(
            len(sinal_suavizado) - 1, max(int(candidatos_direita[pos + 1]), idx_dir)
        )
        sobrepostos = (
            int(candidatos_direita[pos]) >= int(candidatos_esquerda[pos + 1]) - 1
        )
        proximos = (idx_dir - idx_esq) <= max(4, (direita - esquerda) // 3)

        if not (sobrepostos or proximos) or direita - esquerda < 8:
            continue

        ajuste = _ajustar_duas_gaussianas_leves(
            tempo[esquerda : direita + 1],
            sinal_suavizado[esquerda : direita + 1],
            centros_iniciais=(float(tempo[idx_esq]), float(tempo[idx_dir])),
        )
        if ajuste is None:
            continue

        vale_x = float(ajuste["vale_x"])
        vale_idx = int(np.argmin(np.abs(tempo - vale_x)))
        vale_idx = int(np.clip(vale_idx, idx_esq, idx_dir))

        candidatos_direita[pos] = min(int(candidatos_direita[pos]), vale_idx)
        candidatos_esquerda[pos + 1] = max(int(candidatos_esquerda[pos + 1]), vale_idx)
        vales_direita[pos] = vale_idx
        vales_esquerda[pos + 1] = vale_idx

    return candidatos_esquerda, candidatos_direita, vales_esquerda, vales_direita


def _modelo_emg(
    x: np.ndarray,
    amplitude: float,
    centro: float,
    sigma: float,
    tau: float,
    baseline: float,
) -> np.ndarray:
    sigma = max(abs(float(sigma)), 1e-9)
    tau = max(abs(float(tau)), 1e-9)
    argumento = (sigma / tau - (x - centro) / sigma) / np.sqrt(2.0)
    expoente = np.clip((sigma**2) / (2.0 * tau**2) - (x - centro) / tau, -700, 700)

    with np.errstate(over="ignore", invalid="ignore"):
        resposta = baseline + (amplitude / (2.0 * tau)) * np.exp(expoente) * erfc(
            argumento
        )

    return np.nan_to_num(
        resposta, nan=baseline, posinf=baseline + amplitude, neginf=baseline
    )


def ajustar_modelo_pico(
    dados: pd.DataFrame,
    pico: pd.Series,
    modelo: str,
) -> dict[str, np.ndarray | str | float] | None:
    """Ajusta um modelo paramétrico local (Gaussiano ou EMG) ao pico."""
    modelo_normalizado = str(modelo).strip().lower()
    if modelo_normalizado not in {"gaussiano", "emg"}:
        return None

    inicio = float(pico.get("Início", np.nan))
    fim = float(pico.get("Fim", np.nan))
    if not np.isfinite(inicio) or not np.isfinite(fim) or fim <= inicio:
        return None

    segmento = dados[(dados["Tempo"] >= inicio) & (dados["Tempo"] <= fim)].copy()
    if len(segmento) < 5:
        return None

    x = segmento["Tempo"].to_numpy(dtype=float)
    y = segmento["Intensidade"].to_numpy(dtype=float)
    largura = max(float(fim - inicio), 1e-6)
    baseline0 = max(min(float(y[0]), float(y[-1])), 0.0)
    amplitude0 = max(float(np.max(y) - baseline0), 1e-6)
    centro0 = float(x[np.argmax(y)])
    sigma0 = max(largura / 6.0, 1e-4)

    try:
        if modelo_normalizado == "gaussiano":
            parametros, _ = curve_fit(
                _modelo_gaussiano,
                x,
                y,
                p0=[amplitude0, centro0, sigma0, baseline0],
                bounds=(
                    [0.0, float(x[0]), 1e-6, 0.0],
                    [np.inf, float(x[-1]), largura * 3.0, np.inf],
                ),
                maxfev=10000,
            )
            x_fit = np.linspace(float(x[0]), float(x[-1]), 200)
            y_fit = _modelo_gaussiano(x_fit, *parametros)
        else:
            tau0 = max(largura / 4.0, 1e-4)
            parametros, _ = curve_fit(
                _modelo_emg,
                x,
                y,
                p0=[amplitude0, centro0, sigma0, tau0, baseline0],
                bounds=(
                    [0.0, float(x[0]), 1e-6, 1e-6, 0.0],
                    [np.inf, float(x[-1]), largura * 3.0, largura * 4.0, np.inf],
                ),
                maxfev=15000,
            )
            x_fit = np.linspace(float(x[0]), float(x[-1]), 240)
            y_fit = _modelo_emg(x_fit, *parametros)
    except Exception:
        return None

    return {
        "modelo": "Gaussiano" if modelo_normalizado == "gaussiano" else "EMG",
        "x_fit": x_fit,
        "y_fit": y_fit,
        "centro": float(x_fit[np.argmax(y_fit)]),
        "altura": float(np.max(y_fit)),
    }


def ajustar_modelos_picos(
    dados: pd.DataFrame,
    picos: pd.DataFrame,
    modelo: str,
) -> list[dict[str, np.ndarray | str | float]]:
    """Ajusta modelos paramétricos para todos os picos visíveis."""
    if picos.empty or str(modelo).strip().lower() == "nenhum":
        return []

    ajustes: list[dict[str, np.ndarray | str | float]] = []
    for _, pico in picos.iterrows():
        ajuste = ajustar_modelo_pico(dados, pico, modelo)
        if ajuste is not None:
            ajustes.append(ajuste)
    return ajustes


def _obter_baseline_local_pico(
    sinal_suavizado: np.ndarray,
    propriedades: dict[str, np.ndarray | float],
    pos: int,
    idx_pico: int,
    *,
    left_candidate: int | None = None,
    right_candidate: int | None = None,
) -> tuple[float, int, int]:
    """Resolve a baseline local do pico a partir das bases do `peak_prominences`."""
    ultimo_idx = len(sinal_suavizado) - 1
    left_bases = np.asarray(
        propriedades.get("left_bases", np.array([], dtype=int)),
        dtype=int,
    )
    right_bases = np.asarray(
        propriedades.get("right_bases", np.array([], dtype=int)),
        dtype=int,
    )

    if pos < len(left_bases):
        left_base_idx = int(np.clip(left_bases[pos], 0, idx_pico))
    elif left_candidate is not None:
        left_base_idx = int(np.clip(left_candidate, 0, idx_pico))
    else:
        left_base_idx = int(np.clip(idx_pico, 0, ultimo_idx))

    if pos < len(right_bases):
        right_base_idx = int(np.clip(right_bases[pos], idx_pico, ultimo_idx))
    elif right_candidate is not None:
        right_base_idx = int(np.clip(right_candidate, idx_pico, ultimo_idx))
    else:
        right_base_idx = int(np.clip(idx_pico, 0, ultimo_idx))

    baseline_local = float(
        min(
            sinal_suavizado[left_base_idx],
            sinal_suavizado[right_base_idx],
        )
    )
    return baseline_local, left_base_idx, right_base_idx


def estimar_candidatos_limites(
    indices_picos: np.ndarray,
    propriedades: dict[str, np.ndarray | float],
    sinal_suavizado: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estima candidatos de início/fim entre meia-altura e base do pico.

    A versão anterior usava diretamente a largura quase na base (`rel_height=0.98`),
    o que tendia a superestimar o intervalo principalmente no padrão, onde os picos
    são limpos e bem resolvidos. Aqui interpolamos entre meia-altura e base usando
    a assimetria do pico: picos simétricos ficam mais compactos; picos com tailing
    preservam mais cauda no lado assimétrico.
    """
    if len(indices_picos) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    indices = np.asarray(indices_picos, dtype=int)
    left_base_ips = np.asarray(propriedades.get("left_ips", indices), dtype=float)
    right_base_ips = np.asarray(propriedades.get("right_ips", indices), dtype=float)
    prominencias = np.asarray(
        propriedades.get("prominences", np.zeros(len(indices))),
        dtype=float,
    )
    bases_esquerda = np.asarray(
        propriedades.get("left_bases", np.zeros(len(indices))),
        dtype=int,
    )
    bases_direita = np.asarray(
        propriedades.get("right_bases", np.zeros(len(indices))),
        dtype=int,
    )

    try:
        _, _, left_half_ips, right_half_ips = peak_widths(
            sinal_suavizado,
            indices,
            rel_height=0.5,
            prominence_data=(prominencias, bases_esquerda, bases_direita),
        )
    except Exception:
        left_half_ips = left_base_ips
        right_half_ips = right_base_ips

    candidatos_esquerda: list[int] = []
    candidatos_direita: list[int] = []
    ultimo_idx = len(sinal_suavizado) - 1

    for pos, idx_pico in enumerate(indices):
        semi_largura_esq = max(idx_pico - left_half_ips[pos], 1e-6)
        semi_largura_dir = max(right_half_ips[pos] - idx_pico, 1e-6)
        assimetria = semi_largura_dir / semi_largura_esq

        if assimetria > 1.20:
            fator_esq, fator_dir = 0.45, 0.68
        elif assimetria < 0.83:
            fator_esq, fator_dir = 0.68, 0.45
        else:
            fator_esq = fator_dir = 0.42

        candidato_esq = int(
            np.floor(
                left_half_ips[pos]
                + (left_base_ips[pos] - left_half_ips[pos]) * fator_esq
            )
        )
        candidato_dir = int(
            np.ceil(
                right_half_ips[pos]
                + (right_base_ips[pos] - right_half_ips[pos]) * fator_dir
            )
        )

        candidatos_esquerda.append(int(np.clip(candidato_esq, 0, idx_pico)))
        candidatos_direita.append(int(np.clip(candidato_dir, idx_pico, ultimo_idx)))

    return np.asarray(candidatos_esquerda, dtype=int), np.asarray(
        candidatos_direita,
        dtype=int,
    )


def detectar_picos_dataframe(
    df: pd.DataFrame,
    colunas_amostra: list[str],
    altura_minima_pico: float,
    janela_suavizacao: int,
    divisor_distancia_minima: int,
    *,
    min_prominence: float | None = None,
    min_width: int | None = None,
    noise_factor: float = 4.0,
    rel_height: float = 0.98,
    curvature_factor: float = 1.5,
) -> pd.DataFrame:
    """Detecta picos com heurísticas robustas para picos isolados e sobrepostos.

    Melhorias principais em relação à versão original:
    - suavização Savitzky-Golay para reduzir ruído sem deslocar o ápice;
    - filtros por proeminência e largura para reduzir falsos positivos;
    - uso de derivada/curvatura e vales locais para delimitar início/fim;
    - integração trapezoidal no sinal corrigido e não-negativo.
    """
    tempo = df["Tempo"].to_numpy(dtype=float)
    intensidade = df["Intensidade"].to_numpy(dtype=float)

    if len(intensidade) < 5:
        return criar_dataframe_picos_vazio(colunas_amostra)

    config = PeakDetectionConfig.from_legacy(
        signal_length=len(intensidade),
        min_height=altura_minima_pico,
        smoothing_window=janela_suavizacao,
        distance_divisor=divisor_distancia_minima,
    )
    config.min_prominence = (
        None
        if min_prominence is None or float(min_prominence) <= 0
        else float(min_prominence)
    )
    if min_width is not None:
        config.min_width = max(2, int(min_width))
    config.noise_factor = max(float(noise_factor), 0.5)
    config.rel_height = min(max(float(rel_height), 0.50), 0.995)
    config.curvature_factor = max(float(curvature_factor), 0.1)

    indices_picos, propriedades = detectar_picos(
        intensidade,
        height=config.min_height,
        distance=config.min_distance,
        prominence=config.min_prominence,
        width=config.min_width,
        window_length=config.smoothing_window,
        polyorder=config.smoothing_polyorder,
        rel_height=config.rel_height,
        noise_factor=config.noise_factor,
        return_properties=True,
    )

    if len(indices_picos) == 0:
        return criar_dataframe_picos_vazio(colunas_amostra)

    sinal_suavizado = np.asarray(propriedades["signal_smooth"], dtype=float)
    indices_picos, propriedades = _expandir_ombros_por_deconvolucao(
        tempo,
        sinal_suavizado,
        np.asarray(indices_picos, dtype=int),
        propriedades,
        config,
    )
    sinal_suavizado = np.asarray(propriedades["signal_smooth"], dtype=float)
    derivada = np.gradient(sinal_suavizado, tempo)
    curvatura = np.gradient(derivada, tempo)
    slope_tolerance = max(float(np.quantile(np.abs(derivada), 0.25) * 1.5), 1e-9)

    candidatos_esquerda, candidatos_direita = estimar_candidatos_limites(
        indices_picos,
        propriedades,
        sinal_suavizado,
    )
    vales_esquerda: list[int | None] = [None] * len(indices_picos)
    vales_direita: list[int | None] = [None] * len(indices_picos)

    for pos in range(len(indices_picos) - 1):
        idx_esq = int(indices_picos[pos])
        idx_dir = int(indices_picos[pos + 1])
        vale = int(idx_esq + np.argmin(sinal_suavizado[idx_esq : idx_dir + 1]))
        vales_direita[pos] = vale
        vales_esquerda[pos + 1] = vale

    (
        candidatos_esquerda,
        candidatos_direita,
        vales_esquerda,
        vales_direita,
    ) = _ajustar_limites_sobrepostos_por_deconvolucao(
        tempo,
        sinal_suavizado,
        np.asarray(indices_picos, dtype=int),
        np.asarray(candidatos_esquerda, dtype=int),
        np.asarray(candidatos_direita, dtype=int),
        vales_esquerda,
        vales_direita,
    )

    registros_picos: list[dict[str, float]] = []
    raio_refino = max(2, config.smoothing_window)

    for pos, idx_pico in enumerate(indices_picos):
        idx_pico = int(idx_pico)
        left_candidate = (
            int(candidatos_esquerda[pos])
            if pos < len(candidatos_esquerda)
            else idx_pico
        )
        right_candidate = (
            int(candidatos_direita[pos]) if pos < len(candidatos_direita) else idx_pico
        )

        baseline_local, left_base_idx, right_base_idx = _obter_baseline_local_pico(
            sinal_suavizado,
            propriedades,
            pos,
            idx_pico,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
        )

        inicio_idx, fim_idx = get_peak_limits(
            sinal_suavizado,
            idx_pico,
            baseline_local,
            tol_rise=0.03,
            trend_window=max(3, config.smoothing_window // 2),
            derivative=derivada,
            curvature=curvatura,
            left_candidate=left_candidate,
            right_candidate=right_candidate,
            left_valley=vales_esquerda[pos],
            right_valley=vales_direita[pos],
            left_base_idx=left_base_idx,
            right_base_idx=right_base_idx,
            slope_tolerance=slope_tolerance,
            min_height_ratio=max(1.0 - config.rel_height, 0.002),
            curvature_factor=config.curvature_factor,
        )

        local_inicio = max(0, idx_pico - raio_refino)
        local_fim = min(len(intensidade), idx_pico + raio_refino + 1)
        idx_apice = local_inicio + int(np.argmax(intensidade[local_inicio:local_fim]))

        inicio_idx = min(inicio_idx, idx_apice)
        fim_idx = max(fim_idx, idx_apice)
        if fim_idx <= inicio_idx:
            continue

        area = calcular_area_pico_local(tempo, intensidade, inicio_idx, fim_idx)
        largura = max(float(tempo[fim_idx] - tempo[inicio_idx]), 0.0)
        registros_picos.append(
            {
                "Pico": float(len(registros_picos) + 1),
                "Tempo (min)": float(tempo[idx_apice]),
                "Altura": float(intensidade[idx_apice]),
                "Início": float(tempo[inicio_idx]),
                "Fim": float(tempo[fim_idx]),
                "Largura": largura,
                "Área": float(area),
            }
        )

    if not registros_picos:
        return criar_dataframe_picos_vazio(colunas_amostra)

    resultado = (
        pd.DataFrame(registros_picos).sort_values("Tempo (min)").reset_index(drop=True)
    )
    resultado["Pico"] = np.arange(1, len(resultado) + 1, dtype=int)
    return resultado[colunas_amostra]


def detectar_picos_padrao_referenciados(
    df: pd.DataFrame,
    referencias_base: pd.DataFrame,
    colunas_amostra: list[str],
    altura_minima_pico: float,
    janela_suavizacao: int,
    divisor_distancia_minima: int,
    *,
    min_prominence: float | None = None,
    min_width: int | None = None,
    noise_factor: float = 4.0,
    rel_height: float = 0.98,
    curvature_factor: float = 1.5,
) -> pd.DataFrame:
    """Associa os picos detectados às janelas analíticas do padrão."""
    picos_detectados = detectar_picos_dataframe(
        df,
        colunas_amostra,
        altura_minima_pico,
        janela_suavizacao,
        divisor_distancia_minima,
        min_prominence=min_prominence,
        min_width=min_width,
        noise_factor=noise_factor,
        rel_height=rel_height,
        curvature_factor=curvature_factor,
    ).copy()
    if picos_detectados.empty:
        return criar_dataframe_padrao_vazio(colunas_amostra)

    referencias_ordenadas = referencias_base.sort_values("Tempo (min)")
    picos_detectados = picos_detectados.sort_values("Tempo (min)").reset_index(
        drop=True
    )

    registros_padrao: list[dict[str, float | str]] = []
    indices_usados: set[int] = set()

    for numero, (_, referencia) in enumerate(referencias_ordenadas.iterrows(), start=1):
        composto = str(referencia["Composto"])
        tempo_esperado = float(referencia["Tempo (min)"])
        inicio_ref = float(referencia["Início"])
        fim_ref = float(referencia["Fim"])

        largura_ref = max(fim_ref - inicio_ref, 0.20)
        tolerancia = max(0.35, largura_ref * 3.0)

        candidatos = picos_detectados.loc[
            ~picos_detectados.index.isin(indices_usados)
        ].copy()
        if candidatos.empty:
            break

        candidatos["Desvio"] = (candidatos["Tempo (min)"] - tempo_esperado).abs()
        candidatos = candidatos[candidatos["Desvio"] <= tolerancia].copy()
        if candidatos.empty:
            continue

        altura_maxima = max(float(candidatos["Altura"].max()), 1e-12)
        candidatos["Score"] = candidatos["Desvio"] / tolerancia - 0.15 * (
            candidatos["Altura"] / altura_maxima
        )
        melhor_idx = int(
            candidatos.sort_values(
                ["Score", "Desvio", "Altura"],
                ascending=[True, True, False],
            ).index[0]
        )
        melhor = picos_detectados.loc[melhor_idx]

        registros_padrao.append(
            {
                "Composto": composto,
                "Pico": numero,
                "Tempo (min)": float(melhor["Tempo (min)"]),
                "Altura": float(melhor["Altura"]),
                "Início": float(melhor["Início"]),
                "Fim": float(melhor["Fim"]),
                "Largura": float(melhor.get("Largura", 0.0)),
                "Área": float(melhor["Área"]),
            }
        )
        indices_usados.add(melhor_idx)

    return pd.DataFrame(registros_padrao, columns=["Composto", *colunas_amostra])


def criar_referencias_padrao_iniciais(colunas_padrao: list[str]) -> pd.DataFrame:
    """Retorna as referências cromatográficas padrão iniciais."""
    return pd.DataFrame(
        [
            {
                "Composto": "Hexadecano",
                "Tempo (min)": 5.00,
                "Início": 4.80,
                "Fim": 5.20,
            },
            {"Composto": "Tolueno", "Tempo (min)": 6.90, "Início": 6.70, "Fim": 7.10},
            {"Composto": "THN", "Tempo (min)": 9.05, "Início": 8.85, "Fim": 9.25},
            {
                "Composto": "Naftaleno",
                "Tempo (min)": 10.40,
                "Início": 10.20,
                "Fim": 10.60,
            },
            {
                "Composto": "Dibenzotiofeno",
                "Tempo (min)": 16.95,
                "Início": 16.70,
                "Fim": 17.20,
            },
        ],
        columns=colunas_padrao,
    )


def carregar_referencias_padrao(
    settings: QSettings,
    colunas_padrao: list[str],
) -> pd.DataFrame:
    """Lê do `QSettings` as referências persistidas do padrão."""
    base = criar_referencias_padrao_iniciais(colunas_padrao)
    bruto = settings.value("padrao/referencias", "")
    if not bruto:
        return base

    try:
        salvos = pd.DataFrame(json.loads(str(bruto)))
    except Exception:
        return base

    if salvos.empty or "Composto" not in salvos.columns:
        return base

    for _, row in salvos.iterrows():
        nome = str(row.get("Composto", "")).strip()
        if not nome:
            continue
        mascara = base["Composto"] == nome
        if not mascara.any():
            continue
        for coluna in ["Tempo (min)", "Início", "Fim"]:
            valor = pd.to_numeric(row.get(coluna), errors="coerce")
            if pd.notna(valor):
                base.loc[mascara, coluna] = float(valor)

    return base


def salvar_referencias_padrao(
    settings: QSettings, referencias_padrao: pd.DataFrame
) -> None:
    """Persiste as referências do padrão no `QSettings`."""
    registros = referencias_padrao.to_dict(orient="records")
    settings.setValue("padrao/referencias", json.dumps(registros, ensure_ascii=False))
    settings.sync()


def chave_settings_amostra(chave: str) -> str:
    """Gera uma chave segura para persistência dos picos por amostra."""
    segura = chave.replace(":", "_").replace("\\", "__").replace("/", "__")
    return f"amostras/{segura}/picos"


def aplicar_picos_salvos(
    settings: QSettings, chave: str, picos: pd.DataFrame
) -> pd.DataFrame:
    """Sobrescreve picos detectados com edições persistidas do usuário."""
    bruto = settings.value(chave_settings_amostra(chave), "")
    if not bruto:
        return picos

    try:
        salvos = pd.DataFrame(json.loads(str(bruto)))
    except Exception:
        return picos

    if salvos.empty or "Pico" not in salvos.columns:
        return picos

    colunas_padrao = [
        "Pico",
        "Tempo (min)",
        "Altura",
        "Início",
        "Fim",
        "Largura",
        "Área",
    ]

    if picos.empty:
        resultado = salvos.copy()
        for coluna in colunas_padrao:
            if coluna not in resultado.columns:
                resultado[coluna] = 0.0 if coluna != "Pico" else np.nan
        resultado = resultado[colunas_padrao].copy()
        resultado = resultado.sort_values("Tempo (min)").reset_index(drop=True)
        if not resultado.empty:
            resultado["Pico"] = np.arange(1, len(resultado) + 1, dtype=int)
        return resultado

    resultado = picos.copy()
    for _, row in salvos.iterrows():
        pico_id = pd.to_numeric(row.get("Pico"), errors="coerce")
        if pd.isna(pico_id):
            continue
        mascara = resultado["Pico"] == int(pico_id)
        if not mascara.any():
            continue
        for coluna in ["Tempo (min)", "Altura", "Início", "Fim", "Largura", "Área"]:
            valor = pd.to_numeric(row.get(coluna), errors="coerce")
            if pd.notna(valor):
                resultado.loc[mascara, coluna] = float(valor)

    return resultado


def salvar_picos_amostra(settings: QSettings, chave: str, picos: pd.DataFrame) -> None:
    """Persiste os picos editados de uma amostra específica."""
    registros = picos.to_dict(orient="records")
    settings.setValue(
        chave_settings_amostra(chave), json.dumps(registros, ensure_ascii=False)
    )
    settings.sync()


def criar_dataframe_regioes(
    regioes_analiticas: list[tuple[str, float, float]],
) -> pd.DataFrame:
    """Cria o `DataFrame` base com as regiões analíticas configuradas."""
    return pd.DataFrame(
        regioes_analiticas, columns=["Grupos", "Início (min)", "Fim (min)"]
    )


def calcular_areas_por_regiao(
    picos: pd.DataFrame,
    regioes_analiticas: list[tuple[str, float, float]],
    colunas_regioes: list[str],
    dados: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcula as áreas por faixa cromatográfica configurada."""
    resumo = criar_dataframe_regioes(regioes_analiticas).copy()
    resumo["Área Total"] = 0.0

    if dados is not None and not dados.empty:
        tempos_dados = pd.to_numeric(dados.get("Tempo"), errors="coerce")
        intensidades = pd.to_numeric(dados.get("Intensidade"), errors="coerce").fillna(
            0.0
        )
        tempos_np = tempos_dados.to_numpy(dtype=float)
        intensidades_np = intensidades.to_numpy(dtype=float)

        for indice, linha in resumo.iterrows():
            inicio = float(linha["Início (min)"])
            fim = float(linha["Fim (min)"])
            if fim < inicio:
                inicio, fim = fim, inicio

            mascara = (tempos_np >= inicio) & (tempos_np <= fim)
            idx = np.flatnonzero(mascara)
            if len(idx) < 2:
                continue

            resumo.at[indice, "Área Total"] = calcular_area_pico_local(
                tempos_np,
                intensidades_np,
                int(idx[0]),
                int(idx[-1]),
            )

        return resumo[colunas_regioes]

    if picos.empty:
        return resumo[colunas_regioes]

    tempos = pd.to_numeric(picos["Tempo (min)"], errors="coerce")
    areas = pd.to_numeric(picos["Área"], errors="coerce").fillna(0.0)

    for indice, linha in resumo.iterrows():
        inicio = float(linha["Início (min)"])
        fim = float(linha["Fim (min)"])
        eh_ultima = indice == len(resumo) - 1
        mascara = (tempos >= inicio) & (tempos <= fim if eh_ultima else tempos < fim)
        resumo.at[indice, "Área Total"] = float(areas.loc[mascara].sum())

    return resumo[colunas_regioes]


def criar_pico_por_intervalo(
    dados: pd.DataFrame,
    inicio: float,
    fim: float,
    *,
    pico_id: int | None = None,
) -> dict[str, float]:
    """Cria um pico manual a partir do intervalo definido pelo usuário.

    O ápice é calculado automaticamente como o máximo local dentro do intervalo,
    enquanto altura, largura e área são recalculadas a partir do próprio sinal.
    """
    tempos = dados["Tempo"].to_numpy(dtype=float)
    intensidades = dados["Intensidade"].to_numpy(dtype=float)
    if len(tempos) < 2:
        raise ValueError("Amostra sem pontos suficientes para criar um pico.")

    tempo_min = float(tempos.min())
    tempo_max = float(tempos.max())
    inicio = min(max(float(inicio), tempo_min), tempo_max)
    fim = min(max(float(fim), tempo_min), tempo_max)

    if inicio > fim:
        inicio, fim = fim, inicio

    mascara = (tempos >= inicio) & (tempos <= fim)
    indices = np.flatnonzero(mascara)
    if len(indices) < 2:
        raise ValueError("Intervalo muito curto para definir um pico manual.")

    idx_inicio = int(indices[0])
    idx_fim = int(indices[-1])
    idx_apice = int(indices[np.argmax(intensidades[indices])])

    return {
        "Pico": float(pico_id) if pico_id is not None else np.nan,
        "Tempo (min)": float(tempos[idx_apice]),
        "Altura": float(intensidades[idx_apice]),
        "Início": float(tempos[idx_inicio]),
        "Fim": float(tempos[idx_fim]),
        "Largura": float(tempos[idx_fim] - tempos[idx_inicio]),
        "Área": calcular_area_pico_local(tempos, intensidades, idx_inicio, idx_fim),
    }


def recalcular_pico_editado(
    dados: pd.DataFrame, registro: pd.Series
) -> dict[str, float]:
    """Recalcula altura, largura e área após edição manual de um pico."""
    tempo = float(registro["Tempo (min)"])
    inicio = float(registro["Início"])
    fim = float(registro["Fim"])

    tempo_min = float(dados["Tempo"].min())
    tempo_max = float(dados["Tempo"].max())

    if inicio > fim:
        inicio, fim = fim, inicio

    tempo = min(max(tempo, tempo_min), tempo_max)
    inicio = min(max(inicio, tempo_min), tempo_max)
    fim = min(max(fim, tempo_min), tempo_max)

    if tempo < inicio:
        inicio = tempo
    if tempo > fim:
        fim = tempo

    tempos = dados["Tempo"].to_numpy(dtype=float)
    intensidades = dados["Intensidade"].to_numpy(dtype=float)
    idx_pico = int(np.abs(tempos - tempo).argmin())
    altura = float(intensidades[idx_pico])

    idx_inicio = int(np.abs(tempos - inicio).argmin())
    idx_fim = int(np.abs(tempos - fim).argmin())
    if idx_inicio > idx_fim:
        idx_inicio, idx_fim = idx_fim, idx_inicio

    area = calcular_area_pico_local(tempos, intensidades, idx_inicio, idx_fim)
    largura = max(fim - inicio, 0.0)

    return {
        "Tempo (min)": tempo,
        "Altura": altura,
        "Início": inicio,
        "Fim": fim,
        "Largura": largura,
        "Área": area,
    }
