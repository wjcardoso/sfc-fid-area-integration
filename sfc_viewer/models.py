"""Modelos de dados utilizados pela interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Amostra:
    """Representa uma amostra carregada com seus dados, picos e metadados."""

    caminho: Path
    dados: pd.DataFrame
    picos: pd.DataFrame
    metodo_analise: str = "ASTM D5186"
    eh_diesel: bool = False
    ensaio: str = "E037"
