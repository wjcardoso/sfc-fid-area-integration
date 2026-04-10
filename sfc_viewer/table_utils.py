"""Funções auxiliares para configuração e preenchimento de tabelas Qt."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)


def configurar_tabela(tabela: QTableWidget, colunas: list[str]) -> None:
    """Aplica a configuração visual padrão de uma tabela da interface."""
    tabela.setColumnCount(len(colunas))
    tabela.setHorizontalHeaderLabels(colunas)
    tabela.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    tabela.verticalHeader().setVisible(False)
    tabela.setAlternatingRowColors(True)
    tabela.setSelectionBehavior(QAbstractItemView.SelectRows)
    tabela.setSelectionMode(QAbstractItemView.SingleSelection)
    tabela.setEditTriggers(
        QAbstractItemView.DoubleClicked
        | QAbstractItemView.SelectedClicked
        | QAbstractItemView.EditKeyPressed
    )


def preencher_tabela(
    tabela: QTableWidget,
    df: pd.DataFrame,
    colunas: list[str],
    colunas_editaveis: set[str],
    mensagem_vazia: str,
) -> None:
    """Preenche um `QTableWidget` a partir de um `DataFrame`."""
    tabela.clearContents()
    tabela.setRowCount(0)

    if df.empty:
        tabela.setRowCount(1)
        item = QTableWidgetItem(mensagem_vazia)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        tabela.setItem(0, 0, item)
        return

    tabela.setRowCount(len(df))
    for linha, (_, registro) in enumerate(df.iterrows()):
        for coluna, nome_coluna in enumerate(colunas):
            valor = registro[nome_coluna]
            item = QTableWidgetItem(formatar_valor_tabela(valor))
            item.setTextAlignment(Qt.AlignCenter)
            if nome_coluna not in colunas_editaveis:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            tabela.setItem(linha, coluna, item)


def formatar_valor_tabela(valor: object) -> str:
    """Formata números com 4 casas decimais para exibição na interface."""
    if isinstance(valor, (float, np.floating)):
        return f"{float(valor):.4f}"
    return str(valor)


def texto_para_float(texto: str) -> float | None:
    """Converte texto digitado pelo usuário em número decimal."""
    try:
        return float(str(texto).strip().replace(",", "."))
    except ValueError:
        return None
