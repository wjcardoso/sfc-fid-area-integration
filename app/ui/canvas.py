"""Componentes visuais baseados em Matplotlib."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CromatogramaCanvas(FigureCanvas):
    """Canvas Matplotlib dedicado à visualização do cromatograma."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 4))
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.12)
        super().__init__(self.figure)
