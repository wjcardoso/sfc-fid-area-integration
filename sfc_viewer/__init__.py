"""Pacote principal da aplicação SFC Viewer."""

from .models import Amostra

__all__ = ["Amostra", "JanelaPrincipal", "main"]


def __getattr__(name: str):
    """Carrega componentes da interface sob demanda para evitar imports pesados."""
    if name in {"JanelaPrincipal", "main"}:
        from .main_window import JanelaPrincipal, main

        return {"JanelaPrincipal": JanelaPrincipal, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
