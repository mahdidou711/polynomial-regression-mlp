"""Disposition des fichiers exportes, parametree par une racine (depot, repertoire temporaire ou isole)."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Layout:
    root: Path

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def fig_poly(self) -> Path:
        return self.root / "results" / "figures" / "polynomial"

    @property
    def fig_mlp(self) -> Path:
        return self.root / "results" / "figures" / "mlp"

    @property
    def figures_manifest(self) -> Path:
        return self.root / "results" / "figures" / "figures_manifest.json"

    @property
    def preview(self) -> Path:
        return self.root / "results" / "tables_preview.tex"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def manifest(self) -> Path:
        return self.processed / "export_manifest.json"

    def figure_dir(self, folder: str) -> Path:
        return {"polynomial": self.fig_poly, "mlp": self.fig_mlp}[folder]


REPO = Layout(PROJECT_ROOT)
