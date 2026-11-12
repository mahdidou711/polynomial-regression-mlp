"""Formateur CSV vers ``tabular`` / ``longtable`` LaTeX (etape 8, ``EXG-RAP-B4-02``).

Contrat des fichiers produits (consigne de l'avenant 7.6 et de l'utilisateur) :

* fragment ``\\input{}`` : un seul environnement ``tabular`` ou ``longtable``, sans ``caption``
  ni ``label`` (ils appartiennent au rapport, etape 9) ;
* ``booktabs`` (``\\toprule``, ``\\midrule``, ``\\bottomrule``), colonnes numeriques ``S`` de
  ``siunitx`` avec ``table-format`` calcule sur les donnees, aucun ``\\sisetup`` requis ;
* point decimal dans les donnees : la presentation francaise (virgule) se regle dans le
  preambule par ``\\sisetup{output-decimal-marker={,}}`` ;
* ASCII pur (accents ecrits en macros), aucun ``$`` (mathematiques en ``\\( \\)``), aucun
  chemin, aucune commande de mise en page globale.

Les fonctions sont pures (aucune lecture de fichier hors ``read_csv``).
"""

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

LONGTABLE_ROWS = 25  # au-dela, ``longtable`` (un ``tabular`` ne se coupe pas entre deux pages)
MISSING = {"", "nan", "NaN", "None", "none"}
REQUIRED_PACKAGES = ("booktabs", "siunitx", "longtable")
INFINITY = r"\(\infty\)"
KINDS = ("int", "fixed", "sci", "text", "code", "bool")


@dataclass(frozen=True)
class Col:
    """Colonne d'une table : cle de la ligne, en-tete LaTeX, type et precision.

    ``digits`` : chiffres significatifs pour ``sci``, decimales pour ``fixed``.
    """

    key: str
    header: str
    kind: str = "sci"
    digits: int = 3

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"type de colonne inconnu : {self.kind}")


def read_csv(path) -> list[dict]:
    """Lignes d'un CSV (valeurs en chaines ; le typage est decide par les colonnes)."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, header, rows) -> Path:
    """CSV deterministe : ``repr`` des flottants (aller-retour exact), fins de ligne ``\\n``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(row.get(k)) for k in header])
    return path


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return str(value).strip() in MISSING


ACCENTS = {"é": r"\'e", "è": r"\`e", "ê": r"\^e", "ë": r'\"e', "à": r"\`a", "â": r"\^a",
           "î": r"\^\i{}", "ï": r'\"\i{}', "ô": r"\^o", "ù": r"\`u", "û": r"\^u",
           "ü": r'\"u', "ç": r"\c{c}", "É": r"\'E", "È": r"\`E", "À": r"\`A",
           "Ç": r"\c{C}", "œ": r"\oe{}", "’": "'", "×": r"\(\times\)"}


def escape_tex(text) -> str:
    """Echappe les caracteres speciaux de LaTeX ; la sortie est toujours ASCII.

    Les lettres accentuees courantes du francais deviennent des macros ; tout autre caractere
    non ASCII est refuse (``ValueError``) plutot qu'ecrit tel quel.
    """
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}", "$": r"\$"}
    out = []
    for ch in str(text):
        if ch in replacements:
            out.append(replacements[ch])
        elif ch in ACCENTS:
            out.append(ACCENTS[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            raise ValueError(f"caractere non ASCII sans equivalent LaTeX : {ch!r}")
    return "".join(out)


def format_sci(value, digits: int = 3) -> str:
    """Notation scientifique a ``digits`` chiffres significatifs, exposant signe : ``1.23e-04``."""
    if digits < 1:
        raise ValueError("digits >= 1 requis")
    return f"{float(value):.{digits - 1}e}"


def format_fixed(value, decimals: int = 4) -> str:
    return f"{float(value):.{decimals}f}"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if text in ("True", "true", "1"):
        return True
    if text in ("False", "false", "0"):
        return False
    raise ValueError(f"booleen attendu, recu {value!r}")


def format_cell(value, col: Col) -> str:
    """Texte d'une cellule (hors accolades de colonne ``S``) ; ``---`` si la valeur manque."""
    if is_missing(value):
        return "---"
    if col.kind in ("sci", "fixed") and math.isinf(float(value)):
        return INFINITY if float(value) > 0 else "\\(-\\infty\\)"  # kappa d'une matrice singuliere
    if col.kind == "sci":
        return format_sci(value, col.digits)
    if col.kind == "fixed":
        return format_fixed(value, col.digits)
    if col.kind == "int":
        return str(int(float(value)))
    if col.kind == "bool":
        return "oui" if _as_bool(value) else "non"
    if col.kind == "code":
        return r"\texttt{" + escape_tex(value) + "}"
    return escape_tex(value)


_SCI_RE = re.compile(r"^(-?)(\d+)(?:\.(\d+))?e([+-])(\d+)$")


def table_format(strings, col: Col) -> str:
    """``table-format`` de siunitx calcule sur les chaines effectivement ecrites."""
    values = [s for s in strings if s != "---" and not s.startswith("\\")]
    if not values:
        return "1.0"
    if col.kind == "sci":
        parts = [_SCI_RE.fullmatch(s) for s in values]
        if not all(parts):
            raise ValueError(f"valeur scientifique mal formee dans {col.key} : {values}")
        neg = any(p.group(1) for p in parts)
        ints = max(len(p.group(2)) for p in parts)
        decs = max(len(p.group(3) or "") for p in parts)
        exps = max(len(p.group(5)) for p in parts)
        return f"{'-' if neg else ''}{ints}{'.' + str(decs) if decs else ''}e-{exps}"
    neg = any(s.startswith("-") for s in values)
    ints = max(len(s.lstrip("-").split(".")[0]) for s in values)
    decs = max(len(s.split(".")[1]) if "." in s else 0 for s in values)
    return f"{'-' if neg else ''}{ints}{'.' + str(decs) if decs else ''}"


def column_spec(cols, cells) -> str:
    """Specification de colonnes : ``S[table-format=...]`` pour ``sci`` et ``fixed``."""
    out = []
    for j, col in enumerate(cols):
        if col.kind in ("sci", "fixed"):
            out.append(f"S[table-format={table_format([row[j] for row in cells], col)}]")
        elif col.kind == "int":
            out.append("r")
        elif col.kind == "bool":
            out.append("c")
        else:
            out.append("l")
    return " ".join(out)


def _cell_text(text, col: Col) -> str:
    # Colonne S : toute cellule non numerique doit etre entre accolades (siunitx).
    if col.kind in ("sci", "fixed") and (text == "---" or text.startswith("\\")):
        return "{" + text + "}"
    return text


def render_table(cols, rows, source: str, *, longtable: bool | None = None) -> str:
    """Fragment LaTeX d'une table ; ``longtable`` par defaut si plus de ``LONGTABLE_ROWS`` lignes."""
    cols = list(cols)
    if not cols:
        raise ValueError("table sans colonne")
    rows = list(rows)
    cells = [[format_cell(row.get(c.key), c) for c in cols] for row in rows]
    spec = column_spec(cols, cells)
    use_long = len(rows) > LONGTABLE_ROWS if longtable is None else longtable
    header = " & ".join("{" + c.header + "}" for c in cols) + r" \\"
    body = [" & ".join(_cell_text(t, c) for t, c in zip(line, cols)) + r" \\" for line in cells]
    lines = [f"% Genere par src/export (ne pas editer a la main). Source : {source}.",
             f"% Colonnes : {', '.join(c.key for c in cols)} ; {len(rows)} lignes."]
    if use_long:
        lines += [rf"\begin{{longtable}}{{{spec}}}", r"\toprule", header, r"\midrule",
                  r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead",
                  r"\bottomrule", r"\endlastfoot", *body, r"\end{longtable}"]
    else:
        lines += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", header, r"\midrule", *body,
                  r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def environment_of(tex: str) -> str:
    """``tabular`` ou ``longtable`` selon le fragment."""
    return "longtable" if r"\begin{longtable}" in tex else "tabular"
