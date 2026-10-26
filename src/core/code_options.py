"""Customization layer for generated code: options, data loaders, packaging, and combining.

Snippet text uses ``# %%`` cell markers. Two special comment prefixes are resolved when the
final text is produced:

* ``#~ text``  teaching comment, kept only when ``comments == "teaching"``
* ``#? code``  sanity check, kept (as code) only when ``checks`` is enabled
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

PLACEHOLDER_FILE = "your_file.csv"
CELL = "# %%"

PLOT_LIBS = ("plotly", "matplotlib", "seaborn")
COMMENT_LEVELS = ("none", "brief", "teaching")
STYLES = ("notebook", "function")
LOADERS = {
    "local": "Local file",
    "colab": "Colab upload",
    "drive": "Google Drive",
    "url": "URL",
    "gsheet": "Google Sheet",
}
PALETTES = ("default", "viridis", "blues", "teal", "plasma", "sunsetdark")
TEMPLATES = ("dark", "light")

# Third-party modules that Google Colab does not ship with -> pip package names
_NEEDS_INSTALL = {
    "pandera": "pandera",
    "duckdb": "duckdb",
    "pymysql": "pymysql",
    "pyodbc": "pyodbc",
    "great_expectations": "great_expectations",
}


@dataclass(frozen=True)
class CodeOptions:
    plot_lib: str = "plotly"
    comments: str = "brief"
    checks: bool = False
    style: str = "notebook"
    loader: str = "local"
    source_url: str = ""
    template: str = "dark"
    palette: str = "default"
    width: int = 900
    height: int = 450
    title: str = ""
    x_label: str = ""
    y_label: str = ""

    def sanitized(self) -> CodeOptions:
        """Replace unknown values with defaults (options often come from widgets/session)."""
        return replace(
            self,
            plot_lib=self.plot_lib if self.plot_lib in PLOT_LIBS else "plotly",
            comments=self.comments if self.comments in COMMENT_LEVELS else "brief",
            style=self.style if self.style in STYLES else "notebook",
            loader=self.loader if self.loader in LOADERS else "local",
            template=self.template if self.template in TEMPLATES else "dark",
            palette=self.palette if self.palette in PALETTES else "default",
            width=min(max(int(self.width), 300), 2000),
            height=min(max(int(self.height), 200), 1500),
        )


DEFAULT_OPTIONS = CodeOptions()


def resolve(options: CodeOptions | None) -> CodeOptions:
    return (options or DEFAULT_OPTIONS).sanitized()


# --------------------------------------------------------------------------- literals
def q(value: Any) -> str:
    return repr(value)


def lst(values: Iterable[Any]) -> str:
    return "[" + ", ".join(q(v) for v in values) + "]"


# --------------------------------------------------------------------------- loading
_SHEETS_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)")


def sheets_export_url(url: str) -> str:
    match = _SHEETS_RE.match(url.strip())
    if not match:
        return url.strip()
    gid = re.search(r"[#&?]gid=(\d+)", url)
    return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=csv" + (
        f"&gid={gid.group(1)}" if gid else ""
    )


def read_call(
    path_expr: str,
    filename: str,
    sheet: str | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
) -> str:
    lower = filename.lower().split("?")[0]
    if lower.endswith((".xlsx", ".xls")):
        extra = f", sheet_name={q(sheet)}" if sheet else ""
        return f"pd.read_excel({path_expr}{extra})"
    if lower.endswith(".parquet"):
        return f"pd.read_parquet({path_expr})"
    extra = ""
    if delimiter and delimiter != ",":
        extra += f", sep={q(delimiter)}"
    if encoding and encoding.lower().replace("-", "") != "utf8":
        extra += f", encoding={q(encoding)}"
    return f"pd.read_csv({path_expr}{extra})"


def loader_lines(
    filename: str,
    sheet: str | None,
    encoding: str | None,
    delimiter: str | None,
    options: CodeOptions,
) -> list[str]:
    """Statements that define `df` for the chosen loader."""
    name = filename or PLACEHOLDER_FILE
    if options.loader == "colab":
        return [
            "from google.colab import files",
            "uploaded = files.upload()          # pick your file in the dialog",
            "uploaded_name = next(iter(uploaded))",
            f"df = {read_call('uploaded_name', name, sheet, encoding, delimiter)}",
        ]
    if options.loader == "drive":
        return [
            "from google.colab import drive",
            "drive.mount('/content/drive')",
            f"df = {read_call(q('/content/drive/MyDrive/' + name), name, sheet, encoding, delimiter)}",
        ]
    if options.loader == "url":
        url = options.source_url.strip() or "https://example.com/data.csv"
        return [f"df = {read_call(q(url), url, sheet, encoding, delimiter)}"]
    if options.loader == "gsheet":
        url = (
            sheets_export_url(options.source_url)
            if options.source_url.strip()
            else ("https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv")
        )
        return [
            f"df = pd.read_csv({q(url)})   # the sheet must be shared as 'Anyone with the link'"
        ]
    return [f"df = {read_call(q(name), name, sheet, encoding, delimiter)}"]


def load_cell(
    filename: str = PLACEHOLDER_FILE,
    sheet: str | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
    options: CodeOptions | None = None,
) -> str:
    opts = resolve(options)
    lines = loader_lines(filename, sheet, encoding, delimiter, opts)
    hint = (
        "# Load the data. In Colab: upload the file with the Files panel (left sidebar),\n"
        "# or run:  from google.colab import files; files.upload()\n"
        if opts.loader == "local"
        else "# Load the data\n"
    )
    return (
        f"{CELL}\n{hint}" + "\n".join(lines) + "\n"
        "#? assert not df.empty, 'the file loaded no rows'\n"
        "print(df.shape)\ndf.head()\n"
    )


# --------------------------------------------------------------------------- assembling
def split_cells(code: str) -> list[tuple[bool, str]]:
    """[(is_markdown, text)] from ``# %%``-delimited source."""
    cells: list[tuple[bool, str]] = []
    current: list[str] = []
    markdown = False
    started = False
    for line in code.splitlines():
        if line.startswith(CELL):
            if started and "\n".join(current).strip():
                cells.append((markdown, "\n".join(current).strip("\n")))
            current, markdown, started = [], "[markdown]" in line, True
            continue
        current.append(line)
    if started and "\n".join(current).strip():
        cells.append((markdown, "\n".join(current).strip("\n")))
    return cells


def join_cells(cells: Sequence[tuple[bool, str]]) -> str:
    out = []
    for is_md, text in cells:
        out.append(f"{CELL} [markdown]\n{text}" if is_md else f"{CELL}\n{text}")
    return "\n\n".join(out) + "\n"


_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z0-9_]+)", re.M)


def _pip_packages(code: str) -> list[str]:
    mods = {m.group(1) for m in _IMPORT_RE.finditer(code)}
    return sorted(_NEEDS_INSTALL[m] for m in mods if m in _NEEDS_INSTALL)


def _auto_imports(code: str, imports: list[str]) -> list[str]:
    """Add plotting imports implied by the body (px./plt./sns./go.)."""
    extras = []
    for needle, line in (
        ("px.", "import plotly.express as px"),
        ("go.", "import plotly.graph_objects as go"),
        ("plt.", "import matplotlib.pyplot as plt"),
        ("sns.", "import seaborn as sns"),
    ):
        if needle in code and line not in imports and line not in code:
            extras.append(line)
    return imports + extras


def finalize(code: str, options: CodeOptions) -> str:
    """Resolve ``#~`` / ``#?`` markers and drop comments when requested."""
    out: list[str] = []
    in_markdown = False
    for line in code.splitlines():
        stripped = line.lstrip()
        if line.startswith(CELL):
            in_markdown = "[markdown]" in line
            out.append(line)
            continue
        if in_markdown:
            out.append(line)
            continue
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("#~ "):
            if options.comments == "teaching":
                out.append(f"{indent}# {stripped[3:]}")
            continue
        if stripped.startswith("#? "):
            if options.checks:
                out.append(f"{indent}{stripped[3:]}")
            continue
        if options.comments == "none" and stripped.startswith("#"):
            continue
        out.append(line)
    text = "\n".join(out)
    return re.sub(r"\n{4,}", "\n\n\n", text).rstrip("\n") + "\n"


def make_snippet(
    title: str,
    body: str,
    imports: Sequence[str] = ("import pandas as pd", "import numpy as np"),
    filename: str = PLACEHOLDER_FILE,
    sheet: str | None = None,
    encoding: str | None = None,
    delimiter: str | None = None,
    include_load: bool = True,
    options: CodeOptions | None = None,
) -> str:
    """Assemble header, imports, data loading and analysis cells honoring `options`."""
    opts = resolve(options)
    body_cells = body if body.startswith(CELL) else f"{CELL}\n{body}"
    all_imports = _auto_imports(body_cells, list(imports))

    header = (
        f"{CELL} [markdown]\n# # {title}\n"
        "# Generated by CelestiusAI. Runs as-is in Google Colab or Jupyter\n"
        "# (needs pandas, numpy, scipy, statsmodels, scikit-learn, plotly - all preinstalled on Colab).\n"
    )
    pip = _pip_packages("\n".join(all_imports) + body_cells)
    pip_cell = f"{CELL}\n# !pip install -q {' '.join(pip)}\n" if pip else ""
    import_cell = (
        f"{CELL}\n"
        + "\n".join(all_imports)
        + "\n\ntry:\n    display  # available in Jupyter/Colab\n"
        "except NameError:\n    display = print\n"
    )

    if opts.style == "function":
        code = _function_style(
            title,
            all_imports,
            body_cells,
            filename,
            sheet,
            encoding,
            delimiter,
            include_load,
            opts,
            pip,
        )
        return finalize(code, opts)

    parts = [header, pip_cell, import_cell]
    if include_load:
        parts.append(load_cell(filename, sheet, encoding, delimiter, opts))
    parts.append(body_cells)
    return finalize("\n".join(p.rstrip("\n") + "\n" for p in parts if p), opts)


def _function_style(
    title: str,
    imports: list[str],
    body_cells: str,
    filename: str,
    sheet: str | None,
    encoding: str | None,
    delimiter: str | None,
    include_load: bool,
    opts: CodeOptions,
    pip: list[str],
) -> str:
    analysis = []
    for is_md, text in split_cells(body_cells):
        if is_md:
            analysis += [f"# {ln[2:] if ln.startswith('# ') else ln}" for ln in text.splitlines()]
        else:
            analysis += text.splitlines()
        analysis.append("")
    lines = [f'"""{title} - generated by CelestiusAI."""']
    if pip:
        lines.append(f"# !pip install -q {' '.join(pip)}")
    lines += imports + [
        "",
        "try:",
        "    display  # available in Jupyter/Colab",
        "except NameError:",
        "    display = print",
        "",
    ]
    if include_load:
        lines += ["", "def load_data():"]
        lines += [f"    {ln}" for ln in loader_lines(filename, sheet, encoding, delimiter, opts)]
        lines += ["    #? assert not df.empty, 'the file loaded no rows'", "    return df", ""]
    args = "df" if include_load else ""
    lines += ["", f"def analyze({args}):", f'    """{title}."""']
    lines += [("    " + ln) if ln.strip() else "" for ln in analysis] or ["    pass"]
    lines += ["", ""]
    if include_load:
        lines += ['if __name__ == "__main__":', "    analyze(load_data())"]
    else:
        lines += ['if __name__ == "__main__":', "    analyze()"]
    return "\n".join(lines) + "\n"


def combine_snippets(items: Sequence[tuple[str, str]], title: str = "Analysis") -> str:
    """Merge several notebook-style snippets into one: shared setup, then each analysis."""
    imports: list[str] = []
    load_cell_text = ""
    pip_lines: list[str] = []
    sections: list[tuple[bool, str]] = []
    for item_title, code in items:
        cells = split_cells(code)
        analysis: list[tuple[bool, str]] = []
        for idx, (is_md, text) in enumerate(cells):
            if is_md and idx == 0:
                continue  # per-snippet header
            if not is_md and text.startswith("# !pip install"):
                pip_lines.append(text)
                continue
            if (
                not is_md
                and ("import " in text.split("\n", 1)[0] or text.startswith("from "))
                and "display" in text
            ):
                for ln in text.splitlines():
                    if (
                        ln.strip()
                        and ln not in imports
                        and not ln.startswith(("try:", "except", "    display", "display ="))
                    ):
                        imports.append(ln)
                continue
            if not is_md and "# Load the data" in text:
                load_cell_text = load_cell_text or text
                continue
            analysis.append((is_md, text))
        if not analysis:
            continue
        sections.append((True, f"# ## {item_title}"))
        sections += analysis
    setup: list[tuple[bool, str]] = [
        (
            True,
            f"# # {title}\n# Combined export from CelestiusAI. Run the cells top to bottom.",
        )
    ]
    pip_pkgs = sorted(
        {p for text in pip_lines for p in text.replace("# !pip install -q", "").split()}
    )
    if pip_pkgs:
        setup.append((False, f"# !pip install -q {' '.join(pip_pkgs)}"))
    setup.append(
        (
            False,
            "\n".join(imports) + "\n\ntry:\n    display\nexcept NameError:\n    display = print",
        )
    )
    if load_cell_text:
        setup.append((False, load_cell_text))
    return join_cells(setup + sections)


# --------------------------------------------------------------------------- notebook
def _src(text: str) -> list[str]:
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def build_notebook(code: str) -> str:
    """nbformat-4 JSON; ``# !cmd`` lines become real shell escapes."""
    cells: list[dict[str, Any]] = []
    for is_md, text in split_cells(code):
        if is_md:
            body = "\n".join(
                ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in text.splitlines()
            )
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(body)})
        else:
            body = "\n".join(ln[2:] if ln.startswith("# !") else ln for ln in text.splitlines())
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": _src(body),
                }
            )
    return json.dumps(
        {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=1,
    )


def colab_cell(code: str) -> str:
    """Everything as ONE pasteable Colab cell (pip line becomes a real `!pip`)."""
    lines = []
    for is_md, text in split_cells(code):
        if is_md:
            lines += [f"# {ln[2:] if ln.startswith('# ') else ln}" for ln in text.splitlines()]
        else:
            lines += [ln[2:] if ln.startswith("# !") else ln for ln in text.splitlines()]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
