"""Chart code generation for plotly, matplotlib and seaborn from one small spec."""

from __future__ import annotations

from typing import Any

from src.core.code_options import CodeOptions, q, resolve

PLOTLY_SEQ = {
    "viridis": "px.colors.sequential.Viridis",
    "blues": "px.colors.sequential.Blues",
    "teal": "px.colors.sequential.Teal",
    "plasma": "px.colors.sequential.Plasma",
    "sunsetdark": "px.colors.sequential.Sunsetdark",
}
MPL_CMAP = {
    "viridis": "viridis",
    "blues": "Blues",
    "teal": "GnBu",
    "plasma": "plasma",
    "sunsetdark": "magma",
}


def _plotly_template(o: CodeOptions) -> str:
    return "plotly_dark" if o.template == "dark" else "plotly_white"


def _titles(spec: dict[str, Any], o: CodeOptions) -> tuple[str, str, str]:
    return (
        o.title or spec.get("title", ""),
        o.x_label or spec.get("x_label", ""),
        o.y_label or spec.get("y_label", ""),
    )


def _plotly_layout(fig_var: str, title: str, xl: str, yl: str, o: CodeOptions) -> str:
    args = [f"template={q(_plotly_template(o))}", f"width={o.width}", f"height={o.height}"]
    if title:
        args.append(f"title={q(title)}")
    if xl:
        args.append(f"xaxis_title={q(xl)}")
    if yl:
        args.append(f"yaxis_title={q(yl)}")
    return f"{fig_var}.update_layout({', '.join(args)})"


def _mpl_setup(o: CodeOptions) -> list[str]:
    style = "dark_background" if o.template == "dark" else "default"
    return [
        f"plt.style.use({q(style)})",
        f"fig, ax = plt.subplots(figsize=({o.width / 100:g}, {o.height / 100:g}))",
    ]


def _mpl_finish(title: str, xl: str, yl: str, rotate: bool = False) -> list[str]:
    out = []
    if title:
        out.append(f"ax.set_title({q(title)})")
    if xl:
        out.append(f"ax.set_xlabel({q(xl)})")
    if yl:
        out.append(f"ax.set_ylabel({q(yl)})")
    if rotate:
        out.append("plt.xticks(rotation=45, ha='right')")
    out += ["plt.tight_layout()", "plt.show()"]
    return out


def _kw(**kwargs: Any) -> str:
    """Render keyword args, skipping None values."""
    parts = []
    for key, value in kwargs.items():
        if value is None:
            continue
        parts.append(f"{key}={value.text if isinstance(value, _Raw) else q(value)}")
    return ", ".join(parts)


class _Raw:
    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def raw(text: str) -> _Raw:  # a Python expression emitted verbatim (e.g. a variable name)
    return _Raw(text)


def plot_code(kind: str, spec: dict[str, Any], options: CodeOptions | None = None) -> str:
    """Statements that draw one chart. `spec['data']` is a Python expression for the data."""
    o = resolve(options)
    title, xl, yl = _titles(spec, o)
    data = spec.get("data", "df")
    lib = o.plot_lib
    seq = PLOTLY_SEQ.get(o.palette)
    cmap = MPL_CMAP.get(o.palette)
    color = spec.get("color")

    facet = spec.get("facet")
    if lib == "plotly":
        extra = f", color_discrete_sequence={seq}" if seq and color else ""
        cont = f", color_continuous_scale={seq}" if seq else ""
        if kind == "hist":
            args = _kw(
                x=spec["x"],
                nbins=spec.get("bins", 30),
                color=color,
                marginal=spec.get("marginal"),
                facet_col=facet,
            )
            code = [f"fig = px.histogram({data}, {args}{extra})"]
        elif kind == "box":
            code = [
                f"fig = px.box({data}, {_kw(y=spec['y'], x=spec.get('x'), color=color, points=spec.get('points', 'outliers'), facet_col=facet)}{extra})"
            ]
        elif kind == "violin":
            code = [
                f"fig = px.violin({data}, {_kw(y=spec['y'], x=spec.get('x'), color=color, box=True, facet_col=facet)}{extra})"
            ]
        elif kind == "bar":
            horizontal = spec.get("horizontal")
            xk, yk = (spec["y"], spec["x"]) if horizontal else (spec["x"], spec["y"])
            code = [
                f"fig = px.bar({data}, {_kw(x=xk, y=yk, color=color, barmode='group', orientation='h' if horizontal else None, facet_col=facet)}{extra}{cont if not color else ''})"
            ]
        elif kind == "line":
            code = [
                f"fig = px.line({data}, {_kw(x=spec['x'], y=spec['y'], color=color, facet_col=facet)}{extra})"
            ]
        elif kind == "scatter":
            trend = ", trendline='ols'" if spec.get("trend") else ""
            code = [
                f"fig = px.scatter({data}, {_kw(x=spec['x'], y=spec['y'], color=color, size=spec.get('size'), facet_col=facet)}{trend}{extra})"
            ]
        elif kind == "heatmap":
            scale = f"color_continuous_scale={seq}" if seq else "color_continuous_scale='RdBu_r'"
            zr = ", zmin=-1, zmax=1" if spec.get("diverging", True) else ""
            code = [f"fig = px.imshow({data}, {scale}{zr}, text_auto='.2f')"]
        elif kind == "pie":
            code = [f"fig = px.pie({data}, {_kw(names=spec['names'], values=spec.get('values'))})"]
        else:
            raise ValueError(f"Unknown chart kind '{kind}'")
        code += [_plotly_layout("fig", title, xl, yl, o), "fig.show()"]
        return "\n".join(code)

    sns_pal = f", palette={q(cmap)}" if cmap else ""
    code = _mpl_setup(o)
    if lib == "matplotlib":
        if kind == "hist":
            code += [
                f"ax.hist(pd.to_numeric({data}[{q(spec['x'])}], errors='coerce').dropna(), bins={spec.get('bins', 30)}, color={q('#38bdf8')})"
            ]
            code += _mpl_finish(title, xl or spec["x"], yl or "count")
        elif kind in ("box", "violin"):
            fn = "boxplot" if kind == "box" else "violinplot"
            if spec.get("x"):
                code += [
                    f"groups = [g[{q(spec['y'])}].dropna().values for _, g in {data}.groupby({q(spec['x'])})]",
                    f"ax.{fn}(groups)",
                    "ax.set_xticks(range(1, len(groups) + 1))",
                    f"ax.set_xticklabels([str(k) for k, _ in {data}.groupby({q(spec['x'])})])",
                ]
            else:
                code += [f"ax.{fn}([{data}[{q(spec['y'])}].dropna().values])"]
            code += _mpl_finish(title, xl or spec.get("x", ""), yl or spec["y"])
        elif kind == "bar":
            if color:
                code += [
                    f"{data}.pivot_table(index={q(spec['x'])}, columns={q(color)}, values={q(spec['y'])}, aggfunc='sum').plot(kind={q('barh' if spec.get('horizontal') else 'bar')}, ax=ax)"
                ]
            else:
                code += [
                    f"ax.{'barh' if spec.get('horizontal') else 'bar'}({data}[{q(spec['x'])}].astype(str), {data}[{q(spec['y'])}])"
                ]
            code += _mpl_finish(
                title, xl or spec["x"], yl or spec["y"], rotate=not spec.get("horizontal")
            )
        elif kind == "line":
            if color:
                code += [
                    f"for key, grp in {data}.groupby({q(color)}):",
                    f"    ax.plot(grp[{q(spec['x'])}], grp[{q(spec['y'])}], label=str(key))",
                    "ax.legend()",
                ]
            else:
                code += [f"ax.plot({data}[{q(spec['x'])}], {data}[{q(spec['y'])}])"]
            code += _mpl_finish(title, xl or spec["x"], yl or spec["y"])
        elif kind == "scatter":
            c_arg = (
                f", c={data}[{q(color)}].astype('category').cat.codes, cmap={q(cmap or 'viridis')}"
                if color
                else ""
            )
            s_arg = (
                f", s=({data}[{q(spec['size'])}] / {data}[{q(spec['size'])}].max() * 200 + 10)"
                if spec.get("size")
                else ""
            )
            code += [
                f"ax.scatter({data}[{q(spec['x'])}], {data}[{q(spec['y'])}], alpha=0.7{c_arg}{s_arg})"
            ]
            if spec.get("trend"):
                code += [
                    f"_xy = {data}[[{q(spec['x'])}, {q(spec['y'])}]].dropna()",
                    "_m, _b = np.polyfit(_xy.iloc[:, 0], _xy.iloc[:, 1], 1)",
                    "ax.plot(_xy.iloc[:, 0], _m * _xy.iloc[:, 0] + _b, color='red')",
                ]
            code += _mpl_finish(title, xl or spec["x"], yl or spec["y"])
        elif kind == "heatmap":
            code += [
                f"im = ax.imshow({data}, cmap={q(cmap or 'RdBu_r')}{', vmin=-1, vmax=1' if spec.get('diverging', True) else ''})",
                f"ax.set_xticks(range(len({data}.columns))); ax.set_xticklabels({data}.columns, rotation=45, ha='right')",
                f"ax.set_yticks(range(len({data}.index))); ax.set_yticklabels({data}.index)",
                "fig.colorbar(im)",
            ]
            code += _mpl_finish(title, xl, yl)
        elif kind == "pie":
            vals = f"{data}[{q(spec['values'])}]" if spec.get("values") else f"{data}.iloc[:, -1]"
            code += [
                f"ax.pie({vals}, labels={data}[{q(spec['names'])}].astype(str), autopct='%1.0f%%')"
            ]
            code += _mpl_finish(title, "", "")
        else:
            raise ValueError(f"Unknown chart kind '{kind}'")
        return "\n".join(code)

    # seaborn
    if kind == "hist":
        code += [
            f"sns.histplot(data={data}, x={q(spec['x'])}, bins={spec.get('bins', 30)}, kde=True{', hue=' + q(color) if color else ''}{sns_pal if color else ''}, ax=ax)"
        ]
        code += _mpl_finish(title, xl, yl)
    elif kind in ("box", "violin"):
        fn = "boxplot" if kind == "box" else "violinplot"
        code += [
            f"sns.{fn}(data={data}, y={q(spec['y'])}{', x=' + q(spec['x']) if spec.get('x') else ''}{', hue=' + q(color) if color else ''}{sns_pal if color else ''}, ax=ax)"
        ]
        code += _mpl_finish(title, xl, yl)
    elif kind == "bar":
        horizontal = spec.get("horizontal")
        xk, yk = (spec["y"], spec["x"]) if horizontal else (spec["x"], spec["y"])
        code += [
            f"sns.barplot(data={data}, x={q(xk)}, y={q(yk)}{', hue=' + q(color) if color else ''}{sns_pal if color else ''}, estimator='sum', errorbar=None, ax=ax)"
        ]
        code += _mpl_finish(title, xl, yl, rotate=not horizontal)
    elif kind == "line":
        code += [
            f"sns.lineplot(data={data}, x={q(spec['x'])}, y={q(spec['y'])}{', hue=' + q(color) if color else ''}{sns_pal if color else ''}, ax=ax)"
        ]
        code += _mpl_finish(title, xl, yl)
    elif kind == "scatter":
        code += [
            f"sns.scatterplot(data={data}, x={q(spec['x'])}, y={q(spec['y'])}{', hue=' + q(color) if color else ''}{', size=' + q(spec['size']) if spec.get('size') else ''}{sns_pal if color else ''}, ax=ax)"
        ]
        if spec.get("trend"):
            code += [
                f"sns.regplot(data={data}, x={q(spec['x'])}, y={q(spec['y'])}, scatter=False, color='red', ax=ax)"
            ]
        code += _mpl_finish(title, xl, yl)
    elif kind == "heatmap":
        code += [
            f"sns.heatmap({data}, annot=True, fmt='.2f', cmap={q(cmap or 'RdBu_r')}{', vmin=-1, vmax=1' if spec.get('diverging', True) else ''}, ax=ax)"
        ]
        code += _mpl_finish(title, xl, yl)
    elif kind == "pie":
        vals = f"{data}[{q(spec['values'])}]" if spec.get("values") else f"{data}.iloc[:, -1]"
        code += [
            f"ax.pie({vals}, labels={data}[{q(spec['names'])}].astype(str), autopct='%1.0f%%', colors=sns.color_palette({q(cmap or 'deep')}, len({data})))"
        ]
        code += _mpl_finish(title, "", "")
    else:
        raise ValueError(f"Unknown chart kind '{kind}'")
    return "\n".join(code)
