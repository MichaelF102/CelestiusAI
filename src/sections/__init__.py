"""Sections: page bodies that a parent page shows one at a time."""

from __future__ import annotations

import importlib

import streamlit as st

from src.core.ui_components import page_setup, render_code_export, require_dataset


def render_sections(
    title: str, icon: str, subtitle: str, sections: dict[str, str], key: str
) -> None:
    """Show a page header and a section switcher; only the chosen section runs.

    `sections` maps a label to a module name inside `src.sections`. Unlike st.tabs, the
    sections that are not selected are never executed, so switching is cheap.
    """
    page_setup(title, icon, subtitle)
    require_dataset(title)
    label = st.radio(
        "Section",
        list(sections),
        horizontal=True,
        label_visibility="collapsed",
        key=f"section_{key}",
    )
    module = importlib.import_module(f"src.sections.{sections[label]}")
    if getattr(module, "SUBTITLE", ""):
        st.caption(module.SUBTITLE)
    module.render()
    render_code_export()
