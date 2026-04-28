"""Tiny vendored helpers from https://github.com/pklesk/mcts_numba_cuda.

Vendored at commit c9bb9eff20feadf3a5c64632aa2d1a463b640e47 (2026-03-20).
Copyright (c) Przemysław Klęsk; licensed under CC-BY 4.0.
See ATTRIBUTION.md for the full attribution notice.

Only ``dict_to_str`` and ``list_to_str`` are vendored — the rest of the
upstream ``utils.py`` (cpuinfo, psutil, zipfile, experiment hashing, etc.)
is unrelated to the MCTS engine and intentionally not copied.
"""

from __future__ import annotations


def dict_to_str(d, indent: int = 0) -> str:
    """Vertically formatted string representation of a dictionary."""
    indent_str = indent * " "
    dict_str = indent_str + "{"
    for i, key in enumerate(d):
        dict_str += (
            "\n" + indent_str + "  " + str(key) + ": " + str(d[key])
            + ("," if i < len(d) - 1 else "")
        )
    dict_str += "\n" + indent_str + "}"
    return dict_str


def list_to_str(l, indent: int = 0) -> str:
    """Vertically formatted string representation of a list."""
    indent_str = indent * " "
    list_str = ""
    for i, elem in enumerate(l):
        list_str += indent_str
        list_str += "[" if i == 0 else " "
        list_str += str(elem) + (",\n" if i < len(l) - 1 else "]")
    return list_str
