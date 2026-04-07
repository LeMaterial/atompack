# Copyright 2026 Entalpic
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


BACKEND_ORDER = [
    "atompack",
    "atompack_ase_batch",
    "hdf5_soa",
    "lmdb_packed",
    "lmdb_pickle",
    "ase_lmdb",
    "ase_sqlite",
]
BACKEND_LABELS = {
    "atompack": "Atompack",
    "atompack_ase_batch": "ASE Batch",
    "hdf5_soa": "HDF5 SOA",
    "lmdb_packed": "LMDB Packed",
    "lmdb_pickle": "LMDB Pickle",
    "ase_lmdb": "ASE LMDB",
    "ase_sqlite": "ASE SQLite",
}
BACKEND_COLORS = {
    "atompack": "#4F2AF5",
    "atompack_ase_batch": "#6D3AF0",
    "hdf5_soa": "#B03EDC",
    "lmdb_packed": "#D14BBF",
    "lmdb_pickle": "#F05A8A",
    "ase_lmdb": "#8A7DB2",
    "ase_sqlite": "#A98BC8",
}
BACKEND_MARKERS = {
    "atompack": "o",
    "atompack_ase_batch": "h",
    "hdf5_soa": "v",
    "lmdb_packed": "D",
    "lmdb_pickle": "^",
    "ase_lmdb": "P",
    "ase_sqlite": "X",
}
BACKEND_LINESTYLES = {
    "atompack": "solid",
    "atompack_ase_batch": (0, (6, 2, 2, 2)),
    "hdf5_soa": (0, (8, 3)),
    "lmdb_packed": (0, (1, 2)),
    "lmdb_pickle": (0, (8, 3, 2, 3)),
    "ase_lmdb": (0, (2, 4)),
    "ase_sqlite": (0, (6, 3, 2, 3, 2, 3)),
}
FONT_STACK = [
    "IBM Plex Sans",
    "Source Sans 3",
    "Manrope",
    "Inter",
    "Avenir Next",
    "Liberation Sans",
    "Helvetica Neue",
    "Helvetica",
    "DejaVu Sans",
    "Arial",
    "sans-serif",
]
SUPPORTED_FORMATS = {"png", "svg"}
PREFERRED_ATOMS = 64
PREFERRED_SCALING_ATOMS = 12

BG = "#ffffff"
PANEL = "#fbfbfd"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#e5e7eb"
ATOM_COUNT_COLORS = {
    16: "#4F2AF5",
    64: "#A852D9",
    256: "#E05AA8",
}


@dataclass(frozen=True)
class BenchmarkRow:
    source_name: str
    source_path: str
    scenario: str
    backend: str
    atoms: int | None
    workers: int | None
    with_custom: bool | None
    mol_s: float
    atoms_s: float | None
    ci95_atoms_s: float
    normalized_size_bytes: float | None
    size_ratio_vs_atompack: float | None
    bytes_per_mol: float | None


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc


def _normalize_benchmark_rows(path: Path) -> list[BenchmarkRow]:
    data = _load_json(path)
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Unsupported benchmark rows in {path}")

    source_name = path.stem
    out: list[BenchmarkRow] = []
    for row in rows:
        scenario = row.get("benchmark")
        if scenario is None:
            raise ValueError(f"Missing benchmark field in {path}")
        atoms = row.get("atoms")
        atoms_s = row.get("atoms_s")
        if atoms_s is None and atoms is not None:
            atoms_s = float(row["mol_s"]) * float(atoms)
        ci95_atoms_s = row.get("ci95_atoms_s")
        if ci95_atoms_s is None and atoms is not None:
            ci95_atoms_s = float(row.get("ci95_mol_s", 0.0)) * float(atoms)
        out.append(
            BenchmarkRow(
                source_name=source_name,
                source_path=str(path),
                scenario=str(scenario),
                backend=str(row["backend"]),
                atoms=int(atoms) if atoms is not None else None,
                workers=int(row["workers"]) if row.get("workers") is not None else None,
                with_custom=bool(row["with_custom"]) if row.get("with_custom") is not None else None,
                mol_s=float(row["mol_s"]),
                atoms_s=float(atoms_s) if atoms_s is not None else None,
                ci95_atoms_s=float(ci95_atoms_s or 0.0),
                normalized_size_bytes=float(row["normalized_size_bytes"]) if row.get("normalized_size_bytes") is not None else None,
                size_ratio_vs_atompack=float(row["size_ratio_vs_atompack"]) if row.get("size_ratio_vs_atompack") is not None else None,
                bytes_per_mol=float(row["bytes_per_mol"]) if row.get("bytes_per_mol") is not None else None,
            )
        )
    return out


def load_benchmark_inputs(paths: list[Path]) -> list[BenchmarkRow]:
    if not paths:
        raise ValueError("No input paths provided")
    rows: list[BenchmarkRow] = []
    for path in paths:
        rows.extend(_normalize_benchmark_rows(path))
    if not rows:
        raise ValueError("No benchmark rows found")
    return rows


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required. Run with `uv run --group docs --project atompack-py ...`."
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 12,
            "figure.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 17,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.labelsize": 12.5,
            "axes.labelweight": "medium",
            "text.color": TEXT,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.1,
            "lines.markersize": 6,
            "figure.dpi": 160,
        }
    )
    return plt


def _backend_label(name: str) -> str:
    return BACKEND_LABELS.get(name, name.replace("_", " ").title())


def _backend_color(name: str) -> str:
    return BACKEND_COLORS.get(name, "#64748b")


def _errorbar_style(backend: str) -> dict[str, object]:
    is_atompack = backend == "atompack"
    is_ase = backend.startswith("ase_")
    return {
        "ecolor": TEXT,
        "elinewidth": 1.7 if is_atompack else 1.35 if is_ase else 1.45,
        "capsize": 4.0 if is_atompack else 3.2,
        "capthick": 1.7 if is_atompack else 1.35 if is_ase else 1.45,
        "alpha": 0.78 if is_atompack else 0.60,
        "zorder": 6,
    }


EXCLUDED_BACKENDS: set[str] = set()
WRITE_EXCLUDED_BACKENDS = {"atompack_ase_batch"}


def _present_backends(rows: list[BenchmarkRow]) -> list[str]:
    present = {row.backend for row in rows} - EXCLUDED_BACKENDS
    ordered = [backend for backend in BACKEND_ORDER if backend in present]
    extras = sorted(present - set(BACKEND_ORDER))
    return ordered + extras


def _format_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def _format_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def _filesystem_label(name: str) -> str:
    return name.replace("_", " ").upper()


def _preferred_atoms(rows: list[BenchmarkRow], *, scenario: str, workers: int | None = None) -> int | None:
    atoms = sorted(
        {
            row.atoms
            for row in rows
            if row.scenario == scenario and row.atoms is not None and (workers is None or row.workers == workers)
        }
    )
    if not atoms:
        return None
    return min(atoms, key=lambda value: abs(value - PREFERRED_ATOMS))


def _shared_atoms_by_source(rows: list[BenchmarkRow], *, scenario: str, workers: int | None = None) -> int | None:
    atoms_by_source: dict[str, set[int]] = {}
    for row in rows:
        if row.scenario != scenario or row.atoms is None:
            continue
        if workers is not None and row.workers != workers:
            continue
        atoms_by_source.setdefault(row.source_name, set()).add(row.atoms)
    if not atoms_by_source:
        return None
    shared = set.intersection(*(values for values in atoms_by_source.values()))
    if not shared:
        return None
    return min(sorted(shared), key=lambda value: abs(value - PREFERRED_ATOMS))


def _callout_text(rows: list[BenchmarkRow], *, target_backend: str = "ase_lmdb") -> str | None:
    lookup = {row.backend: row for row in rows if row.atoms_s is not None and row.atoms_s > 0}
    atompack = lookup.get("atompack")
    target = lookup.get(target_backend)
    if atompack is None or target is None or target.atoms_s is None or target.atoms_s <= 0:
        return None
    return f"{_format_speedup(atompack.atoms_s / target.atoms_s)} over {_backend_label(target_backend)}"


def _style_axes(ax, *, title: str, xlabel: str, ylabel: str = "", log_x: bool = False, log_y: bool = False) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator

    ax.set_title(title, pad=22)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", pad=6, length=0)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("medium")
    ax.set_axisbelow(True)
    _rate_fmt = FuncFormatter(lambda x, _: _format_rate(x) if x > 0 else "")
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(_rate_fmt)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(_rate_fmt)
    if log_x and log_y:
        ax.grid(True, axis="both", color=BORDER, linewidth=0.50, alpha=0.28, linestyle="--")
    elif log_x:
        ax.grid(True, axis="x", color=BORDER, linewidth=0.55, alpha=0.35, linestyle="--")
        ax.grid(False, axis="y")
    elif log_y:
        ax.grid(True, axis="y", color=BORDER, linewidth=0.55, alpha=0.35, linestyle="--")
        ax.grid(True, axis="x", color=BORDER, linewidth=0.35, alpha=0.20, linestyle=":")
    else:
        ax.grid(True, axis="y", color=BORDER, linewidth=0.55, alpha=0.35, linestyle="--")


def _add_header(fig, *, title: str, subtitle: str = "") -> None:
    fig.text(0.5, 0.945, title, ha="center", va="center", fontsize=22, fontweight="semibold", color=TEXT)
    if subtitle:
        fig.text(0.5, 0.912, subtitle, ha="center", va="center", fontsize=11, fontweight="regular", color=MUTED)


def _add_panel_note(ax, text: str) -> None:
    ax.text(
        0.98,
        1.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="medium",
        color=MUTED,
        clip_on=False,
    )


def _add_footer(fig) -> None:
    fig.text(
        0.5, 0.005, "atompack benchmark suite",
        ha="center", va="bottom", fontsize=8, color=MUTED, style="italic",
    )


def _add_panel_dividers(fig, plt, axes) -> None:
    for ax in (axes[1:] if hasattr(axes, '__len__') else []):
        pos = ax.get_position()
        x = pos.x0 - 0.008
        fig.add_artist(plt.Line2D(
            [x, x], [pos.y0, pos.y1],
            transform=fig.transFigure, color=BORDER, linewidth=0.6, alpha=0.4,
        ))


def _plot_ranked_bars(
    ax,
    rows: list[BenchmarkRow],
    *,
    title: str,
    show_xlabel: bool,
    show_errorbars: bool = False,
    title_pad: float = 22,
) -> None:
    ranked = sorted(
        [row for row in rows if row.atoms_s is not None and row.atoms_s > 0 and row.backend not in EXCLUDED_BACKENDS],
        key=lambda row: row.atoms_s or 0.0,
        reverse=True,
    )
    if not ranked:
        return

    values = [float(row.atoms_s or 0.0) for row in ranked]
    cis = [max(float(row.ci95_atoms_s), 0.0) for row in ranked]
    lower_errs = [min(ci, value * 0.95) for value, ci in zip(values, cis)]
    upper_errs = cis
    y = list(range(len(ranked)))
    colors = [_backend_color(row.backend) for row in ranked]
    edgecolors = ["#6d28d9" if row.backend == "atompack" else "none" for row in ranked]
    linewidths = [1.4 if row.backend == "atompack" else 0.0 for row in ranked]

    ax.barh(y, values, height=0.66, color=colors, edgecolor=edgecolors, linewidth=linewidths)
    ax.set_yticks(y)
    ax.set_yticklabels([_backend_label(row.backend) for row in ranked])
    for tick, row in zip(ax.get_yticklabels(), ranked):
        if row.backend == "atompack":
            tick.set_fontweight("medium")
            tick.set_color(TEXT)
    ax.invert_yaxis()
    _style_axes(ax, title=title, xlabel="atoms/s" if show_xlabel else "", log_x=True)
    ax.set_title(title, pad=title_pad)
    min_visible = min(max(value - lower, value * 0.05) for value, lower in zip(values, lower_errs))
    max_visible = max(value + upper for value, upper in zip(values, upper_errs))
    ax.set_xlim(min_visible * 0.72, max_visible * 1.65)

    if show_errorbars:
        for yi, row, value, lower, upper in zip(y, ranked, values, lower_errs, upper_errs):
            ax.errorbar(
                value,
                yi,
                xerr=[[lower], [upper]],
                fmt="none",
                barsabove=True,
                **_errorbar_style(row.backend),
            )

    for i, row, upper in zip(y, ranked, upper_errs):
        value = float(row.atoms_s or 0.0)
        label_anchor = value + upper
        ax.text(
            label_anchor * 1.05,
            i,
            _format_rate(value),
            ha="left",
            va="center",
            fontsize=11,
            fontweight="semibold" if row.backend == "atompack" else "medium",
            color=_backend_color(row.backend),
        )


def _plot_scaling_lines(
    ax,
    rows: list[BenchmarkRow],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    normalize_to_1_worker: bool,
    show_errorbars: bool = False,
) -> tuple[list, list]:
    handles = []
    labels = []
    for backend in _present_backends(rows):
        series = sorted([row for row in rows if row.backend == backend], key=lambda row: row.workers or row.atoms or 0)
        if not series:
            continue
        if normalize_to_1_worker:
            baseline = next((row for row in series if row.workers == 1 and row.atoms_s and row.atoms_s > 0), None)
            if baseline is None or baseline.atoms_s is None:
                continue
            xs = [int(row.workers or 0) for row in series]
            ys = [float((row.atoms_s or 0.0) / baseline.atoms_s) for row in series]
            lower_errs = None
            upper_errs = None
        else:
            xs = [int(row.atoms or 0) for row in series]
            ys = [float(row.atoms_s or 0.0) for row in series]
            cis = [max(float(row.ci95_atoms_s), 0.0) for row in series]
            lower_errs = [min(ci, y * 0.95) for y, ci in zip(ys, cis)]
            upper_errs = cis
        is_atompack = backend == "atompack"
        is_ase = backend.startswith("ase_")
        color = _backend_color(backend)
        line = ax.plot(
            xs,
            ys,
            color=color,
            linestyle=BACKEND_LINESTYLES.get(backend, "solid"),
            marker=BACKEND_MARKERS.get(backend, "o"),
            linewidth=3.0 if is_atompack else 1.5 if is_ase else 2.0,
            markersize=7.2 if is_atompack else 4.8 if is_ase else 5.4,
            markeredgecolor="white",
            markeredgewidth=1.0 if is_atompack else 0.5,
            alpha=1.0 if is_atompack else 0.78 if is_ase else 0.9,
            solid_capstyle="round",
            label=_backend_label(backend),
        )[0]
        if show_errorbars and not normalize_to_1_worker and lower_errs is not None and upper_errs is not None:
            ax.errorbar(
                xs,
                ys,
                yerr=[lower_errs, upper_errs],
                fmt="none",
                barsabove=True,
                **_errorbar_style(backend),
            )
        handles.append(line)
        labels.append(_backend_label(backend))

    _style_axes(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        log_x=not normalize_to_1_worker,
        log_y=not normalize_to_1_worker,
    )
    if normalize_to_1_worker:
        ax.set_ylim(bottom=0.8)
    return handles, labels


def _plot_storage_ratio_lines(ax, rows: list[BenchmarkRow], *, title: str) -> tuple[list, list]:
    from matplotlib.lines import Line2D

    backend_series: list[tuple[str, list[BenchmarkRow]]] = []
    for backend in _present_backends(rows):
        if backend == "atompack":
            continue
        series = sorted(
            [
                row
                for row in rows
                if row.backend == backend and row.atoms is not None and row.size_ratio_vs_atompack is not None
            ],
            key=lambda row: row.atoms or 0,
        )
        if series:
            backend_series.append((backend, series))
    if not backend_series:
        return [], []

    backend_series.sort(
        key=lambda item: sum(float(row.size_ratio_vs_atompack or 0.0) for row in item[1]) / len(item[1])
    )
    y_positions = list(range(len(backend_series)))

    all_values = []
    for y, (backend, series) in zip(y_positions, backend_series):
        ratios = [float(row.size_ratio_vs_atompack or 0.0) for row in series]
        all_values.extend(ratios)
        ax.hlines(y, min(ratios), max(ratios), color=BORDER, linewidth=2.0, zorder=1)
        for row in series:
            atoms = int(row.atoms or 0)
            ax.scatter(
                float(row.size_ratio_vs_atompack or 0.0),
                y,
                s=70,
                color=ATOM_COUNT_COLORS.get(atoms, _backend_color("atompack")),
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_backend_label(backend) for backend, _ in backend_series])
    ax.invert_yaxis()
    _style_axes(
        ax,
        title=title,
        xlabel="dataset size vs Atompack",
        ylabel="",
        log_x=False,
        log_y=False,
    )
    ax.grid(True, axis="x", color=BORDER, linewidth=0.55, alpha=0.35, linestyle="--")
    ax.grid(False, axis="y")
    ax.axvline(1.0, color=TEXT, linewidth=1.1, alpha=0.45, linestyle="--", zorder=0)
    ax.text(
        1.04,
        0.02,
        "Atompack = 1.0x",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )
    ax.set_xlim(0.78, max(all_values) * 1.08)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.5,
            markerfacecolor=ATOM_COUNT_COLORS[atoms],
            markeredgecolor="white",
            markeredgewidth=1.0,
            label=f"{atoms} atoms",
        )
        for atoms in sorted({int(row.atoms or 0) for row in rows if row.atoms is not None})
    ]
    labels = [handle.get_label() for handle in handles]
    return handles, labels


def _save_figure(fig, out_base: Path, formats: list[str], dpi: int) -> list[str]:
    outputs = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "facecolor": BG, "pad_inches": 0.18}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out_path, **kwargs)
        outputs.append(str(out_path))
    fig.clf()
    return outputs


def render_hero_figure(rows: list[BenchmarkRow], out_base: Path, *, title_prefix: str | None, formats: list[str], dpi: int) -> list[str]:
    primary_source = rows[0].source_name
    primary_rows = [row for row in rows if row.source_name == primary_source]
    atoms = _preferred_atoms(primary_rows, scenario="sequential")
    if atoms is None:
        return []

    sequential_rows = [row for row in primary_rows if row.scenario == "sequential" and row.atoms == atoms and row.atoms_s]
    random_rows = [row for row in primary_rows if row.scenario == "multiprocessing" and row.workers == 1 and row.atoms == atoms and row.atoms_s]
    if not sequential_rows or not random_rows:
        return []

    plt = _import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), facecolor=BG)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.06, top=0.89, wspace=0.22)
    for ax in axes:
        ax.set_facecolor(PANEL)

    _plot_ranked_bars(axes[0], sequential_rows, title="Sequential Read", show_xlabel=True, show_errorbars=True)
    _plot_ranked_bars(axes[1], random_rows, title="Random Read", show_xlabel=True, show_errorbars=True)

    _add_panel_dividers(fig, plt, axes)
    return _save_figure(fig, out_base, formats, dpi)


def render_size_figure(rows: list[BenchmarkRow], out_base: Path, *, title_prefix: str | None, formats: list[str], dpi: int) -> list[str]:
    primary_source = rows[0].source_name
    primary_rows = [row for row in rows if row.source_name == primary_source]
    sequential_rows = [row for row in primary_rows if row.scenario == "sequential" and row.atoms is not None and row.atoms_s]
    random_rows = [row for row in primary_rows if row.scenario == "multiprocessing" and row.workers == 1 and row.atoms is not None and row.atoms_s]
    if len({row.atoms for row in sequential_rows}) < 2 or len({row.atoms for row in random_rows}) < 2:
        return []

    plt = _import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.15, top=0.89, wspace=0.22)
    for ax in axes:
        ax.set_facecolor(PANEL)

    handles, labels = _plot_scaling_lines(
        axes[0],
        sequential_rows,
        title="Sequential Throughput",
        xlabel="atoms / molecule",
        ylabel="atoms/s",
        normalize_to_1_worker=False,
        show_errorbars=True,
    )
    _plot_scaling_lines(
        axes[1],
        random_rows,
        title="Random Throughput",
        xlabel="atoms / molecule",
        ylabel="atoms/s",
        normalize_to_1_worker=False,
        show_errorbars=True,
    )
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels), fontsize=11, handlelength=2.4, columnspacing=1.8,
    )

    _add_panel_dividers(fig, plt, axes)
    return _save_figure(fig, out_base, formats, dpi)


def render_filesystem_figure(rows: list[BenchmarkRow], out_base: Path, *, title_prefix: str | None, formats: list[str], dpi: int) -> list[str]:
    atoms = _shared_atoms_by_source(rows, scenario="multiprocessing", workers=1)
    if atoms is None:
        return []

    source_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row.source_name in seen:
            continue
        seen.add(row.source_name)
        source_names.append(row.source_name)

    panel_rows_by_source = {
        source: [
            row
            for row in rows
            if row.source_name == source and row.scenario == "multiprocessing" and row.workers == 1 and row.atoms == atoms and row.atoms_s
        ]
        for source in source_names
    }
    valid_sources = [source for source, panel_rows in panel_rows_by_source.items() if panel_rows]
    if len(valid_sources) < 2:
        return []

    all_values = [float(row.atoms_s or 0.0) for source in valid_sources for row in panel_rows_by_source[source]]
    x_limits = (min(all_values) * 0.72, max(all_values) * 1.45)

    plt = _import_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.91, wspace=0.18, hspace=0.30)
    flat_axes = list(axes.flatten())
    for ax in flat_axes:
        ax.set_facecolor(PANEL)

    for idx, source in enumerate(valid_sources):
        ax = flat_axes[idx]
        _plot_ranked_bars(
            ax,
            panel_rows_by_source[source],
            title=_filesystem_label(source),
            show_xlabel=idx >= 2,
            show_errorbars=True,
            title_pad=10,
        )
        ax.set_xlim(*x_limits)
    for ax in flat_axes[len(valid_sources):]:
        ax.axis("off")

    return _save_figure(fig, out_base, formats, dpi)


def render_write_figure(rows: list[BenchmarkRow], out_base: Path, *, title_prefix: str | None, formats: list[str], dpi: int) -> list[str]:
    write_rows = [
        row
        for row in rows
        if row.scenario == "write_throughput"
        and row.atoms is not None
        and row.atoms_s
        and row.backend not in WRITE_EXCLUDED_BACKENDS
    ]
    if not write_rows:
        return []

    builtins_rows = [row for row in write_rows if row.with_custom is False]
    custom_rows = [row for row in write_rows if row.with_custom is True]
    if not builtins_rows and not custom_rows:
        return []

    plt = _import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.89, wspace=0.22)
    for ax in axes:
        ax.set_facecolor(PANEL)

    panel_specs = [
        (axes[0], builtins_rows, "Built-in Properties"),
        (axes[1], custom_rows, "With Custom Properties"),
    ]
    legend_by_label: dict[str, object] = {}
    for ax, panel_rows, title in panel_specs:
        if not panel_rows:
            ax.axis("off")
            continue
        handles, labels = _plot_scaling_lines(
            ax,
            panel_rows,
            title=title,
            xlabel="atoms / molecule",
            ylabel="atoms/s",
            normalize_to_1_worker=False,
            show_errorbars=True,
        )
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)

    if legend_by_label:
        fig.legend(
            list(legend_by_label.values()),
            list(legend_by_label.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_by_label),
            fontsize=9.2,
            handlelength=1.8,
            columnspacing=1.0,
        )

    _add_panel_dividers(fig, plt, axes)
    return _save_figure(fig, out_base, formats, dpi)


def render_write_storage_figure(
    rows: list[BenchmarkRow],
    out_base: Path,
    *,
    title_prefix: str | None,
    formats: list[str],
    dpi: int,
) -> list[str]:
    storage_rows = [
        row
        for row in rows
        if row.scenario == "write_storage"
        and row.atoms is not None
        and row.size_ratio_vs_atompack is not None
        and row.backend not in WRITE_EXCLUDED_BACKENDS
    ]
    if not storage_rows:
        return []

    builtins_rows = [row for row in storage_rows if row.with_custom is False]
    custom_rows = [row for row in storage_rows if row.with_custom is True]
    if not builtins_rows and not custom_rows:
        return []

    plt = _import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), facecolor=BG)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.89, wspace=0.22)
    for ax in axes:
        ax.set_facecolor(PANEL)

    panel_specs = [
        (axes[0], builtins_rows, "Built-in Properties"),
        (axes[1], custom_rows, "With Custom Properties"),
    ]
    legend_by_label: dict[str, object] = {}
    for ax, panel_rows, title in panel_specs:
        if not panel_rows:
            ax.axis("off")
            continue
        handles, labels = _plot_storage_ratio_lines(ax, panel_rows, title=title)
        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(label, handle)

    if legend_by_label:
        fig.legend(
            list(legend_by_label.values()),
            list(legend_by_label.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_by_label),
            fontsize=9.2,
            handlelength=1.8,
            columnspacing=1.0,
        )

    _add_panel_dividers(fig, plt, axes)
    return _save_figure(fig, out_base, formats, dpi)


def generate_story_report(
    inputs: list[Path],
    out_dir: Path,
    *,
    title_prefix: str | None,
    formats: list[str],
    dpi: int,
) -> dict[str, object]:
    rows = load_benchmark_inputs(inputs)
    figures = [
        {"key": "story_read_hero", "outputs": render_hero_figure(rows, out_dir / "story_read_hero", title_prefix=title_prefix, formats=formats, dpi=dpi)},
        {"key": "story_size_scaling", "outputs": render_size_figure(rows, out_dir / "story_size_scaling", title_prefix=title_prefix, formats=formats, dpi=dpi)},
        {"key": "story_random_filesystems", "outputs": render_filesystem_figure(rows, out_dir / "story_random_filesystems", title_prefix=title_prefix, formats=formats, dpi=dpi)},
        {"key": "story_write_overview", "outputs": render_write_figure(rows, out_dir / "story_write_overview", title_prefix=title_prefix, formats=formats, dpi=dpi)},
        {"key": "story_write_storage", "outputs": render_write_storage_figure(rows, out_dir / "story_write_storage", title_prefix=title_prefix, formats=formats, dpi=dpi)},
    ]
    for item in figures:
        item["status"] = "produced" if item["outputs"] else "skipped"
    manifest = {
        "inputs": [str(path) for path in inputs],
        "figures": figures,
    }
    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _parse_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [item for item in formats if item not in SUPPORTED_FORMATS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported formats: {', '.join(invalid)}")
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required")
    return formats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render launch-focused benchmark figures from benchmark.py JSON outputs")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", type=str, default=None)
    parser.add_argument("--formats", type=_parse_formats, default=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args(argv)

    manifest = generate_story_report(
        args.inputs,
        args.out_dir,
        title_prefix=args.title_prefix,
        formats=args.formats,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "figures": {item["key"]: item["status"] for item in manifest["figures"]},
                "out_dir": str(args.out_dir),
                "manifest": manifest["manifest_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
