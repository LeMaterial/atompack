# Copyright 2026 Entalpic
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


BACKEND_ORDER = [
    "atompack",
    "hdf5_soa",
    "lmdb_packed",
    "lmdb_pickle",
    "ase_sqlite",
    "ase_lmdb",
    "atompack_ase_batch",
]
DEFAULT_BACKENDS = set(BACKEND_ORDER)
WRITE_EXCLUDED_BACKENDS = {"atompack_ase_batch"}
METHOD_ORDER = ["loop", "batch", "flat"]
SUPPORTED_FORMATS = {"svg", "png"}
SUPPORTED_REPORT_MODES = {"full", "blog"}
READ_OVERVIEW_SCENARIOS = [
    ("sequential", "Sequential Read"),
    ("random", "Random Read"),
    ("multiprocessing", "Multiprocessing"),
]
BLOG_PREFERRED_ATOMS = 64

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
LEMAT_BG = "#ffffff"
LEMAT_PANEL = "#fafbfc"
LEMAT_PANEL_SOFT = "#f2f4f7"
LEMAT_TEXT = "#111827"
LEMAT_MUTED = "#6b7280"
LEMAT_BORDER = "#e5e7eb"

# Vibrant, web-friendly palette — each pair is (fill, dark-accent).
LEMAT_PRIMARY = "#7c3aed"      # atompack — vivid purple
LEMAT_PRIMARY_DARK = "#6d28d9"
LEMAT_AMBER = "#d97706"        # lmdb_packed
LEMAT_ROSE = "#e11d48"         # lmdb_pickle
LEMAT_BLUE = "#2563eb"         # hdf5_soa
LEMAT_TEAL = "#0f766e"
LEMAT_SLATE = "#475569"        # ase_lmdb
LEMAT_STEEL = "#64748b"        # ase_sqlite
LEMAT_FUCHSIA = "#c026d3"      # atompack_ase_batch

BACKEND_COLORS = {
    "atompack": LEMAT_PRIMARY,
    "lmdb_packed": LEMAT_AMBER,
    "lmdb_pickle": LEMAT_ROSE,
    "hdf5_soa": LEMAT_BLUE,
    "ase_lmdb": LEMAT_SLATE,
    "ase_sqlite": LEMAT_STEEL,
    "atompack_ase_batch": LEMAT_FUCHSIA,
}

# Distinct line dashes — redundant encoding so plots survive grayscale print.
# Format: (offset, (on, off, ...))  or string name.
BACKEND_DASHES: dict[str, Any] = {
    "atompack": "solid",
    "lmdb_packed": (0, (1, 1.5)),       # dotted
    "lmdb_pickle": (0, (5, 2, 1, 2)),   # dash-dot
    "hdf5_soa": (0, (8, 3)),            # long dash
    "ase_lmdb": (0, (3, 1, 1, 1)),      # dash-dot-dot
    "ase_sqlite": (0, (2, 3)),           # sparse dots
    "atompack_ase_batch": (0, (6, 2, 2, 2)),
}

# Distinct marker shapes per backend.
BACKEND_MARKERS = {
    "atompack": "o",
    "lmdb_packed": "D",
    "lmdb_pickle": "^",
    "hdf5_soa": "v",
    "ase_lmdb": "P",
    "ase_sqlite": "X",
    "atompack_ase_batch": "h",
}
BACKEND_LABELS = {
    "atompack": "Atompack",
    "hdf5_soa": "HDF5 SOA",
    "lmdb_packed": "LMDB Packed",
    "lmdb_pickle": "LMDB Pickle",
    "ase_lmdb": "ASE LMDB",
    "ase_sqlite": "ASE SQLite",
    "atompack_ase_batch": "ASE Batch",
}

METHOD_COLORS = {
    "loop": "#818cf8",
    "batch": LEMAT_PRIMARY,
    "flat": LEMAT_FUCHSIA,
}
THREAD_COLORS = [
    LEMAT_PRIMARY,
    LEMAT_BLUE,
    LEMAT_FUCHSIA,
    LEMAT_ROSE,
    LEMAT_AMBER,
    LEMAT_TEAL,
    LEMAT_STEEL,
]

# Font stack — first available wins.  Liberation Sans is metrically
# identical to Helvetica and ships on most Linux distros.
_FONT_STACK = [
    "Avenir Next",
    "Manrope",
    "IBM Plex Sans",
    "Inter",
    "Source Sans 3",
    "Liberation Sans",
    "Helvetica Neue",
    "Helvetica",
    "DejaVu Sans",
    "Arial",
    "sans-serif",
]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from {path}: {exc}") from exc


def _rows_schema(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("Unsupported benchmark JSON schema: empty row list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Unsupported benchmark JSON schema: expected list[dict]")

    benchmarks = {row.get("benchmark") for row in rows if row.get("benchmark") is not None}
    if benchmarks:
        if benchmarks == {"atompack_batch_api"}:
            return "batch"
        if benchmarks <= {"write_throughput", "write_scaling", "write_batch_scaling"}:
            return "write"
        return "benchmark"

    row_keys = set(rows[0])
    if {"atoms_per_molecule", "backend", "n_workers", "mean_mol_s"} <= row_keys:
        return "scaling"

    raise ValueError("Unsupported benchmark JSON schema")


def _detect_input_schema(data: Any) -> str:
    if isinstance(data, list):
        return _rows_schema(data)

    if isinstance(data, dict):
        if "open_connection_memory" in data and "streaming_read_memory" in data:
            return "memory"
        if "results" in data and isinstance(data["results"], list):
            return _rows_schema(data["results"])

    raise ValueError(
        "Unsupported benchmark JSON schema. Supported inputs are benchmark.py, "
        "scaling_benchmark.py, memory_benchmark.py, atompack_batch_benchmark.py, "
        "and write_benchmark.py outputs."
    )


def _require_keys(row: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"Missing required keys for {context}: {', '.join(missing)}")


def _normalize_benchmark_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        _require_keys(row, ["benchmark", "backend", "mol_s"], "benchmark row")
        atoms = row.get("atoms")
        workers = row.get("workers")
        atoms_s = row.get("atoms_s")
        ci95_atoms_s = row.get("ci95_atoms_s")
        if atoms_s is None and atoms is not None:
            atoms_s = float(row["mol_s"]) * float(atoms)
        if ci95_atoms_s is None and atoms is not None:
            ci95_atoms_s = float(row.get("ci95_mol_s", 0.0)) * float(atoms)
        out.append(
            {
                "source": "benchmark",
                "scenario": str(row["benchmark"]),
                "backend": str(row["backend"]),
                "atoms": int(atoms) if atoms is not None else None,
                "workers": int(workers) if workers is not None else None,
                "mol_s": float(row["mol_s"]),
                "atoms_s": float(atoms_s) if atoms_s is not None else None,
                "ci95_atoms_s": float(ci95_atoms_s) if ci95_atoms_s is not None else 0.0,
                "sample": int(row["sample"]) if row.get("sample") is not None else None,
                "n_mols": int(row["n_mols"]) if row.get("n_mols") is not None else None,
            }
        )
    return out


def _attach_row_source(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    source_path = str(path)
    source_name = path.stem
    for row in rows:
        row["source_path"] = source_path
        row["source_name"] = source_name
    return rows


def _normalize_scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        _require_keys(
            row,
            ["atoms_per_molecule", "backend", "n_workers", "mean_mol_s"],
            "scaling row",
        )
        atoms = int(row["atoms_per_molecule"])
        ci95_mol_s = float(row.get("ci95", 0.0))
        out.append(
            {
                "source": "scaling",
                "scenario": "read_scaling",
                "backend": str(row["backend"]),
                "atoms": atoms,
                "workers": int(row["n_workers"]),
                "mol_s": float(row["mean_mol_s"]),
                "atoms_s": float(row.get("mean_atoms_s", float(row["mean_mol_s"]) * atoms)),
                "ci95_atoms_s": float(row.get("ci95_atoms_s", ci95_mol_s * atoms)),
                "sample": None,
                "n_mols": int(row["num_molecules"]) if row.get("num_molecules") is not None else None,
            }
        )
    return out


def _private_total(row: dict[str, Any], per_worker: bool = False) -> int:
    key = "private_total_kib_per_worker" if per_worker else "private_total_kib"
    if row.get(key) is not None:
        return int(row[key])
    clean_key = "private_clean_kib_per_worker" if per_worker else "private_clean_kib"
    dirty_key = "private_dirty_kib_per_worker" if per_worker else "private_dirty_kib"
    return int(row.get(clean_key, 0)) + int(row.get(dirty_key, 0))


def _normalize_memory_rows(
    rows: list[dict[str, Any]],
    *,
    per_worker: bool,
) -> list[dict[str, Any]]:
    out = []
    suffix = "_per_worker" if per_worker else ""
    for row in rows:
        _require_keys(row, ["backend"], "memory row")
        out.append(
            {
                "backend": str(row["backend"]),
                "workers": int(row["workers"]) if row.get("workers") is not None else None,
                "rss_kib": int(row.get(f"rss_kib{suffix}", 0)),
                "private_total_kib": _private_total(row, per_worker=per_worker),
                "rss_anon_kib": int(row.get(f"rss_anon_kib{suffix}", 0)),
                "rss_file_kib": int(row.get(f"rss_file_kib{suffix}", 0)),
            }
        )
    return out


def _normalize_batch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        _require_keys(
            row,
            ["benchmark", "method", "batch_size", "threads", "mol_s"],
            "batch row",
        )
        out.append(
            {
                "method": str(row["method"]),
                "batch_size": int(row["batch_size"]),
                "threads": str(row["threads"]),
                "mol_s": float(row["mol_s"]),
                "ci95_mol_s": float(row.get("ci95_mol_s", 0.0)),
                "atoms": int(row["atoms"]) if row.get("atoms") is not None else None,
                "n_mols": int(row["n_mols"]) if row.get("n_mols") is not None else None,
            }
        )
    return out


def _normalize_write_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        _require_keys(
            row,
            ["benchmark", "backend", "atoms", "n_mols", "mol_s"],
            "write row",
        )
        benchmark = str(row["benchmark"])
        if benchmark not in {"write_throughput", "write_scaling", "write_batch_scaling"}:
            raise ValueError(f"Unsupported write benchmark type: {benchmark}")
        out.append(
            {
                "source": "write",
                "scenario": benchmark,
                "backend": str(row["backend"]),
                "atoms": int(row["atoms"]),
                "n_mols": int(row["n_mols"]),
                "n_mols_requested": int(row.get("n_mols_requested", row["n_mols"])),
                "with_custom": bool(row.get("with_custom", False)),
                "trials": int(row.get("trials", 1)),
                "mol_s": float(row["mol_s"]),
                "ci95_mol_s": float(row.get("ci95_mol_s", 0.0)),
                "size_bytes": int(row.get("size_bytes", 0)),
                "batch_size": (
                    int(row["batch_size"])
                    if row.get("batch_size") is not None
                    else None
                ),
            }
        )
    return out


def load_inputs(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("No input paths provided")

    bundle = {
        "benchmark_rows": [],
        "scaling_rows": [],
        "memory_rows_open": [],
        "memory_rows_stream": [],
        "batch_rows": [],
        "write_rows": [],
        "sources": defaultdict(list),
    }

    for path in paths:
        data = _load_json(path)
        schema = _detect_input_schema(data)
        bundle["sources"][schema].append(str(path))

        if schema == "memory":
            bundle["memory_rows_open"].extend(
                _attach_row_source(
                    _normalize_memory_rows(data["open_connection_memory"], per_worker=True),
                    path,
                )
            )
            bundle["memory_rows_stream"].extend(
                _attach_row_source(
                    _normalize_memory_rows(data["streaming_read_memory"], per_worker=False),
                    path,
                )
            )
            continue

        rows = data["results"] if isinstance(data, dict) else data
        if schema == "benchmark":
            bundle["benchmark_rows"].extend(_attach_row_source(_normalize_benchmark_rows(rows), path))
        elif schema == "scaling":
            bundle["scaling_rows"].extend(_attach_row_source(_normalize_scaling_rows(rows), path))
        elif schema == "batch":
            bundle["batch_rows"].extend(_attach_row_source(_normalize_batch_rows(rows), path))
        elif schema == "write":
            bundle["write_rows"].extend(_attach_row_source(_normalize_write_rows(rows), path))

    bundle["sources"] = dict(bundle["sources"])
    return bundle


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plot_benchmark_report.py. "
            "Run with `uv run --group docs --project atompack-py ...`."
        ) from exc

    # Apply seaborn whitegrid base, then override with our branding.
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", font_scale=1.0, rc={"grid.linestyle": "--"})
    except ModuleNotFoundError:
        pass  # graceful fallback — seaborn is optional

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_STACK,
            "font.weight": "regular",
            "figure.facecolor": LEMAT_BG,
            "savefig.facecolor": LEMAT_BG,
            "axes.facecolor": LEMAT_PANEL,
            "axes.edgecolor": LEMAT_BORDER,
            "axes.labelcolor": LEMAT_TEXT,
            "axes.titlecolor": LEMAT_TEXT,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.color": LEMAT_MUTED,
            "ytick.color": LEMAT_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": LEMAT_BORDER,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            "text.color": LEMAT_TEXT,
            "legend.frameon": False,
            "legend.labelcolor": LEMAT_TEXT,
            "legend.fontsize": 10.5,
            "font.size": 12,
            "axes.labelsize": 12.5,
            "axes.labelweight": "medium",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "figure.dpi": 150,
        }
    )
    return plt


def _format_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def _format_mib(value_kib: int) -> str:
    return f"{value_kib / 1024.0:.1f} MiB"


def _mib(value_kib: int) -> float:
    return float(value_kib) / 1024.0


def _sorted_backends(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["backend"] for row in rows}
    ordered = [backend for backend in BACKEND_ORDER if backend in present]
    extras = sorted(present - set(BACKEND_ORDER))
    return ordered + extras


def _build_title(prefix: str | None, base: str) -> str:
    return f"{prefix} - {base}" if prefix else base


def _backend_color(name: str) -> str:
    return BACKEND_COLORS.get(name, LEMAT_SLATE)


def _backend_dash(name: str):
    return BACKEND_DASHES.get(name, "solid")


def _backend_marker(name: str) -> str:
    return BACKEND_MARKERS.get(name, "o")


def _backend_label(name: str) -> str:
    return BACKEND_LABELS.get(name, name.replace("_", " ").title())


def _backend_style(name: str) -> dict[str, Any]:
    """Full style dict suitable for ``ax.plot()``."""
    is_atompack = name == "atompack"
    is_ase = name.startswith("ase_")
    return {
        "color": _backend_color(name),
        "linestyle": _backend_dash(name),
        "marker": _backend_marker(name),
        "linewidth": 3.0 if is_atompack else 1.5 if is_ase else 1.9,
        "markersize": 7.4 if is_atompack else 4.8 if is_ase else 5.2,
        "markeredgecolor": "white",
        "markeredgewidth": 1.2 if is_atompack else 0.5,
        "alpha": 1.0 if is_atompack else 0.75 if is_ase else 0.88,
        "zorder": 4 if is_atompack else 3,
        "solid_capstyle": "round",
    }


def _backend_errorbar_style(name: str) -> dict[str, Any]:
    """Style dict for ``ax.errorbar()`` — excludes keys it doesn't accept."""
    s = _backend_style(name)
    for key in ("solid_capstyle", "markeredgecolor", "markeredgewidth"):
        s.pop(key, None)
    return s


def _thread_color(threads: str, order: list[str]) -> str:
    try:
        idx = order.index(threads)
    except ValueError:
        idx = 0
    return THREAD_COLORS[idx % len(THREAD_COLORS)]


def _is_logworthy(values: list[float], *, threshold: float = 10.0) -> bool:
    positives = [value for value in values if value > 0]
    if len(positives) < 2:
        return False
    return max(positives) / min(positives) >= threshold


def _filesystem_label(name: str) -> str:
    return name.replace("_", " ").upper()


def _format_speedup(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}x"
    return f"{value:.1f}x"


def _setup_axis(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    ax.set_title(title, pad=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", color=LEMAT_BORDER, linewidth=0.5, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=6)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")


def _callout(ax, text: str) -> None:
    ax.text(
        0.98,
        0.94,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="semibold",
        color=LEMAT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.18",
            "facecolor": LEMAT_PANEL_SOFT,
            "edgecolor": LEMAT_BORDER,
            "linewidth": 0.6,
            "alpha": 0.96,
        },
    )


def _figure_subtitle(fig, text: str, *, y: float = 0.94) -> None:
    fig.text(
        0.5,
        y,
        text,
        ha="center",
        va="center",
        fontsize=11.2,
        color=LEMAT_MUTED,
        fontweight="medium",
    )


def _collect_legend_items(axes: list[Any]) -> tuple[list[Any], list[str]]:
    seen: set[str] = set()
    handles_out = []
    labels_out = []
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label in seen:
                continue
            seen.add(label)
            handles_out.append(handle)
            labels_out.append(label)
    return handles_out, labels_out


def _save_figure(fig, out_base: Path, formats: list[str], dpi: int) -> list[str]:
    outputs = []
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = out_base.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "facecolor": LEMAT_BG, "pad_inches": 0.3}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out_path, **kwargs)
        outputs.append(str(out_path))
    fig.clf()
    return outputs


def _subplots_row(
    plt,
    n_panels: int,
    *,
    width: float,
    height: float,
    constrained_layout: bool = True,
):
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width, height),
        squeeze=False,
        constrained_layout=constrained_layout,
    )
    return fig, list(axes[0])


def _subplots_grid(
    plt,
    nrows: int,
    ncols: int,
    *,
    width: float,
    height: float,
    constrained_layout: bool = True,
):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        squeeze=False,
        constrained_layout=constrained_layout,
    )
    return fig, [ax for row in axes for ax in row]


def _figure_status(
    *,
    key: str,
    kind: str,
    sources: list[str],
    rows: list[dict[str, Any]] | None = None,
    no_batch_api: bool = False,
) -> dict[str, Any]:
    if kind == "batch_api" and no_batch_api:
        return {
            "key": key,
            "kind": kind,
            "status": "skipped",
            "reason": "disabled by --no-batch-api",
            "sources": sources,
            "outputs": [],
        }
    if rows is not None and not rows:
        return {
            "key": key,
            "kind": kind,
            "status": "skipped",
            "reason": f"no {kind} rows found in the provided inputs",
            "sources": sources,
            "outputs": [],
        }
    return {
        "key": key,
        "kind": kind,
        "status": "produced",
        "reason": "",
        "sources": sources,
        "outputs": [],
    }


def _read_overview_panels(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    panels = []
    for scenario, label in READ_OVERVIEW_SCENARIOS:
        if scenario == "random":
            # Random read = multiprocessing with 1 worker (single-threaded random access).
            panel_rows = [
                row for row in rows
                if row["scenario"] == "multiprocessing"
                and row["workers"] == 1
                and row["atoms_s"] is not None
            ]
        else:
            panel_rows = [
                row
                for row in rows
                if row["scenario"] == scenario and row["atoms_s"] is not None
            ]
        if not panel_rows:
            continue
        if scenario == "multiprocessing":
            max_workers = max(int(row["workers"]) for row in panel_rows if row["workers"] is not None)
            panel_rows = [row for row in panel_rows if row["workers"] == max_workers]
            label = f"{label} ({max_workers} workers)"
        panels.append((scenario, label, panel_rows))
    return panels


def _comparison_note(rows: list[dict[str, Any]], *, x_key: str, y_key: str) -> str | None:
    if not rows:
        return None
    latest_x = max(row[x_key] for row in rows if row.get(x_key) is not None)
    latest_rows = [row for row in rows if row.get(x_key) == latest_x]
    if len(latest_rows) < 2:
        return None
    ranked = sorted(latest_rows, key=lambda row: row[y_key], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    if runner_up[y_key] <= 0:
        return f"{best['backend']} leads at {latest_x}"
    speedup = best[y_key] / runner_up[y_key]
    return f"{best['backend']} leads by {speedup:.1f}x at {latest_x}"


def _preferred_atoms(rows: list[dict[str, Any]], *, scenario: str) -> int | None:
    atoms_values = sorted(
        {
            int(row["atoms"])
            for row in rows
            if row.get("scenario") == scenario and row.get("atoms") is not None
        }
    )
    if not atoms_values:
        return None
    return min(atoms_values, key=lambda value: abs(value - BLOG_PREFERRED_ATOMS))


def _shared_atoms_by_source(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    workers: int | None = None,
) -> int | None:
    atoms_by_source: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("scenario") != scenario or row.get("atoms") is None:
            continue
        if workers is not None and row.get("workers") != workers:
            continue
        source_name = str(row.get("source_name") or "")
        atoms_by_source[source_name].add(int(row["atoms"]))
    if not atoms_by_source:
        return None
    shared = set.intersection(*(values for values in atoms_by_source.values() if values))
    if not shared:
        return None
    shared_values = sorted(shared)
    return min(shared_values, key=lambda value: abs(value - BLOG_PREFERRED_ATOMS))


def _panel_callout(rows: list[dict[str, Any]], *, target_backend: str = "ase_lmdb") -> str | None:
    by_backend = {
        row["backend"]: row
        for row in rows
        if row.get("atoms_s") is not None and row.get("atoms_s", 0) > 0
    }
    atompack_row = by_backend.get("atompack")
    target_row = by_backend.get(target_backend)
    if atompack_row is None or target_row is None or target_row["atoms_s"] <= 0:
        return None
    speedup = atompack_row["atoms_s"] / target_row["atoms_s"]
    return f"{_format_speedup(speedup)} over {_backend_label(target_backend)}"


def _speedup_rows(rows: list[dict[str, Any]], *, atoms: int) -> list[dict[str, Any]]:
    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("scenario") == "multiprocessing" and row.get("atoms") == atoms:
            by_backend[str(row["backend"])].append(row)

    speedup_rows = []
    for backend, series in by_backend.items():
        baseline = next((row for row in series if row.get("workers") == 1 and row.get("atoms_s", 0) > 0), None)
        if baseline is None:
            continue
        base_rate = float(baseline["atoms_s"])
        for row in series:
            value = float(row.get("atoms_s", 0.0))
            if value <= 0:
                continue
            speedup_rows.append(
                {
                    **row,
                    "speedup": value / base_rate,
                    "baseline_atoms_s": base_rate,
                }
            )
    return speedup_rows


def _blog_random_rows(rows: list[dict[str, Any]], *, atoms: int) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("scenario") == "multiprocessing"
        and row.get("workers") == 1
        and row.get("atoms") == atoms
        and row.get("atoms_s") is not None
    ]


def _annotate_series_end(ax, xs: list[float], ys: list[float], label: str, color: str) -> None:
    if not xs or not ys:
        return
    ax.annotate(
        label,
        xy=(xs[-1], ys[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=10,
        color=color,
        va="center",
        ha="left",
    )


def _draw_rounded_barh(
    ax, y: float, width: float, height: float, *,
    color: str, alpha: float = 1.0, radius: float = 0.08,
    edgecolor: str = "none", linewidth: float = 0, label: str | None = None,
    zorder: int = 3,
) -> Any:
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (0, y - height / 2), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
        alpha=alpha, zorder=zorder, label=label,
    )
    ax.add_patch(patch)
    return patch


def _draw_rounded_bar(
    ax, x: float, bottom: float, width: float, height: float, *,
    color: str, alpha: float = 1.0, radius: float = 0.06,
    edgecolor: str = "none", linewidth: float = 0, label: str | None = None,
    zorder: int = 3,
) -> Any:
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x, bottom), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor=edgecolor, linewidth=linewidth,
        alpha=alpha, zorder=zorder, label=label,
    )
    ax.add_patch(patch)
    return patch


def _plot_ranked_panel(
    ax,
    rows: list[dict[str, Any]],
    *,
    title: str,
    value_key: str = "atoms_s",
    x_label: str = "atoms/s",
    log_x: bool = True,
    callout: str | None = None,
) -> None:
    ranked = sorted(
        [row for row in rows if row.get(value_key) is not None and row.get(value_key, 0) > 0],
        key=lambda row: row[value_key],
        reverse=True,
    )
    if not ranked:
        return

    values = [float(row[value_key]) for row in ranked]
    y_positions = list(range(len(ranked)))
    max_val = max(values)

    for idx, row in enumerate(ranked):
        backend = str(row["backend"])
        value = float(row[value_key])
        color = _backend_color(backend)
        is_hero = backend == "atompack"
        _draw_rounded_barh(
            ax,
            idx,
            value,
            0.58,
            color=color,
            alpha=1.0 if is_hero else 0.82,
            edgecolor=LEMAT_PRIMARY_DARK if is_hero else "none",
            linewidth=1.5 if is_hero else 0,
            radius=0.12,
            zorder=4 if is_hero else 3,
        )
        ax.text(
            value * (1.02 if log_x else 1.01),
            idx,
            _format_rate(value),
            va="center",
            ha="left",
            fontsize=11,
            fontweight="semibold" if is_hero else "regular",
            color=color,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_backend_label(str(row["backend"])) for row in ranked])
    for tick_label, row in zip(ax.get_yticklabels(), ranked):
        if row["backend"] == "atompack":
            tick_label.set_fontweight("semibold")
            tick_label.set_color(LEMAT_TEXT)
    ax.invert_yaxis()
    _setup_axis(ax, title=title, xlabel=x_label, ylabel="")
    ax.grid(True, axis="x", color=LEMAT_BORDER, linewidth=0.5, alpha=0.4, linestyle="--")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(left=min(values) * 0.75, right=max_val * 1.55)
    else:
        ax.set_xlim(left=0, right=max_val * 1.15)
    if callout:
        _callout(ax, callout)


def _label_line_series(ax, xs: list[float], ys: list[float], label: str, color: str) -> None:
    if not xs or not ys:
        return
    ax.annotate(
        label,
        xy=(xs[-1], ys[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="medium",
        color=color,
        va="center",
        ha="left",
    )


def _label_series_ends(
    ax,
    series: list[tuple[float, float, str, str]],
    *,
    x_offset_points: float = 8.0,
    min_gap_px: float = 14.0,
) -> None:
    if not series:
        return

    transformed = []
    for x, y, label, color in series:
        x_disp, y_disp = ax.transData.transform((x, y))
        transformed.append([x, y, label, color, x_disp, y_disp])
    transformed.sort(key=lambda item: item[5])

    placed: list[list[float | str]] = []
    last_y = -float("inf")
    for item in transformed:
        if item[5] < last_y + min_gap_px:
            item[5] = last_y + min_gap_px
        last_y = item[5]
        placed.append(item)

    dpi = ax.figure.dpi if ax.figure is not None else 150.0
    for x, y, label, color, _x_disp, y_disp in placed:
        original_disp = ax.transData.transform((x, y))[1]
        dy_points = (y_disp - original_disp) * 72.0 / dpi
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x_offset_points, dy_points),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="medium",
            color=color,
            va="center",
            ha="left",
        )


def _plot_hbar_panel(ax, panel_rows: list[dict[str, Any]], *, title: str) -> None:
    """Horizontal bar chart for single-x-value panels — much cleaner than dot columns."""
    backends = _sorted_backends(panel_rows)
    by_backend = {row["backend"]: row for row in panel_rows}

    ys = list(range(len(backends)))
    bar_height = 0.55
    values = []
    errors = []
    colors = []
    for backend in backends:
        row = by_backend.get(backend, {})
        values.append(row.get("atoms_s", 0))
        errors.append(row.get("ci95_atoms_s", 0))
        colors.append(_backend_color(backend))

    max_val = max(values) if values else 1

    for i, (backend, val, err, color) in enumerate(zip(backends, values, errors, colors)):
        is_hero = backend == "atompack"
        _draw_rounded_barh(
            ax, i, val, bar_height,
            color=color,
            alpha=1.0 if is_hero else 0.82,
            edgecolor=LEMAT_PRIMARY_DARK if is_hero else "none",
            linewidth=1.5 if is_hero else 0,
            label=backend,
            radius=0.12,
            zorder=4 if is_hero else 3,
        )
        if err > 0:
            ax.errorbar(
                val, i, xerr=err,
                fmt="none", ecolor=LEMAT_MUTED, elinewidth=0.8,
                capsize=2.5, alpha=0.45, zorder=5,
            )
        is_best = val == max_val
        ax.text(
            val + max_val * 0.02, i,
            _format_rate(val),
            va="center", ha="left", fontsize=9,
            fontweight="semibold" if is_best else "regular",
            color=LEMAT_TEXT if is_best else LEMAT_MUTED,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(backends)
    ax.invert_yaxis()
    ax.set_title(title, pad=18)
    ax.set_xlabel("atoms/s")
    ax.grid(True, axis="x", color=LEMAT_BORDER, linewidth=0.5, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=6)
    ax.set_xlim(right=max_val * 1.18)


SCENARIO_MARKERS = {
    "sequential": "o",
    "random": "D",
    "compression": "s",
    "compression_custom": "^",
    "multiprocessing": "P",
    "tensor_pipeline": "X",
}
SCENARIO_SHORT_LABELS = {
    "sequential": "Sequential",
    "random": "Random",
    "compression": "Compressed",
    "compression_custom": "Compr + Custom",
    "multiprocessing": "Multiprocessing",
    "tensor_pipeline": "Tensor Pipeline",
}


def _plot_read_overview(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    panels = _read_overview_panels(rows)
    if not panels:
        return []

    # Collect all backends and scenarios across panels.
    all_backends = []
    seen = set()
    for _, _, panel_rows in panels:
        for b in _sorted_backends(panel_rows):
            if b not in seen:
                all_backends.append(b)
                seen.add(b)
    n_backends = len(all_backends)
    n_scenarios = len(panels)

    use_hbar = all(
        len({row["atoms"] for row in pr if row["atoms"] is not None}) <= 1
        for _, _, pr in panels
    )

    # When every panel is single-atom-count, use the grouped bar chart.
    if use_hbar:
        return _plot_read_overview_dotplot(
            panels, all_backends,
            title_prefix=title_prefix, width=width,
            formats=formats, dpi=dpi, out_base=out_base,
        )

    plt = _import_plotting()
    effective_width = max(width, n_scenarios * 5.5)
    if use_hbar:
        panel_height = max(4.5, 0.55 * n_backends + 2.5)
        fig, axes = _subplots_row(plt, n_scenarios, width=effective_width, height=panel_height)
    else:
        fig, axes = _subplots_row(plt, n_scenarios, width=effective_width, height=max(5.5, effective_width * 0.40))

    for ax, (_, title, panel_rows) in zip(axes, panels):
        if use_hbar:
            _plot_hbar_panel(ax, panel_rows, title=title)
        else:
            grouped = defaultdict(list)
            for row in panel_rows:
                grouped[row["backend"]].append(row)

            for backend in _sorted_backends(panel_rows):
                series = sorted(grouped[backend], key=lambda row: row["atoms"] or 0)
                xs = [row["atoms"] for row in series if row["atoms"] is not None]
                ys = [row["atoms_s"] for row in series if row["atoms_s"] is not None]
                yerr = [row["ci95_atoms_s"] for row in series if row["atoms_s"] is not None]
                if not xs or not ys:
                    continue
                style = _backend_errorbar_style(backend)
                ax.errorbar(xs, ys, yerr=yerr, label=backend,
                            capsize=2.5, elinewidth=0.8, **style)
                if backend == "atompack":
                    _annotate_series_end(ax, xs, ys, "atompack", style["color"])

            _setup_axis(ax, title=title, xlabel="atoms", ylabel="atoms/s", log_y=True)
            xticks = sorted({row["atoms"] for row in panel_rows if row["atoms"] is not None})
            if xticks:
                ax.set_xticks(xticks)
            note = _comparison_note(panel_rows, x_key="atoms", y_key="atoms_s")
            if note:
                _callout(ax, note)

    handles, labels = _collect_legend_items(axes)
    n_legend = len(labels)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(n_legend, 7),
                   bbox_to_anchor=(0.5, 1.06), fontsize=10, columnspacing=1.5,
                   handletextpad=0.5, handlelength=1.2, framealpha=0, borderpad=0.6)
    fig.suptitle(_build_title(title_prefix, "Read Performance Overview"), fontsize=20,
                 fontweight="bold", y=1.12)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_read_overview_dotplot(
    panels: list[tuple[str, str, list[dict[str, Any]]]],
    all_backends: list[str],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
    suptitle_override: str | None = None,
) -> list[str]:
    """Grouped horizontal bar chart on a shared log-scale x-axis."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D

    plt = _import_plotting()

    n_backends = len(all_backends)
    n_scenarios = len(panels)

    # Build look-up: (backend, scenario) -> row
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario, _label, panel_rows in panels:
        for row in panel_rows:
            lookup[(row["backend"], scenario)] = row

    scenario_keys = [s for s, _, _ in panels]
    scenario_labels = {s: SCENARIO_SHORT_LABELS.get(s, label) for s, label, _ in panels}
    scenario_colors = {}
    palette = [LEMAT_PRIMARY, LEMAT_BLUE, LEMAT_ROSE, LEMAT_TEAL, LEMAT_AMBER, LEMAT_FUCHSIA]
    for i, key in enumerate(scenario_keys):
        scenario_colors[key] = palette[i % len(palette)]

    # Figure sizing — give each backend row enough vertical space for grouped bars.
    bar_h = 0.20
    bar_gap = 0.03
    group_gap = 0.40
    group_height = n_scenarios * bar_h + (n_scenarios - 1) * bar_gap
    row_pitch = group_height + group_gap
    fig_w = max(width, 10)
    fig_h = max(5.0, n_backends * row_pitch + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(LEMAT_BG)
    ax.set_facecolor(LEMAT_PANEL)

    # Compute the y-centre for each backend group.
    y_centres = [i * row_pitch for i in range(n_backends)]

    # We need a log-scale x-axis. matplotlib log-scaled barh doesn't render
    # FancyBboxPatch well, so we draw bars in log-space manually.
    # Collect all positive values to determine x-limits first.
    all_vals = [
        row.get("atoms_s", 0)
        for row in lookup.values()
        if row.get("atoms_s", 0) > 0
    ]
    if not all_vals:
        return []
    x_min_data = min(all_vals)
    x_max_data = max(all_vals)
    # Pad the log range so bars have room for labels.
    log_min = math.floor(math.log10(x_min_data)) - 0.3
    log_max = math.ceil(math.log10(x_max_data)) + 0.45

    def to_log(v: float) -> float:
        return math.log10(max(v, 10 ** log_min))

    # Draw grouped bars. For each backend, only lay out bars for scenarios
    # that have data, so missing scenarios don't leave gaps.
    for bi, backend in enumerate(all_backends):
        present = [
            (si, sk) for si, sk in enumerate(scenario_keys)
            if lookup.get((backend, sk)) is not None
            and lookup[(backend, sk)].get("atoms_s", 0) > 0
        ]
        n_present = len(present)
        if n_present == 0:
            continue
        local_height = n_present * bar_h + (n_present - 1) * bar_gap
        local_offsets = np.linspace(
            -local_height / 2 + bar_h / 2,
            local_height / 2 - bar_h / 2,
            n_present,
        )

        for li, (_si, scenario_key) in enumerate(present):
            color = scenario_colors[scenario_key]
            row = lookup[(backend, scenario_key)]
            val = row["atoms_s"]
            yc = y_centres[bi] + local_offsets[li]
            log_val = to_log(val)
            bar_width = log_val - log_min
            is_hero = backend == "atompack"

            patch = FancyBboxPatch(
                (log_min, yc - bar_h / 2), bar_width, bar_h,
                boxstyle=f"round,pad=0,rounding_size={min(0.08, bar_h / 3):.3f}",
                facecolor=color,
                edgecolor="white" if is_hero else "none",
                linewidth=1.2 if is_hero else 0,
                alpha=1.0 if is_hero else 0.85,
                zorder=4 if is_hero else 3,
            )
            ax.add_patch(patch)

            # CI error bar.
            err = row.get("ci95_atoms_s", 0)
            if err > 0 and val - err > 0:
                log_lo = to_log(val - err)
                log_hi = to_log(val + err)
                ax.plot(
                    [log_lo, log_hi], [yc, yc],
                    color=color, linewidth=1.0, alpha=0.4, zorder=2,
                    solid_capstyle="round",
                )

            # Value label at the end of the bar.
            ax.text(
                log_val + 0.06, yc,
                _format_rate(val),
                va="center", ha="left", fontsize=9,
                fontweight="semibold" if is_hero else "regular",
                color=color,
            )

    # Alternating row shading.
    for bi in range(n_backends):
        yc = y_centres[bi]
        half = row_pitch / 2
        if bi % 2 == 0:
            ax.axhspan(yc - half, yc + half, color=LEMAT_PANEL_SOFT, zorder=0)

    # Axis configuration — we're in manual log-space so set ticks accordingly.
    ax.set_xlim(log_min, log_max)
    # Major ticks at integer powers of 10.
    major_pows = list(range(math.ceil(log_min), math.floor(log_max) + 1))
    ax.set_xticks(major_pows)
    ax.set_xticklabels([f"$10^{{{p}}}$" for p in major_pows])
    # Minor ticks at 2, 5 within each decade.
    minor_ticks = []
    for p in range(math.floor(log_min), math.ceil(log_max) + 1):
        for m in (2, 5):
            t = math.log10(m * 10**p)
            if log_min < t < log_max:
                minor_ticks.append(t)
    ax.set_xticks(minor_ticks, minor=True)

    ax.set_yticks(y_centres)
    ax.set_yticklabels(all_backends, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("atoms/s  (log scale)", fontsize=12, color=LEMAT_MUTED)
    y_pad = row_pitch / 2
    ax.set_ylim(y_centres[-1] + y_pad, y_centres[0] - y_pad)

    # Grid.
    ax.grid(True, axis="x", which="major", color=LEMAT_BORDER, linewidth=0.6,
            alpha=0.5, linestyle="--")
    ax.grid(True, axis="x", which="minor", color=LEMAT_BORDER, linewidth=0.3,
            alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend.
    legend_handles = [
        Line2D(
            [0], [0],
            color=scenario_colors[s], linewidth=6,
            solid_capstyle="round", label=scenario_labels[s],
        )
        for s in scenario_keys
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor=LEMAT_BORDER,
        borderpad=0.8,
        handletextpad=0.6,
        labelspacing=0.5,
    )

    fig.suptitle(
        suptitle_override or _build_title(title_prefix, "Read Performance Overview"),
        fontsize=20, fontweight="bold", y=1.02,
    )

    return _save_figure(fig, out_base, formats, dpi)


_PREFERRED_MP_ATOMS = 64


def _plot_multiprocessing_scaling(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    """Single-panel line plot: atoms/s vs workers for the representative molecule size."""
    mp_rows = [
        row for row in rows
        if row["scenario"] == "multiprocessing"
        and row["workers"] is not None
        and row["atoms_s"] is not None
    ]
    if not mp_rows:
        return []

    atoms_values = sorted({row["atoms"] for row in mp_rows if row["atoms"] is not None})
    if not atoms_values:
        return []
    # Pick 64 atoms if available, otherwise closest.
    target = min(atoms_values, key=lambda a: abs(a - _PREFERRED_MP_ATOMS))
    panel_rows = [row for row in mp_rows if row["atoms"] == target]

    plt = _import_plotting()
    fig_w = max(width, 8)
    fig_h = fig_w * 5 / 14  # compact wide ratio
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor(LEMAT_BG)
    ax.set_facecolor(LEMAT_PANEL)

    grouped = defaultdict(list)
    for row in panel_rows:
        grouped[row["backend"]].append(row)

    use_log_y = _is_logworthy([row["atoms_s"] for row in panel_rows])

    for backend in _sorted_backends(panel_rows):
        series = sorted(grouped[backend], key=lambda row: row["workers"] or 0)
        xs = [row["workers"] for row in series]
        ys = [row["atoms_s"] for row in series]
        yerr = [row["ci95_atoms_s"] for row in series]
        style = _backend_style(backend)

        if any(e > 0 for e in yerr):
            ax.errorbar(
                xs, ys, yerr=yerr,
                fmt="none", ecolor=style["color"], elinewidth=0.8,
                capsize=2.5, alpha=0.35, zorder=2,
            )

        ax.plot(xs, ys, label=backend, **style)

    if use_log_y:
        ax.set_yscale("log")
    ax.set_xlabel("workers", fontsize=11, color=LEMAT_MUTED)
    ax.set_ylabel("atoms/s", fontsize=11, color=LEMAT_MUTED)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="y", which="major", color=LEMAT_BORDER, linewidth=0.5,
            alpha=0.4, linestyle="--")
    ax.grid(True, axis="x", color=LEMAT_BORDER, linewidth=0.3, alpha=0.25,
            linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="best", fontsize=9, frameon=True, fancybox=True,
            framealpha=0.90, edgecolor=LEMAT_BORDER, borderpad=0.6,
        )

    fig.suptitle(
        _build_title(title_prefix, f"Multiprocessing Scaling ({target} atoms/mol)"),
        fontsize=16, fontweight="bold",
    )
    return _save_figure(fig, out_base, formats, dpi)


def _plot_atom_scaling(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    """Polished line plot: atoms/s vs molecule size, one line per backend."""
    seq_rows = [
        row for row in rows
        if row["scenario"] == "sequential"
        and row["atoms"] is not None
        and row["atoms_s"] is not None
    ]
    atoms_values = sorted({row["atoms"] for row in seq_rows})
    if len(atoms_values) < 2:
        return []

    plt = _import_plotting()
    fig_w = max(width, 8)
    fig_h = fig_w * 5 / 14  # compact wide ratio
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor(LEMAT_BG)
    ax.set_facecolor(LEMAT_PANEL)

    grouped = defaultdict(list)
    for row in seq_rows:
        grouped[row["backend"]].append(row)

    use_log_x = _is_logworthy(atoms_values)
    use_log_y = _is_logworthy([row["atoms_s"] for row in seq_rows])

    for backend in _sorted_backends(seq_rows):
        series = sorted(grouped[backend], key=lambda row: row["atoms"] or 0)
        xs = [row["atoms"] for row in series]
        ys = [row["atoms_s"] for row in series]
        yerr = [row["ci95_atoms_s"] for row in series]
        style = _backend_style(backend)

        if any(e > 0 for e in yerr):
            ax.errorbar(
                xs, ys, yerr=yerr,
                fmt="none", ecolor=style["color"], elinewidth=0.8,
                capsize=2.5, alpha=0.35, zorder=2,
            )

        ax.plot(xs, ys, label=backend, **style)

    # Axis styling.
    if use_log_x:
        ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")
    ax.set_xlabel("atoms / molecule", fontsize=11, color=LEMAT_MUTED)
    ax.set_ylabel("atoms/s", fontsize=11, color=LEMAT_MUTED)
    ax.grid(True, axis="y", which="major", color=LEMAT_BORDER, linewidth=0.5,
            alpha=0.4, linestyle="--")
    ax.grid(True, axis="x", color=LEMAT_BORDER, linewidth=0.3, alpha=0.25,
            linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if ax.get_legend_handles_labels()[0]:
        ax.legend(
            loc="best", fontsize=10, frameon=True, fancybox=True,
            framealpha=0.90, edgecolor=LEMAT_BORDER, borderpad=0.8,
        )
    fig.suptitle(
        _build_title(title_prefix, "Read Throughput vs Molecule Size"),
        fontsize=18, fontweight="bold", y=1.02,
    )
    return _save_figure(fig, out_base, formats, dpi)


def _plot_read_scaling(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    atoms_values = sorted({row["atoms"] for row in rows if row["atoms"] is not None})
    if not atoms_values:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(plt, len(atoms_values), width=width, height=max(5.5, width * 0.40))

    for ax, atoms in zip(axes, atoms_values):
        panel_rows = [row for row in rows if row["atoms"] == atoms]
        grouped = defaultdict(list)
        for row in panel_rows:
            grouped[row["backend"]].append(row)

        for backend in _sorted_backends(panel_rows):
            series = sorted(grouped[backend], key=lambda row: row["workers"] or 0)
            xs = [row["workers"] for row in series if row["workers"] is not None]
            ys = [row["atoms_s"] for row in series]
            yerr = [row["ci95_atoms_s"] for row in series]
            style = _backend_style(backend)
            ax.plot(xs, ys, label=backend, **style)
            if any(e > 0 for e in yerr):
                ax.errorbar(
                    xs, ys, yerr=yerr,
                    fmt="none", ecolor=style["color"], elinewidth=0.8,
                    capsize=2.5, alpha=0.35, zorder=style["zorder"] - 1,
                )
            if backend == "atompack":
                _annotate_series_end(ax, xs, ys, "atompack", style["color"])

        _setup_axis(ax, title=f"{atoms} atoms", xlabel="workers", ylabel="atoms/s", log_y=True)
        note = _comparison_note(panel_rows, x_key="workers", y_key="atoms_s")
        if note:
            _callout(ax, note)

    handles, labels = _collect_legend_items(axes)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   bbox_to_anchor=(0.5, 1.07), fontsize=11, columnspacing=2.0,
                   handletextpad=0.6, handlelength=1.5, framealpha=0, borderpad=0.8)
    fig.suptitle(_build_title(title_prefix, "Random Read Scaling"), fontsize=20,
                 fontweight="bold", y=1.15)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_memory_panel(ax, rows: list[dict[str, Any]], title: str) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["private_total_kib"])
    ys = list(range(len(sorted_rows)))
    labels = [row["backend"] for row in sorted_rows]
    private = [_mib(row["private_total_kib"]) for row in sorted_rows]
    file_backed = [_mib(row["rss_file_kib"]) for row in sorted_rows]
    colors = [_backend_color(label) for label in labels]
    max_val = max(private) if private else 1

    for i, (label, val, fb_val, color) in enumerate(zip(labels, private, file_backed, colors)):
        is_hero = label == "atompack"
        _draw_rounded_barh(
            ax, i, val, 0.55,
            color=color,
            alpha=1.0 if is_hero else 0.82,
            edgecolor=LEMAT_PRIMARY_DARK if is_hero else "none",
            linewidth=1.5 if is_hero else 0,
            label="private" if i == 0 else None,
            radius=0.1,
        )
        _draw_rounded_barh(
            ax, i, fb_val, 0.25,
            color=LEMAT_PANEL_SOFT,
            edgecolor="none",
            label="file-backed RSS" if i == 0 else None,
            radius=0.06,
        )
        ax.text(
            val + max_val * 0.02, i,
            f"{val:.1f}",
            va="center", ha="left", fontsize=9,
            color=LEMAT_MUTED,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, pad=18)
    ax.set_xlabel("MiB")
    ax.grid(True, axis="x", color=LEMAT_BORDER, linewidth=0.5, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=6)
    ax.set_xlim(right=max_val * 1.15)

    if sorted_rows:
        _callout(
            ax,
            f"Lowest: {sorted_rows[0]['backend']} ({_format_mib(sorted_rows[0]['private_total_kib'])})",
        )


def _plot_memory_profile(
    open_rows: list[dict[str, Any]],
    stream_rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    panels = []
    if open_rows:
        panels.append(("Open Connection Memory (per worker)", open_rows))
    if stream_rows:
        panels.append(("Streaming Read Memory", stream_rows))
    if not panels:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(plt, len(panels), width=width, height=max(5.2, width * 0.40))

    for ax, (title, rows) in zip(axes, panels):
        _plot_memory_panel(ax, rows, title)

    handles, labels = _collect_legend_items(axes)
    if handles:
        fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.07), fontsize=11, columnspacing=2.0,
                   handletextpad=0.6, handlelength=1.5, framealpha=0, borderpad=0.8)
    fig.suptitle(_build_title(title_prefix, "Memory Profile"), fontsize=20,
                 fontweight="bold", y=1.15)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_batch_api(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in rows)]
    if not methods:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(plt, len(methods), width=width, height=max(5.5, width * 0.40))

    for ax, method in zip(axes, methods):
        panel_rows = [row for row in rows if row["method"] == method]
        grouped = defaultdict(list)
        for row in panel_rows:
            grouped[row["threads"]].append(row)
        thread_order = sorted(grouped, key=lambda value: (value != "native", value))

        for threads in thread_order:
            color = _thread_color(threads, thread_order)
            series = sorted(grouped[threads], key=lambda row: row["batch_size"])
            xs = [row["batch_size"] for row in series]
            ys = [row["mol_s"] for row in series]
            yerr = [row["ci95_mol_s"] for row in series]
            z = 4 if threads == "native" else 3
            ax.plot(
                xs, ys,
                color=color,
                linewidth=2.5 if threads == "native" else 1.8,
                markersize=6.5 if threads == "native" else 5.5,
                marker="o",
                label=threads,
                zorder=z,
            )
            if any(e > 0 for e in yerr):
                ax.errorbar(
                    xs, ys, yerr=yerr,
                    fmt="none", ecolor=color, elinewidth=0.8,
                    capsize=2.5, alpha=0.35, zorder=z - 1,
                )

        _setup_axis(
            ax,
            title=method.capitalize(),
            xlabel="batch size",
            ylabel="mol/s",
            log_x=True,
            log_y=_is_logworthy([row["mol_s"] for row in panel_rows]),
        )
        ax.set_title(
            method.capitalize(),
            color=METHOD_COLORS.get(method, LEMAT_TEXT),
            pad=18,
        )
        note = _comparison_note(panel_rows, x_key="batch_size", y_key="mol_s")
        if note:
            _callout(ax, note)

    handles, labels = _collect_legend_items(axes)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   title="threads", bbox_to_anchor=(0.5, 1.07), fontsize=11,
                   columnspacing=2.0, handletextpad=0.6, handlelength=1.5,
                   framealpha=0, borderpad=0.8)
    fig.suptitle(_build_title(title_prefix, "Atompack Batch API"), fontsize=20,
                 fontweight="bold", y=1.15)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_blog_read_hero(
    rows: list[dict[str, Any]],
    *,
    source_name: str | None,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    blog_rows = [row for row in rows if source_name is None or row.get("source_name") == source_name]
    atoms = _preferred_atoms(blog_rows, scenario="sequential")
    if atoms is None:
        return []

    sequential_rows = [
        row for row in blog_rows
        if row.get("scenario") == "sequential" and row.get("atoms") == atoms and row.get("atoms_s") is not None
    ]
    random_rows = _blog_random_rows(blog_rows, atoms=atoms)
    speedup_rows = _speedup_rows(blog_rows, atoms=atoms)
    if not sequential_rows or not random_rows or not speedup_rows:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(
        plt,
        3,
        width=max(width, 16.0),
        height=6.2,
        constrained_layout=False,
    )
    fig.patch.set_facecolor(LEMAT_BG)
    fig.subplots_adjust(top=0.82, wspace=0.32)
    for ax in axes:
        ax.set_facecolor(LEMAT_PANEL)

    _plot_ranked_panel(
        axes[0],
        sequential_rows,
        title="Sequential Read",
        callout=_panel_callout(sequential_rows),
    )
    _plot_ranked_panel(
        axes[1],
        random_rows,
        title="Random Read",
        callout=_panel_callout(random_rows),
    )

    speedup_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    abs_mp_rows = [
        row
        for row in blog_rows
        if row.get("scenario") == "multiprocessing" and row.get("atoms") == atoms and row.get("atoms_s") is not None
    ]
    for row in speedup_rows:
        speedup_by_backend[str(row["backend"])].append(row)

    ax = axes[2]
    line_endpoints: list[tuple[float, float, str, str]] = []
    for backend in _sorted_backends(speedup_rows):
        series = sorted(speedup_by_backend[backend], key=lambda row: int(row.get("workers") or 0))
        xs = [int(row["workers"]) for row in series if row.get("workers") is not None]
        ys = [float(row["speedup"]) for row in series]
        if not xs:
            continue
        style = _backend_style(backend)
        ax.plot(xs, ys, **style)
        line_endpoints.append((xs[-1], ys[-1], _backend_label(backend), style["color"]))

    _setup_axis(
        ax,
        title="Multiprocessing Scaling",
        xlabel="workers",
        ylabel="speedup vs 1 worker",
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, axis="both", color=LEMAT_BORDER, linewidth=0.4, alpha=0.35, linestyle="--")
    _label_series_ends(ax, line_endpoints, x_offset_points=8.0, min_gap_px=16.0)

    shared_workers = sorted(
        set(int(row["workers"]) for row in abs_mp_rows if row["backend"] == "atompack" and row.get("workers") is not None)
        & set(int(row["workers"]) for row in abs_mp_rows if row["backend"] == "ase_lmdb" and row.get("workers") is not None)
    )
    if shared_workers:
        target_worker = shared_workers[-1]
        panel_rows = [row for row in abs_mp_rows if row.get("workers") == target_worker]
        callout = _panel_callout(panel_rows)
        if callout:
            _callout(ax, f"{callout} at {target_worker} workers")

    fig.suptitle(
        _build_title(title_prefix, "Fast Reads For Dataset Access"),
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )
    _figure_subtitle(
        fig,
        f"{atoms} atoms/mol. Random read = shuffled single-item access. Multiprocessing shown as speedup vs 1 worker.",
        y=0.90,
    )
    return _save_figure(fig, out_base, formats, dpi)


def _plot_blog_size_scaling(
    rows: list[dict[str, Any]],
    *,
    source_name: str | None,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    blog_rows = [row for row in rows if source_name is None or row.get("source_name") == source_name]
    sequential_rows = [
        row for row in blog_rows
        if row.get("scenario") == "sequential" and row.get("atoms") is not None and row.get("atoms_s") is not None
    ]
    random_rows = [
        row
        for row in blog_rows
        if row.get("scenario") == "multiprocessing"
        and row.get("workers") == 1
        and row.get("atoms") is not None
        and row.get("atoms_s") is not None
    ]
    seq_atoms = {int(row["atoms"]) for row in sequential_rows}
    random_atoms = {int(row["atoms"]) for row in random_rows}
    if len(seq_atoms) < 2 or len(random_atoms) < 2:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(
        plt,
        2,
        width=max(width, 13.2),
        height=6.1,
        constrained_layout=False,
    )
    fig.patch.set_facecolor(LEMAT_BG)
    fig.subplots_adjust(top=0.81, wspace=0.30)

    panel_specs = [
        ("Sequential Throughput vs Molecule Size", sequential_rows),
        ("Random Throughput vs Molecule Size", random_rows),
    ]
    for ax, (title, panel_rows) in zip(axes, panel_specs):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in panel_rows:
            grouped[str(row["backend"])].append(row)

        line_endpoints: list[tuple[float, float, str, str]] = []
        for backend in _sorted_backends(panel_rows):
            series = sorted(grouped[backend], key=lambda row: int(row["atoms"]))
            xs = [int(row["atoms"]) for row in series]
            ys = [float(row["atoms_s"]) for row in series]
            style = _backend_style(backend)
            ax.plot(xs, ys, **style)
            line_endpoints.append((xs[-1], ys[-1], _backend_label(backend), style["color"]))

        _setup_axis(
            ax,
            title=title,
            xlabel="atoms / molecule",
            ylabel="atoms/s",
            log_x=True,
            log_y=True,
        )
        _label_series_ends(ax, line_endpoints, x_offset_points=8.0, min_gap_px=14.0)
        callout = _panel_callout(
            [row for row in panel_rows if int(row["atoms"]) == max(int(item["atoms"]) for item in panel_rows)]
        )
        if callout:
            _callout(ax, callout)

    fig.suptitle(
        _build_title(title_prefix, "Throughput That Holds As Molecules Grow"),
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )
    _figure_subtitle(
        fig,
        "Sequential and shuffled single-item reads stay strong as molecule size increases.",
        y=0.90,
    )
    return _save_figure(fig, out_base, formats, dpi)


def _plot_blog_filesystem_random(
    rows: list[dict[str, Any]],
    *,
    source_order: list[str],
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    atoms = _shared_atoms_by_source(rows, scenario="multiprocessing", workers=1)
    if atoms is None:
        return []

    sources = [
        source_name
        for source_name in source_order
        if any(
            row.get("source_name") == source_name
            and row.get("scenario") == "multiprocessing"
            and row.get("workers") == 1
            and row.get("atoms") == atoms
            for row in rows
        )
    ]
    if len(sources) < 2:
        return []

    plt = _import_plotting()
    ncols = 2 if len(sources) > 2 else len(sources)
    nrows = math.ceil(len(sources) / max(1, ncols))
    fig, axes = _subplots_grid(
        plt,
        nrows,
        ncols,
        width=max(width, 13.8),
        height=9.6 if len(sources) > 2 else 5.8,
        constrained_layout=False,
    )
    fig.patch.set_facecolor(LEMAT_BG)
    fig.subplots_adjust(top=0.86, hspace=0.38, wspace=0.22)

    panel_rows_by_source = {
        source_name: [
            row
            for row in rows
            if row.get("source_name") == source_name
            and row.get("scenario") == "multiprocessing"
            and row.get("workers") == 1
            and row.get("atoms") == atoms
            and row.get("atoms_s") is not None
        ]
        for source_name in sources
    }
    all_values = [
        float(row["atoms_s"])
        for panel_rows in panel_rows_by_source.values()
        for row in panel_rows
        if row.get("atoms_s", 0) > 0
    ]
    if not all_values:
        return []
    x_limits = (min(all_values) * 0.75, max(all_values) * 1.55)

    for idx, (ax, source_name) in enumerate(zip(axes, sources)):
        panel_rows = panel_rows_by_source[source_name]
        _plot_ranked_panel(
            ax,
            panel_rows,
            title=_filesystem_label(source_name),
            callout=_panel_callout(panel_rows),
        )
        ax.set_xlim(*x_limits)
        if nrows > 1 and idx < ncols:
            ax.set_xlabel("")
    for ax in axes[len(sources):]:
        ax.axis("off")

    fig.suptitle(
        _build_title(title_prefix, "Fast Random Reads Across Storage Environments"),
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )
    _figure_subtitle(fig, f"{atoms} atoms/mol. Random read shown at 1 worker across storage environments.", y=0.92)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_grouped_bars(ax, rows: list[dict[str, Any]], *, title: str) -> None:
    backends = _sorted_backends(rows)
    atoms_values = sorted({row["atoms"] for row in rows})
    x_positions = list(range(len(atoms_values)))
    group_width = 0.8
    bar_width = group_width / max(1, len(backends))
    all_values = [row["mol_s"] for row in rows]

    for index, backend in enumerate(backends):
        backend_rows = {row["atoms"]: row for row in rows if row["backend"] == backend}
        offsets = [x + (index - (len(backends) - 1) / 2.0) * bar_width for x in x_positions]
        heights = [backend_rows.get(atoms, {}).get("mol_s", float("nan")) for atoms in atoms_values]
        yerr = [backend_rows.get(atoms, {}).get("ci95_mol_s", 0.0) for atoms in atoms_values]
        is_hero = backend == "atompack"

        for i, (off, h) in enumerate(zip(offsets, heights)):
            if math.isnan(h):
                continue
            _draw_rounded_bar(
                ax, off - bar_width * 0.85 / 2, 0, bar_width * 0.85, h,
                color=_backend_color(backend),
                alpha=1.0 if is_hero else 0.82,
                edgecolor=LEMAT_PRIMARY_DARK if is_hero else "none",
                linewidth=1.5 if is_hero else 0,
                label=backend if i == 0 else None,
                zorder=4 if is_hero else 3,
            )

        valid_pairs = [(o, h, e) for o, h, e in zip(offsets, heights, yerr) if not math.isnan(h) and e > 0]
        if valid_pairs:
            ax.errorbar(
                [p[0] for p in valid_pairs],
                [p[1] for p in valid_pairs],
                yerr=[p[2] for p in valid_pairs],
                fmt="none", ecolor=LEMAT_MUTED, elinewidth=0.8,
                capsize=2.5, alpha=0.45, zorder=5,
            )

        max_height = max((v for v in heights if not math.isnan(v)), default=0)
        for off, value in zip(offsets, heights):
            if math.isnan(value):
                continue
            is_best = value == max_height
            ax.text(
                off, value,
                _format_rate(value),
                ha="center", va="bottom", fontsize=9,
                fontweight="semibold" if is_best else "regular",
                color=LEMAT_TEXT if is_best else LEMAT_MUTED,
            )

    max_val = max((v for v in all_values if not math.isnan(v)), default=1)
    ax.set_ylim(0, max_val * 1.15)
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)

    _setup_axis(
        ax,
        title=title,
        xlabel="atoms",
        ylabel="mol/s",
        log_y=_is_logworthy(all_values, threshold=12.0),
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in atoms_values])
    note = _comparison_note(rows, x_key="atoms", y_key="mol_s")
    if note:
        _callout(ax, note)


def _plot_scaling_lines(ax, rows: list[dict[str, Any]], *, title: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["backend"]].append(row)

    for backend in _sorted_backends(rows):
        series = sorted(grouped[backend], key=lambda row: row["n_mols"])
        xs = [row["n_mols"] for row in series]
        ys = [row["mol_s"] for row in series]
        yerr = [row["ci95_mol_s"] for row in series]
        style = _backend_style(backend)
        ax.plot(xs, ys, label=backend, **style)
        if any(e > 0 for e in yerr):
            ax.errorbar(
                xs, ys, yerr=yerr,
                fmt="none", ecolor=style["color"], elinewidth=0.8,
                capsize=2.5, alpha=0.35, zorder=style["zorder"] - 1,
            )
        if backend == "atompack":
            _annotate_series_end(ax, xs, ys, "atompack", style["color"])

    _setup_axis(ax, title=title, xlabel="molecules", ylabel="mol/s", log_x=True, log_y=True)
    note = _comparison_note(rows, x_key="n_mols", y_key="mol_s")
    if note:
        _callout(ax, note)


def _plot_compression_impact(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    """Grouped bar chart comparing read throughput with/without compression."""
    builtin = [r for r in rows if r["scenario"] == "compression" and r["atoms_s"] is not None]
    custom = [r for r in rows if r["scenario"] == "compression_custom" and r["atoms_s"] is not None]

    panels: list[tuple[str, str, list[dict[str, Any]]]] = []
    if builtin:
        panels.append(("compression", "Compression Impact", builtin))
    if custom:
        panels.append(("compression_custom", "Compression + Custom Props", custom))
    if not panels:
        return []

    all_backends = []
    seen: set[str] = set()
    for _, _, pr in panels:
        for b in _sorted_backends(pr):
            if b not in seen:
                all_backends.append(b)
                seen.add(b)

    return _plot_read_overview_dotplot(
        panels, all_backends,
        title_prefix=title_prefix, width=width,
        formats=formats, dpi=dpi, out_base=out_base,
        suptitle_override=_build_title(title_prefix, "Compression Impact"),
    )


def _plot_write_overview(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str | None,
    width: float,
    formats: list[str],
    dpi: int,
    out_base: Path,
) -> list[str]:
    rows = [row for row in rows if row["backend"] not in WRITE_EXCLUDED_BACKENDS]
    panels: list[tuple[str, list[dict[str, Any]], str]] = []
    throughput_builtin = [
        row for row in rows if row["scenario"] == "write_throughput" and not row["with_custom"]
    ]
    throughput_custom = [
        row for row in rows if row["scenario"] == "write_throughput" and row["with_custom"]
    ]
    scaling_rows = [row for row in rows if row["scenario"] == "write_scaling"]

    if throughput_builtin:
        panels.append(("Builtins Write Throughput", throughput_builtin, "bars"))
    if throughput_custom:
        panels.append(("Custom Write Throughput", throughput_custom, "bars"))
    if scaling_rows:
        panels.append(("Write Scaling Law", scaling_rows, "scaling"))
    if not panels:
        return []

    plt = _import_plotting()
    fig, axes = _subplots_row(plt, len(panels), width=width, height=max(5.5, width * 0.40))

    for ax, (title, panel_rows, kind) in zip(axes, panels):
        if kind == "bars":
            _plot_grouped_bars(ax, panel_rows, title=title)
        else:
            _plot_scaling_lines(ax, panel_rows, title=title)

    handles, labels = _collect_legend_items(axes)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)),
                   bbox_to_anchor=(0.5, 1.07), fontsize=11, columnspacing=2.0,
                   handletextpad=0.6, handlelength=1.5, framealpha=0, borderpad=0.8)
    fig.suptitle(_build_title(title_prefix, "Write Performance"), fontsize=20,
                 fontweight="bold", y=1.15)
    return _save_figure(fig, out_base, formats, dpi)


def _primary_source_name(bundle: dict[str, Any], schema: str) -> str | None:
    sources = bundle["sources"].get(schema, [])
    if not sources:
        return None
    return Path(sources[0]).stem


def _select_figures(
    bundle: dict[str, Any],
    *,
    no_batch_api: bool,
    report_mode: str,
) -> list[dict[str, Any]]:
    bm_rows = bundle["benchmark_rows"]
    bm_sources = bundle["sources"].get("benchmark", [])
    primary_benchmark = _primary_source_name(bundle, "benchmark")
    primary_rows = [
        row for row in bm_rows
        if primary_benchmark is None or row.get("source_name") == primary_benchmark
    ]

    if report_mode == "blog":
        seq_atoms = {
            int(row["atoms"])
            for row in primary_rows
            if row.get("scenario") == "sequential" and row.get("atoms") is not None
        }
        random_atoms = {
            int(row["atoms"])
            for row in primary_rows
            if row.get("scenario") == "multiprocessing"
            and row.get("workers") == 1
            and row.get("atoms") is not None
        }
        mp_rows = [
            row
            for row in primary_rows
            if row.get("scenario") == "multiprocessing" and row.get("workers") is not None
        ]
        filesystem_rows = [
            row
            for row in bm_rows
            if row.get("scenario") == "multiprocessing" and row.get("workers") == 1
        ]
        source_names = {str(row.get("source_name")) for row in filesystem_rows}
        return [
            _figure_status(
                key="blog_read_hero",
                kind="blog_read_hero",
                rows=primary_rows if seq_atoms and random_atoms and mp_rows else [],
                sources=bm_sources[:1],
            ),
            _figure_status(
                key="blog_size_scaling",
                kind="blog_size_scaling",
                rows=primary_rows if len(seq_atoms) >= 2 and len(random_atoms) >= 2 else [],
                sources=bm_sources[:1],
            ),
            _figure_status(
                key="blog_random_filesystems",
                kind="blog_random_filesystems",
                rows=filesystem_rows if len(source_names) >= 2 else [],
                sources=bm_sources,
            ),
        ]

    # multiprocessing_scaling: only if we have multiprocessing rows with >1 distinct worker count.
    mp_rows = [r for r in bm_rows if r["scenario"] == "multiprocessing" and r["workers"] is not None]
    mp_workers = {r["workers"] for r in mp_rows}

    # atom_scaling: only if sequential rows span >1 distinct atom count.
    seq_rows = [r for r in bm_rows if r["scenario"] == "sequential" and r["atoms"] is not None]
    seq_atoms = {r["atoms"] for r in seq_rows}

    # compression: rows from compression / compression_custom scenarios.
    comp_rows = [r for r in bm_rows if r["scenario"] in ("compression", "compression_custom")]

    return [
        _figure_status(
            key="read_overview",
            kind="read_overview",
            rows=bm_rows,
            sources=bm_sources,
        ),
        _figure_status(
            key="multiprocessing_scaling",
            kind="multiprocessing_scaling",
            rows=mp_rows if len(mp_workers) >= 2 else [],
            sources=bm_sources,
        ),
        _figure_status(
            key="atom_scaling",
            kind="atom_scaling",
            rows=seq_rows if len(seq_atoms) >= 2 else [],
            sources=bm_sources,
        ),
        _figure_status(
            key="compression_impact",
            kind="compression_impact",
            rows=comp_rows,
            sources=bm_sources,
        ),
        _figure_status(
            key="read_scaling",
            kind="read_scaling",
            rows=bundle["scaling_rows"],
            sources=bundle["sources"].get("scaling", []),
        ),
        _figure_status(
            key="memory_profile",
            kind="memory_profile",
            rows=(bundle["memory_rows_open"] + bundle["memory_rows_stream"]),
            sources=bundle["sources"].get("memory", []),
        ),
        _figure_status(
            key="batch_api",
            kind="batch_api",
            rows=bundle["batch_rows"],
            sources=bundle["sources"].get("batch", []),
            no_batch_api=no_batch_api,
        ),
        _figure_status(
            key="write_overview",
            kind="write_overview",
            rows=bundle["write_rows"],
            sources=bundle["sources"].get("write", []),
        ),
    ]


def _build_manifest(
    bundle: dict[str, Any],
    figures: list[dict[str, Any]],
    *,
    report_mode: str,
) -> dict[str, Any]:
    return {
        "report_mode": report_mode,
        "inputs": bundle["sources"],
        "figures": figures,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _filter_backends(bundle: dict[str, Any], allowed: set[str]) -> None:
    """Remove rows whose backend is not in *allowed*, in-place."""
    for key in ("benchmark_rows", "scaling_rows"):
        bundle[key] = [r for r in bundle[key] if r["backend"] in allowed]
    for key in ("memory_rows_open", "memory_rows_stream"):
        bundle[key] = [r for r in bundle[key] if r["backend"] in allowed]
    bundle["write_rows"] = [r for r in bundle["write_rows"] if r["backend"] in allowed]


def generate_report(
    inputs: list[Path],
    out_dir: Path,
    *,
    title_prefix: str | None,
    formats: list[str],
    dpi: int,
    width: float,
    overview_width: float,
    no_batch_api: bool,
    report_mode: str = "full",
    all_backends: bool = False,
) -> dict[str, Any]:
    bundle = load_inputs(inputs)
    if not all_backends:
        _filter_backends(bundle, DEFAULT_BACKENDS)
    figures = _select_figures(bundle, no_batch_api=no_batch_api, report_mode=report_mode)
    primary_benchmark = _primary_source_name(bundle, "benchmark")
    source_order = [Path(path).stem for path in bundle["sources"].get("benchmark", [])]

    for figure in figures:
        if figure["status"] != "produced":
            continue

        out_base = out_dir / figure["key"]
        if figure["kind"] == "read_overview":
            outputs = _plot_read_overview(
                bundle["benchmark_rows"],
                title_prefix=title_prefix,
                width=overview_width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "multiprocessing_scaling":
            outputs = _plot_multiprocessing_scaling(
                bundle["benchmark_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "atom_scaling":
            outputs = _plot_atom_scaling(
                bundle["benchmark_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "compression_impact":
            outputs = _plot_compression_impact(
                bundle["benchmark_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "read_scaling":
            outputs = _plot_read_scaling(
                bundle["scaling_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "memory_profile":
            outputs = _plot_memory_profile(
                bundle["memory_rows_open"],
                bundle["memory_rows_stream"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "batch_api":
            outputs = _plot_batch_api(
                bundle["batch_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "write_overview":
            outputs = _plot_write_overview(
                bundle["write_rows"],
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "blog_read_hero":
            outputs = _plot_blog_read_hero(
                bundle["benchmark_rows"],
                source_name=primary_benchmark,
                title_prefix=title_prefix,
                width=overview_width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "blog_size_scaling":
            outputs = _plot_blog_size_scaling(
                bundle["benchmark_rows"],
                source_name=primary_benchmark,
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        elif figure["kind"] == "blog_random_filesystems":
            outputs = _plot_blog_filesystem_random(
                bundle["benchmark_rows"],
                source_order=source_order,
                title_prefix=title_prefix,
                width=width,
                formats=formats,
                dpi=dpi,
                out_base=out_base,
            )
        else:
            raise ValueError(f"Unknown figure kind: {figure['kind']}")

        if not outputs:
            figure["status"] = "skipped"
            figure["reason"] = f"{figure['kind']} had no plottable rows"
        figure["outputs"] = outputs

    manifest = _build_manifest(bundle, figures, report_mode=report_mode)
    manifest["manifest_path"] = _write_manifest(out_dir / "figure_manifest.json", manifest)
    return manifest


def _parse_formats(value: str) -> list[str]:
    formats = [fmt.strip().lower() for fmt in value.split(",") if fmt.strip()]
    invalid = [fmt for fmt in formats if fmt not in SUPPORTED_FORMATS]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported formats: {', '.join(invalid)}")
    if not formats:
        raise argparse.ArgumentTypeError("At least one output format is required")
    return formats


def _parse_report_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode not in SUPPORTED_REPORT_MODES:
        raise argparse.ArgumentTypeError(
            f"Unsupported report mode: {value}. Expected one of {', '.join(sorted(SUPPORTED_REPORT_MODES))}."
        )
    return mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render publication-friendly benchmark figures from JSON outputs"
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", type=str, default=None)
    parser.add_argument("--formats", type=_parse_formats, default=["svg", "png"])
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--width", type=float, default=14.0)
    parser.add_argument("--overview-width", type=float, default=12.0)
    parser.add_argument(
        "--report-mode",
        type=_parse_report_mode,
        default="full",
        help="Choose 'full' for the broader report or 'blog' for the launch-story figure set.",
    )
    parser.add_argument("--no-batch-api", action="store_true")
    parser.add_argument(
        "--all-backends", action="store_true",
        help="Keep every backend present in the inputs instead of applying the default filtering.",
    )
    args = parser.parse_args(argv)

    manifest = generate_report(
        args.inputs,
        args.out_dir,
        title_prefix=args.title_prefix,
        formats=args.formats,
        dpi=args.dpi,
        width=args.width,
        overview_width=args.overview_width,
        no_batch_api=args.no_batch_api,
        report_mode=args.report_mode,
        all_backends=args.all_backends,
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
