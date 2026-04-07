# Copyright 2026 Entalpic
"""Convert legacy atompack shards into the current SOA format."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import atompack
import numpy as np

LEGACY_MODULE_NAME = "atompack_legacy"


@dataclass(frozen=True)
class ShardJob:
    source: Path
    target: Path
    label: str


@dataclass
class ShardStats:
    molecules: int = 0
    fast_batches: int = 0
    fast_molecules: int = 0
    fallback_batches: int = 0
    fallback_molecules: int = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert legacy atompack files into the current SOA format."
    )
    parser.add_argument("input_path", help="Legacy .atp file or dataset directory")
    parser.add_argument("--output", help="Output file or output directory root")
    parser.add_argument(
        "--compression",
        choices=("none", "lz4", "zstd"),
        default="none",
        help="Compression codec for the converted database",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Compression level for zstd output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of legacy molecules to inspect per conversion batch",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=100_000,
        help="Emit progress every N converted molecules (0 disables periodic reports)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next shard when one shard fails",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of shard worker processes for directory conversions",
    )
    return parser


def load_legacy_atompack(module_name: str = LEGACY_MODULE_NAME) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise SystemExit(
            "Could not import 'atompack_legacy'. Install the renamed legacy package in the "
            "same environment before running this converter."
        ) from exc


def _open_legacy_database(legacy_module: ModuleType, path: Path) -> Any:
    database_cls = legacy_module.Database
    open_mmap = getattr(database_cls, "open_mmap", None)
    if callable(open_mmap):
        return open_mmap(str(path))
    return database_cls.open(str(path))


def _load_legacy_batch(legacy_db: Any, start: int, stop: int) -> list[Any]:
    get_molecules = getattr(legacy_db, "get_molecules", None)
    if callable(get_molecules):
        return list(get_molecules(list(range(start, stop))))
    return [legacy_db[index] for index in range(start, stop)]


def _module_name(module: ModuleType | None) -> str | None:
    if module is None:
        return None
    name = getattr(module, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return None


def _load_manifest_shards(dataset_dir: Path) -> list[Path]:
    manifest_path = dataset_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    output_files = payload.get("output_files")
    if not isinstance(output_files, list) or not output_files:
        raise ValueError(f"Manifest {manifest_path} is missing a non-empty 'output_files' list")

    shards: list[Path] = []
    for raw_path in output_files:
        if not isinstance(raw_path, str):
            raise ValueError(f"Manifest {manifest_path} contains a non-string output file entry")
        shard_path = dataset_dir / raw_path
        shards.append(shard_path)
    return shards


def _discover_source_shards(input_path: Path) -> tuple[Path, list[Path]]:
    if input_path.is_file():
        return input_path.parent, [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    manifest_path = input_path / "manifest.json"
    if manifest_path.is_file():
        shards = _load_manifest_shards(input_path)
    else:
        shards = sorted(path for path in input_path.rglob("*.atp") if path.is_file())

    if not shards:
        raise ValueError(f"No .atp shards found under {input_path}")

    return input_path, shards


def _default_file_output(source: Path) -> Path:
    if source.suffix == ".atp":
        return source.with_name(f"{source.stem}.soa.atp")
    return source.with_name(f"{source.name}.soa")


def _ensure_safe_target(source: Path, target: Path, overwrite: bool) -> None:
    if source.resolve() == target.resolve():
        raise ValueError(f"Refusing to overwrite source shard in place: {source}")
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists at '{target}'. Pass --overwrite to replace it."
        )


def plan_jobs(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> list[ShardJob]:
    input_path = Path(input_path)
    root, shards = _discover_source_shards(input_path)

    if input_path.is_file():
        source = shards[0]
        if output_path is None:
            target = _default_file_output(source)
        else:
            candidate = Path(output_path)
            target = (
                candidate / source.name
                if candidate.exists() and candidate.is_dir()
                else candidate
            )
        _ensure_safe_target(source, target, overwrite)
        return [ShardJob(source=source, target=target, label=source.name)]

    output_root = (
        Path(output_path)
        if output_path is not None
        else input_path.with_name(f"{input_path.name}_soa")
    )

    jobs: list[ShardJob] = []
    for source in shards:
        rel_path = source.relative_to(root)
        target = output_root / rel_path
        _ensure_safe_target(source, target, overwrite)
        jobs.append(ShardJob(source=source, target=target, label=str(rel_path)))
    return jobs


def _property_keys(molecule: Any) -> list[str]:
    method = getattr(molecule, "property_keys", None)
    if callable(method):
        return list(method())

    raw_properties = getattr(molecule, "properties", None)
    if isinstance(raw_properties, dict):
        return list(raw_properties.keys())

    return []


def _get_property(molecule: Any, key: str) -> Any:
    method = getattr(molecule, "get_property", None)
    if callable(method):
        return method(key)

    raw_properties = getattr(molecule, "properties", None)
    if isinstance(raw_properties, dict):
        return raw_properties[key]

    try:
        return molecule[key]
    except Exception as exc:  # pragma: no cover - defensive adapter fallback
        raise KeyError(f"Could not extract legacy property '{key}'") from exc


def _as_array(
    name: str,
    value: Any,
    *,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def _positions(molecule: Any) -> np.ndarray:
    positions = _as_array("positions", molecule.positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (n_atoms, 3), got {positions.shape}")
    return positions


def _atomic_numbers(molecule: Any, n_atoms: int) -> np.ndarray:
    atomic_numbers = _as_array("atomic_numbers", molecule.atomic_numbers, dtype=np.uint8)
    if atomic_numbers.shape != (n_atoms,):
        raise ValueError(
            f"atomic_numbers must have shape ({n_atoms},), got {atomic_numbers.shape}"
        )
    return atomic_numbers


def _optional_array(
    molecule: Any,
    field: str,
    *,
    dtype: Any,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    value = getattr(molecule, field, None)
    if value is None:
        return None
    return _as_array(field, value, dtype=dtype, shape=shape)


def _optional_matrix(molecule: Any, field: str) -> np.ndarray | None:
    value = getattr(molecule, field, None)
    if value is None:
        return None
    return _as_array(field, value, dtype=np.float64, shape=(3, 3))


def _optional_scalar(molecule: Any, field: str) -> float | None:
    value = getattr(molecule, field, None)
    if value is None:
        return None
    return float(value)


def _optional_pbc(molecule: Any) -> tuple[bool, bool, bool] | None:
    value = getattr(molecule, "pbc", None)
    if value is None:
        return None
    array = np.asarray(value, dtype=bool)
    if array.shape != (3,):
        raise ValueError(f"pbc must have shape (3,), got {array.shape}")
    return (bool(array[0]), bool(array[1]), bool(array[2]))


def _optional_name(molecule: Any) -> str | None:
    value = getattr(molecule, "name", None)
    if value is None:
        return None
    return str(value)


def _normalized_custom_properties(molecule: Any) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for key in sorted(_property_keys(molecule)):
        properties[key] = _normalize_custom_property(_get_property(molecule, key))
    return properties


def _custom_value_signature(value: Any) -> tuple[str, tuple[int, ...]]:
    if isinstance(value, str):
        return ("str", ())
    if isinstance(value, int):
        return ("int", ())
    if isinstance(value, float):
        return ("float", ())
    array = np.asarray(value)
    return (str(array.dtype), tuple(array.shape))


def _custom_signature(molecule: Any) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    properties = _normalized_custom_properties(molecule)
    return tuple(
        (key, *(_custom_value_signature(value)))
        for key, value in properties.items()
    )


def _builtin_signature(molecule: Any) -> tuple[int, tuple[bool, ...]]:
    n_atoms = len(_positions(molecule))
    flags = (
        getattr(molecule, "energy", None) is not None,
        getattr(molecule, "forces", None) is not None,
        getattr(molecule, "charges", None) is not None,
        getattr(molecule, "velocities", None) is not None,
        getattr(molecule, "cell", None) is not None,
        getattr(molecule, "stress", None) is not None,
        getattr(molecule, "pbc", None) is not None,
        getattr(molecule, "name", None) is not None,
    )
    return n_atoms, flags


def _conversion_signature(
    molecule: Any,
) -> tuple[tuple[int, tuple[bool, ...]], tuple[tuple[str, str, tuple[int, ...]], ...]]:
    return _builtin_signature(molecule), _custom_signature(molecule)


def _iter_conversion_runs(molecules: list[Any]) -> list[list[Any]]:
    runs: list[list[Any]] = []
    current_run: list[Any] = []
    current_signature: (
        tuple[tuple[int, tuple[bool, ...]], tuple[tuple[str, str, tuple[int, ...]], ...]] | None
    ) = None

    for molecule in molecules:
        signature = _conversion_signature(molecule)
        if current_signature is None or signature == current_signature:
            current_run.append(molecule)
            current_signature = signature
            continue
        runs.append(current_run)
        current_run = [molecule]
        current_signature = signature

    if current_run:
        runs.append(current_run)

    return runs


def _stack_custom_property(values: list[Any]) -> list[str] | np.ndarray:
    first = values[0]
    if isinstance(first, str):
        return [str(value) for value in values]
    if isinstance(first, int):
        return np.asarray(values, dtype=np.int64)
    if isinstance(first, float):
        return np.asarray(values, dtype=np.float64)

    template = np.asarray(first)
    stacked = np.stack([np.asarray(value, dtype=template.dtype) for value in values])
    return stacked.astype(template.dtype, copy=False)


def _batched_custom_properties(molecules: list[Any]) -> dict[str, list[str] | np.ndarray]:
    first_properties = _normalized_custom_properties(molecules[0])
    if not first_properties:
        return {}

    batched: dict[str, list[str] | np.ndarray] = {}
    normalized_rows = [_normalized_custom_properties(molecule) for molecule in molecules]
    for key in first_properties:
        batched[key] = _stack_custom_property([properties[key] for properties in normalized_rows])
    return batched


def _write_batch_run(db: Any, molecules: list[Any]) -> None:
    first = molecules[0]
    positions0 = _positions(first)
    n_atoms = positions0.shape[0]

    positions = np.stack([_positions(molecule) for molecule in molecules]).astype(
        np.float32, copy=False
    )
    atomic_numbers = np.stack(
        [_atomic_numbers(molecule, n_atoms) for molecule in molecules]
    ).astype(np.uint8, copy=False)

    kwargs: dict[str, Any] = {}

    if getattr(first, "energy", None) is not None:
        kwargs["energy"] = np.asarray(
            [_optional_scalar(molecule, "energy") for molecule in molecules], dtype=np.float64
        )
    if getattr(first, "forces", None) is not None:
        kwargs["forces"] = np.stack(
            [
                _optional_array(molecule, "forces", dtype=np.float32, shape=(n_atoms, 3))
                for molecule in molecules
            ]
        ).astype(np.float32, copy=False)
    if getattr(first, "charges", None) is not None:
        kwargs["charges"] = np.stack(
            [
                _optional_array(molecule, "charges", dtype=np.float64, shape=(n_atoms,))
                for molecule in molecules
            ]
        ).astype(np.float64, copy=False)
    if getattr(first, "velocities", None) is not None:
        kwargs["velocities"] = np.stack(
            [
                _optional_array(molecule, "velocities", dtype=np.float32, shape=(n_atoms, 3))
                for molecule in molecules
            ]
        ).astype(np.float32, copy=False)
    if getattr(first, "cell", None) is not None:
        kwargs["cell"] = np.stack(
            [_optional_matrix(molecule, "cell") for molecule in molecules]
        ).astype(np.float64, copy=False)
    if getattr(first, "stress", None) is not None:
        kwargs["stress"] = np.stack(
            [_optional_matrix(molecule, "stress") for molecule in molecules]
        ).astype(np.float64, copy=False)
    if getattr(first, "pbc", None) is not None:
        kwargs["pbc"] = np.asarray([_optional_pbc(molecule) for molecule in molecules], dtype=bool)
    if getattr(first, "name", None) is not None:
        kwargs["name"] = [_optional_name(molecule) for molecule in molecules]
    custom_properties = _batched_custom_properties(molecules)
    if custom_properties:
        kwargs["properties"] = custom_properties

    db.add_arrays_batch(positions, atomic_numbers, **kwargs)


def _normalize_custom_property(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)

    array = np.asarray(value)
    if array.ndim == 0:
        if array.dtype.kind in {"b", "i", "u"}:
            return int(array.item())
        if array.dtype.kind == "f":
            return float(array.item())
        raise ValueError(f"Unsupported scalar property dtype: {array.dtype}")

    if array.ndim == 1:
        if array.dtype == np.float32:
            return array.astype(np.float32, copy=False)
        if array.dtype.kind == "f":
            return array.astype(np.float64, copy=False)
        if array.dtype == np.int32:
            return array.astype(np.int32, copy=False)
        if array.dtype.kind in {"b", "i", "u"}:
            return array.astype(np.int64, copy=False)
        raise ValueError(f"Unsupported vector property dtype: {array.dtype}")

    if array.ndim == 2 and array.shape[1] == 3:
        if array.dtype == np.float32:
            return array.astype(np.float32, copy=False)
        if array.dtype.kind == "f":
            return array.astype(np.float64, copy=False)
        raise ValueError(f"Unsupported (n, 3) property dtype: {array.dtype}")

    raise ValueError(f"Unsupported property shape {array.shape}")


def _materialize_current_molecule(atompack_module: ModuleType, legacy_molecule: Any) -> Any:
    positions = _positions(legacy_molecule)
    n_atoms = positions.shape[0]
    molecule = atompack_module.Molecule.from_arrays(
        positions,
        _atomic_numbers(legacy_molecule, n_atoms),
        energy=_optional_scalar(legacy_molecule, "energy"),
        forces=_optional_array(legacy_molecule, "forces", dtype=np.float32, shape=(n_atoms, 3)),
        charges=_optional_array(legacy_molecule, "charges", dtype=np.float64, shape=(n_atoms,)),
        velocities=_optional_array(
            legacy_molecule,
            "velocities",
            dtype=np.float32,
            shape=(n_atoms, 3),
        ),
        cell=_optional_matrix(legacy_molecule, "cell"),
        stress=_optional_matrix(legacy_molecule, "stress"),
        pbc=_optional_pbc(legacy_molecule),
        name=_optional_name(legacy_molecule),
    )
    for key in _property_keys(legacy_molecule):
        molecule.set_property(key, _normalize_custom_property(_get_property(legacy_molecule, key)))
    return molecule


def _print_progress(job: ShardJob, stats: ShardStats, total: int, started_at: float) -> None:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    throughput = stats.molecules / elapsed
    print(
        f"[{job.label}] {stats.molecules:,}/{total:,} mols | {throughput:,.0f} mol/s | "
        f"fast={stats.fast_molecules:,} fallback={stats.fallback_molecules:,}",
        flush=True,
    )


def convert_shard(
    job: ShardJob,
    *,
    legacy_module: ModuleType,
    atompack_module: ModuleType = atompack,
    compression: str = "none",
    level: int = 3,
    batch_size: int = 4096,
    report_every: int = 100_000,
) -> ShardStats:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    job.target.parent.mkdir(parents=True, exist_ok=True)
    legacy_db = _open_legacy_database(legacy_module, job.source)
    total = len(legacy_db)
    writer = atompack_module.Database(
        str(job.target),
        compression=compression,
        level=level,
        overwrite=True,
    )
    stats = ShardStats()
    started_at = time.perf_counter()
    next_report = report_every if report_every > 0 else None

    try:
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            molecules = _load_legacy_batch(legacy_db, start, stop)
            try:
                for run in _iter_conversion_runs(molecules):
                    _write_batch_run(writer, run)
                stats.fast_batches += 1
                stats.fast_molecules += len(molecules)
            except ValueError:
                converted = [
                    _materialize_current_molecule(atompack_module, molecule)
                    for molecule in molecules
                ]
                writer.add_molecules(converted)
                stats.fallback_batches += 1
                stats.fallback_molecules += len(molecules)

            stats.molecules += len(molecules)
            if next_report is not None and stats.molecules >= next_report:
                _print_progress(job, stats, total, started_at)
                next_report += report_every

        writer.flush()
    except Exception:
        del writer
        try:
            job.target.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    _print_progress(job, stats, total, started_at)
    return stats


def _convert_job_in_subprocess(
    job: ShardJob,
    *,
    legacy_module_name: str,
    atompack_module_name: str,
    compression: str,
    level: int,
    batch_size: int,
    report_every: int,
) -> ShardStats:
    legacy_module = load_legacy_atompack(legacy_module_name)
    atompack_module = importlib.import_module(atompack_module_name)
    return convert_shard(
        job,
        legacy_module=legacy_module,
        atompack_module=atompack_module,
        compression=compression,
        level=level,
        batch_size=batch_size,
        report_every=report_every,
    )


def _convert_jobs_parallel(
    jobs: list[ShardJob],
    *,
    legacy_module_name: str,
    atompack_module_name: str,
    compression: str,
    level: int,
    batch_size: int,
    report_every: int,
    continue_on_error: bool,
    workers: int,
) -> int:
    exit_code = 0
    max_workers = min(workers, len(jobs))
    future_to_job: dict[concurrent.futures.Future[ShardStats], ShardJob] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for job in jobs:
            print(f"Converting {job.source} -> {job.target}", flush=True)
            future = executor.submit(
                _convert_job_in_subprocess,
                job,
                legacy_module_name=legacy_module_name,
                atompack_module_name=atompack_module_name,
                compression=compression,
                level=level,
                batch_size=batch_size,
                report_every=report_every,
            )
            future_to_job[future] = job

        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Failed to convert {job.source}: {exc}", file=sys.stderr, flush=True)
                if not continue_on_error:
                    raise
                exit_code = 1

    return exit_code


def convert_jobs(
    jobs: list[ShardJob],
    *,
    legacy_module: ModuleType,
    atompack_module: ModuleType = atompack,
    compression: str = "none",
    level: int = 3,
    batch_size: int = 4096,
    report_every: int = 100_000,
    continue_on_error: bool = False,
    workers: int = 1,
) -> int:
    if workers <= 0:
        raise ValueError("--workers must be positive")

    legacy_module_name = _module_name(legacy_module)
    atompack_module_name = _module_name(atompack_module)
    can_parallelize = (
        workers > 1
        and len(jobs) > 1
        and legacy_module_name is not None
        and atompack_module_name is not None
    )
    if can_parallelize:
        return _convert_jobs_parallel(
            jobs,
            legacy_module_name=legacy_module_name,
            atompack_module_name=atompack_module_name,
            compression=compression,
            level=level,
            batch_size=batch_size,
            report_every=report_every,
            continue_on_error=continue_on_error,
            workers=workers,
        )

    if workers > 1 and len(jobs) > 1:
        print(
            "Parallel shard conversion requested, but custom in-process modules are not "
            "re-importable in worker processes; falling back to sequential conversion.",
            file=sys.stderr,
            flush=True,
        )

    exit_code = 0
    for job in jobs:
        print(f"Converting {job.source} -> {job.target}", flush=True)
        try:
            convert_shard(
                job,
                legacy_module=legacy_module,
                atompack_module=atompack_module,
                compression=compression,
                level=level,
                batch_size=batch_size,
                report_every=report_every,
            )
        except Exception as exc:
            print(f"Failed to convert {job.source}: {exc}", file=sys.stderr, flush=True)
            if not continue_on_error:
                raise
            exit_code = 1
    return exit_code


def main(
    argv: list[str] | None = None,
    *,
    legacy_module: ModuleType | None = None,
    atompack_module: ModuleType = atompack,
) -> int:
    args = _parser().parse_args(argv)
    legacy_module = legacy_module or load_legacy_atompack()
    jobs = plan_jobs(args.input_path, args.output, overwrite=args.overwrite)
    return convert_jobs(
        jobs,
        legacy_module=legacy_module,
        atompack_module=atompack_module,
        compression=args.compression,
        level=args.level,
        batch_size=args.batch_size,
        report_every=args.report_every,
        continue_on_error=args.continue_on_error,
        workers=args.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
