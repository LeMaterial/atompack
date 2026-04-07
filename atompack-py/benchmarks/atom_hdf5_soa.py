# Copyright 2026 Entalpic
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py as _h5py  # type: ignore
except ImportError:
    _h5py = None

_HDF5_SOA_VERSION = 2


def hdf5_available() -> bool:
    return _h5py is not None


def _require_hdf5() -> None:
    if _h5py is None:
        raise RuntimeError("h5py is required for hdf5_soa benchmarks")


@dataclass(frozen=True)
class AtomHdf5SoaConfig:
    atoms_per_molecule: int
    with_props: bool
    with_cell_pbc: bool = False
    with_stress: bool = False
    with_charges: bool = False
    compression: str = "none"
    compression_opts: int | None = None
    chunk_size: int = 1024
    rdcc_nbytes_mib: int = 128
    rdcc_nslots: int = 1_000_003
    reader_cache_chunks: int = 8


@dataclass
class AtomHdf5SoaReader:
    file: Any
    meta: dict[str, Any]
    positions: Any
    atomic_numbers: Any
    energy: Any | None
    forces: Any | None
    cell: Any | None
    pbc: Any | None
    stress: Any | None
    charges: Any | None
    properties: dict[str, Any]
    atom_properties: dict[str, Any]
    chunk_size: int
    cache_chunks: int
    chunk_cache: OrderedDict[int, dict[str, Any]]

    def close(self) -> None:
        self.file.close()


class AtomHdf5Soa:
    """Fixed-shape SOA layout stored as conventional HDF5 datasets."""

    def __init__(self, path: Path, cfg: AtomHdf5SoaConfig):
        self.path = Path(path)
        self.cfg = cfg

    def reset_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def open_file(self, mode: str) -> Any:
        _require_hdf5()
        kwargs: dict[str, Any] = {}
        if mode == "r":
            kwargs["rdcc_nbytes"] = int(self.cfg.rdcc_nbytes_mib) * 1024 * 1024
            kwargs["rdcc_nslots"] = int(self.cfg.rdcc_nslots)
        return _h5py.File(self.path, mode, **kwargs)

    def _compression_kwargs(self) -> dict[str, Any]:
        if self.cfg.compression == "none":
            return {}
        kwargs: dict[str, Any] = {"compression": self.cfg.compression}
        if self.cfg.compression_opts is not None:
            kwargs["compression_opts"] = self.cfg.compression_opts
        return kwargs

    def _dataset_shape_dtype(
        self,
        field: str,
        num_molecules: int,
    ) -> tuple[tuple[int, ...], np.dtype]:
        atoms = self.cfg.atoms_per_molecule
        if field == "positions":
            return (num_molecules, atoms, 3), np.dtype(np.float32)
        if field == "atomic_numbers":
            return (num_molecules, atoms), np.dtype(np.uint8)
        if field == "energy":
            return (num_molecules,), np.dtype(np.float64)
        if field == "forces":
            return (num_molecules, atoms, 3), np.dtype(np.float32)
        if field == "cell":
            return (num_molecules, 3, 3), np.dtype(np.float64)
        if field == "pbc":
            return (num_molecules, 3), np.dtype(np.uint8)
        if field == "stress":
            return (num_molecules, 3, 3), np.dtype(np.float64)
        if field == "charges":
            return (num_molecules, atoms), np.dtype(np.float64)
        raise KeyError(f"Unknown field: {field}")

    def _chunk_shape(self, field: str, num_molecules: int) -> tuple[int, ...]:
        shape, _dtype = self._dataset_shape_dtype(field, num_molecules)
        return (min(self.cfg.chunk_size, num_molecules), *shape[1:])

    def _custom_chunk_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        return (min(self.cfg.chunk_size, shape[0]), *shape[1:])

    def _fields(self) -> list[str]:
        fields = ["positions", "atomic_numbers"]
        if self.cfg.with_props:
            fields.extend(["energy", "forces"])
        if self.cfg.with_cell_pbc:
            fields.extend(["cell", "pbc"])
        if self.cfg.with_stress:
            fields.append("stress")
        if self.cfg.with_charges:
            fields.append("charges")
        return fields

    def create_file(self, num_molecules: int) -> None:
        _require_hdf5()
        self.reset_file()
        compression_kwargs = self._compression_kwargs()
        with self.open_file("w") as handle:
            handle.attrs["version"] = _HDF5_SOA_VERSION
            handle.attrs["atoms_per_molecule"] = int(self.cfg.atoms_per_molecule)
            handle.attrs["num_molecules"] = int(num_molecules)
            handle.attrs["chunk_size"] = int(self.cfg.chunk_size)
            handle.attrs["with_props"] = bool(self.cfg.with_props)
            handle.attrs["with_cell_pbc"] = bool(self.cfg.with_cell_pbc)
            handle.attrs["with_stress"] = bool(self.cfg.with_stress)
            handle.attrs["with_charges"] = bool(self.cfg.with_charges)
            handle.attrs["compression"] = self.cfg.compression
            handle.require_group("properties")
            handle.require_group("atom_properties")

            for field in self._fields():
                shape, dtype = self._dataset_shape_dtype(field, num_molecules)
                handle.create_dataset(
                    field,
                    shape=shape,
                    dtype=dtype,
                    chunks=self._chunk_shape(field, num_molecules),
                    **compression_kwargs,
                )

    def open_reader(self) -> AtomHdf5SoaReader:
        handle = self.open_file("r")
        meta = {str(key): handle.attrs[key] for key in handle.attrs.keys()}
        properties_group = handle.get("properties")
        atom_properties_group = handle.get("atom_properties")
        return AtomHdf5SoaReader(
            file=handle,
            meta=meta,
            positions=handle["positions"],
            atomic_numbers=handle["atomic_numbers"],
            energy=handle.get("energy"),
            forces=handle.get("forces"),
            cell=handle.get("cell"),
            pbc=handle.get("pbc"),
            stress=handle.get("stress"),
            charges=handle.get("charges"),
            properties=(
                {
                    str(name): dataset
                    for name, dataset in properties_group.items()
                }
                if properties_group is not None
                else {}
            ),
            atom_properties=(
                {
                    str(name): dataset
                    for name, dataset in atom_properties_group.items()
                }
                if atom_properties_group is not None
                else {}
            ),
            chunk_size=int(meta["chunk_size"]),
            cache_chunks=max(0, int(self.cfg.reader_cache_chunks)),
            chunk_cache=OrderedDict(),
        )

    def _normalize_custom_value(self, value: Any, *, key: str) -> np.ndarray:
        if isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"custom dataset '{key}' must have a batch dimension")
        return arr

    def _custom_dataset_spec(
        self,
        value: Any,
        *,
        key: str,
        num_molecules: int,
        batch_n: int,
        atom_property: bool,
    ) -> tuple[np.ndarray, tuple[int, ...], Any]:
        arr = self._normalize_custom_value(value, key=key)
        if arr.shape[0] != batch_n:
            raise ValueError(
                f"custom dataset '{key}' first dimension ({arr.shape[0]}) "
                f"doesn't match batch size ({batch_n})"
            )
        if atom_property:
            atoms = self.cfg.atoms_per_molecule
            if arr.ndim < 2 or arr.shape[1] != atoms:
                raise ValueError(
                    f"atom property '{key}' must have shape (batch, {atoms}, ...)"
                )
        if arr.dtype.kind in {"U", "S", "O"}:
            if atom_property:
                raise ValueError(f"atom property '{key}' string dtype is not supported")
            if arr.ndim != 1:
                raise ValueError(f"string property '{key}' must have shape (batch,)")
            dtype = _h5py.string_dtype(encoding="utf-8")  # type: ignore[union-attr]
            arr = arr.astype(str, copy=False)
        else:
            dtype = arr.dtype
        shape = (num_molecules, *arr.shape[1:])
        return arr, shape, dtype

    def _ensure_custom_datasets(
        self,
        handle: Any,
        group_name: str,
        values: dict[str, Any] | None,
        *,
        start_id: int,
        end_id: int,
        atom_property: bool,
    ) -> None:
        if not values:
            return

        group = handle.require_group(group_name)
        compression_kwargs = self._compression_kwargs()
        num_molecules = int(handle.attrs["num_molecules"])
        batch_n = end_id - start_id

        for key, value in values.items():
            arr, shape, dtype = self._custom_dataset_spec(
                value,
                key=key,
                num_molecules=num_molecules,
                batch_n=batch_n,
                atom_property=atom_property,
            )
            if key in group:
                dataset = group[key]
                if tuple(dataset.shape) != shape:
                    raise ValueError(
                        f"custom dataset '{group_name}/{key}' has shape {dataset.shape}, "
                        f"expected {shape}"
                    )
            else:
                dataset = group.create_dataset(
                    key,
                    shape=shape,
                    dtype=dtype,
                    chunks=self._custom_chunk_shape(shape),
                    **compression_kwargs,
                )
            dataset[start_id:end_id] = arr

    def put_batch(
        self,
        handle: Any,
        start_id: int,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None,
        pbc: np.ndarray | None,
        energy: np.ndarray | None,
        forces: np.ndarray | None,
        stress: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        properties: dict[str, Any] | None = None,
        atom_properties: dict[str, Any] | None = None,
    ) -> None:
        start_id = int(start_id)
        batch_n = int(positions.shape[0])
        end_id = start_id + batch_n
        atoms = self.cfg.atoms_per_molecule
        if positions.shape != (batch_n, atoms, 3):
            raise ValueError(f"positions must have shape ({batch_n}, {atoms}, 3)")
        if atomic_numbers.shape != (batch_n, atoms):
            raise ValueError(f"atomic_numbers must have shape ({batch_n}, {atoms})")

        handle["positions"][start_id:end_id] = np.asarray(positions, dtype=np.float32)
        handle["atomic_numbers"][start_id:end_id] = np.asarray(atomic_numbers, dtype=np.uint8)

        if self.cfg.with_props:
            if energy is None or forces is None:
                raise ValueError("energy and forces are required when with_props=True")
            handle["energy"][start_id:end_id] = np.asarray(energy, dtype=np.float64)
            handle["forces"][start_id:end_id] = np.asarray(forces, dtype=np.float32)

        if self.cfg.with_cell_pbc:
            if cell is None or pbc is None:
                raise ValueError("cell and pbc are required when with_cell_pbc=True")
            handle["cell"][start_id:end_id] = np.asarray(cell, dtype=np.float64)
            handle["pbc"][start_id:end_id] = np.asarray(pbc, dtype=np.uint8)

        if self.cfg.with_stress:
            if stress is None:
                raise ValueError("stress is required when with_stress=True")
            handle["stress"][start_id:end_id] = np.asarray(stress, dtype=np.float64)

        if self.cfg.with_charges:
            if charges is None:
                raise ValueError("charges are required when with_charges=True")
            handle["charges"][start_id:end_id] = np.asarray(charges, dtype=np.float64)

        self._ensure_custom_datasets(
            handle,
            "properties",
            properties,
            start_id=start_id,
            end_id=end_id,
            atom_property=False,
        )
        self._ensure_custom_datasets(
            handle,
            "atom_properties",
            atom_properties,
            start_id=start_id,
            end_id=end_id,
            atom_property=True,
        )

    def _decode_custom_item(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray) and value.dtype.kind == "S":
            return value.astype(str)
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _validate_idx(self, reader: AtomHdf5SoaReader, idx: int) -> None:
        total = int(reader.meta["num_molecules"])
        if idx < 0 or idx >= total:
            raise IndexError(f"Index {idx} out of bounds for HDF5 SOA of length {total}")

    def _remember_chunk(
        self,
        reader: AtomHdf5SoaReader,
        chunk_id: int,
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        if reader.cache_chunks <= 0:
            return chunk
        reader.chunk_cache[chunk_id] = chunk
        reader.chunk_cache.move_to_end(chunk_id)
        while len(reader.chunk_cache) > reader.cache_chunks:
            reader.chunk_cache.popitem(last=False)
        return chunk

    def _load_chunk(self, reader: AtomHdf5SoaReader, chunk_id: int) -> dict[str, Any]:
        cached = reader.chunk_cache.get(chunk_id)
        if cached is not None:
            reader.chunk_cache.move_to_end(chunk_id)
            return cached

        total = int(reader.meta["num_molecules"])
        start = chunk_id * reader.chunk_size
        stop = min(total, start + reader.chunk_size)
        if start >= total:
            raise IndexError(f"Chunk {chunk_id} out of range")

        chunk: dict[str, Any] = {
            "positions": reader.positions[start:stop],
            "atomic_numbers": reader.atomic_numbers[start:stop],
        }
        if reader.energy is not None:
            chunk["energy"] = reader.energy[start:stop]
        if reader.forces is not None:
            chunk["forces"] = reader.forces[start:stop]
        if reader.cell is not None:
            chunk["cell"] = reader.cell[start:stop]
        if reader.pbc is not None:
            chunk["pbc"] = reader.pbc[start:stop]
        if reader.stress is not None:
            chunk["stress"] = reader.stress[start:stop]
        if reader.charges is not None:
            chunk["charges"] = reader.charges[start:stop]
        if reader.properties:
            chunk["properties"] = {
                key: dataset[start:stop]
                for key, dataset in reader.properties.items()
            }
        if reader.atom_properties:
            chunk["atom_properties"] = {
                key: dataset[start:stop]
                for key, dataset in reader.atom_properties.items()
            }
        return self._remember_chunk(reader, chunk_id, chunk)

    def _payload_from_chunk(self, chunk: dict[str, Any], local: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "positions": chunk["positions"][local],
            "atomic_numbers": chunk["atomic_numbers"][local],
        }
        if "energy" in chunk:
            payload["energy"] = float(chunk["energy"][local])
        if "forces" in chunk:
            payload["forces"] = chunk["forces"][local]
        if "cell" in chunk:
            payload["cell"] = chunk["cell"][local]
        if "pbc" in chunk:
            payload["pbc"] = chunk["pbc"][local].astype(bool, copy=False)
        if "stress" in chunk:
            payload["stress"] = chunk["stress"][local]
        if "charges" in chunk:
            payload["charges"] = chunk["charges"][local]
        if "properties" in chunk:
            payload["properties"] = {
                key: self._decode_custom_item(values[local])
                for key, values in chunk["properties"].items()
            }
        if "atom_properties" in chunk:
            payload["atom_properties"] = {
                key: self._decode_custom_item(values[local])
                for key, values in chunk["atom_properties"].items()
            }
        return payload

    def get_payload(self, reader: AtomHdf5SoaReader, idx: int) -> dict[str, Any]:
        idx = int(idx)
        self._validate_idx(reader, idx)
        chunk_id, local = divmod(idx, reader.chunk_size)
        chunk = self._load_chunk(reader, chunk_id)
        return self._payload_from_chunk(chunk, local)

    def get_payloads(self, reader: AtomHdf5SoaReader, idxs: list[int]) -> list[dict[str, Any]]:
        grouped: OrderedDict[int, list[tuple[int, int]]] = OrderedDict()
        payloads: list[dict[str, Any] | None] = [None] * len(idxs)

        for out_pos, raw_idx in enumerate(idxs):
            idx = int(raw_idx)
            self._validate_idx(reader, idx)
            chunk_id, local = divmod(idx, reader.chunk_size)
            grouped.setdefault(chunk_id, []).append((out_pos, local))

        for chunk_id, group in grouped.items():
            chunk = self._load_chunk(reader, chunk_id)
            for out_pos, local in group:
                payloads[out_pos] = self._payload_from_chunk(chunk, local)

        if any(payload is None for payload in payloads):
            raise RuntimeError("Missing payload while reconstructing HDF5 batch")
        return [payload for payload in payloads if payload is not None]
