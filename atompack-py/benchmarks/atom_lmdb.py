# Copyright 2026 Entalpic
from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_LMDB_MAGIC = b"APLM"
_LMDB_HEADER = struct.Struct("<4sBBI")  # magic, version, flags, n_atoms
_LMDB_VERSION = 2
_LMDB_FLAG_WITH_PROPS = 1 << 0
_LMDB_FLAG_WITH_CELL_PBC = 1 << 1
_LMDB_FLAG_WITH_STRESS = 1 << 2
_LMDB_FLAG_WITH_CHARGES = 1 << 3
_LMDB_FLAG_WITH_CUSTOM = 1 << 4

_SUPPORTED_LMDB_VERSIONS = {_LMDB_VERSION, 3}
_CUSTOM_EIGENVALUES_LEN = 20


def _pack_custom_payload(custom: dict[str, Any], n_atoms: int) -> bytes:
    required = {
        "bandgap",
        "formation_energy",
        "eigenvalues",
        "mulliken_charges",
        "hirshfeld_volumes",
    }
    missing = sorted(required - set(custom))
    if missing:
        raise ValueError(f"Missing required custom LMDB fields: {', '.join(missing)}")

    eigenvalues = np.ascontiguousarray(custom["eigenvalues"], dtype=np.float64).reshape(-1)
    if eigenvalues.shape != (_CUSTOM_EIGENVALUES_LEN,):
        raise ValueError(
            f"eigenvalues must have shape ({_CUSTOM_EIGENVALUES_LEN},), got {eigenvalues.shape}"
        )
    mulliken = np.ascontiguousarray(custom["mulliken_charges"], dtype=np.float64).reshape(-1)
    if mulliken.shape != (n_atoms,):
        raise ValueError(f"mulliken_charges must have shape ({n_atoms},), got {mulliken.shape}")
    hirshfeld = np.ascontiguousarray(custom["hirshfeld_volumes"], dtype=np.float64).reshape(-1)
    if hirshfeld.shape != (n_atoms,):
        raise ValueError(f"hirshfeld_volumes must have shape ({n_atoms},), got {hirshfeld.shape}")

    payload = bytearray()
    payload += struct.pack("<d", float(custom["bandgap"]))
    payload += struct.pack("<d", float(custom["formation_energy"]))
    payload += eigenvalues.tobytes(order="C")
    payload += mulliken.tobytes(order="C")
    payload += hirshfeld.tobytes(order="C")
    return bytes(payload)


def lmdb_codec_fns(
    codec: str,
) -> tuple[Callable[[bytes], bytes] | None, Callable[[bytes], bytes] | None]:
    codec = (codec or "none").lower()
    if codec == "none":
        return None, None
    if codec.startswith("zstd:"):
        try:
            import zstandard as zstd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("zstandard is required for lmdb:zstd (pip install zstandard)") from e
        level = int(codec.split(":", 1)[1])
        compressor = zstd.ZstdCompressor(level=level)
        decompressor = zstd.ZstdDecompressor()
        return compressor.compress, decompressor.decompress
    if codec == "lz4":
        try:
            import lz4.frame  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("lz4 is required for lmdb:lz4 (pip install lz4)") from e
        return lz4.frame.compress, lz4.frame.decompress
    raise ValueError(f"Unsupported lmdb codec: {codec}")


def pack_record(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray | None,
    pbc: np.ndarray | None,
    energy: float | None,
    forces: np.ndarray | None,
    stress: np.ndarray | None = None,
    charges: np.ndarray | None = None,
    custom: dict[str, Any] | None = None,
) -> bytes:
    n_atoms = int(positions.shape[0])
    flags = 0
    if energy is not None and forces is not None:
        flags |= _LMDB_FLAG_WITH_PROPS
    if cell is not None and pbc is not None:
        flags |= _LMDB_FLAG_WITH_CELL_PBC
    if stress is not None:
        flags |= _LMDB_FLAG_WITH_STRESS
    if charges is not None:
        flags |= _LMDB_FLAG_WITH_CHARGES
    custom_payload = b""
    if custom is not None:
        custom_payload = _pack_custom_payload(custom, n_atoms)
        flags |= _LMDB_FLAG_WITH_CUSTOM
    version = 3 if custom_payload else _LMDB_VERSION
    header = _LMDB_HEADER.pack(_LMDB_MAGIC, version, flags, n_atoms)

    payload = bytearray()
    payload += header
    payload += np.ascontiguousarray(positions, dtype=np.float32).tobytes(order="C")
    payload += np.ascontiguousarray(atomic_numbers, dtype=np.uint8).tobytes(order="C")
    if flags & _LMDB_FLAG_WITH_CELL_PBC:
        payload += np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3).tobytes(order="C")  # type: ignore[arg-type]
        payload += np.ascontiguousarray(pbc, dtype=np.uint8).reshape(3).tobytes(order="C")  # type: ignore[arg-type]
    if flags & _LMDB_FLAG_WITH_PROPS:
        payload += struct.pack("<d", float(energy))
        payload += np.ascontiguousarray(forces, dtype=np.float32).tobytes(order="C")
    if flags & _LMDB_FLAG_WITH_STRESS:
        payload += np.ascontiguousarray(stress, dtype=np.float64).reshape(3, 3).tobytes(order="C")  # type: ignore[arg-type]
    if flags & _LMDB_FLAG_WITH_CHARGES:
        payload += (
            np.ascontiguousarray(charges, dtype=np.float64).reshape(n_atoms).tobytes(order="C")
        )  # type: ignore[arg-type]
    if custom_payload:
        payload += custom_payload
    return bytes(payload)


def unpack_arrays(
    value: bytes,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    float | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    magic, version, flags, n_atoms = _LMDB_HEADER.unpack_from(value, 0)
    if magic != _LMDB_MAGIC or version not in _SUPPORTED_LMDB_VERSIONS:
        raise ValueError("Invalid LMDB record")

    offset = _LMDB_HEADER.size
    pos_nbytes = int(n_atoms) * 3 * 4
    z_nbytes = int(n_atoms)

    positions_b = memoryview(value)[offset : offset + pos_nbytes]
    offset += pos_nbytes
    atomic_numbers_b = memoryview(value)[offset : offset + z_nbytes]
    offset += z_nbytes

    positions = np.frombuffer(positions_b, dtype=np.float32).reshape(n_atoms, 3)
    atomic_numbers = np.frombuffer(atomic_numbers_b, dtype=np.uint8)

    cell: np.ndarray | None = None
    pbc: np.ndarray | None = None
    if flags & _LMDB_FLAG_WITH_CELL_PBC:
        cell_nbytes = 9 * 8
        cell_b = memoryview(value)[offset : offset + cell_nbytes]
        offset += cell_nbytes
        pbc_b = memoryview(value)[offset : offset + 3]
        offset += 3
        cell = np.frombuffer(cell_b, dtype=np.float64).reshape(3, 3)
        pbc = np.frombuffer(pbc_b, dtype=np.uint8).astype(bool, copy=False)

    energy: float | None = None
    forces: np.ndarray | None = None
    if flags & _LMDB_FLAG_WITH_PROPS:
        energy = struct.unpack_from("<d", value, offset)[0]
        offset += 8
        forces_b = memoryview(value)[offset : offset + pos_nbytes]
        forces = np.frombuffer(forces_b, dtype=np.float32).reshape(n_atoms, 3)
        offset += pos_nbytes

    stress: np.ndarray | None = None
    if flags & _LMDB_FLAG_WITH_STRESS:
        stress_nbytes = 9 * 8
        stress_b = memoryview(value)[offset : offset + stress_nbytes]
        offset += stress_nbytes
        stress = np.frombuffer(stress_b, dtype=np.float64).reshape(3, 3)

    charges: np.ndarray | None = None
    if flags & _LMDB_FLAG_WITH_CHARGES:
        charges_nbytes = int(n_atoms) * 8
        charges_b = memoryview(value)[offset : offset + charges_nbytes]
        offset += charges_nbytes
        charges = np.frombuffer(charges_b, dtype=np.float64)

    return n_atoms, positions, atomic_numbers, cell, pbc, energy, forces, stress, charges


def unpack_custom_fields(value: bytes) -> dict[str, Any] | None:
    magic, version, flags, n_atoms = _LMDB_HEADER.unpack_from(value, 0)
    if magic != _LMDB_MAGIC or version not in _SUPPORTED_LMDB_VERSIONS:
        raise ValueError("Invalid LMDB record")
    if not (flags & _LMDB_FLAG_WITH_CUSTOM):
        return None

    offset = _LMDB_HEADER.size
    offset += int(n_atoms) * 3 * 4  # positions
    offset += int(n_atoms)  # atomic_numbers

    if flags & _LMDB_FLAG_WITH_CELL_PBC:
        offset += 9 * 8 + 3
    if flags & _LMDB_FLAG_WITH_PROPS:
        offset += 8
        offset += int(n_atoms) * 3 * 4
    if flags & _LMDB_FLAG_WITH_STRESS:
        offset += 9 * 8
    if flags & _LMDB_FLAG_WITH_CHARGES:
        offset += int(n_atoms) * 8

    bandgap = struct.unpack_from("<d", value, offset)[0]
    offset += 8
    formation_energy = struct.unpack_from("<d", value, offset)[0]
    offset += 8

    eigenvalues_nbytes = _CUSTOM_EIGENVALUES_LEN * 8
    eigenvalues = np.frombuffer(
        memoryview(value)[offset : offset + eigenvalues_nbytes],
        dtype=np.float64,
    )
    offset += eigenvalues_nbytes

    per_atom_nbytes = int(n_atoms) * 8
    mulliken = np.frombuffer(
        memoryview(value)[offset : offset + per_atom_nbytes],
        dtype=np.float64,
    )
    offset += per_atom_nbytes
    hirshfeld = np.frombuffer(
        memoryview(value)[offset : offset + per_atom_nbytes],
        dtype=np.float64,
    )

    return {
        "bandgap": bandgap,
        "formation_energy": formation_energy,
        "eigenvalues": eigenvalues,
        "mulliken_charges": mulliken,
        "hirshfeld_volumes": hirshfeld,
    }


def decode_record(value: bytes, decode_numpy: bool) -> None:
    _, pos, z, cell, pbc, _energy, forces, stress, charges = unpack_arrays(value)
    if decode_numpy:
        _ = pos
        _ = z
        _ = cell
        _ = pbc
        _ = forces
        _ = stress
        _ = charges


def env_used_bytes(env: Any) -> int | None:
    # Estimate bytes used by pages up to last_pgno (not the full map_size / data.mdb file size).
    try:
        info = env.info()
        stat = env.stat()
        last_pgno = int(info.get("last_pgno", -1))
        psize = int(stat.get("psize", 0))
        if last_pgno >= 0 and psize > 0:
            return (last_pgno + 1) * psize
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class AtomLMDBConfig:
    atoms_per_molecule: int
    with_props: bool
    with_cell_pbc: bool = False
    with_stress: bool = False
    with_charges: bool = False
    codec: str = "none"
    durable: bool = True
    readahead: bool = True
    map_factor: float = 3.0
    map_slack_mib: int = 1024


class AtomLMDB:
    """
    Minimal LMDB-backed store for fixed-size atomistic structures.

    Key: big-endian u64 molecule id.
    Value: binary blob: header + positions (float32) + Z (uint8) [+ energy (float64) + forces (float32)].
    Optional compression is applied to the value bytes.
    """

    def __init__(self, path: Path, cfg: AtomLMDBConfig):
        self.path = Path(path)
        self.cfg = cfg
        self._compress, self._decompress = lmdb_codec_fns(cfg.codec)

    @staticmethod
    def estimate_map_size_bytes(
        num_molecules: int,
        atoms_per_molecule: int,
        with_props: bool,
        with_cell_pbc: bool,
        with_stress: bool = False,
        with_charges: bool = False,
        map_factor: float = 3.0,
        map_slack_mib: int = 1024,
    ) -> int:
        per_record = _LMDB_HEADER.size + atoms_per_molecule * (3 * 4 + 1)
        if with_cell_pbc:
            per_record += 9 * 8 + 3
        if with_props:
            per_record += 8 + atoms_per_molecule * 3 * 4
        if with_stress:
            per_record += 9 * 8
        if with_charges:
            per_record += atoms_per_molecule * 8
        map_factor = float(map_factor)
        if map_factor < 1.3:
            map_factor = 1.3
        slack = int(max(0, int(map_slack_mib))) * 1024 * 1024
        return int(per_record * int(num_molecules) * map_factor + slack)

    def open_env(self, *, map_size: int, readonly: bool, lock: bool) -> Any:
        import lmdb  # type: ignore

        return lmdb.open(
            str(self.path),
            map_size=int(map_size),
            subdir=True,
            max_dbs=1,
            lock=bool(lock),
            readonly=bool(readonly),
            sync=bool(self.cfg.durable) if not readonly else True,
            metasync=bool(self.cfg.durable) if not readonly else True,
            readahead=bool(self.cfg.readahead),
            meminit=False,
        )

    def reset_dir(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def put_batch(
        self,
        txn: Any,
        start_id: int,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None,
        pbc: np.ndarray | None,
        energy: np.ndarray | None,
        forces: np.ndarray | None,
    ) -> None:
        batch_n = int(positions.shape[0])
        for i in range(batch_n):
            key = struct.pack(">Q", int(start_id) + i)
            value = pack_record(
                positions[i],
                atomic_numbers[i],
                cell,
                pbc,
                float(energy[i]) if self.cfg.with_props else None,  # type: ignore[index]
                forces[i] if self.cfg.with_props else None,  # type: ignore[index]
            )
            if self._compress is not None:
                value = self._compress(value)
            txn.put(key, value)

    def get_value(self, txn: Any, idx: int) -> bytes | None:
        key = struct.pack(">Q", int(idx))
        value = txn.get(key)
        if value is None:
            return None
        value = self.decompress_value(value)
        return value

    def get_values(self, txn: Any, idxs: list[int]) -> list[bytes | None]:
        # Batch helper to reduce Python overhead in benchmark loops.
        # Note: LMDB itself is still key-by-key; this just centralizes decompression.
        out: list[bytes | None] = []
        for idx in idxs:
            key = struct.pack(">Q", int(idx))
            value = txn.get(key)
            if value is None:
                out.append(None)
                continue
            out.append(self.decompress_value(value))
        return out

    def decompress_value(self, value: bytes) -> bytes:
        if self._decompress is None:
            return value
        return self._decompress(value)
