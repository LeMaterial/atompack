# Copyright 2026 Entalpic
from __future__ import annotations

import pickle
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _load_codec_fns():
    try:
        from atom_lmdb import lmdb_codec_fns
    except ModuleNotFoundError:
        this_dir = Path(__file__).resolve().parent
        if str(this_dir) not in sys.path:
            sys.path.insert(0, str(this_dir))
        from atom_lmdb import lmdb_codec_fns
    return lmdb_codec_fns


lmdb_codec_fns = _load_codec_fns()


@dataclass(frozen=True)
class PickleLMDBConfig:
    codec: str = "none"
    durable: bool = True
    readahead: bool = True
    map_factor: float = 3.0
    map_slack_mib: int = 1024
    pickle_protocol: int = 5


class PickleAtomLMDB:
    """
    LMDB store with a common Python-style payload:
    pickled dict with numpy arrays and optional scalar properties.
    """

    def __init__(self, path: Path, cfg: PickleLMDBConfig):
        self.path = Path(path)
        self.cfg = cfg
        self._compress, self._decompress = lmdb_codec_fns(cfg.codec)

    @staticmethod
    def estimate_map_size_bytes(
        num_molecules: int,
        atoms_per_molecule: int,
        with_props: bool,
        with_cell_pbc: bool,
        map_factor: float,
        map_slack_mib: int,
    ) -> int:
        base = int(atoms_per_molecule) * (3 * 4 + 1) + 256
        if with_cell_pbc:
            base += 9 * 8 + 3
        if with_props:
            base += 8 + int(atoms_per_molecule) * 3 * 4
        factor = max(1.3, float(map_factor))
        slack = int(max(0, int(map_slack_mib))) * 1024 * 1024
        return int(int(num_molecules) * base * factor + slack)

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

    def encode_payload(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: np.ndarray | None = None,
        energy: float | None = None,
        forces: np.ndarray | None = None,
        stress: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        custom: dict[str, Any] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "positions": np.ascontiguousarray(positions, dtype=np.float32),
            "atomic_numbers": np.ascontiguousarray(atomic_numbers, dtype=np.uint8),
        }
        if cell is not None:
            payload["cell"] = np.ascontiguousarray(cell, dtype=np.float64).reshape(3, 3)
        if pbc is not None:
            payload["pbc"] = np.ascontiguousarray(pbc, dtype=np.uint8).reshape(3)
        if energy is not None:
            payload["energy"] = float(energy)
        if forces is not None:
            payload["forces"] = np.ascontiguousarray(forces, dtype=np.float32)
        if stress is not None:
            payload["stress"] = np.ascontiguousarray(stress, dtype=np.float64).reshape(3, 3)
        if charges is not None:
            payload["charges"] = np.ascontiguousarray(charges, dtype=np.float64)
        if custom:
            for k, v in custom.items():
                if isinstance(v, np.ndarray):
                    payload[k] = np.ascontiguousarray(v)
                else:
                    payload[k] = v
        encoded = pickle.dumps(payload, protocol=self.cfg.pickle_protocol)
        if self._compress is not None:
            return self._compress(encoded)
        return encoded

    def decode_payload(self, value: bytes) -> dict[str, Any]:
        raw = self.decompress_value(value)
        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Pickle LMDB payload must be a dict")
        if "positions" not in payload or "atomic_numbers" not in payload:
            raise ValueError("Pickle LMDB payload missing required keys")
        return payload

    def decompress_value(self, value: bytes) -> bytes:
        if self._decompress is None:
            return value
        return self._decompress(value)

    def put_molecule(
        self,
        txn: Any,
        idx: int,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: np.ndarray | None = None,
        energy: float | None = None,
        forces: np.ndarray | None = None,
        stress: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        custom: dict[str, Any] | None = None,
    ) -> None:
        key = struct.pack(">Q", int(idx))
        value = self.encode_payload(
            positions,
            atomic_numbers,
            cell,
            pbc,
            energy,
            forces,
            stress,
            charges,
            custom,
        )
        txn.put(key, value)

    def get_payload(self, txn: Any, idx: int) -> dict[str, Any] | None:
        key = struct.pack(">Q", int(idx))
        value = txn.get(key)
        if value is None:
            return None
        return self.decode_payload(value)

    def get_payloads(self, txn: Any, idxs: list[int]) -> list[dict[str, Any] | None]:
        out: list[dict[str, Any] | None] = []
        for idx in idxs:
            key = struct.pack(">Q", int(idx))
            value = txn.get(key)
            if value is None:
                out.append(None)
                continue
            out.append(self.decode_payload(value))
        return out
