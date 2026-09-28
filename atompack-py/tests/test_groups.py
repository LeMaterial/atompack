# Copyright 2026 Entalpic
from __future__ import annotations

from pathlib import Path

import atompack
import numpy as np
import pytest


def _db(path: Path, n: int = 4) -> atompack.Database:
    db = atompack.Database(str(path), compression="zstd")
    for i in range(n):
        mol = atompack.Molecule(
            np.array([[float(i), 0.0, 0.0]], dtype=np.float32),
            np.array([6], dtype=np.uint8),
        )
        mol.energy = float(i)
        db.add_molecule(mol)
    return db


def test_named_groups_round_trip_and_share_records(tmp_path: Path) -> None:
    path = tmp_path / "groups.atp"
    db = _db(path)
    db.add_groups(
        "adsorption",
        [{"adslab": 1, "slab": 0}, {"adslab": 2, "slab": 0, "gas": 3}],
        {"e_ads": [-1.5, -0.25], "id": ["a", "b"], "n": np.array([2, 3])},
    )
    db.add_groups("ordered", [[3, 1, 2]])
    db.flush()

    for db in (atompack.Database.open(str(path)), atompack.Database.open(str(path), mmap=False)):
        assert len(db) == 4
        assert list(db.groups) == ["adsorption", "ordered"]
        assert "adsorption" in db.groups and "missing" not in db.groups
        ads = db.groups["adsorption"]
        assert (ads.name, len(ads), ads.roles) == ("adsorption", 2, ["adslab", "slab", "gas"])
        assert db.groups["ordered"].roles == []

        props = ads.properties
        np.testing.assert_array_equal(props["e_ads"], [-1.5, -0.25])
        assert props["e_ads"].dtype == np.float64
        assert props["n"].dtype == np.int64
        assert props["id"] == ["a", "b"]

        g = ads[1]
        assert isinstance(g, atompack.Group)
        assert g.indices == {"adslab": 2, "slab": 0, "gas": 3}
        assert g.properties == {"e_ads": -0.25, "id": "b", "n": 3}
        assert g["gas"].energy == 3.0
        assert "slab" in g and len(g) == 3 and list(g) == ["adslab", "slab", "gas"]
        assert ads[-1].indices == g.indices

        # Batches (list / slice / numpy) read the shared clean slab once.
        for batch in (ads[[0, 1]], ads[:], ads[np.array([0, 1])]):
            assert [grp.indices["slab"] for grp in batch] == [0, 0]
            assert [grp["slab"].energy for grp in batch] == [0.0, 0.0]
        assert [grp.properties["id"] for grp in ads] == ["a", "b"]
        assert ads[::-1][0].properties["id"] == "b"

        ordered = db.groups["ordered"][0]
        assert ordered.indices == [3, 1, 2]
        assert [m.energy for m in ordered] == [3.0, 1.0, 2.0]
        assert ordered[0].energy == 3.0


def test_append_groups_across_sessions(tmp_path: Path) -> None:
    path = tmp_path / "groups.atp"
    db = _db(path)
    db.add_groups("pairs", [{"a": 0, "b": 1}], {"y": [1.0]})
    db.flush()

    db = atompack.Database.open(str(path), mmap=False)
    db.add_molecule(db[0])
    db.add_groups(
        "pairs",
        np.array([[4, -1, 2], [3, 1, -1]]),
        {"y": np.array([2.0, 3.0])},
        roles=["a", "c", "b"],
    )
    db.flush()

    db = atompack.Database.open(str(path))
    pairs = db.groups["pairs"]
    assert len(pairs) == 3
    assert pairs.roles == ["a", "b", "c"]
    assert pairs[1].indices == {"a": 4, "b": 2}
    assert pairs[2].indices == {"a": 3, "c": 1}
    np.testing.assert_array_equal(pairs.properties["y"], [1.0, 2.0, 3.0])


def test_files_without_groups(tmp_path: Path) -> None:
    path = tmp_path / "plain.atp"
    _db(path).flush()
    db = atompack.Database.open(str(path))
    assert len(db.groups) == 0 and list(db.groups) == []
    with pytest.raises(KeyError):
        db.groups["missing"]


@pytest.mark.parametrize(
    ("members", "properties", "error"),
    [
        ([[0, 4]], None, ValueError),  # record does not exist
        ([[0, -2]], None, ValueError),  # negative index
        ([[]], None, ValueError),  # empty group
        ([{"a": 0}, [1]], None, TypeError),  # mixed named/ordered
        ([[0], [1]], {"y": [1.0]}, ValueError),  # one value per group
        ([[0]], {"y": [{"x": 1}]}, TypeError),  # unsupported value
    ],
)
def test_invalid_groups_are_rejected(tmp_path: Path, members, properties, error) -> None:
    db = _db(tmp_path / "bad.atp")
    with pytest.raises(error):
        db.add_groups("g", members, properties)
    assert list(db.groups) == []


def test_group_errors(tmp_path: Path) -> None:
    path = tmp_path / "groups.atp"
    db = _db(path)
    db.add_groups("g", [[0, 1]], {"y": [1.0]})
    with pytest.raises(ValueError, match="properties"):
        db.add_groups("g", [[2]], {"z": [1.0]})
    with pytest.raises(ValueError, match="entries"):
        db.add_groups("h", [[0, 1]], roles=["a"])
    db.flush()

    db = atompack.Database.open(str(path))
    for bad in (1, -2, [0, 1]):
        with pytest.raises(IndexError):
            db.groups["g"][bad]
    with pytest.raises(ValueError, match="read-only"):
        db.add_groups("g", [[0]], {"y": [1.0]})
