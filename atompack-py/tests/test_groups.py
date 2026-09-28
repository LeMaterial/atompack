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


def test_iteration_and_shared_records_in_batches(tmp_path: Path) -> None:
    db = _db(tmp_path / "many.atp")
    n_groups = 600  # crosses the iterator's 256-group prefetch boundary
    members = [[i % 4, (i + 1) % 4] for i in range(n_groups)]
    db.add_groups("many", members, {"i": list(range(n_groups))})
    db.add_groups("repeat", [[0, 0, 1]])
    many = db.groups["many"]

    assert [g.indices for g in many] == members
    assert [g.properties["i"] for g in many] == list(range(n_groups))
    batch = many[[0, 4, 0, 599]]  # same group twice, same records across groups
    assert [g.indices for g in batch] == [members[0], members[4], members[0], members[599]]
    assert [[m.energy for m in g] for g in batch] == [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [3.0, 0.0],
    ]
    assert [m.energy for m in db.groups["repeat"][0]] == [0.0, 0.0, 1.0]


def _shard(path: Path, groups: list | None, props: dict | None = None) -> None:
    db = _db(path)  # 4 records with energies 0..3
    if groups is not None:
        db.add_groups("pairs", groups, props)
    db.flush()


def test_sharded_reader_groups(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    _shard(shards / "a.atp", [{"x": 0, "y": 1}, {"x": 2, "y": 3}], {"e": [0.5, 1.5]})
    _shard(shards / "b.atp", None)  # no groups in this shard
    _shard(shards / "c.atp", [{"x": 3, "z": 0}], {"e": [2.5]})

    reader = atompack.hub.open_path(shards)
    assert list(reader.groups) == ["pairs"] and "pairs" in reader.groups
    pairs = reader.groups["pairs"]
    assert len(pairs) == 3
    assert pairs.roles == ["x", "y", "z"]
    np.testing.assert_array_equal(pairs.properties["e"], [0.5, 1.5, 2.5])

    # Record indices are global: reader[index] is the same record as the member.
    assert pairs[2].indices == {"x": 11, "z": 8}
    assert pairs[-1].indices == pairs[2].indices
    for group in pairs:
        for role, index in group.indices.items():
            assert reader[index].energy == group[role].energy

    batch = pairs[[2, 0, 2]]  # crosses shards, keeps order
    assert [g.properties["e"] for g in batch] == [2.5, 0.5, 2.5]
    assert [g.properties["e"] for g in pairs[1:]] == [1.5, 2.5]
    with pytest.raises(IndexError):
        pairs[3]
    with pytest.raises(KeyError):
        reader.groups["missing"]


def test_sharded_reader_rejects_inconsistent_groupings(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    _shard(shards / "a.atp", [[0, 1]], {"e": [0.5]})
    _shard(shards / "b.atp", [[2]], {"e": [1]})  # int column vs float column
    pairs = atompack.hub.open_path(shards).groups["pairs"]
    with pytest.raises(ValueError, match="different types"):
        _ = pairs.properties

    (shards / "b.atp").unlink()
    _shard(shards / "b.atp", [{"x": 2}], {"e": [1.0]})  # named vs ordered
    with pytest.raises(ValueError, match="mixes"):
        atompack.hub.open_path(shards).groups["pairs"]
