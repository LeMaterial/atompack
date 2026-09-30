"""Write test/fixtures/groups.atp: 6 records (molecules and crystals) and two groupings.

Run from atompack-py: uv run --no-sync python ../atompack-vscode/test/make_fixture.py
"""

from pathlib import Path

import atompack
import numpy as np

path = Path(__file__).parent / "fixtures" / "groups.atp"
path.parent.mkdir(exist_ok=True)
path.unlink(missing_ok=True)

db = atompack.Database(str(path), compression="zstd")
for i in range(6):
    n = i + 1
    mol = atompack.Molecule(
        np.arange(3 * n, dtype=np.float32).reshape(n, 3) * 0.5 + i,
        np.full(n, 6 + i % 3, dtype=np.uint8),
    )
    mol.energy = -float(i)
    mol.forces = np.full((n, 3), 0.1 * i, dtype=np.float32)
    if i % 2:
        mol.cell = np.eye(3) * (5.0 + i)
        mol.pbc = (True, True, i != 5)
    mol.set_property("tag", f"r{i}")
    db.add_molecule(mol)
db.add_groups(
    "adsorption",
    [{"slab": 1, "adslab": 3}, {"slab": 1, "adslab": 5, "gas": 0}],
    {"e_ads": [-1.5, 0.25], "id": ["a", "b"], "n": np.array([2, 3])},
)
db.add_groups("ordered", [[4, 2], [0]])
db.flush()
print(path, path.stat().st_size, "bytes")
