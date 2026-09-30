# Copyright 2026 Entalpic
"""Generate the documentation viewer demo; energies/forces are illustrative, not DFT.

From atompack-py: uv run --no-sync --with ase==3.26.0 python ../examples/vscode_demo.py
Close the demo in VS Code before regenerating it (the viewer uses mmap).
"""

from collections import defaultdict
from pathlib import Path

import atompack
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111

OUTPUT = Path(__file__).resolve().parents[1] / "docs/source/_static/data/catalysis-demo.atp"
METALS = {"Pt": 3.92, "Pd": 3.89, "Cu": 3.61, "Ni": 3.52, "Au": 4.08, "Ag": 4.09}
ADSORBATES = {
    "CO": Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]]),
    "OH": Atoms("OH", positions=[[0, 0, 0], [0.65, 0, 0.73]]),
    "O": Atoms("O", positions=[[0, 0, 0]]),
    "H": Atoms("H", positions=[[0, 0, 0]]),
    "NH3": Atoms(
        "NH3", positions=[[0, 0, 0], [0.94, 0, 0.38], [-0.47, 0.81, 0.38], [-0.47, -0.81, 0.38]]
    ),
}
SITES = ["ontop", "bridge", "fcc", "hcp"]


def main():
    rng = np.random.default_rng(20260930)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    db = atompack.Database(str(OUTPUT), compression="zstd", overwrite=True)

    def store(atoms, label, energy, force_scale=0.01, **properties):
        mol = atompack.Molecule.from_arrays(
            np.asarray(atoms.positions, dtype=np.float32),
            np.asarray(atoms.numbers, dtype=np.uint8),
            name=label,
        )
        if atoms.pbc.any():
            mol.cell = np.asarray(atoms.cell)
            mol.pbc = tuple(bool(v) for v in atoms.pbc)
        mol.energy = float(energy)
        mol.forces = rng.normal(0, force_scale, (len(atoms), 3)).astype(np.float32)
        for key, value in properties.items():
            mol.set_property(key, value)
        index = len(db)
        db.add_molecule(mol)
        return index

    gas_ids = {}
    gas_energy = {}
    for i, (name, atoms) in enumerate(ADSORBATES.items()):
        gas_energy[name] = -4.0 * len(atoms) - i
        gas_ids[name] = store(atoms, name, gas_energy[name], kind="gas", adsorbate=name)

    adsorption, adsorption_props = [], defaultdict(list)
    site_groups, site_props = [], defaultdict(list)
    series = defaultdict(dict)
    relaxations, relaxation_props = [], defaultdict(list)

    for metal_index, (metal, lattice) in enumerate(METALS.items()):
        for strain in [-0.03, -0.015, 0.0, 0.015, 0.03]:
            slab = fcc111(metal, size=(3, 3, 3), a=lattice * (1 + strain), vacuum=7.0)
            slab_energy = -len(slab) * (3.5 + 0.35 * metal_index) + 400 * strain**2
            slab_id = store(
                slab, f"{metal}(111) slab", slab_energy, kind="slab", metal=metal, strain=strain
            )
            for ads_index, (adsorbate, molecule) in enumerate(ADSORBATES.items()):
                members, energies = {}, []
                for site_index, site in enumerate(SITES):
                    atoms = slab.copy()
                    add_adsorbate(
                        atoms, molecule.copy(), height=1.75 + 0.05 * ads_index, position=site
                    )
                    # A deterministic illustrative landscape, deliberately not a physical model.
                    binding = -0.55 - 0.24 * ads_index - 0.12 * metal_index - 0.16 * site_index
                    binding += 5 * strain + rng.normal(0, 0.06)
                    energy = slab_energy + gas_energy[adsorbate] + binding
                    record = store(
                        atoms,
                        f"{adsorbate} on {metal}(111) · {site}",
                        energy,
                        force_scale=0.02 + 0.5 * abs(strain) + 0.007 * site_index,
                        kind="adsorbed",
                        metal=metal,
                        adsorbate=adsorbate,
                        site=site,
                        strain=strain,
                        adsorption_energy=float(binding),
                    )
                    members[site] = record
                    energies.append(binding)
                    series[(adsorbate, site, strain)][metal] = record
                    adsorption.append(
                        {"adsorbed": record, "slab": slab_id, "gas": gas_ids[adsorbate]}
                    )
                    for key, value in {
                        "metal": metal,
                        "adsorbate": adsorbate,
                        "site": site,
                        "strain": strain,
                        "adsorption_energy": binding,
                    }.items():
                        adsorption_props[key].append(value)

                    # A short synthetic approach/relaxation sequence for one site at zero strain.
                    if strain == 0 and site == "ontop":
                        frames = []
                        for step in range(7):
                            frame = atoms.copy()
                            frame.positions[len(slab) :, 2] += 0.8 * (1 - step / 6)
                            frames.append(
                                store(
                                    frame,
                                    f"{adsorbate}/{metal} · step {step}",
                                    energy + 0.3 * (1 - step / 6) ** 2,
                                    force_scale=0.12 * (1 - step / 6) + 0.005,
                                    kind="trajectory",
                                    metal=metal,
                                    adsorbate=adsorbate,
                                    step=step,
                                )
                            )
                        relaxations.append(frames)
                        relaxation_props["metal"].append(metal)
                        relaxation_props["adsorbate"].append(adsorbate)
                site_groups.append(members)
                for key, value in {
                    "metal": metal,
                    "adsorbate": adsorbate,
                    "strain": strain,
                    "energy_spread": max(energies) - min(energies),
                }.items():
                    site_props[key].append(value)

    db.add_groups("adsorption", adsorption, dict(adsorption_props))
    db.add_groups("site_comparison", site_groups, dict(site_props))
    db.add_groups(
        "metal_series",
        list(series.values()),
        {
            "adsorbate": [key[0] for key in series],
            "site": [key[1] for key in series],
            "strain": [key[2] for key in series],
        },
    )
    db.add_groups("relaxation", relaxations, dict(relaxation_props))
    db.flush()
    print(f"{OUTPUT}: {len(db)} records, {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
