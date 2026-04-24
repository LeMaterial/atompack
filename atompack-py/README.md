# Atompack

Fast, compressed storage for atomic structures with properties.

Atompack is part of the open-source [LeMaterial](https://lematerial.org) effort for
large-scale materials and molecular ML datasets.

## Install

```bash
uv pip install atompack-db
```

Import the package as `atompack` after installation.

The Hugging Face Hub upload/download helpers are included in the main package.

## Quick start

```python
import atompack
import numpy as np

positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
atomic_numbers = np.array([6, 8], dtype=np.uint8)
mol = atompack.Molecule.from_arrays(positions, atomic_numbers)
mol.energy = -123.456
mol.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

db = atompack.Database("data.atp", overwrite=True)
db.add_molecule(mol)
db.flush()
```

Reopen with mmap for read-only access, or disable mmap when you want to append:

```python
db = atompack.Database.open("data.atp")             # read-only, mmap=True by default
db = atompack.Database.open("data.atp", mmap=False)  # writable append mode
```

## Hugging Face Hub

```python
import atompack

db = atompack.hub.open("LeMaterial/Atompack", "omat/train")
print(db[0].energy)
db.close()
```

Remote datasets such as `omat/train` and `omol/train` can be opened directly from
[`LeMaterial/Atompack`](https://huggingface.co/datasets/LeMaterial/Atompack).

## More

Full documentation and benchmark notes live at
`https://entalpic-atompack.readthedocs-hosted.com/en/latest/`.

If you are working on the Rust crate directly, the maintained Rust entrypoints are:

- `cargo run -p atompack --example basic_usage`
- `cargo run -p atompack --release --bin atompack-bench -- --help`

## License

Apache-2.0
