from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_STUB = ROOT / "python" / "atompack" / "_atompack_rs.pyi"
PUBLIC_STUB = ROOT / "python" / "atompack" / "__init__.pyi"
HUB_STUB = ROOT / "python" / "atompack" / "hub.pyi"


def _class_method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def _class_docstring(path: Path, class_name: str) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_docstring(node)
    raise AssertionError(f"Class {class_name!r} not found in {path}")


def _function_docstring(path: Path, function_name: str) -> str | None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return ast.get_docstring(node)
    raise AssertionError(f"Function {function_name!r} not found in {path}")


def _function_arg_names(path: Path, function_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [arg.arg for arg in node.args.args]
            args.extend(arg.arg for arg in node.args.kwonlyargs)
            return args
    raise AssertionError(f"Function {function_name!r} not found in {path}")


def test_private_stub_tracks_low_level_surface() -> None:
    molecule_methods = _class_method_names(PRIVATE_STUB, "PyMolecule")
    assert {
        "__init__",
        "from_arrays",
        "to_owned",
        "_ase_builtin_tuple_fast",
        "_ase_payload",
        "__getitem__",
    } <= molecule_methods

    database_methods = _class_method_names(PRIVATE_STUB, "PyAtomDatabase")
    assert {"add_arrays_batch", "get_molecules_flat"} <= database_methods

    text = PRIVATE_STUB.read_text(encoding="utf-8")
    assert 'compression: str = "none"' in text
    assert "overwrite: bool = False" in text
    assert "Parameters" in (_class_docstring(PRIVATE_STUB, "PyAtom") or "")
    assert "Atomic positions" in (_class_docstring(PRIVATE_STUB, "PyMolecule") or "")
    assert "Compression type" in (_class_docstring(PRIVATE_STUB, "PyAtomDatabase") or "")


def test_public_stub_exposes_flat_batch_reader() -> None:
    database_methods = _class_method_names(PUBLIC_STUB, "Database")
    assert "get_molecules_flat" in database_methods


def test_hub_stub_has_public_docstrings() -> None:
    reader_doc = _class_docstring(HUB_STUB, "AtompackReader") or ""
    assert "lexicographically ordered shard set" in reader_doc

    download_doc = _function_docstring(HUB_STUB, "download") or ""
    assert "shard directory" in download_doc

    upload_doc = _function_docstring(HUB_STUB, "upload") or ""
    assert "Xet" in upload_doc
    assert "use_xet" in _function_arg_names(HUB_STUB, "upload")

    open_doc = _function_docstring(HUB_STUB, "open") or ""
    assert "download" in open_doc.lower()

    open_path_doc = _function_docstring(HUB_STUB, "open_path") or ""
    assert "Directories are scanned recursively" in open_path_doc
