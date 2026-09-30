# Atompack Viewer

Open `.atp` databases in VS Code to browse records and groups, plot properties, and
view atomic structures. The viewer uses the native Atompack mmap reader; installing
the extension requires neither Rust nor Python.

![Six-metal adsorption comparison](https://raw.githubusercontent.com/LeMaterial/atompack/main/docs/source/_static/img/viewer/compare.png)

Try the [catalysis demo](https://github.com/LeMaterial/atompack/blob/main/docs/source/_static/data/catalysis-demo.atp) with the
[viewer walkthrough](https://entalpic-atompack.readthedocs-hosted.com/en/latest/vscode-viewer.html). Its 845 structures include
four groupings; energies and forces are synthetic demonstration values.

## Install

Download the VSIX matching your extension host from the **VS Code extension** GitHub
Actions artifacts, then run **Extensions: Install from VSIX** in VS Code. Packages
are built for macOS (Intel and Apple Silicon), Linux with glibc 2.35 or newer (x64
and ARM64), and Windows x64. Windows ARM64, Alpine/musl, and browser-only VS Code
are not currently packaged.

For Remote SSH, WSL, or Dev Containers, install into the remote workspace and choose
the remote machine's OS and architecture. `extensionKind: workspace` runs the native
reader alongside your files. The repository's workflow builds downloadable packages;
it does not publish to the Marketplace.

Local files are memory mapped read-only. Resources supplied by a virtual filesystem
are copied into a private temporary directory, which is removed when the document
closes. Treat open databases as immutable: close the viewer before modifying or
replacing the underlying file.

## Known limitations

- Overwriting or truncating a database while it is open can crash the shared VS Code
  extension host. Close all viewer tabs for that file before regenerating it.
  Process isolation or non-mmap reading is tracked in [#51](https://github.com/LeMaterial/atompack/issues/51).
- Sending a group with the same record in multiple roles to Compare can crash that
  view. Avoid comparing repeated members until [#52](https://github.com/LeMaterial/atompack/issues/52) is fixed.
- Synchronized cameras share rotation and position, but orthographic zoom currently
  needs adjusting in each pane. Tracked in [#53](https://github.com/LeMaterial/atompack/issues/53).

## Develop

Install Node.js 22.13+ and stable Rust with the native C/C++ build tools for your OS.
From `atompack-vscode/`:

```sh
npm ci
npm run build
npm run check
npm test
npm run package
```

Packaging automatically tags the VSIX for the current OS and CPU. CI builds and
tests each supported target on a matching runner. The Node-API 8 binding lives in
`atompack-node/` and calls the existing `AtomDatabase` API without modifying the
core library. Plot scans run in Node worker threads with independent readers.
MatterViz renders structures in the webview and retains its own browser WASM assets.

To include a larger database in the reader and worker smoke tests, set `ATP_FILE`
to its absolute path when running `npm test`.
