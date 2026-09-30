# Atompack Viewer

Open `.atp` databases in VS Code to browse records and groups, plot properties, and
view atomic structures. The viewer uses the native Atompack mmap reader; installing
the extension requires neither Rust nor Python.

![Six-metal adsorption comparison](https://raw.githubusercontent.com/LeMaterial/atompack/main/docs/source/_static/img/viewer/compare.png)

Try the [catalysis demo](https://github.com/LeMaterial/atompack/blob/main/docs/source/_static/data/catalysis-demo.atp) with the
[viewer walkthrough](https://entalpic-atompack.readthedocs-hosted.com/en/latest/vscode-viewer.html). Its 845 structures include
four groupings; energies and forces are synthetic demonstration values.

## Tour

**Records.** Filter by composition and properties, for example
`formula = COCu27, adsorption_energy < -1`, then click a row to see its structure and
metadata. **copy #** copies the matching record numbers as a Python list, ready for
`db.get_molecules(indices)`.

![Eleven filtered CO-on-copper records beside the selected structure and its metadata](https://raw.githubusercontent.com/LeMaterial/atompack/main/docs/source/_static/img/viewer/records.png)

**Groups.** Browse related records with their roles, such as an adsorbed structure, its
clean slab, and the gas-phase reference.

![A filtered adsorption group displaying the combined structure, copper slab, and gas-phase CO](https://raw.githubusercontent.com/LeMaterial/atompack/main/docs/source/_static/img/viewer/groups.png)

**Compare.** Place up to 12 records or group members side by side, as in the image
above. With **sync cameras** on, rotating, panning, or zooming one pane moves the others,
while each stays framed on its own structure. **Reset view** reframes every pane.

**Plots.** Draw histograms or scatter plots of any numeric property. Drag a box to zoom,
click a point to see its structure, and **Open in Records** to carry the visible bounds
and sort order into the record table.

![Adsorption energy versus maximum force, with 584 records in the zoomed region and a selected gold structure](https://raw.githubusercontent.com/LeMaterial/atompack/main/docs/source/_static/img/viewer/plots.png)

## Install

Install **Atompack Viewer** from the
[Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Ramlaoui.atompack-vscode),
or run:

```sh
code --install-extension Ramlaoui.atompack-vscode
```

Packages are published for macOS (Intel and Apple Silicon), Linux with glibc 2.35 or
newer (x64 and ARM64), and Windows x64. Windows ARM64, Alpine/musl, and browser-only
VS Code are not currently supported.

For Remote SSH, WSL, or Dev Containers, use **Install in SSH: …** (or the equivalent
remote action) in the Extensions view: `extensionKind: workspace` runs the native reader
on the remote machine, next to your files, and VS Code fetches the package for its
platform. Preview builds are also available as VSIX files from the **VS Code extension**
GitHub Actions runs; install them with **Extensions: Install from VSIX**.

## Reading files

Local files are memory mapped read-only. Resources supplied by a virtual filesystem
are copied into a private temporary directory, which is removed when the document
closes. The reader runs in its own process, so overwriting or truncating a database
while it is open stops only that process: pending requests report the error, and the
next request restarts the reader. Reopen the file to see its new contents.

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
core library. Each open database has a reader process (`src/reader-process.ts`), whose
plot scans run in worker threads with independent readers.
MatterViz renders structures in the webview; `vite.config.ts` leaves out its file-parsing
and glTF decoder payloads, which the viewer never loads.

To include a larger database in the reader and worker smoke tests, set `ATP_FILE`
to its absolute path when running `npm test`.
