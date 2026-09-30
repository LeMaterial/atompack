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
reader alongside your files. CI builds downloadable packages on pull requests and
main; extension release tags also publish those packages to the VS Code Marketplace.

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

## Release to the VS Code Marketplace

The **VS Code extension** workflow builds, tests, and packages all five supported
platforms before publishing the resulting VSIX files. Pull requests, pushes to main,
and manual runs on branches only build packages. Publishing runs on tags named
`atompack-vscode-v<VERSION>`; the tag must match `version` in `package.json`.
These tags are separate from the Rust/Python `v*` release tags.

Before the first release, register the `publisher` ID from `package.json` (currently
`atompack`) on the [Marketplace publisher management page](https://marketplace.visualstudio.com/manage).
Configure its [GitHub trusted publishing policy](https://github.com/microsoft/vscode-vsce#trusted-publishing)
for repository `LeMaterial/atompack`, workflow `vscode.yml`, and GitHub environment
`vscode-marketplace`. Create that environment under repository **Settings → Environments**.
Publishing uses GitHub OIDC and a short-lived Marketplace credential; no stored PAT
or Azure credentials are required. The workflow performs the documented exchange
directly because `vsce` 4.0.0 predates Microsoft's
[final token-exchange contract](https://github.com/microsoft/vscode-vsce/commit/c960f2e97da3360899f2bfe93390fa4890c69327).
Once a released CLI includes that fix, replace the exchange step with `vsce publish --oidc`.

For example, after merging the extension and its `0.1.0` manifest into main:

```sh
git switch main
git pull --ff-only
git tag atompack-vscode-v0.1.0
git push origin atompack-vscode-v0.1.0
```

For later releases, update `version` in `package.json` and `package-lock.json` together
(for example, `npm version patch --no-git-tag-version` from `atompack-vscode/`), merge
the change, then tag the matching version. If publishing stops after uploading some
platforms, rerun the failed job: `--skip-duplicate` keeps already-published targets
and publishes the remaining ones. Release runs are not canceled by newer builds.
