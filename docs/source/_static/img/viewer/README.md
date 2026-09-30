# Viewer figures

Real screenshots captured in a fresh VS Code window using the **Atompack PR50 Review**
profile and the CI-built `darwin-arm64` VSIX from commit
`69dca5c43561c9f4616ba0ecdcb64ddc6ec0b586`:
[CI run](https://github.com/LeMaterial/atompack/actions/runs/36698467256).

Captured on 2026-09-30 in VS Code 1.139.1, dark theme, side panels hidden.
The 2400 × 1600 captures were cropped to the viewer area
(`x=106, y=190, width=2284, height=1344`), removing editor chrome and temporary paths.
No interface elements or data were composited or retouched. For the web, the crops were
scaled to 1600 px wide and saved as WebP (`cwebp -q 90 -m 6 -sharp_yuv`), about 100 KB each.

| Asset | View and selection | Suggested use |
| --- | --- | --- |
| `compare.webp` | `metal_series` group 0; records 6, 146, 286, 426, 566, 706; three columns; synchronized cameras | README and walkthrough hero |
| `groups.webp` | `adsorption`, metal `Cu`, adsorbate `CO`, group 240; records 328, 327, 0 | Group relationships |
| `records.webp` | `formula = COCu27, adsorption_energy < -1`; 11 matches; record 287 | Filtering and inspection |
| `plots.webp` | Scatter `adsorption_energy` versus `fmax`; zoomed to 584 records; record 606 | Plot selection workflow |

Force arrows are disabled for clarity. Structures retain the viewer's default camera
and periodic images. The demo has 845 records: 5 gas references, 30 clean slabs,
600 adsorption structures, and 210 trajectory frames. Group counts are 600 adsorption,
150 site comparison, 100 metal series, and 30 relaxation.

Source: `examples/vscode_demo.py`, ASE 3.26.0, seed 20260930.
The dataset is `docs/source/_static/data/catalysis-demo.atp` (708,876 bytes), SHA-256:
`75db68bdc3ca9083b1c3ca9d8bbedd1245732971e6a26fc7e0c1171e61dc842e`.
Energies and forces are synthetic, and geometries are illustrative.

Manual verification included every viewer tab on the six-record CI fixture,
logarithmic `fmax` selection (5 records in both Plots and Records), and this demo's
zoomed scatter (584 records in both views). The latter transferred bounds without
rounding and sorted Records by adsorption energy.
