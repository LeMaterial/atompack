import type { Vec3 } from 'matterviz/math'
import { merge_compare } from './compare'
import type { Sort } from './query'

// Browsers cap live WebGL contexts (~16); keep headroom for the other tabs' viewers.
export const MAX_COMPARE = 12

// Records tab position, kept across tab switches. With a filter or sort, pages come from the scan.
export const records_view = $state({
  page: 0,
  selected: 0,
  column_filter: ``,
  filter: ``,
  sort: null as Sort | null,
})

// Plots tab choices, kept across tab switches.
export const plot_view = $state({
  kind: `histogram` as `histogram` | `scatter`,
  x: `energy_per_atom`,
  y: `n_atoms`,
  log_x: false,
  log_y: false,
  // Show far outliers instead of fitting the axes to the bulk of the data.
  outliers: false,
  // Order of the records-in-view list, by x.
  desc: false,
  selected: null as number | null,
})

// Viewer arrows for per-atom forces.
export const viewer_settings = $state({ forces: true })

export const compare = $state({ records: [] as number[], labels: {} as Record<number, string> })

export function toggle_compare(index: number, label?: string) {
  const at = compare.records.indexOf(index)
  if (at >= 0) {
    compare.records.splice(at, 1)
    delete compare.labels[index]
  } else if (compare.records.length < MAX_COMPARE) {
    compare.records.push(index)
    if (label) compare.labels[index] = label
  }
}

export function set_compare(entries: { record: number; label?: string }[]) {
  Object.assign(compare, merge_compare(entries, MAX_COMPARE))
}

/** Copy record numbers as a Python list, e.g. to index the same file from atompack. */
export async function copy_records(indices: ArrayLike<number>) {
  await navigator.clipboard.writeText(`[${Array.from(indices).join(`, `)}]`)
  return `copied ${indices.length.toLocaleString()} record numbers`
}

/** Shared camera pose for synchronized viewers (same coordinate frame). */
export type CameraPose = { position?: Vec3; target?: Vec3 }
