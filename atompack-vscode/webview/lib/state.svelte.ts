import type { Vec3 } from 'matterviz/math'

// Browsers cap live WebGL contexts (~16); keep headroom for the other tabs' viewers.
export const MAX_COMPARE = 12

// Records tab position, kept across tab switches.
export const records_view = $state({ page: 0, selected: 0, column_filter: `` })

// Plots tab choices, kept across tab switches.
export const plot_view = $state({
  kind: `histogram` as `histogram` | `scatter`,
  x: `energy`,
  y: `n_atoms`,
  log_x: false,
  log_y: false,
  selected: null as number | null,
})

export const compare = $state({ records: [] as number[], labels: {} as Record<number, string> })

export function toggle_compare(index: number, label?: string) {
  const at = compare.records.indexOf(index)
  if (at >= 0) compare.records.splice(at, 1)
  else if (compare.records.length < MAX_COMPARE) {
    compare.records.push(index)
    if (label) compare.labels[index] = label
  }
}

export function set_compare(entries: { record: number; label?: string }[]) {
  compare.records = entries.slice(0, MAX_COMPARE).map((e) => e.record)
  compare.labels = Object.fromEntries(entries.filter((e) => e.label).map((e) => [e.record, e.label!]))
}

/** Shared camera pose for synchronized viewers (same coordinate frame). */
export type CameraPose = { position?: Vec3; target?: Vec3 }
