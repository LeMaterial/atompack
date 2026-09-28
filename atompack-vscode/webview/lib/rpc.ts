// Requests to the extension host, which answers from the WASM reader.
// Shapes mirror atompack-wasm/src/lib.rs.

export type Scalar = number | string | null
export type Overview = {
  num_records: number
  compression: { kind: string; level?: number }
  record_format: number
  schema: {
    positions_dtype: string | null
    sections: { kind: string; key: string; dtype: string; per_atom: boolean }[]
  } | null
  groupings: {
    name: string
    count: number
    n_members: number
    roles: string[]
    properties: { key: string; dtype: string }[]
  }[]
}
export type RecordRow = {
  index: number
  n_atoms: number
  name: string | null
  composition: [number, number][]
  energy: number | null
  periodic: boolean
  properties: Record<string, Scalar>
}
export type MoleculeData = {
  index: number
  name: string | null
  numbers: number[]
  positions: number[]
  cell: number[] | null
  pbc: [boolean, boolean, boolean] | null
  energy: number | null
  forces: number[] | null
  charges: number[] | null
  velocities: number[] | null
  stress: number[] | null
  properties: Record<string, unknown>
  atom_properties: Record<string, unknown>
}
export type GroupRow = {
  index: number
  members: { record: number; role: string | null }[]
  properties: Record<string, Scalar>
}
export type Columns = Record<string, Scalar[]>
// Float32, NaN where a record has no value; compositions as codes into `keys`.
export type NumericColumns = {
  count: number
  columns: Record<string, Float32Array>
  compositions: { keys: string[]; codes: Int32Array }
}

declare function acquireVsCodeApi(): { postMessage(message: unknown): void }
const vscode = acquireVsCodeApi()
const pending = new Map<number, { resolve(value: any): void; reject(err: Error): void }>()
let next_id = 0

window.addEventListener(`message`, ({ data }) => {
  const request = pending.get(data?.id)
  if (!request) return
  pending.delete(data.id)
  if (`error` in data) request.reject(new Error(data.error))
  else request.resolve(data.result)
})

function call<T>(method: string, ...params: number[]): Promise<T> {
  const id = next_id++
  vscode.postMessage({ id, method, params })
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
}

const molecules = new Map<number, Promise<MoleculeData>>()

export const api = {
  overview: () => call<Overview>(`overview`),
  records: (start: number, count: number) => call<RecordRow[]>(`records`, start, count),
  records_at: (indices: ArrayLike<number>) => call<RecordRow[]>(`records_at`, ...Array.from(indices)),
  record_columns: (start: number, count: number) =>
    call<NumericColumns>(`record_columns`, start, count),
  groups: (grouping: number, start: number, count: number) =>
    call<GroupRow[]>(`groups`, grouping, start, count),
  group_columns: (grouping: number) => call<Columns>(`group_columns`, grouping),
  /** Cached: viewers re-request the same records while browsing. */
  molecule(index: number): Promise<MoleculeData> {
    let mol = molecules.get(index)
    if (!mol) {
      mol = call<MoleculeData>(`molecule`, index)
      mol.catch(() => molecules.delete(index))
      if (molecules.size >= 128) molecules.delete(molecules.keys().next().value!)
      molecules.set(index, mol)
    }
    return mol
  },
}
