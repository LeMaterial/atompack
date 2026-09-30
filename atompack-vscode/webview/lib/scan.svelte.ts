import { api } from './rpc'

// Per-record numeric columns and compositions for plotting, sorting and filtering. Records are
// decoded whole, so this reads the file in chunks on the extension host's scan workers, with
// several requests in flight to hide the round trips of remote sessions. Bounded so a huge
// file can't exhaust the webview: past MAX_ROWS records it reads whole chunks spread evenly
// over the file (a sample). Chunks are visited in a scattered order, so plots of a partial scan
// already cover the whole file; rows stay in file order.
const CHUNK = 4096
const IN_FLIGHT = 8
const MAX_ROWS = 5_000_000
// Float32 values over all columns (160 MB); columns first seen past it are skipped.
const MAX_VALUES = 40_000_000
// Publishing progress re-renders the plots; don't do it for every chunk.
const PUBLISH_MS = 500

export const scan = $state({
  done: 0,
  planned: 0,
  total: 0,
  finished: false,
  keys: [] as string[],
  skipped: 0,
  error: ``,
})
// Row i holds record ids[i] (-1 until read); missing values are NaN. Its composition is
// compositions[composition[i]], a "Z:count ..." key (see chem.composition_key).
export const table = {
  ids: new Int32Array(0),
  columns: {} as Record<string, Float32Array>,
  composition: new Int32Array(0),
  compositions: [] as string[],
}

let scanning: Promise<void> | undefined

/** Starts the scan once; resolves when it is over (see scan.error). */
export const start_scan = (total: number) => (scanning ??= run(total))

async function run(total: number) {
  const n_chunks = Math.ceil(total / CHUNK)
  const n_read = Math.min(n_chunks, Math.floor(MAX_ROWS / CHUNK))
  const rows = n_read === n_chunks ? total : n_read * CHUNK
  table.ids = new Int32Array(rows).fill(-1)
  table.composition = new Int32Array(rows).fill(-1)
  scan.total = total
  scan.planned = rows
  // Visit order k -> k * step mod n_read, a permutation when step and n_read are coprime.
  let step = Math.max(1, Math.round(n_read * 0.618))
  while (gcd(step, n_read) > 1) step++

  const skipped = new Set<string>()
  const codes = new Map<string, number>()
  let [done, next, published] = [0, 0, 0]
  const publish = () => {
    scan.keys = Object.keys(table.columns).sort()
    scan.skipped = skipped.size
    scan.done = done
    published = performance.now()
  }
  const code = (key: string) => {
    let c = codes.get(key)
    if (c === undefined) codes.set(key, (c = table.compositions.push(key) - 1))
    return c
  }
  const read = async (k: number) => {
    const first = Math.floor((k * n_chunks) / n_read) * CHUNK
    const { count, columns, compositions } = await api.record_columns(first, CHUNK)
    const row = k * CHUNK
    const global = compositions.keys.map(code)
    for (let i = 0; i < count; i++) {
      table.ids[row + i] = first + i
      table.composition[row + i] = global[compositions.codes[i]]
    }
    for (const [key, values] of Object.entries(columns)) {
      if (!table.columns[key] && (Object.keys(table.columns).length + 1) * rows > MAX_VALUES) skipped.add(key)
      else (table.columns[key] ??= new Float32Array(rows).fill(NaN)).set(values, row)
    }
    done += count
    if (performance.now() - published > PUBLISH_MS) publish()
  }
  const lane = async () => {
    while (next < n_read && !scan.error) await read((next++ * step) % n_read)
  }
  try {
    await Promise.all(Array.from({ length: IN_FLIGHT }, lane))
  } catch (err) {
    scan.error = err instanceof Error ? err.message : String(err)
  }
  publish()
  scan.finished = true
}

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a)
