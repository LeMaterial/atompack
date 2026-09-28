// Queries over the scanned table (scan.svelte.ts): filters, sorting and default plot ranges.
// They work on table rows; table.ids maps rows to record indices.
import { composition_key } from './chem'
import { table } from './scan.svelte'

export type Sort = { key: string; desc: boolean }
export type Test = (row: number) => boolean

const CONDITION = /^\s*([\w.-]+)\s*(<=|>=|!=|==|=|<|>)\s*(\S+)\s*$/

/**
 * Parses `fmax > 1, formula = H2O` (joined by `,`, `and` or `&&`) into a row test and the
 * numeric columns it uses; throws when invalid.
 */
export function parse_filter(text: string): { test: Test; keys: string[] } {
  const keys: string[] = []
  const tests = text
    .split(/,|&&|\band\b/)
    .filter((part) => part.trim())
    .map((part): Test => {
      const match = part.match(CONDITION)
      if (!match) throw new Error(`can't read "${part.trim()}", try e.g. fmax > 1 or formula = H2O`)
      const [, key, op, raw] = match
      if (key === `formula`) {
        if (![`=`, `==`, `!=`].includes(op)) throw new Error(`formula takes = or !=`)
        const want = table.compositions.indexOf(composition_key(raw))
        return op === `!=` ? (row) => table.composition[row] !== want : (row) => table.composition[row] === want
      }
      const values = table.columns[key]
      if (!values) throw new Error(`no numeric column ${key}`)
      const x = Number(raw)
      if (Number.isNaN(x)) throw new Error(`${raw} is not a number`)
      keys.push(key)
      return {
        '<': (row: number) => values[row] < x,
        '<=': (row: number) => values[row] <= x,
        '>': (row: number) => values[row] > x,
        '>=': (row: number) => values[row] >= x,
        '=': (row: number) => values[row] === Math.fround(x),
        '==': (row: number) => values[row] === Math.fround(x),
        '!=': (row: number) => values[row] !== Math.fround(x),
      }[op]!
    })
  return { test: (row) => tests.every((test) => test(row)), keys }
}

/** Read rows passing `test`, ordered by `sort` (rows without a value last), else in file order. */
export function select_rows(test: Test, sort: Sort | null): Int32Array {
  const out = new Int32Array(table.ids.length)
  let n = 0
  for (let row = 0; row < table.ids.length; row++) if (table.ids[row] >= 0 && test(row)) out[n++] = row
  const rows = out.subarray(0, n)
  const values = sort && table.columns[sort.key]
  return values ? sorted(rows, values, sort.desc) : rows
}

// Sorts (key bits, row) pairs as native u64s, several times faster than a comparator sort:
// flipping the sign bit (and the rest for negatives) makes float order match integer order.
function sorted(rows: Int32Array, values: Float32Array, desc: boolean): Int32Array {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length)
  const finite = rows.filter((row) => Number.isFinite(values[row]))
  const pairs = new BigUint64Array(finite.length)
  const words = new Uint32Array(pairs.buffer) // little-endian: [row, key] per pair
  finite.forEach((row, i) => {
    const b = bits[row]
    words[2 * i] = row
    words[2 * i + 1] = b & 0x80000000 ? ~b : b | 0x80000000
  })
  pairs.sort()
  const order = Int32Array.from({ length: finite.length }, (_, i) => words[2 * i])
  if (desc) order.reverse()
  const out = new Int32Array(rows.length)
  out.set(order)
  out.set(rows.filter((row) => !Number.isFinite(values[row])), order.length)
  return out
}

/**
 * Default axis range: the data without far outliers, i.e. the 0.1–99.9th percentiles widened
 * by 5%. Null (automatic, all data) when that would hide nothing.
 */
export function fit(values: Float32Array | undefined, log: boolean): [number, number] | null {
  if (!values) return null
  const [to, from] = log ? [Math.log10, (v: number) => 10 ** v] : [(v: number) => v, (v: number) => v]
  const stride = Math.max(1, Math.floor(values.length / 100_000))
  const sample: number[] = []
  let [min, max] = [Infinity, -Infinity]
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v) || (log && v <= 0)) continue
    ;[min, max] = [Math.min(min, v), Math.max(max, v)]
    if (i % stride === 0) sample.push(to(v))
  }
  if (sample.length < 2) return null
  sample.sort((a, b) => a - b)
  const at = (q: number) => sample[Math.round(q * (sample.length - 1))]
  const [lo, hi] = [at(0.001), at(0.999)]
  const pad = 0.05 * (hi - lo)
  const range: [number, number] = [Math.max(min, from(lo - pad)), Math.min(max, from(hi + pad))]
  return range[0] < range[1] && (range[0] > min || range[1] < max) ? range : null
}
