import assert from 'node:assert/strict'
import { test } from 'node:test'
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'

// Exercise the real query parser/sorter with a plain scan table, without a webview runtime.
const { outputFiles } = await build({
  stdin: {
    contents: `export * from './query'; export { table } from './scan.svelte'`,
    resolveDir: fileURLToPath(new URL('../webview/lib', import.meta.url)),
  },
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
  plugins: [{
    name: 'scan-table',
    setup(builder) {
      builder.onResolve({ filter: /^\.\/scan\.svelte$/ }, () => ({ path: 'table', namespace: 'test' }))
      builder.onLoad({ filter: /.*/, namespace: 'test' }, () => ({
        contents: `export const table = { ids: [], columns: {}, compositions: [], composition: [] }`,
      }))
    },
  }],
})
const { table, plot_filter, parse_filter, select_rows } = await import(
  `data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`
)

function records(columns, axes, desc = false) {
  table.columns = Object.fromEntries(Object.entries(columns).map(([key, values]) => [key, Float32Array.from(values)]))
  // Non-contiguous IDs model a sampled scan; the user copies record IDs, not table rows.
  table.ids = Int32Array.from(columns[axes[0].key], (_, i) => 10 + i * 3)
  const { test } = parse_filter(plot_filter(axes))
  return Array.from(select_rows(test, { key: axes[0].key, desc }), (row) => table.ids[row])
}

test('automatic scatter ranges exclude missing and non-finite coordinates in Records', () => {
  const axes = ['energy', 'fmax'].map((key) => ({ key, range: null, log: false }))
  assert.deepEqual(records({ energy: [1, 2, 3, Infinity, -Infinity], fmax: [1, NaN, 2, 3, 4] }, axes), [10, 16])
  assert.deepEqual(records({ energy: [1, 2, 3], fmax: [1, NaN, 2] }, axes, true), [16, 10])
})

test('log axes exclude zero and negative values with automatic or explicit ranges', () => {
  for (const range of [null, [-5, 5]]) {
    assert.deepEqual(records({ energy: [-1, 0, 1, NaN, Infinity] }, [{ key: 'energy', range, log: true }]), [16])
  }
  assert.deepEqual(records({ x: [1, 2, 3], y: [0, -1, 1] }, [
    { key: 'x', range: null, log: false }, { key: 'y', range: null, log: true },
  ]), [16])
})

test('plot bounds keep full Float32 precision and include both visible boundaries', () => {
  const lo = Math.fround(0.12345671), hi = Math.fround(0.98765432)
  assert.deepEqual(records({ x: [lo - 1e-7, lo, hi, hi + 1e-7] }, [
    { key: 'x', range: [lo, hi], log: false },
  ]), [13, 16])
})
