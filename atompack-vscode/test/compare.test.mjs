import assert from 'node:assert/strict'
import { test } from 'node:test'
import { build } from 'esbuild'
import { fileURLToPath } from 'node:url'

const { outputFiles } = await build({
  entryPoints: [fileURLToPath(new URL('../webview/lib/compare.ts', import.meta.url))],
  bundle: true,
  write: false,
  format: 'esm',
  platform: 'node',
})
const { merge_compare } = await import(
  `data:text/javascript;base64,${Buffer.from(outputFiles[0].text).toString('base64')}`
)

test('a record in several group roles gets one pane with every role', () => {
  const merged = merge_compare([
    { record: 4, label: 'initial' },
    { record: 7, label: 'final' },
    { record: 4, label: 'reference' },
    { record: 4, label: 'initial' },
    { record: 9 },
  ], 12)
  assert.deepEqual(merged.records, [4, 7, 9])
  assert.deepEqual(merged.labels, { 4: 'initial, reference', 7: 'final' })
})

test('the pane cap counts distinct records', () => {
  const entries = [1, 1, 2, 2, 3, 3].map((record, i) => ({ record, label: `role ${i}` }))
  const merged = merge_compare(entries, 2)
  assert.deepEqual(merged.records, [1, 2])
  assert.deepEqual(merged.labels, { 1: 'role 0, role 1', 2: 'role 2, role 3' })
})
