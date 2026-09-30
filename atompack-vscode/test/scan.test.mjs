// Scan chunks served from worker threads must match the in-process reader, and failures
// must reject rather than hang or crash the host.
import assert from 'node:assert/strict'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { AtpReader } from '../dist/reader.js'
import { Scanner, to_binary } from '../dist/scan.js'

const fixture = fileURLToPath(new URL(`fixtures/groups.atp`, import.meta.url))

test(`worker chunks match the reader`, async () => {
  const reader = new AtpReader(fixture)
  const scanner = new Scanner(fixture)
  try {
    const chunks = await Promise.all([0, 2, 4, 6].map((start) => scanner.record_columns(start, 2)))
    chunks.forEach((chunk, i) => assert.deepEqual(chunk, to_binary(reader.record_columns(2 * i, 2))))
    assert.deepEqual(chunks[3], { count: 0, columns: {}, compositions: { keys: [], codes: new Int32Array() } })
    assert.ok(chunks[0].columns.energy instanceof Float32Array)
    assert.deepEqual([...chunks[2].columns.energy], [-4, -5])
    assert.deepEqual(chunks[2].compositions, { keys: [`7:5`, `8:6`], codes: Int32Array.of(0, 1) })
  } finally {
    scanner.dispose()
    reader.dispose()
  }
})

test(`an unreadable file rejects every queued chunk`, async () => {
  const reader = new AtpReader(fixture)
  const scanner = new Scanner(`/nonexistent.atp`)
  const results = await Promise.allSettled([0, 1, 2, 3, 4, 5, 6, 7].map((i) => scanner.record_columns(i, 1)))
  assert.ok(results.every((r) => r.status === `rejected` && /ENOENT/.test(r.reason.message)))
  scanner.dispose()
  reader.dispose()
})

test(`dispose rejects pending chunks`, async () => {
  const reader = new AtpReader(fixture)
  const scanner = new Scanner(fixture)
  const pending = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => scanner.record_columns(i, 1))
  scanner.dispose()
  const results = await Promise.allSettled(pending)
  assert.ok(results.every((r) => r.status === `rejected`))
  reader.dispose()
})

// Chunks span several of the reader's 64-record decode batches: ATP_FILE=/path/to/file.atp
test(`real file chunks line up with records`, { skip: !process.env.ATP_FILE }, async () => {
  const reader = new AtpReader(process.env.ATP_FILE)
  const scanner = new Scanner(process.env.ATP_FILE)
  try {
    const n = reader.overview().num_records
    for (const start of [0, 30, Math.max(0, n - 150)]) {
      const rows = reader.records(start, 200)
      const { count, columns } = await scanner.record_columns(start, 200)
      assert.equal(count, rows.length)
      if (!count) continue
      assert.deepEqual([...columns.n_atoms], rows.map((r) => r.n_atoms))
      assert.deepEqual([...columns.energy], rows.map((r) => Math.fround(r.energy ?? NaN)))
    }
  } finally {
    scanner.dispose()
    reader.dispose()
  }
})
