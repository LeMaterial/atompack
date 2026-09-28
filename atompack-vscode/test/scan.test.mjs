// Scan chunks served from worker threads must match the in-process reader, and failures
// must reject rather than hang or crash the host.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { AtpReader, bytesSource, fileSource } from '../dist/reader.js'
import { Scanner, to_binary } from '../dist/scan.js'

const fixture = new URL(`fixtures/groups.atp`, import.meta.url).pathname

test(`worker chunks match the reader`, async () => {
  const reader = new AtpReader(fileSource(fixture))
  const scanner = new Scanner(fixture, reader)
  try {
    const chunks = await Promise.all([0, 2, 4, 6].map((start) => scanner.record_columns(start, 2)))
    chunks.forEach((chunk, i) => assert.deepEqual(chunk, to_binary(reader.record_columns(2 * i, 2))))
    assert.deepEqual(chunks[3], { count: 0, columns: {} })
    assert.ok(chunks[0].columns.energy instanceof Float32Array)
    assert.deepEqual([...chunks[2].columns.energy], [-4, -5])
  } finally {
    scanner.dispose()
    reader.dispose()
  }
})

test(`non-file sources scan in process`, async () => {
  const reader = new AtpReader(bytesSource(readFileSync(fixture)))
  const scanner = new Scanner(undefined, reader)
  assert.deepEqual(await scanner.record_columns(0, 6), to_binary(reader.record_columns(0, 6)))
  scanner.dispose()
  reader.dispose()
})

test(`an unreadable file rejects every queued chunk`, async () => {
  const reader = new AtpReader(fileSource(fixture))
  const scanner = new Scanner(`/nonexistent.atp`, reader)
  const results = await Promise.allSettled([0, 1, 2, 3, 4, 5, 6, 7].map((i) => scanner.record_columns(i, 1)))
  assert.ok(results.every((r) => r.status === `rejected` && /ENOENT/.test(r.reason.message)))
  scanner.dispose()
  reader.dispose()
})

test(`dispose rejects pending chunks`, async () => {
  const reader = new AtpReader(fileSource(fixture))
  const scanner = new Scanner(fixture, reader)
  const pending = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => scanner.record_columns(i, 1))
  scanner.dispose()
  const results = await Promise.allSettled(pending)
  assert.ok(results.every((r) => r.status === `rejected`))
  reader.dispose()
})

// Chunks span several of the reader's 64-record decode batches: ATP_FILE=/path/to/file.atp
test(`real file chunks line up with records`, { skip: !process.env.ATP_FILE }, async () => {
  const reader = new AtpReader(fileSource(process.env.ATP_FILE))
  const scanner = new Scanner(process.env.ATP_FILE, reader)
  try {
    const n = reader.overview().num_records
    for (const start of [0, 30, Math.max(0, n - 150)]) {
      const rows = reader.records(start, 200)
      const { count, columns } = await scanner.record_columns(start, 200)
      assert.equal(count, rows.length)
      assert.deepEqual([...columns.n_atoms], rows.map((r) => r.n_atoms))
      assert.deepEqual([...columns.energy], rows.map((r) => Math.fround(r.energy ?? NaN)))
    }
  } finally {
    scanner.dispose()
    reader.dispose()
  }
})
