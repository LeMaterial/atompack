// The reader process behind the extension: requests, temporary copies, and crashes.
import assert from 'node:assert/strict'
import { copyFileSync, mkdtempSync, readFileSync, readdirSync, rmSync, truncateSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { after, test } from 'node:test'
import { ReaderClient } from '../dist/client.js'

const fixture = fileURLToPath(new URL(`fixtures/groups.atp`, import.meta.url))
// Test files run in separate processes, so this directory only holds this file's copies.
const temporary = mkdtempSync(join(tmpdir(), `atompack-client-test-`))
process.env.TMPDIR = process.env.TMP = process.env.TEMP = temporary
after(() => rmSync(temporary, { recursive: true, force: true }))

for (const [kind, source] of [
  [`file`, () => ({ path: fixture })],
  [`bytes`, () => ({ bytes: readFileSync(fixture) })],
]) {
  test(`serves reader methods and scan chunks from a ${kind} source`, async () => {
    const reader = new ReaderClient(source())
    try {
      assert.equal((await reader.call(`overview`)).num_records, 6)
      assert.deepEqual((await reader.call(`records_at`, [5, 0])).map((r) => r.index), [5, 0])
      assert.deepEqual((await reader.call(`molecule`, [3])).numbers, [6, 6, 6, 6])
      const chunk = await reader.call(`record_columns`, [4, 10])
      assert.ok(chunk.columns.energy instanceof Float32Array)
      assert.deepEqual([...chunk.columns.energy], [-4, -5])
      await assert.rejects(reader.call(`molecule`, [6]), /out of bounds/i)
      await assert.rejects(reader.call(`dispose`), /unknown method/)
    } finally {
      await reader.dispose()
    }
    await assert.rejects(reader.call(`overview`), /file closed/)
  })
}

test(`temporary copies are removed after close and failed open`, async () => {
  const reader = new ReaderClient({ bytes: readFileSync(fixture) })
  assert.equal(readdirSync(temporary).length, 1)
  await reader.call(`overview`)
  await reader.dispose()
  await reader.dispose()
  assert.deepEqual(readdirSync(temporary), [])

  const invalid = new ReaderClient({ bytes: new Uint8Array(10000) })
  await assert.rejects(invalid.call(`overview`))
  await invalid.dispose()
  assert.deepEqual(readdirSync(temporary), [])
})

test(`a file truncated while open stops only the reader process`, async () => {
  const dir = mkdtempSync(join(tmpdir(), `atompack-truncate-`))
  const file = join(dir, `live.atp`)
  copyFileSync(fixture, file)
  const reader = new ReaderClient({ path: file })
  try {
    assert.equal((await reader.call(`overview`)).num_records, 6)
    // The index stays in memory; reading a record touches unmapped pages past the new end.
    truncateSync(file, 0)
    const results = await Promise.allSettled([
      reader.call(`molecule`, [5]),
      reader.call(`record_columns`, [0, 6]),
      reader.call(`records`, [0, 6]),
    ])
    for (const result of results) {
      assert.equal(result.status, `rejected`)
      assert.match(result.reason.message, /atompack reader stopped \(SIG\w+\).*reopen/)
    }
    // The next request restarts the reader, which reports the broken file instead of crashing.
    await assert.rejects(reader.call(`overview`), (err) => {
      assert.doesNotMatch(err.message, /stopped/)
      return true
    })
    copyFileSync(fixture, file)
    assert.equal((await reader.call(`molecule`, [5])).numbers.length, 6)
  } finally {
    await reader.dispose()
    rmSync(dir, { recursive: true, force: true })
  }
})
