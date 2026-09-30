// Exercises the native reader through the same loader the extension uses.
// Fixture from test/make_fixture.py; `npm test` bundles src/reader.ts first.
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import { AtpReader } from '../dist/reader.js'

const fixture = fileURLToPath(new URL(`fixtures/groups.atp`, import.meta.url))

test(`reads records and groups`, () => {
  const reader = new AtpReader(fixture)
  try {
    const overview = reader.overview()
    assert.equal(overview.num_records, 6)
    assert.deepEqual(overview.compression, { kind: `zstd`, level: 3 })
    assert.deepEqual(
      overview.groupings.map((g) => [g.name, g.count, g.n_members, g.roles]),
      [
        [`adsorption`, 2, 5, [`slab`, `adslab`, `gas`]],
        [`ordered`, 2, 3, []],
      ],
    )

    const rows = reader.records(4, 10)
    assert.deepEqual(
      rows.map((r) => [r.index, r.n_atoms, r.composition, r.energy, r.periodic, r.properties.tag]),
      [
        [4, 5, [[7, 5]], -4, false, `r4`],
        [5, 6, [[8, 6]], -5, true, `r5`],
      ],
    )

    assert.deepEqual(
      reader.records_at(5, 0, 4).map((r) => r.index),
      [5, 0, 4],
    )

    const cols = reader.record_columns(4, 10)
    assert.deepEqual(Object.keys(cols), [`composition`, `energy`, `energy_per_atom`, `fmax`, `n_atoms`])
    assert.deepEqual(cols.energy, [-4, -5])
    assert.deepEqual(cols.energy_per_atom, [-4 / 5, -5 / 6])
    assert.deepEqual(cols.n_atoms, [5, 6])
    assert.deepEqual(cols.composition, [`7:5`, `8:6`])
    assert.deepEqual(cols.fmax.map((f) => f.toFixed(4)), [0.4, 0.5].map((f) => (f * Math.sqrt(3)).toFixed(4)))

    const mol = reader.molecule(3)
    assert.deepEqual(mol.numbers, [6, 6, 6, 6])
    assert.deepEqual(mol.positions.slice(0, 6), [3, 3.5, 4, 4.5, 5, 5.5])
    assert.deepEqual(mol.cell, [8, 0, 0, 0, 8, 0, 0, 0, 8])
    assert.deepEqual(mol.pbc, [true, true, true])
    assert.equal(mol.forces.length, 12)

    const [group] = reader.groups(0, 1, 1)
    assert.deepEqual(group.members, [
      { record: 1, role: `slab` },
      { record: 5, role: `adslab` },
      { record: 0, role: `gas` },
    ])
    assert.deepEqual(group.properties, { e_ads: 0.25, id: `b`, n: 3 })
    assert.deepEqual(reader.group_columns(0), { e_ads: [-1.5, 0.25], id: [`a`, `b`], n: [2, 3] })
    assert.deepEqual(reader.groups(1, 0, 5).map((g) => g.members.map((m) => m.record)), [[4, 2], [0]])

    assert.throws(() => reader.molecule(6), /out of bounds/i)
    assert.throws(() => reader.group_columns(2), /out of bounds/i)
  } finally {
    reader.dispose()
  }
})

test(`rejects files that are not atompack`, () => {
  const dir = mkdtempSync(join(tmpdir(), `atompack-test-`))
  try {
    writeFileSync(join(dir, `zeros.atp`), new Uint8Array(10000))
    assert.throws(() => new AtpReader(join(dir, `zeros.atp`)))
    assert.throws(() => new AtpReader(join(dir, `missing.atp`)), /ENOENT/)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test(`corrupt record offsets throw JS errors without aborting the host`, () => {
  // Run in another Node process so a regression reports a failed test, not a dead test runner.
  execFileSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict'
    import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
    import { tmpdir } from 'node:os'
    import { join } from 'node:path'
    import { AtpReader } from ${JSON.stringify(new URL('../dist/reader.js', import.meta.url).href)}
    const bytes = readFileSync(${JSON.stringify(fixture)})
    // The fixture ends with [u64 count][six 20-byte index entries]. Corrupt record 2.
    assert.equal(bytes.readBigUInt64LE(bytes.length - 6 * 20 - 8), 6n)
    bytes.writeBigUInt64LE(1_000_000_000n, bytes.length - 4 * 20)
    const dir = mkdtempSync(join(tmpdir(), 'atompack-test-'))
    writeFileSync(join(dir, 'corrupt.atp'), bytes)
    const reader = new AtpReader(join(dir, 'corrupt.atp'))
    try {
      for (const read of [() => reader.molecule(2), () => reader.records(0, 6), () => reader.record_columns(0, 6)]) {
        assert.throws(read, /out of range|out of bounds/)
        assert.equal(reader.molecule(0).numbers.length, 1)
      }
    } finally { reader.dispose(); rmSync(dir, { recursive: true, force: true }) }
  `], { stdio: ['ignore', 'pipe', 'pipe'] })
})

test(`several readers stay independent`, () => {
  const a = new AtpReader(fixture)
  const b = new AtpReader(fixture)
  a.dispose()
  assert.equal(b.overview().num_records, 6)
  assert.throws(() => a.overview(), /not open/)
  b.dispose()
})

// Optional smoke test on a real database: ATP_FILE=/path/to/file.atp npm test
test(`real file smoke`, { skip: !process.env.ATP_FILE }, () => {
  const reader = new AtpReader(process.env.ATP_FILE)
  const { num_records } = reader.overview()
  for (const i of num_records ? [0, Math.floor(num_records / 2), num_records - 1] : []) {
    const mol = reader.molecule(i)
    assert.equal(mol.positions.length, 3 * mol.numbers.length)
  }
  assert.equal(reader.records(0, 100).length, Math.min(100, num_records))
  reader.dispose()
})
