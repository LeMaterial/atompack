// Exercises the WASM reader through the same loader the extension uses.
// Fixture from test/make_fixture.py; `npm test` bundles src/reader.ts first.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { AtpReader, bytesSource, fileSource } from '../dist/reader.js'

const fixture = new URL(`fixtures/groups.atp`, import.meta.url).pathname

for (const [kind, open] of [
  [`file`, () => fileSource(fixture)],
  [`bytes`, () => bytesSource(readFileSync(fixture))],
]) {
  test(`reads records and groups from ${kind} source`, () => {
    const reader = new AtpReader(open())
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
}

test(`rejects files that are not atompack`, () => {
  assert.throws(() => new AtpReader(bytesSource(new Uint8Array(10000))))
})

test(`several readers stay independent`, () => {
  const a = new AtpReader(fileSource(fixture))
  const b = new AtpReader(bytesSource(readFileSync(fixture)))
  a.dispose()
  assert.equal(b.overview().num_records, 6)
  assert.throws(() => a.overview(), /not open/)
  b.dispose()
})

// Optional smoke test on a real database: ATP_FILE=/path/to/file.atp npm test
test(`real file smoke`, { skip: !process.env.ATP_FILE }, () => {
  const reader = new AtpReader(fileSource(process.env.ATP_FILE))
  const { num_records } = reader.overview()
  for (const i of [0, Math.floor(num_records / 2), num_records - 1]) {
    const mol = reader.molecule(i)
    assert.equal(mol.positions.length, 3 * mol.numbers.length)
  }
  assert.equal(reader.records(0, 100).length, Math.min(100, num_records))
  reader.dispose()
})
