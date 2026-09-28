import element_data from 'matterviz/element/data'
import { calc_lattice_params, create_cart_to_frac, type Matrix3x3, type Vec3 } from 'matterviz/math'
import type { AnyStructure, Site } from 'matterviz/structure'
import type { MoleculeData, Scalar } from './rpc'

export const symbol = (z: number) => element_data[z - 1]?.symbol ?? `X`

/** Hill formula from [Z, count] pairs. */
export function formula(composition: [number, number][]): string {
  const counts = new Map(composition.map(([z, n]) => [symbol(z), n]))
  const order = [...counts.keys()].sort()
  if (counts.has(`C`)) {
    order.splice(order.indexOf(`C`), 1)
    order.unshift(`C`)
    if (counts.has(`H`)) order.splice(order.indexOf(`H`), 1), order.splice(1, 0, `H`)
  }
  return order.map((el) => (counts.get(el)! > 1 ? `${el}${counts.get(el)}` : el)).join(``)
}

export function composition(numbers: number[]): [number, number][] {
  const counts = new Map<number, number>()
  for (const z of numbers) counts.set(z, (counts.get(z) ?? 0) + 1)
  return [...counts]
}

const triples = (flat: number[]): Vec3[] =>
  Array.from({ length: flat.length / 3 }, (_, i) => [flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]])

export function to_structure(mol: MoleculeData): AnyStructure {
  const matrix = mol.cell ? (triples(mol.cell) as Matrix3x3) : null
  const to_frac = matrix ? create_cart_to_frac(matrix) : null
  const forces = mol.forces ? triples(mol.forces) : null
  const sites: Site[] = triples(mol.positions).map((xyz, i) => {
    const element = symbol(mol.numbers[i]) as Site[`species`][number][`element`]
    return {
      species: [{ element, occu: 1, oxidation_state: 0 }],
      xyz,
      abc: to_frac ? to_frac(xyz) : [0, 0, 0],
      label: `${element}${i}`,
      properties: forces ? { force: forces[i] } : {},
    }
  })
  const id = `atompack-${mol.index}`
  if (!matrix) return { id, sites }
  const pbc = mol.pbc ?? [true, true, true]
  return { id, sites, lattice: { matrix, pbc, ...calc_lattice_params(matrix) } }
}

export function fmt(value: Scalar | undefined, digits = 5): string {
  if (value === null || value === undefined) return ``
  if (typeof value === `number`) {
    if (Number.isInteger(value)) return String(value)
    const abs = Math.abs(value)
    return abs !== 0 && (abs < 1e-3 || abs >= 1e6) ? value.toExponential(3) : value.toPrecision(digits)
  }
  return value
}
