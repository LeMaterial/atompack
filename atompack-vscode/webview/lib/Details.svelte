<script lang="ts">
  import { composition, fmt, formula } from './chem'
  import { api } from './rpc'

  let { index }: { index: number } = $props()

  const mol = $derived(api.molecule(index))

  function describe(value: unknown): string {
    if (value === null || value === undefined) return `—`
    if (typeof value === `number` || typeof value === `string`) return fmt(value)
    if (Array.isArray(value)) {
      const flat = value.flat(2) as number[]
      const shape = Array.isArray(value[0]) ? `${value.length}×${value[0].length}` : `${value.length}`
      const head = flat.slice(0, 6).map((v) => fmt(v, 4)).join(`, `)
      return `[${shape}] ${head}${flat.length > 6 ? `, …` : ``}`
    }
    if (typeof value === `object` && `shape` in value && `values` in value) {
      const { shape, values } = value as { shape: number[]; values: number[] }
      return `tensor[${shape.join(`×`)}] ${values.slice(0, 6).map((v) => fmt(v, 4)).join(`, `)}${values.length > 6 ? `, …` : ``}`
    }
    return JSON.stringify(value)
  }
</script>

{#await mol then m}
  {@const rows = [
    [`name`, m.name],
    [`formula`, formula(composition(m.numbers))],
    [`atoms`, m.numbers.length],
    [`energy`, m.energy],
    [`cell`, m.cell ? [m.cell.slice(0, 3), m.cell.slice(3, 6), m.cell.slice(6, 9)] : null],
    [`pbc`, m.pbc ? m.pbc.map((p) => (p ? `T` : `F`)).join(``) : null],
    [`stress`, m.stress],
    [`forces`, m.forces ? `[${m.forces.length / 3}×3]` : null],
    [`charges`, m.charges],
    [`velocities`, m.velocities ? `[${m.velocities.length / 3}×3]` : null],
  ].filter(([, v]) => v !== null && v !== undefined)}
  <table class="w-full text-xs">
    <tbody>
      {#each rows as [key, value] (key)}
        <tr class="border-b border-line/50">
          <td class="py-0.5 pr-3 align-top text-muted">{key}</td>
          <td class="font-mono break-all">{describe(value)}</td>
        </tr>
      {/each}
      {#each Object.entries(m.properties) as [key, value] (key)}
        <tr class="border-b border-line/50">
          <td class="py-0.5 pr-3 align-top text-muted">{key}</td>
          <td class="font-mono break-all">{describe(value)}</td>
        </tr>
      {/each}
      {#each Object.entries(m.atom_properties) as [key, value] (key)}
        <tr class="border-b border-line/50">
          <td class="py-0.5 pr-3 align-top text-muted">{key} <span class="opacity-60">(atom)</span></td>
          <td class="font-mono break-all">{describe(value)}</td>
        </tr>
      {/each}
    </tbody>
  </table>
{/await}
