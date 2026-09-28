<script lang="ts">
  import type { Overview } from './rpc'

  let { overview }: { overview: Overview } = $props()
  const { compression, schema } = $derived(overview)
</script>

<div class="h-full space-y-6 overflow-auto p-4 *:max-w-4xl">
  <dl class="grid grid-cols-[repeat(auto-fit,minmax(10rem,1fr))] gap-3">
    {#each [
      [`records`, overview.num_records.toLocaleString()],
      [`compression`, compression.level === undefined ? compression.kind : `${compression.kind} (${compression.level})`],
      [`record format`, `SOA v${overview.record_format}`],
      [`positions`, schema?.positions_dtype ?? `—`],
      [`groupings`, overview.groupings.length],
    ] as [label, value] (label)}
      <div class="rounded border border-line bg-panel p-3">
        <dt class="text-xs text-muted">{label}</dt>
        <dd class="text-lg font-semibold">{value}</dd>
      </div>
    {/each}
  </dl>

  {#if schema}
    <section>
      <h2 class="mb-2 font-semibold">Schema</h2>
      <table class="w-full text-xs">
        <thead class="text-left text-muted">
          <tr><th class="py-1">key</th><th>level</th><th>dtype</th><th>per atom</th></tr>
        </thead>
        <tbody class="font-mono">
          {#each schema.sections as s (s.kind + s.key)}
            <tr class="border-t border-line/50">
              <td class="py-0.5">{s.key}</td><td>{s.kind}</td><td>{s.dtype}</td><td>{s.per_atom ? `✓` : ``}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}

  {#if overview.groupings.length}
    <section>
      <h2 class="mb-2 font-semibold">Groupings</h2>
      <table class="w-full text-xs">
        <thead class="text-left text-muted">
          <tr><th class="py-1">name</th><th>groups</th><th>members</th><th>roles</th><th>properties</th></tr>
        </thead>
        <tbody class="font-mono">
          {#each overview.groupings as g (g.name)}
            <tr class="border-t border-line/50">
              <td class="py-0.5">{g.name}</td>
              <td>{g.count}</td>
              <td>{g.n_members}</td>
              <td>{g.roles.join(`, `)}</td>
              <td>{g.properties.map((p) => `${p.key}: ${p.dtype}`).join(`, `)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}
</div>
