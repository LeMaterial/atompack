<script lang="ts">
  import { untrack } from 'svelte'
  import { fmt } from './chem'
  import Histogram from './Histogram.svelte'
  import { api, type Columns, type Overview } from './rpc'
  import { type CameraPose, MAX_COMPARE, set_compare } from './state.svelte'
  import Viewer from './Viewer.svelte'

  const PAGE = 200
  let { groupings, on_compare }: { groupings: Overview[`groupings`]; on_compare: () => void } =
    $props()

  let grouping = $state(0)
  let columns = $state<Columns>({})
  let ranges = $state<Record<string, [number, number] | null>>({})
  let texts = $state<Record<string, string>>({})
  let selected = $state(0)
  let page = $state(0)
  let sync = $state(true)
  let camera = $state<CameraPose>({})

  const info = $derived(groupings[grouping])
  const numeric = $derived(info.properties.filter((p) => p.dtype !== `string`).map((p) => p.key))
  const strings = $derived(info.properties.filter((p) => p.dtype === `string`).map((p) => p.key))

  $effect(() => {
    const g = grouping
    columns = {}
    ranges = {}
    texts = {}
    selected = 0
    api.group_columns(g).then((c) => (columns = c))
  })

  const filtered = $derived.by(() => {
    const out: number[] = []
    const active_ranges = Object.entries(ranges).filter(([, r]) => r) as [string, [number, number]][]
    const active_texts = Object.entries(texts).filter(([, t]) => t)
    outer: for (let i = 0; i < info.count; i++) {
      for (const [key, [a, b]] of active_ranges) {
        const v = columns[key]?.[i] as number
        if (!(v >= a && v <= b)) continue outer
      }
      for (const [key, text] of active_texts)
        if (!String(columns[key]?.[i] ?? ``).toLowerCase().includes(text.toLowerCase())) continue outer
      out.push(i)
    }
    return out
  })
  const subset = (key: string) => filtered.map((i) => columns[key]?.[i] as number)
  // New filter: back to the first page, and keep the viewer on a group that still matches.
  $effect(() => {
    page = 0
    const keep = filtered
    untrack(() => {
      if (keep.length && !keep.includes(selected)) selected = keep[0]
    })
  })

  const members = $derived(api.groups(grouping, selected, 1).then((rows) => rows[0]?.members ?? []))
  $effect(() => {
    void selected
    camera = {}
  })

  async function compare_members() {
    const list = await members
    set_compare(list.map((m) => ({ record: m.record, label: m.role ?? undefined })))
    on_compare()
  }
</script>

<div class="flex h-full min-h-0">
  <aside class="w-64 shrink-0 space-y-3 overflow-auto border-r border-line p-2">
    {#if groupings.length > 1}
      <select class="w-full rounded border border-line bg-input p-1" bind:value={grouping}>
        {#each groupings as g, i (g.name)}<option value={i}>{g.name} ({g.count})</option>{/each}
      </select>
    {/if}
    <div class="text-xs text-muted">
      <b class="text-fg">{info.name}</b>: {info.count} groups, {info.n_members} members
      {#if info.roles.length}· roles {info.roles.join(`, `)}{/if}
    </div>
    <div class="text-xs">
      <b>{filtered.length}</b> / {info.count} match
      {#if Object.values(ranges).some(Boolean) || Object.values(texts).some(Boolean)}
        <button class="btn ml-2" onclick={() => ((ranges = {}), (texts = {}))}>Clear filters</button>
      {/if}
    </div>
    {#each numeric as key (key)}
      <Histogram
        label={key}
        values={(columns[key] ?? []) as number[]}
        subset={subset(key)}
        bind:range={ranges[key]}
      />
    {/each}
    {#each strings as key (key)}
      <label class="block text-xs">
        <span class="font-medium">{key}</span>
        <input
          class="w-full rounded border border-line bg-input px-1"
          placeholder="contains…"
          bind:value={texts[key]}
        />
      </label>
    {/each}
  </aside>

  <section class="flex w-96 min-w-40 flex-col border-r border-line">
    <div class="flex items-center gap-2 border-b border-line px-2 py-1 text-xs">
      <button class="btn" disabled={page === 0} onclick={() => (page -= 1)}>‹</button>
      <span>{page * PAGE}–{Math.min(filtered.length, (page + 1) * PAGE) - 1} of {filtered.length}</span>
      <button class="btn" disabled={(page + 1) * PAGE >= filtered.length} onclick={() => (page += 1)}>›</button>
    </div>
    <div class="min-h-0 flex-1 overflow-auto">
      <table class="w-full text-xs whitespace-nowrap">
        <thead class="sticky top-0 bg-panel text-left">
          <tr>
            <th class="px-2 py-1">group</th>
            {#each info.properties as p (p.key)}<th class="px-2">{p.key}</th>{/each}
          </tr>
        </thead>
        <tbody class="font-mono">
          {#each filtered.slice(page * PAGE, (page + 1) * PAGE) as g (g)}
            <tr
              class="cursor-pointer border-b border-line/40 select-none hover:bg-hover"
              class:!bg-selected={g === selected}
              class:text-selected-fg={g === selected}
              onclick={() => (selected = g)}
              ondblclick={() => ((selected = g), compare_members())}
              title="Double-click to compare this group's members"
            >
              <td class="px-2 py-0.5">{g}</td>
              {#each info.properties as p (p.key)}<td class="px-2">{fmt(columns[p.key]?.[g])}</td>{/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>

  <section class="flex min-w-72 flex-1 flex-col">
    <div class="flex items-center gap-2 border-b border-line px-2 py-1 text-xs">
      <b>Group {selected}</b>
      <label class="flex items-center gap-1"><input type="checkbox" bind:checked={sync} /> sync cameras</label>
      <button class="btn ml-auto" onclick={compare_members}>Members → compare</button>
    </div>
    {#await members then list}
      <div
        class="grid min-h-0 flex-1 gap-2 overflow-auto p-2"
        style="grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); grid-auto-rows: minmax(16rem, 1fr)"
      >
        {#each list.slice(0, MAX_COMPARE) as m, i (i)}
          <Viewer
            index={m.record}
            label={m.role ?? undefined}
            camera={sync ? camera : undefined}
            projection="perspective"
          />
        {/each}
      </div>
      {#if list.length > MAX_COMPARE}
        <p class="px-2 pb-2 text-xs text-muted">Showing {MAX_COMPARE} of {list.length} members.</p>
      {/if}
    {/await}
  </section>
</div>
