<script lang="ts">
  import { fmt, formula } from './chem'
  import Details from './Details.svelte'
  import { api, type RecordRow } from './rpc'
  import { compare, MAX_COMPARE, records_view as view, toggle_compare } from './state.svelte'
  import Viewer from './Viewer.svelte'

  const PAGE = 100
  let { total }: { total: number } = $props()

  let rows = $state<RecordRow[]>([])
  let error = $state(``)
  const pages = $derived(Math.max(1, Math.ceil(total / PAGE)))
  const columns = $derived(
    [...new Set(rows.flatMap((r) => Object.keys(r.properties)))].filter((key) =>
      key.toLowerCase().includes(view.column_filter.toLowerCase()),
    ),
  )

  $effect(() => {
    const start = view.page * PAGE
    api.records(start, PAGE).then(
      (r) => ((rows = r), (error = ``)),
      (err) => (error = err.message),
    )
  })

  function select(index: number) {
    if (!(index >= 0 && index < total)) return
    view.selected = index
    view.page = Math.floor(index / PAGE)
    requestAnimationFrame(() =>
      document.getElementById(`record-${index}`)?.scrollIntoView({ block: `nearest` }),
    )
  }

  function onkeydown(event: KeyboardEvent) {
    const step = { ArrowDown: 1, ArrowUp: -1, PageDown: PAGE, PageUp: -PAGE }[event.key]
    if (step === undefined) return
    event.preventDefault()
    select(Math.min(total - 1, Math.max(0, view.selected + step)))
  }
</script>

<div class="flex h-full min-h-0">
  <section class="flex min-w-0 flex-1 flex-col border-r border-line">
    <div class="flex items-center gap-2 border-b border-line px-2 py-1 text-xs">
      <button class="btn" disabled={view.page === 0} onclick={() => (view.page -= 1)}>‹</button>
      <span>{view.page * PAGE}–{Math.min(total, (view.page + 1) * PAGE) - 1} of {total}</span>
      <button class="btn" disabled={view.page >= pages - 1} onclick={() => (view.page += 1)}>›</button>
      <input
        type="number"
        min="0"
        max={total - 1}
        placeholder="go to #"
        class="w-24"
        onchange={(e) => select(e.currentTarget.valueAsNumber)}
      />
      <input class="ml-auto w-40" placeholder="filter columns…" bind:value={view.column_filter} />
      {#if error}<span class="text-error">{error}</span>{/if}
    </div>
    <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_static_element_interactions -->
    <div class="min-h-0 flex-1 overflow-auto outline-none" tabindex="0" {onkeydown}>
      <table class="w-full text-xs whitespace-nowrap">
        <thead class="sticky top-0 bg-panel text-left">
          <tr>
            <th class="px-2 py-1">#</th>
            <th class="px-2">formula</th>
            <th class="px-2 text-right">atoms</th>
            <th class="px-2 text-right">energy</th>
            <th class="px-2">pbc</th>
            <th class="px-2">name</th>
            {#each columns as key (key)}<th class="px-2">{key}</th>{/each}
          </tr>
        </thead>
        <tbody class="font-mono">
          {#each rows as row (row.index)}
            <tr
              id="record-{row.index}"
              class="cursor-pointer border-b border-line/40 hover:bg-hover"
              class:!bg-selected={row.index === view.selected}
              class:text-selected-fg={row.index === view.selected}
              onclick={() => (view.selected = row.index)}
            >
              <td class="px-2 py-0.5">{row.index}</td>
              <td class="px-2">{formula(row.composition)}</td>
              <td class="px-2 text-right">{row.n_atoms}</td>
              <td class="px-2 text-right">{fmt(row.energy)}</td>
              <td class="px-2">{row.periodic ? `✓` : ``}</td>
              <td class="max-w-48 truncate px-2" title={row.name ?? ``}>{row.name ?? ``}</td>
              {#each columns as key (key)}
                {@const value = fmt(row.properties[key])}
                <td class="max-w-48 truncate px-2" title={value}>{value}</td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </section>
  <aside class="flex w-[45%] min-w-80 flex-col">
    <div class="min-h-0 flex-[3] p-2">
      <Viewer index={view.selected}>
        {#snippet header()}
          {@const added = compare.records.includes(view.selected)}
          <button
            class="btn"
            disabled={!added && compare.records.length >= MAX_COMPARE}
            onclick={() => toggle_compare(view.selected)}
          >
            {added ? `Remove from compare` : `Add to compare`}
          </button>
        {/snippet}
      </Viewer>
    </div>
    <div class="min-h-0 flex-[2] overflow-auto border-t border-line p-2">
      <Details index={view.selected} />
    </div>
  </aside>
</div>
