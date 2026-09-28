<script lang="ts">
  import { composition_key, fmt, formula } from './chem'
  import Details from './Details.svelte'
  import { is_toggle_key, step_key } from './keys'
  import { parse_filter, select_rows } from './query'
  import { api, type RecordRow } from './rpc'
  import { scan, start_scan, table } from './scan.svelte'
  import { compare, MAX_COMPARE, records_view as view, toggle_compare } from './state.svelte'
  import Viewer from './Viewer.svelte'

  const PAGE = 100
  // Sort choices before the scan has found the file's columns.
  const BUILTINS = [`energy`, `energy_per_atom`, `fmax`, `n_atoms`]
  let { total }: { total: number } = $props()

  let rows = $state<RecordRow[]>([])
  let error = $state(``)
  let note = $state(``)

  // Filtering and sorting run over the scan (see scan.svelte.ts), started on first use.
  const active = $derived(!!view.filter.trim() || !!view.sort)
  $effect(() => {
    if (active) start_scan(total)
  })
  // Scan rows in display order, or null for plain file order.
  const query = $derived.by(() => {
    if (!active || !scan.finished) return { order: null, keys: [] as string[], error: `` }
    try {
      const { test, keys } = view.filter.trim() ? parse_filter(view.filter) : { test: () => true, keys: [] }
      return { order: select_rows(test, view.sort), keys, error: `` }
    } catch (err) {
      return { order: null, keys: [], error: (err as Error).message }
    }
  })
  const order = $derived(query.order)
  const count = $derived(order ? order.length : total)
  const pages = $derived(Math.max(1, Math.ceil(count / PAGE)))
  const columns = $derived(
    [...new Set(rows.flatMap((r) => Object.keys(r.properties)))].filter((key) =>
      key.toLowerCase().includes(view.column_filter.toLowerCase()),
    ),
  )
  // Scanned values of the sort and filter columns the table doesn't show already.
  const extra = $derived(
    order
      ? [...new Set([view.sort?.key ?? ``, ...query.keys])].filter(
          (key) => table.columns[key] && ![`n_atoms`, `energy`, ...columns].includes(key),
        )
      : [],
  )
  const sort_keys = $derived([...new Set([...BUILTINS, ...scan.keys])].sort())

  const record_at = (pos: number) => (order ? table.ids[order[pos]] : pos)
  const position = (index: number) => (order ? order.findIndex((row) => table.ids[row] === index) : index)

  let latest = 0
  $effect(() => {
    const start = view.page * PAGE
    const page = order?.subarray(start, start + PAGE)
    const load = page ? api.records_at(Array.from(page, (row) => table.ids[row])) : api.records(start, PAGE)
    const request = ++latest
    load.then(
      (r) => request === latest && ((rows = r), (error = ``)),
      (err) => request === latest && (error = err.message),
    )
  })
  // Scan row of each shown record, for the extra columns.
  const shown_rows = $derived(order?.subarray(view.page * PAGE, view.page * PAGE + PAGE))

  function show(pos: number) {
    if (!(pos >= 0 && pos < count)) return
    view.selected = record_at(pos)
    view.page = Math.floor(pos / PAGE)
  }
  // Keep the selection in sight, also once its page has loaded.
  $effect(() => {
    void rows
    const index = view.selected
    requestAnimationFrame(() => document.getElementById(`record-${index}`)?.scrollIntoView({ block: `nearest` }))
  })

  function sort_by(key: string) {
    const s = view.sort
    view.sort = s?.key !== key ? { key, desc: false } : s.desc ? null : { key, desc: true }
    view.page = 0
  }

  // A record number, or the next record (after the selected one, in display order) with a formula.
  async function go_to(text: string) {
    note = ``
    text = text.trim()
    if (!text) return
    if (/^\d+$/.test(text)) {
      const pos = position(Number(text))
      if (pos >= 0 && pos < count) show(pos)
      else note = order ? `#${text} is not in this view` : `no record #${text}`
      return
    }
    let want: string
    try {
      want = composition_key(text)
    } catch (err) {
      note = (err as Error).message
      return
    }
    note = `searching…`
    await start_scan(total)
    const code = table.compositions.indexOf(want)
    const n = order ? order.length : table.ids.length
    // Without a view, scan rows are in file order but may be a sample of the records.
    const start = Math.max(0, order ? position(view.selected) + 1 : table.ids.findIndex((id) => id > view.selected))
    for (let i = 0; code >= 0 && i < n; i++) {
      const pos = (start + i) % n
      const row = order ? order[pos] : pos
      if (table.ids[row] >= 0 && table.composition[row] === code) {
        note = ``
        show(order ? pos : table.ids[row])
        return
      }
    }
    note = `no ${text}${scan.done < scan.total ? ` in the ${scan.done.toLocaleString()} scanned records` : ``}`
  }

  function onkeydown(event: KeyboardEvent) {
    if (is_toggle_key(event)) {
      event.preventDefault()
      toggle_compare(view.selected)
      return
    }
    const step = step_key(event, PAGE)
    if (step === undefined) return
    event.preventDefault()
    const pos = Math.max(0, position(view.selected))
    show(Math.min(count - 1, Math.max(0, pos + step)))
  }
</script>

<svelte:window {onkeydown} />

<div class="flex h-full min-h-0">
  <section class="flex min-w-0 flex-1 flex-col border-r border-line">
    <div class="flex flex-wrap items-center gap-2 border-b border-line px-2 py-1 text-xs">
      <button class="btn" disabled={view.page === 0} onclick={() => (view.page -= 1)}>‹</button>
      <span>
        {(view.page * PAGE).toLocaleString()}–{(Math.min(count, (view.page + 1) * PAGE) - 1).toLocaleString()} of
        {count.toLocaleString()}
      </span>
      <button class="btn" disabled={view.page >= pages - 1} onclick={() => (view.page += 1)}>›</button>
      <input
        class="w-36"
        placeholder="go to # or formula"
        title="A record number, or a formula (e.g. H2O) to jump to its next record"
        onchange={(e) => go_to(e.currentTarget.value)}
      />
      {#if note}<span class="text-muted">{note}</span>{/if}
      <input
        class="w-56"
        placeholder="filter, e.g. fmax > 1, formula = H2O"
        title="Conditions on scanned columns (<, <=, >, >=, =, !=), joined by commas"
        value={view.filter}
        onchange={(e) => ((view.filter = e.currentTarget.value), (view.page = 0))}
      />
      <label class="flex items-center gap-1">
        <span class="text-muted">sort</span>
        <select
          class="rounded border border-line bg-input px-1"
          value={view.sort?.key ?? ``}
          onchange={(e) => {
            const key = e.currentTarget.value
            view.sort = key ? { key, desc: view.sort?.desc ?? false } : null
            view.page = 0
          }}
        >
          <option value="">file order</option>
          {#each sort_keys as key (key)}<option value={key}>{key}</option>{/each}
        </select>
      </label>
      {#if view.sort}
        <button class="btn" title="Sort direction" onclick={() => view.sort && (view.sort.desc = !view.sort.desc)}>
          {view.sort.desc ? `↓` : `↑`}
        </button>
      {/if}
      {#if active && !scan.finished}
        <span class="text-muted">scanning {Math.round((100 * scan.done) / Math.max(1, scan.planned))}%…</span>
      {:else if order && scan.done < scan.total}
        <span class="text-muted" title="Filters and sorting cover the scanned records only">
          of a {scan.done.toLocaleString()}-record sample
        </span>
      {/if}
      {#if query.error || scan.error}<span class="text-error">{query.error || scan.error}</span>{/if}
      <span class="ml-auto text-muted" title="Toggle the compare checkbox, double-click a row, or press Space">
        {compare.records.length}/{MAX_COMPARE} in compare
      </span>
      <input class="w-32" placeholder="filter columns…" bind:value={view.column_filter} />
      {#if error}<span class="text-error">{error}</span>{/if}
    </div>
    <div class="min-h-0 flex-1 scroll-pt-6 overflow-auto" title="↑/↓ or j/k to step, Space to add to compare">
      <table class="w-full text-xs whitespace-nowrap">
        <thead class="sticky top-0 bg-panel text-left">
          <tr>
            {#snippet sortable(label: string, key: string, align = ``)}
              <th class="px-2 {align}">
                <button class="hover:text-fg" title="Sort by {key}" onclick={() => sort_by(key)}>
                  {label}{view.sort?.key === key ? (view.sort.desc ? ` ↓` : ` ↑`) : ``}
                </button>
              </th>
            {/snippet}
            <th class="pl-2" title="In compare (double-click a row or press Space)">⧉</th>
            <th class="px-2 py-1">#</th>
            <th class="px-2">formula</th>
            {@render sortable(`atoms`, `n_atoms`, `text-right`)}
            {@render sortable(`energy`, `energy`, `text-right`)}
            {#each extra as key (key)}{@render sortable(key, key, `text-right`)}{/each}
            <th class="px-2">pbc</th>
            <th class="px-2">name</th>
            {#each columns as key (key)}{@render sortable(key, key)}{/each}
          </tr>
        </thead>
        <tbody class="font-mono">
          {#each rows as row, i (row.index)}
            {@const in_compare = compare.records.includes(row.index)}
            <tr
              id="record-{row.index}"
              class="cursor-pointer border-b border-line/40 select-none hover:bg-hover"
              class:!bg-selected={row.index === view.selected}
              class:text-selected-fg={row.index === view.selected}
              onclick={() => (view.selected = row.index)}
              ondblclick={() => toggle_compare(row.index)}
            >
              <td class="pl-2">
                <input
                  type="checkbox"
                  class="align-middle"
                  checked={in_compare}
                  disabled={!in_compare && compare.records.length >= MAX_COMPARE}
                  title={in_compare ? `Remove from compare` : `Add to compare`}
                  onclick={(e) => {
                    e.stopPropagation()
                    toggle_compare(row.index)
                  }}
                  ondblclick={(e) => e.stopPropagation()}
                />
              </td>
              <td class="px-2 py-0.5">{row.index}</td>
              <td class="px-2">{formula(row.composition)}</td>
              <td class="px-2 text-right">{row.n_atoms}</td>
              <td class="px-2 text-right">{fmt(row.energy)}</td>
              {#each extra as key (key)}
                <td class="px-2 text-right">{fmt(shown_rows && table.columns[key][shown_rows[i]])}</td>
              {/each}
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
