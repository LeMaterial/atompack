<script lang="ts">
  // BinnedScatterPlot, not ScatterPlot: the latter keeps per-point objects and crashed
  // the webview at ~50k points; this one takes typed arrays and draws density bins.
  import type { Vec2 } from 'matterviz/math'
  import { type AxisConfig, BinnedScatterPlot, Histogram } from 'matterviz/plot'
  import { fmt, key_formula } from './chem'
  import { is_toggle_key, step_key } from './keys'
  import { fit, plot_filter, select_rows } from './query'
  import { scan, start_scan, table } from './scan.svelte'
  import {
    compare,
    MAX_COMPARE,
    plot_view as view,
    records_view,
    toggle_compare,
  } from './state.svelte'
  import Viewer from './Viewer.svelte'

  // Rows shown in the records-in-view list.
  const LIST = 500
  let { total, on_records }: { total: number; on_records(): void } = $props()
  $effect(() => {
    start_scan(total)
  })

  // Pick sensible columns once the first chunk arrives.
  $effect(() => {
    const keys = scan.keys
    const pick = (...want: string[]) => want.find((k) => keys.includes(k)) ?? keys[0]
    if (keys.length && !keys.includes(view.x)) view.x = pick(`energy_per_atom`, `energy`)
    if (keys.length && !keys.includes(view.y)) view.y = pick(`n_atoms`)
  })

  // Axis ranges: the user's zoom (drag a box, double-click resets), else the bulk of the data.
  let zoom = $state<{ x: Vec2 | null; y: Vec2 | null }>({ x: null, y: null })
  $effect(() => {
    void [view.kind, view.x, view.y, view.log_x, view.log_y, view.outliers]
    zoom = { x: null, y: null }
  })
  const fit_x = $derived(scan.done && !view.outliers ? fit(table.columns[view.x], view.log_x) : null)
  const fit_y = $derived(
    scan.done && !view.outliers && view.kind === `scatter` ? fit(table.columns[view.y], view.log_y) : null,
  )
  const x_range = $derived(zoom.x ?? fit_x)
  const y_range = $derived(zoom.y ?? fit_y)
  const axis = (label: string, log: boolean, range: Vec2 | null): AxisConfig => ({
    label,
    format: `.4~g`,
    scale_type: log ? `log` : `linear`,
    range: range ?? [null, null],
  })
  const x_axis = $derived(axis(view.x, view.log_x, x_range))
  const y_axis = $derived(axis(view.kind === `scatter` ? view.y : `count`, view.log_y, y_range))
  const pinned = (range: AxisConfig[`range`]) =>
    range && range[0] !== null && range[1] !== null ? (range as Vec2) : null

  // Rows not read yet are NaN, so the finite checks also skip them. Recomputed as scan.done grows.
  const hist_values = $derived.by(() => {
    const xs = scan.done && table.columns[view.x]
    if (!xs) return []
    const out: number[] = []
    for (const x of xs) if (Number.isFinite(x)) out.push(x)
    return out
  })

  // One point per record with finite x and y; point ids are record indices.
  const scatter = $derived.by(() => {
    const [xs, ys] = [table.columns[view.x], table.columns[view.y]]
    const keep: number[] = []
    if (scan.done && xs && ys)
      for (let i = 0; i < xs.length; i++) if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) keep.push(i)
    return {
      x: Float32Array.from(keep, (i) => xs[i]),
      y: Float32Array.from(keep, (i) => ys[i]),
      point_ids: Int32Array.from(keep, (i) => table.ids[i]),
    }
  })
  const n_plotted = $derived(view.kind === `scatter` ? scatter.point_ids.length : hist_values.length)

  // Rows by x, sorted once per column: zooming then only filters (sorting 5M rows takes ~0.6 s).
  const by_x = $derived(
    scan.finished ? select_rows(() => true, { key: view.x, desc: false }) : new Int32Array(0),
  )
  // Rows inside the visible ranges (x only for histograms): their count and the first LIST.
  const in_view = $derived.by(() => {
    const [xs, ys] = [table.columns[view.x], view.kind === `scatter` ? table.columns[view.y] : undefined]
    const inside = (v: number, range: Vec2 | null, log: boolean) =>
      Number.isFinite(v) && (!log || v > 0) && (!range || (v >= range[0] && v <= range[1]))
    const listed: number[] = []
    let count = 0
    for (let i = 0; i < by_x.length; i++) {
      const row = by_x[view.desc ? by_x.length - 1 - i : i]
      if (!inside(xs[row], x_range, view.log_x) || (ys && !inside(ys[row], y_range, view.log_y))) continue
      if (count++ < LIST) listed.push(row)
    }
    return { count, listed }
  })
  const listed = $derived(in_view.listed)

  function select(index: number) {
    view.selected = index
    requestAnimationFrame(() => document.getElementById(`in-view-${index}`)?.scrollIntoView({ block: `nearest` }))
  }

  // The same ranges and order in the Records tab, which pages through all of them.
  function open_records() {
    records_view.filter = plot_filter([
      { key: view.x, range: x_range, log: view.log_x },
      ...(view.kind === `scatter` ? [{ key: view.y, range: y_range, log: view.log_y }] : []),
    ])
    records_view.sort = { key: view.x, desc: view.desc }
    records_view.page = 0
    on_records()
  }

  function onkeydown(event: KeyboardEvent) {
    if (!listed.length) return
    if (is_toggle_key(event) && view.selected !== null) {
      event.preventDefault()
      toggle_compare(view.selected)
      return
    }
    const step = step_key(event, 20)
    if (step === undefined) return
    event.preventDefault()
    const at = view.selected === null ? -1 : listed.findIndex((row) => table.ids[row] === view.selected)
    const next = at < 0 ? 0 : Math.min(listed.length - 1, Math.max(0, at + step))
    select(table.ids[listed[next]])
  }
</script>

<svelte:window {onkeydown} />

<div class="flex h-full min-h-0 flex-col">
  <div class="flex flex-wrap items-center gap-3 border-b border-line px-2 py-1 text-xs">
    <span class="flex overflow-hidden rounded border border-line">
      {#each [`histogram`, `scatter`] as const as kind (kind)}
        <button
          class="px-2 py-0.5 {view.kind === kind ? `bg-button text-button-fg` : `hover:bg-hover`}"
          onclick={() => (view.kind = kind)}>{kind}</button
        >
      {/each}
    </span>
    {#snippet axis_picker(key: `x` | `y`, log: `log_x` | `log_y`)}
      <label class="flex items-center gap-1">
        <span class="text-muted">{key}</span>
        <select class="rounded border border-line bg-input px-1" bind:value={view[key]}>
          {#each scan.keys as k (k)}<option value={k}>{k}</option>{/each}
        </select>
      </label>
      <label class="flex items-center gap-1 text-muted">
        <input type="checkbox" bind:checked={view[log]} />log
      </label>
    {/snippet}
    {@render axis_picker(`x`, `log_x`)}
    {#if view.kind === `scatter`}
      {@render axis_picker(`y`, `log_y`)}
    {:else}
      <label class="flex items-center gap-1 text-muted">
        <input type="checkbox" bind:checked={view.log_y} />log count
      </label>
    {/if}
    <label
      class="flex items-center gap-1 text-muted"
      title="Axes fit the bulk of the data (0.1–99.9th percentiles); check to include far outliers"
    >
      <input type="checkbox" bind:checked={view.outliers} />outliers
    </label>
    <span class="ml-auto flex items-center gap-2 text-muted">
      {#if scan.error}
        <span class="text-error">{scan.error}</span>
      {:else if !scan.finished}
        <span>scanning {scan.done.toLocaleString()} / {scan.planned.toLocaleString()} records</span>
        <span class="h-1 w-24 overflow-hidden rounded bg-line">
          <span class="block h-full bg-chart" style:width="{(100 * scan.done) / scan.planned}%"></span>
        </span>
      {:else}
        <span>
          {n_plotted.toLocaleString()} of {scan.total.toLocaleString()} records have values
          {#if scan.done < scan.total}(sample of {scan.done.toLocaleString()} spread over the file){/if}
        </span>
      {/if}
      {#if scan.skipped}
        <span title="Memory limit for plot columns">{scan.skipped} columns skipped</span>
      {/if}
    </span>
  </div>

  <div class="flex min-h-0 flex-1">
    <section class="relative min-w-0 flex-1 p-2" title="Drag a box to zoom, double-click to reset">
      {#if !scan.keys.length}
        <p class="p-2 text-muted">{scan.error ? `` : `Reading records…`}</p>
      {:else if view.kind === `histogram`}
        <Histogram
          series={[{ values: hist_values, label: view.x }]}
          bind:x_axis={() => x_axis, (a) => (zoom.x = pinned(a.range))}
          bind:y_axis={() => y_axis, (a) => (zoom.y = pinned(a.range))}
          show_legend={false}
          style="height: 100%"
        />
      {:else}
        <BinnedScatterPlot
          series={[scatter]}
          bind:x_axis={() => x_axis, (a) => (zoom.x = pinned(a.range))}
          bind:y_axis={() => y_axis, (a) => (zoom.y = pinned(a.range))}
          selected_point_id={view.selected}
          on_point_click={({ point }) => select(Number(point.point_id))}
          style="height: 100%"
        >
          {#snippet tooltip({ x, y, point })}
            <div class="font-mono text-xs">
              <div>#{point.point_id}</div>
              <div>{view.x} = {fmt(x)}</div>
              <div>{view.y} = {fmt(y)}</div>
              <div class="text-muted">click to view</div>
            </div>
          {/snippet}
        </BinnedScatterPlot>
      {/if}
    </section>

    <aside class="flex w-[35%] min-w-72 flex-col gap-2 border-l border-line p-2">
      <div class="flex min-h-0 flex-[2] flex-col overflow-hidden rounded border border-line">
        <div class="flex items-center gap-2 border-b border-line bg-panel px-2 py-1 text-xs whitespace-nowrap">
          {#if scan.finished}
            <span>{in_view.count.toLocaleString()} in view</span>
            <button
              class="text-muted hover:text-fg"
              title="Sort order (j/k or ↑/↓ step through the list)"
              onclick={() => (view.desc = !view.desc)}
            >
              by {view.x}
              {view.desc ? `↓` : `↑`}
            </button>
            <button class="btn ml-auto" disabled={!in_view.count} onclick={open_records}>Open in Records</button>
          {:else}
            <span class="text-muted">Records in view are listed once the scan is over</span>
          {/if}
        </div>
        <div class="min-h-0 flex-1 overflow-auto">
          <table class="w-full font-mono text-xs whitespace-nowrap">
            <tbody>
              {#each listed as row (row)}
                {@const index = table.ids[row]}
                <tr
                  id="in-view-{index}"
                  class="cursor-pointer border-b border-line/40 select-none hover:bg-hover"
                  class:!bg-selected={index === view.selected}
                  class:text-selected-fg={index === view.selected}
                  onclick={() => select(index)}
                  ondblclick={() => toggle_compare(index)}
                >
                  <td class="px-2 py-0.5">{index}</td>
                  <td class="max-w-40 truncate px-2">{key_formula(table.compositions[table.composition[row]])}</td>
                  <td class="px-2 text-right">{fmt(table.columns[view.x][row])}</td>
                  {#if view.kind === `scatter`}
                    <td class="px-2 text-right">{fmt(table.columns[view.y]?.[row])}</td>
                  {/if}
                </tr>
              {/each}
            </tbody>
          </table>
          {#if in_view.count > LIST}
            <p class="p-2 text-xs text-muted">
              First {LIST} shown; zoom in or open in Records to page through all.
            </p>
          {/if}
        </div>
      </div>
      {#if view.selected !== null}
        {@const index = view.selected}
        <div class="min-h-0 flex-[3]">
          <Viewer {index}>
            {#snippet header()}
              {@const added = compare.records.includes(index)}
              <button
                class="btn"
                disabled={!added && compare.records.length >= MAX_COMPARE}
                onclick={() => toggle_compare(index)}
              >
                {added ? `Remove from compare` : `Add to compare`}
              </button>
              <button class="btn" title="Close" onclick={() => (view.selected = null)}>✕</button>
            {/snippet}
          </Viewer>
        </div>
      {/if}
    </aside>
  </div>
</div>
