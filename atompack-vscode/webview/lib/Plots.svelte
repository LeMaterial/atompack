<script lang="ts">
  // BinnedScatterPlot, not ScatterPlot: the latter keeps per-point objects and crashed
  // the webview at ~50k points; this one takes typed arrays and draws density bins.
  import { type AxisConfig, BinnedScatterPlot, Histogram } from 'matterviz/plot'
  import { fmt } from './chem'
  import { scan, start_scan, table } from './scan.svelte'
  import { compare, MAX_COMPARE, plot_view as view, toggle_compare } from './state.svelte'
  import Viewer from './Viewer.svelte'

  let { total }: { total: number } = $props()
  $effect(() => {
    start_scan(total)
  })

  // Fresh axes when the plotted columns change; the plots may edit them (zoom, controls).
  let x_axis = $state<AxisConfig>({})
  let y_axis = $state<AxisConfig>({})
  $effect(() => {
    x_axis = { label: view.x, format: `.4~g`, scale_type: view.log_x ? `log` : `linear` }
  })
  $effect(() => {
    const label = view.kind === `scatter` ? view.y : `count`
    y_axis = { label, format: `.4~g`, scale_type: view.log_y ? `log` : `linear` }
  })

  // Pick sensible columns once the first chunk arrives.
  $effect(() => {
    const keys = scan.keys
    const pick = (want: string) => (keys.includes(want) ? want : keys[0])
    if (keys.length && !keys.includes(view.x)) view.x = pick(`energy`)
    if (keys.length && !keys.includes(view.y)) view.y = pick(`n_atoms`)
  })

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
</script>

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
    {#snippet axis(key: `x` | `y`, log: `log_x` | `log_y`)}
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
    {@render axis(`x`, `log_x`)}
    {#if view.kind === `scatter`}
      {@render axis(`y`, `log_y`)}
    {:else}
      <label class="flex items-center gap-1 text-muted">
        <input type="checkbox" bind:checked={view.log_y} />log count
      </label>
    {/if}
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
    <section class="relative min-w-0 flex-1 p-2">
      {#if !scan.keys.length}
        <p class="p-2 text-muted">{scan.error ? `` : `Reading records…`}</p>
      {:else if view.kind === `histogram`}
        <Histogram
          series={[{ values: hist_values, label: view.x }]}
          bind:x_axis
          bind:y_axis
          show_legend={false}
          style="height: 100%"
        />
      {:else}
        <BinnedScatterPlot
          series={[scatter]}
          bind:x_axis
          bind:y_axis
          selected_point_id={view.selected}
          on_point_click={({ point }) => (view.selected = Number(point.point_id))}
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
    {#if view.kind === `scatter` && view.selected !== null}
      {@const index = view.selected}
      <aside class="w-[35%] min-w-72 p-2">
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
      </aside>
    {/if}
  </div>
</div>
