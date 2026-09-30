<script lang="ts">
  import { fmt } from './chem'

  // All values in muted bars, the currently filtered subset on top; drag to select a range,
  // double-click to clear it.
  let {
    label,
    values,
    subset,
    range = $bindable(),
  }: {
    label: string
    values: number[]
    subset: number[]
    range?: [number, number] | null
  } = $props()

  const BINS = 40
  const W = 240
  const H = 56

  const finite = $derived(values.filter(Number.isFinite))
  const lo = $derived(finite.reduce((a, b) => Math.min(a, b), Infinity))
  const hi = $derived(finite.reduce((a, b) => Math.max(a, b), -Infinity))
  const width = $derived(hi > lo ? (hi - lo) / BINS : 1)

  function counts(data: number[]) {
    const bins = new Array(BINS).fill(0)
    for (const v of data)
      if (Number.isFinite(v)) bins[Math.min(BINS - 1, Math.floor((v - lo) / width))] += 1
    return bins
  }
  const all = $derived(counts(finite))
  const sub = $derived(counts(subset))
  const peak = $derived(Math.max(1, ...all))

  const x_of = (v: number) => ((v - lo) / (width * BINS)) * W
  const v_of = (x: number) => lo + (Math.min(W, Math.max(0, x)) / W) * width * BINS

  let drag_start: number | null = null
  const x_in = (e: PointerEvent) => {
    const rect = (e.currentTarget as SVGElement).getBoundingClientRect()
    return ((e.clientX - rect.left) / rect.width) * W
  }
</script>

<div class="text-xs">
  <div class="flex justify-between gap-2">
    <span class="truncate font-medium" title={label}>{label}</span>
    <span class="font-mono text-muted">
      {#if range}{fmt(range[0], 4)} – {fmt(range[1], 4)}{:else}{fmt(lo, 4)} – {fmt(hi, 4)}{/if}
    </span>
  </div>
  {#if finite.length}
    <svg
      viewBox="0 0 {W} {H}"
      class="h-14 w-full cursor-crosshair touch-none select-none"
      preserveAspectRatio="none"
      role="slider"
      aria-valuenow={range?.[0] ?? lo}
      tabindex="-1"
      onpointerdown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId)
        drag_start = v_of(x_in(e))
        range = [drag_start, drag_start]
      }}
      onpointermove={(e) => {
        if (drag_start === null) return
        const v = v_of(x_in(e))
        range = [Math.min(drag_start, v), Math.max(drag_start, v)]
      }}
      onpointerup={() => {
        if (range && range[0] === range[1]) range = null
        drag_start = null
      }}
      ondblclick={() => (range = null)}
    >
      {#each all as n, i (i)}
        <rect x={(i * W) / BINS} width={W / BINS - 0.5} y={H - (n / peak) * H} height={(n / peak) * H} class="fill-muted opacity-25" />
        <rect x={(i * W) / BINS} width={W / BINS - 0.5} y={H - (sub[i] / peak) * H} height={(sub[i] / peak) * H} class="fill-chart" />
      {/each}
      {#if range}
        <rect x={x_of(range[0])} width={Math.max(1, x_of(range[1]) - x_of(range[0]))} y="0" height={H} class="fill-accent opacity-20" />
      {/if}
    </svg>
  {:else}
    <p class="text-muted">no finite values</p>
  {/if}
</div>
