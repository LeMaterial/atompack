<script lang="ts">
  import type { Vec3 } from 'matterviz/math'
  import { type AnyStructure, get_center_of_mass, Structure } from 'matterviz/structure'
  import type { Snippet } from 'svelte'
  import { composition, fmt, formula, to_structure } from './chem'
  import { api } from './rpc'
  import type { CameraPose } from './state.svelte'

  let {
    index,
    label = undefined,
    camera = undefined,
    header = undefined,
  }: {
    index: number
    label?: string
    // When given, the viewer follows and updates this shared pose.
    camera?: CameraPose
    header?: Snippet
  } = $props()

  const Z_UP: Vec3 = [-Math.PI / 2, 0, 0]

  const mol = $derived(api.molecule(index))
  // three.js draws y up; turn the structure -90° about x, (x, y, z) -> (x, z, -y), so z is up.
  // MatterViz turns molecules about their center of mass but aims the camera at their
  // bounding-box center, so aim at where the turn moves that center instead.
  function scene_props(structure: AnyStructure) {
    if (camera?.position) return { rotation: Z_UP, camera_position: camera.position, camera_target: camera.target }
    if (`lattice` in structure || !structure.sites.length) return { rotation: Z_UP }
    const [lo, hi] = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]]
    for (const { xyz } of structure.sites)
      for (const a of [0, 1, 2]) [lo[a], hi[a]] = [Math.min(lo[a], xyz[a]), Math.max(hi[a], xyz[a])]
    const com = get_center_of_mass(structure)
    const [dx, dy, dz] = [0, 1, 2].map((a) => (lo[a] + hi[a]) / 2 - com[a])
    return { rotation: Z_UP, camera_target: [com[0] + dx, com[1] + dz, com[2] - dy] as Vec3 }
  }
</script>

<div class="flex h-full min-h-0 flex-col overflow-hidden rounded border border-line">
  <div class="flex items-center gap-2 border-b border-line bg-panel px-2 py-1 text-xs whitespace-nowrap">
    {#if label}<span class="rounded bg-selected px-1 text-selected-fg">{label}</span>{/if}
    <span class="font-mono">#{index}</span>
    {#await mol then m}
      <span class="truncate" title={m.name ?? ``}>{formula(composition(m.numbers))}</span>
      {#if m.energy !== null}<span class="shrink-0 text-muted">E = {fmt(m.energy)}</span>{/if}
    {/await}
    <span class="ml-auto flex gap-1">{@render header?.()}</span>
  </div>
  <div class="relative min-h-0 flex-1">
    {#await mol}
      <p class="p-4 text-muted">Loading #{index}…</p>
    {:then m}
      {@const structure = to_structure(m)}
      <Structure
        {structure}
        scene_props={scene_props(structure)}
        show_controls={{ mode: `hover`, hidden: [`multi-view`] }}
        enable_info_pane={false}
        allow_file_drop={false}
        persist_settings={false}
        style="height: 100%; width: 100%"
        on_camera_move={camera
          ? (e) => {
              camera.position = e.camera_position
              camera.target = e.camera_target
            }
          : undefined}
      />
    {:catch err}
      <p class="p-4 text-error">{err.message}</p>
    {/await}
  </div>
</div>
