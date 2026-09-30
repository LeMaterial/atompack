<script lang="ts">
  import { add, normalize_vec, scale, subtract, type Vec3 } from 'matterviz/math'
  import {
    type AnyStructure,
    get_center_of_mass,
    Structure,
    type StructureHandlerData,
    structure_fit_frame,
  } from 'matterviz/structure'
  import type { Snippet } from 'svelte'
  import { composition, fmt, formula, to_structure } from './chem'
  import { api } from './rpc'
  import { type CameraPose, viewer_settings } from './state.svelte'

  let {
    index,
    label = undefined,
    camera = undefined,
    projection = undefined,
    header = undefined,
  }: {
    index: number
    label?: string
    // When given, the viewer follows and updates this shared pose.
    camera?: CameraPose
    // Synchronized grids use perspective: its zoom moves the shared camera position, while
    // MatterViz keeps each pane's orthographic zoom internal.
    projection?: `orthographic` | `perspective`
    header?: Snippet
  } = $props()

  // three.js draws y up; turn the structure -90° about x, (x, y, z) -> (x, z, -y), so z is up.
  const Z_UP: Vec3 = [-Math.PI / 2, 0, 0]
  // MatterViz sizes the longest arrow to ~1.35 atom spacings, which hides the structure of
  // molecules with large forces; keep them short and thin.
  const ARROWS = { vector_scale: 0.35, vector_shaft_radius: -0.006, vector_arrow_head_radius: -0.02, vector_arrow_head_length: -0.05 }

  const mol = $derived(api.molecule(index))

  // The content center and size MatterViz frames. It turns the structure about the cell center
  // (crystals) or center of mass (molecules) but aims the camera at the unturned content
  // center, so aim at where the turn moves that center instead.
  function frame(structure: AnyStructure) {
    const { center, extent } = structure_fit_frame(structure)
    const pivot = `lattice` in structure ? scale(add(...structure.lattice.matrix), 0.5) : get_center_of_mass(structure)
    const [dx, dy, dz] = subtract(center, pivot)
    return { center: add<Vec3>(pivot, [dx, dz, -dy]), size: extent }
  }

  // Synchronized panes share the pose relative to their own frame, so a slab and its gas
  // molecule stay framed alike.
  function scene_props(structure: AnyStructure) {
    const base = {
      ...ARROWS,
      rotation: Z_UP,
      ...(projection && { camera_projection: projection }),
      vector_configs: { force: { visible: viewer_settings.forces, color: null, scale: null } },
    }
    if (!structure.sites.length) return base
    const { center, size } = frame(structure)
    if (!camera?.direction) return { ...base, camera_target: center }
    const target = add(center, scale(camera.offset ?? [0, 0, 0], size))
    return { ...base, camera_target: target, camera_position: add(target, scale(camera.direction, camera.distance! * size)) }
  }

  function share_pose(structure: AnyStructure, { camera_position, camera_target }: StructureHandlerData) {
    if (!camera || !camera_position || !structure.sites.length) return
    const { center, size } = frame(structure)
    const target = camera_target ?? center
    const away = subtract(camera_position, target)
    Object.assign(camera, {
      direction: normalize_vec(away),
      distance: Math.hypot(...away) / size,
      offset: scale(subtract(target, center), 1 / size),
    })
  }
</script>

<div class="flex h-full min-h-0 flex-col overflow-hidden rounded border border-line">
  <div class="flex items-center gap-2 border-b border-line bg-panel px-2 py-1 text-xs whitespace-nowrap">
    {#if label}<span class="rounded bg-selected px-1 text-selected-fg">{label}</span>{/if}
    <span class="font-mono">#{index}</span>
    {#await mol then m}
      <span class="truncate" title={m.name ?? ``}>{formula(composition(m.numbers))}</span>
      {#if m.energy !== null}<span class="shrink-0 text-muted">E = {fmt(m.energy)}</span>{/if}
      {#if m.forces}
        <label class="flex shrink-0 items-center gap-1 text-muted" title="Show force arrows">
          <input type="checkbox" bind:checked={viewer_settings.forces} />forces
        </label>
      {/if}
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
        on_camera_move={camera ? (e) => share_pose(structure, e) : undefined}
      />
    {:catch err}
      <p class="p-4 text-error">{err.message}</p>
    {/await}
  </div>
</div>
