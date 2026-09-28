<script lang="ts">
  import { Structure } from 'matterviz/structure'
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

  const mol = $derived(api.molecule(index))
  const scene_props = $derived(
    camera?.position ? { camera_position: camera.position, camera_target: camera.target } : {},
  )
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
      <Structure
        structure={to_structure(m)}
        {scene_props}
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
