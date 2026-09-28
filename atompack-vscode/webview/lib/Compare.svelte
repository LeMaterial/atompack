<script lang="ts">
  import { type CameraPose, compare, MAX_COMPARE, toggle_compare } from './state.svelte'
  import Viewer from './Viewer.svelte'

  let sync = $state(true)
  let camera = $state<CameraPose>({})
  let columns = $state(3)
</script>

<div class="flex h-full min-h-0 flex-col">
  <div class="flex items-center gap-3 border-b border-line px-2 py-1 text-xs">
    <span>{compare.records.length} / {MAX_COMPARE} structures</span>
    <label class="flex items-center gap-1"><input type="checkbox" bind:checked={sync} /> sync cameras</label>
    <label class="flex items-center gap-1">
      columns <input type="range" min="1" max="6" bind:value={columns} />
    </label>
    <button class="btn" onclick={() => (camera = {})}>Reset view</button>
    <button class="btn ml-auto" onclick={() => (compare.records = [])}>Clear</button>
  </div>
  {#if compare.records.length === 0}
    <p class="p-4 text-muted">Add records from the Records tab or group members from the Groups tab.</p>
  {:else}
    <div
      class="grid min-h-0 flex-1 gap-2 overflow-auto p-2"
      style="grid-template-columns: repeat({columns}, minmax(0, 1fr)); grid-auto-rows: minmax(16rem, 1fr)"
    >
      {#each compare.records as index (index)}
        <Viewer {index} label={compare.labels[index]} camera={sync ? camera : undefined}>
          {#snippet header()}
            <button class="btn" title="Remove" onclick={() => toggle_compare(index)}>✕</button>
          {/snippet}
        </Viewer>
      {/each}
    </div>
  {/if}
</div>
