<script lang="ts">
  import Compare from './lib/Compare.svelte'
  import Groups from './lib/Groups.svelte'
  import OverviewTab from './lib/OverviewTab.svelte'
  import Records from './lib/Records.svelte'
  import { api } from './lib/rpc'
  import { compare } from './lib/state.svelte'

  type Tab = `overview` | `records` | `groups` | `compare`
  let tab = $state<Tab>(`records`)
  const overview = api.overview()
</script>

{#await overview}
  <p class="p-4 text-muted">Opening…</p>
{:then info}
  <div class="flex h-full flex-col">
    <nav class="flex items-center gap-1 border-b border-line px-2 text-sm">
      <span class="mr-3 font-semibold">{document.body.dataset.title}</span>
      {#each [
        [`overview`, `Overview`],
        [`records`, `Records (${info.num_records.toLocaleString()})`],
        ...(info.groupings.length ? [[`groups`, `Groups`]] : []),
        [`compare`, `Compare (${compare.records.length})`],
      ] as [id, label] (id)}
        <button
          class="border-b-2 px-3 py-1.5 {tab === id ? `border-accent` : `border-transparent text-muted hover:text-fg`}"
          onclick={() => (tab = id as Tab)}
        >
          {label}
        </button>
      {/each}
    </nav>
    <!-- Tabs stay mounted only while visible: WebGL contexts are a scarce resource. -->
    <main class="min-h-0 flex-1">
      {#if tab === `overview`}
        <OverviewTab overview={info} />
      {:else if tab === `records`}
        <Records total={info.num_records} />
      {:else if tab === `groups`}
        <Groups groupings={info.groupings} on_compare={() => (tab = `compare`)} />
      {:else}
        <Compare />
      {/if}
    </main>
  </div>
{:catch err}
  <p class="p-4 text-error">Failed to open: {err.message}</p>
{/await}
