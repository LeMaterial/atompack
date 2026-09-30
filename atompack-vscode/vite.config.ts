import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig, type Plugin } from 'vite'

// Dependencies the viewers never use, which would otherwise add about 12 MB to the extension:
// MatterViz parses dropped files (disabled here) in a worker, reading HDF5 with h5wasm, and
// three's Draco and KTX2 loaders for glTF models emit their decoders when merely imported.
const unused: Record<string, string> = {
  h5wasm: 'throw new Error(`HDF5 files are not supported`)',
  'three/examples/jsm/loaders/DRACOLoader.js':
    'export class DRACOLoader { constructor() { throw new Error(`Draco is not supported`) } }',
  'three/examples/jsm/loaders/KTX2Loader.js':
    'export class KTX2Loader { constructor() { throw new Error(`KTX2 is not supported`) } }',
}

function without_unused(): Plugin {
  // The parse worker never starts, which MatterViz already handles.
  const worker = 'new Worker(new URL(`./parse-worker.js`, import.meta.url), { type: `module` })'
  return {
    name: `without-unused`,
    enforce: `pre`,
    resolveId: (id) => (id in unused ? `\0${id}` : undefined),
    load: (id) => (id.startsWith(`\0`) ? unused[id.slice(1)] : undefined),
    transform(code, id) {
      if (!id.endsWith(`matterviz/dist/file-viewer/parse-in-worker.js`)) return
      if (!code.includes(worker)) this.error(`MatterViz's parse worker changed; update without_unused`)
      return code.replace(worker, '(() => { throw new Error(`File parsing is disabled`) })()')
    },
  }
}

// The extension HTML loads dist/webview/index.{js,css} by fixed name.
export default defineConfig({
  plugins: [without_unused(), svelte(), tailwindcss()],
  base: `./`,
  build: {
    outDir: `dist/webview`,
    emptyOutDir: true,
    chunkSizeWarningLimit: 8000,
    rollupOptions: {
      input: `webview/main.ts`,
      output: {
        entryFileNames: `index.js`,
        chunkFileNames: `[name]-[hash].js`,
        assetFileNames: (asset) =>
          asset.names.some((name) => name.endsWith(`.css`)) ? `index.css` : `[name]-[hash][extname]`,
      },
    },
  },
})
