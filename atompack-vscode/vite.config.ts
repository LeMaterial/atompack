import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The extension HTML loads dist/webview/index.{js,css} by fixed name.
export default defineConfig({
  plugins: [svelte(), tailwindcss()],
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
