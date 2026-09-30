import { execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { fileURLToPath } from 'node:url'

const require = createRequire(import.meta.url)
const root = fileURLToPath(new URL('../', import.meta.url))
const { target } = JSON.parse(readFileSync(new URL('../dist/native-target.json', import.meta.url)))
if (target !== `${process.platform}-${process.arch}`) throw new Error('Rebuild the native reader on the packaging host')
// Always tag the VSIX: an untagged package could be installed on an incompatible host.
execFileSync(process.execPath, [require.resolve('@vscode/vsce/vsce'), 'package', '--no-dependencies', '--target', target], {
  cwd: root,
  stdio: 'inherit',
})
