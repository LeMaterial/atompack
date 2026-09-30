import { execFileSync } from 'node:child_process'
import { copyFileSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const root = fileURLToPath(new URL('../../', import.meta.url))
const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--no-deps', '--format-version=1'], { cwd: root }))
execFileSync('cargo', ['build', '-p', 'atompack-node', '--release', '--locked'], { cwd: root, stdio: 'inherit' })
const library = { darwin: 'libatompack_node.dylib', linux: 'libatompack_node.so', win32: 'atompack_node.dll' }[process.platform]
if (!library) throw new Error(`Unsupported platform: ${process.platform}`)
const dist = path.join(root, 'atompack-vscode', 'dist')
mkdirSync(dist, { recursive: true })
copyFileSync(path.join(metadata.target_directory, 'release', library), path.join(dist, 'atompack.node'))
writeFileSync(path.join(dist, 'native-target.json'), JSON.stringify({ target: `${process.platform}-${process.arch}` }))
// Remove the obsolete reader artifact when rebuilding an existing checkout.
rmSync(path.join(dist, 'atompack.wasm'), { force: true })
