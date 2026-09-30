// Runs the native reader and its scan workers in a child process. Reads go through a memory
// map, so a file truncated while open faults (SIGBUS) in a way no JS or Rust handler can catch:
// that must end a disposable process, not the shared extension host.
import { type ChildProcess, fork } from 'node:child_process'
import { once } from 'node:events'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

export type Source = { path: string } | { bytes: Uint8Array }
type Pending = { resolve(value: unknown): void; reject(err: Error): void }

export class ReaderClient {
  private file: string
  private temporary?: string
  private child?: ChildProcess
  private pending = new Map<number, Pending>()
  private next_id = 0
  private closed = false

  /** Other filesystems (remote, virtual) are read from a private temporary copy. */
  constructor(source: Source) {
    if (`path` in source) {
      this.file = source.path
      return
    }
    this.temporary = fs.mkdtempSync(path.join(os.tmpdir(), `atompack-`))
    this.file = path.join(this.temporary, `database.atp`)
    try {
      fs.writeFileSync(this.file, source.bytes, { mode: 0o600 })
    } catch (err) {
      fs.rmSync(this.temporary, { recursive: true, force: true })
      throw err
    }
  }

  /** A reader method, or record_columns from the scan workers. Restarts a stopped reader. */
  call(method: string, params: number[] = []): Promise<unknown> {
    if (this.closed) return Promise.reject(new Error(`file closed`))
    const child = this.child ?? this.spawn()
    const id = this.next_id++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      child.send({ id, method, params })
    })
  }

  private spawn(): ChildProcess {
    const child = fork(path.join(__dirname, `reader-process.js`), [this.file], {
      // Structured clone keeps the scan's typed arrays.
      serialization: `advanced`,
      // The VS Code extension host runs in Electron; run the child as plain Node.
      env: { ...process.env, ELECTRON_RUN_AS_NODE: `1` },
      stdio: [`ignore`, `inherit`, `inherit`, `ipc`],
    })
    child.on(`message`, (msg: { id: number; result?: unknown; error?: string }) => {
      const request = this.pending.get(msg.id)
      this.pending.delete(msg.id)
      if (msg.error !== undefined) request?.reject(new Error(msg.error))
      else request?.resolve(msg.result)
    })
    child.on(`error`, (err) => this.fail(err))
    child.on(`exit`, (code, signal) => {
      if (this.child === child) this.child = undefined
      this.fail(new Error(
        `The atompack reader stopped (${signal ?? `exit code ${code}`}). ` +
          `If the file changed while open, reopen it.`,
      ))
    })
    this.child = child
    return child
  }

  private fail(err: Error) {
    for (const request of this.pending.values()) request.reject(err)
    this.pending.clear()
  }

  /** Stops the reader, then removes the temporary copy (Windows cannot delete it while mapped). */
  async dispose() {
    if (this.closed) return
    this.closed = true
    this.fail(new Error(`file closed`))
    const child = this.child
    if (child) {
      const exited = once(child, `exit`)
      child.kill()
      await exited
    }
    if (this.temporary) fs.rmSync(this.temporary, { recursive: true, force: true })
  }
}
