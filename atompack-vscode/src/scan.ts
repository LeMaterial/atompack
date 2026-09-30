// Serves record_columns chunks for the Plots tab from worker threads, so a long scan neither
// blocks the viewer's other requests nor takes over the machine (remote hosts are often shared).
import * as os from 'node:os'
import * as path from 'node:path'
import { Worker } from 'node:worker_threads'

export type BinaryColumns = {
  count: number
  columns: Record<string, Float32Array<ArrayBuffer>>
  // Composition keys ("Z:count ...") of the chunk's records, as indices into `keys`.
  compositions: { keys: string[]; codes: Int32Array<ArrayBuffer> }
}

// At most half the cores, and no more than 4.
const WORKERS = Math.min(4, Math.max(1, Math.floor(os.availableParallelism() / 2)))
// Workers hold an independent native reader each; free them once a scan is over.
const IDLE_MS = 10_000

/** Float32 with NaN for missing values: half the bytes of f64, and far fewer than JSON. */
export function to_binary(cols: unknown): BinaryColumns {
  const { composition = [], ...numeric } = cols as Record<string, (number | null)[]> & { composition?: string[] }
  const columns = Object.fromEntries(
    Object.entries(numeric).map(([key, values]) => [key, Float32Array.from(values, (v) => v ?? NaN)]),
  )
  const keys = [...new Set(composition)]
  const code = new Map(keys.map((key, i) => [key, i]))
  const codes = Int32Array.from(composition, (key) => code.get(key)!)
  return { count: composition.length, columns, compositions: { keys, codes } }
}

/** Buffers to transfer rather than copy when posting a chunk. */
export const buffers = ({ columns, compositions }: BinaryColumns) => [
  ...Object.values(columns).map((c) => c.buffer),
  compositions.codes.buffer,
]

type Job = { start: number; count: number; resolve(value: BinaryColumns): void; reject(err: Error): void }

export class Scanner {
  private queue: Job[] = []
  private idle: Worker[] = []
  private jobs = new Map<Worker, Job | undefined>()
  private timer: NodeJS.Timeout | undefined

  constructor(private file: string) {}

  record_columns(start: number, count: number): Promise<BinaryColumns> {
    return new Promise((resolve, reject) => {
      this.queue.push({ start, count, resolve, reject })
      this.pump()
    })
  }

  private pump() {
    while (this.queue.length && (this.idle.length || this.jobs.size < WORKERS)) {
      const worker = this.idle.pop() ?? this.spawn()
      const job = this.queue.shift()!
      this.jobs.set(worker, job)
      worker.postMessage({ start: job.start, count: job.count })
    }
    clearTimeout(this.timer)
    if (this.idle.length) this.timer = setTimeout(() => this.stop_idle(), IDLE_MS).unref()
  }

  private spawn(): Worker {
    const worker = new Worker(path.join(__dirname, `scan-worker.js`), { workerData: this.file })
    let failure: Error | undefined
    worker.on(`message`, (msg) => {
      const job = this.jobs.get(worker)!
      this.jobs.set(worker, undefined)
      this.idle.push(worker)
      if (`error` in msg) job.reject(new Error(msg.error))
      else job.resolve(msg.result)
      this.pump()
    })
    worker.on(`error`, (err) => (failure = err))
    worker.on(`exit`, () => {
      const job = this.jobs.get(worker)
      this.jobs.delete(worker)
      this.idle = this.idle.filter((w) => w !== worker)
      if (!job && !failure) return // stopped while idle
      // A worker that died will likely die again (unreadable file, out of memory): fail the
      // queue too instead of respawning.
      const err = failure ?? new Error(`scan worker exited`)
      for (const j of [job, ...this.queue.splice(0)]) j?.reject(err)
    })
    return worker
  }

  private stop_idle() {
    for (const worker of this.idle.splice(0)) {
      this.jobs.delete(worker)
      worker.terminate()
    }
  }

  dispose() {
    clearTimeout(this.timer)
    for (const job of this.queue.splice(0)) job.reject(new Error(`file closed`))
    for (const worker of this.jobs.keys()) worker.terminate()
    this.idle = []
  }
}
