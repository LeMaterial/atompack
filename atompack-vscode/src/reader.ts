// Loads the atompack WASM reader and serves byte ranges from files or buffers.
// See atompack-wasm/src/lib.rs for the ABI.
import * as fs from 'node:fs'
import * as path from 'node:path'

export interface Source {
  size: number
  read(offset: number, dst: Uint8Array): void
  close?(): void
}

export function fileSource(file_path: string): Source {
  const fd = fs.openSync(file_path, `r`)
  return {
    size: fs.fstatSync(fd).size,
    read(offset, dst) {
      let done = 0
      while (done < dst.length) {
        const n = fs.readSync(fd, dst, done, dst.length - done, offset + done)
        if (n === 0) throw new Error(`unexpected end of file at ${offset + done}`)
        done += n
      }
    },
    close: () => fs.closeSync(fd),
  }
}

export function bytesSource(bytes: Uint8Array): Source {
  return {
    size: bytes.length,
    read(offset, dst) {
      if (offset + dst.length > bytes.length) throw new Error(`read out of bounds`)
      dst.set(bytes.subarray(offset, offset + dst.length))
    },
  }
}

interface Exports {
  memory: WebAssembly.Memory
  reply_ptr(): number
  reply_len(): number
  open(source: number, size: number): number
  close(source: number): void
  overview(source: number): number
  records(source: number, start: number, count: number): number
  record_columns(source: number, start: number, count: number): number
  molecule(source: number, index: number): number
  groups(source: number, grouping: number, start: number, count: number): number
  group_columns(source: number, grouping: number): number
}

const sources = new Map<number, Source>()
let next_id = 1
let wasm: Exports | undefined

function load(): Exports {
  if (wasm) return wasm
  const bytes = fs.readFileSync(path.join(__dirname, `atompack.wasm`))
  const imports = {
    atompack: {
      read_at(source: number, offset: number, len: number, dst: number): number {
        try {
          sources.get(source)!.read(offset, new Uint8Array(wasm!.memory.buffer, dst, len))
          return 0
        } catch {
          return 1
        }
      },
    },
  }
  const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports)
  return (wasm = instance.exports as unknown as Exports)
}

function reply(exports: Exports): unknown {
  const bytes = new Uint8Array(exports.memory.buffer, exports.reply_ptr(), exports.reply_len())
  const msg = JSON.parse(new TextDecoder().decode(bytes))
  if (`error` in msg) throw new Error(msg.error)
  return msg.ok
}

/** One open .atp file. Every method maps to a WASM export of the same name. */
export class AtpReader {
  private exports = load()
  private id = next_id++

  constructor(private source: Source) {
    sources.set(this.id, source)
    try {
      this.exports.open(this.id, source.size)
      reply(this.exports)
    } catch (err) {
      this.dispose()
      throw err
    }
  }

  overview() {
    this.exports.overview(this.id)
    return reply(this.exports)
  }
  records(start: number, count: number) {
    this.exports.records(this.id, start, count)
    return reply(this.exports)
  }
  /** Records at the given indices, e.g. a sorted page: one message instead of one per row. */
  records_at(...indices: number[]) {
    return indices.map((index) => (this.records(index, 1) as unknown[])[0])
  }
  record_columns(start: number, count: number) {
    this.exports.record_columns(this.id, start, count)
    return reply(this.exports)
  }
  molecule(index: number) {
    this.exports.molecule(this.id, index)
    return reply(this.exports)
  }
  groups(grouping: number, start: number, count: number) {
    this.exports.groups(this.id, grouping, start, count)
    return reply(this.exports)
  }
  group_columns(grouping: number) {
    this.exports.group_columns(this.id, grouping)
    return reply(this.exports)
  }

  dispose() {
    this.exports.close(this.id)
    sources.delete(this.id)
    this.source.close?.()
  }
}

export const READER_METHODS = [
  `overview`,
  `records`,
  `records_at`,
  `record_columns`,
  `molecule`,
  `groups`,
  `group_columns`,
] as const
export type ReaderMethod = (typeof READER_METHODS)[number]
