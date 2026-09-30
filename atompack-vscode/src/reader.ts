// Native Node-API reader; local files use AtomDatabase's existing mmap implementation.
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

export type Source = { path: string } | { bytes: Uint8Array }

export function fileSource(file_path: string): Source {
  fs.accessSync(file_path, fs.constants.R_OK)
  return { path: file_path }
}

export function bytesSource(bytes: Uint8Array): Source {
  return { bytes }
}

interface NativeReader {
  overview(): unknown
  records(start: number, count: number): unknown
  record_columns(start: number, count: number): unknown
  molecule(index: number): unknown
  groups(grouping: number, start: number, count: number): unknown
  group_columns(grouping: number): unknown
  dispose(): void
}

/** Owns one native reader and, for virtual filesystem resources, its temporary copy. */
export class AtpReader {
  private native?: NativeReader
  private temporary?: string

  constructor(source: Source) {
    try {
      const { NativeReader } = require(path.join(__dirname, 'atompack.node')) as {
        NativeReader: new (file_path: string) => NativeReader
      }
      let file_path: string
      if ('path' in source) {
        file_path = source.path
      } else {
        this.temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'atompack-'))
        file_path = path.join(this.temporary, 'database.atp')
        fs.writeFileSync(file_path, source.bytes, { mode: 0o600 })
      }
      this.native = new NativeReader(file_path)
    } catch (err) {
      this.dispose()
      throw err
    }
  }

  private get reader(): NativeReader {
    if (!this.native) throw new Error('Reader is not open')
    return this.native
  }

  overview() { return this.reader.overview() }
  records(start: number, count: number) { return this.reader.records(start, count) }
  records_at(...indices: number[]) {
    return indices.map((index) => (this.records(index, 1) as unknown[])[0])
  }
  record_columns(start: number, count: number) { return this.reader.record_columns(start, count) }
  molecule(index: number) { return this.reader.molecule(index) }
  groups(grouping: number, start: number, count: number) {
    return this.reader.groups(grouping, start, count)
  }
  group_columns(grouping: number) { return this.reader.group_columns(grouping) }

  dispose() {
    this.native?.dispose()
    this.native = undefined
    if (this.temporary) {
      fs.rmSync(this.temporary, { recursive: true, force: true })
      this.temporary = undefined
    }
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
