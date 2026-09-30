// Native Node-API reader over AtomDatabase's existing mmap implementation. The extension only
// opens it in the reader process (see client.ts).
import * as fs from 'node:fs'
import * as path from 'node:path'

interface NativeReader {
  overview(): unknown
  records(start: number, count: number): unknown
  record_columns(start: number, count: number): unknown
  molecule(index: number): unknown
  groups(grouping: number, start: number, count: number): unknown
  group_columns(grouping: number): unknown
  dispose(): void
}

/** Owns one native reader. */
export class AtpReader {
  private native?: NativeReader

  constructor(file_path: string) {
    fs.accessSync(file_path, fs.constants.R_OK)
    const { NativeReader } = require(path.join(__dirname, 'atompack.node')) as {
      NativeReader: new (file_path: string) => NativeReader
    }
    this.native = new NativeReader(file_path)
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
