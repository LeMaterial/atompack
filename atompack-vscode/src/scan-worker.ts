// One scan worker: its own native mmap reader, answering record_columns chunks.
import { parentPort, workerData } from 'node:worker_threads'
import { AtpReader, fileSource } from './reader'
import { buffers, to_binary } from './scan'

const reader = new AtpReader(fileSource(workerData))
parentPort!.on(`message`, ({ start, count }) => {
  try {
    const result = to_binary(reader.record_columns(start, count))
    parentPort!.postMessage({ result }, buffers(result))
  } catch (err) {
    parentPort!.postMessage({ error: err instanceof Error ? err.message : String(err) })
  }
})
