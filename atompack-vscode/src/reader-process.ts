// The child process behind one ReaderClient (see client.ts): a reader and plot scan workers.
import { AtpReader, READER_METHODS, type ReaderMethod } from './reader'
import { Scanner } from './scan'

const file = process.argv[2]
let opened: { reader: AtpReader; scanner: Scanner } | undefined

// Opened on demand: a file that cannot be opened fails each request until it is readable again.
async function answer(method: string, params: number[]) {
  const { reader, scanner } = (opened ??= { reader: new AtpReader(file), scanner: new Scanner(file) })
  if (method === `record_columns`) return scanner.record_columns(params[0], params[1])
  if (!READER_METHODS.includes(method as ReaderMethod)) throw new Error(`unknown method ${method}`)
  const fn = reader[method as ReaderMethod] as (...args: number[]) => unknown
  return fn.apply(reader, params)
}

process.on(`message`, ({ id, method, params }: { id: number; method: string; params: number[] }) => {
  answer(method, params).then(
    (result) => process.send!({ id, result }),
    (err) => process.send!({ id, error: err instanceof Error ? err.message : String(err) }),
  )
})
// Exit with the extension host.
process.on(`disconnect`, () => process.exit())
