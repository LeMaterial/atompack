import * as vscode from 'vscode'
import { AtpReader, bytesSource, fileSource, READER_METHODS, type ReaderMethod } from './reader'
import { Scanner } from './scan'

class AtpDocument implements vscode.CustomDocument {
  constructor(
    readonly uri: vscode.Uri,
    readonly reader: AtpReader,
    readonly scanner: Scanner,
  ) {}

  dispose() {
    this.scanner.dispose()
    this.reader.dispose()
  }
}

class AtpEditorProvider implements vscode.CustomReadonlyEditorProvider<AtpDocument> {
  constructor(private context: vscode.ExtensionContext) {}

  async openCustomDocument(uri: vscode.Uri): Promise<AtpDocument> {
    // Local files are memory mapped; other schemes use a temporary local copy.
    const file = uri.scheme === `file` ? uri.fsPath : undefined
    const reader = new AtpReader(file ? fileSource(file) : bytesSource(await vscode.workspace.fs.readFile(uri)))
    return new AtpDocument(uri, reader, new Scanner(file, reader))
  }

  resolveCustomEditor(document: AtpDocument, panel: vscode.WebviewPanel) {
    panel.iconPath = vscode.Uri.joinPath(this.context.extensionUri, `media`, `icon.png`)
    const webview = panel.webview
    const dist = vscode.Uri.joinPath(this.context.extensionUri, `dist`, `webview`)
    webview.options = { enableScripts: true, localResourceRoots: [dist] }

    webview.onDidReceiveMessage(async ({ id, method, params }) => {
      let reply
      try {
        if (method === `record_columns`) {
          const [start, count] = params
          reply = { id, result: await document.scanner.record_columns(start, count) }
        } else {
          if (!READER_METHODS.includes(method)) throw new Error(`unknown method ${method}`)
          const fn = document.reader[method as ReaderMethod] as (...args: number[]) => unknown
          reply = { id, result: fn.apply(document.reader, params ?? []) }
        }
      } catch (err) {
        reply = { id, error: err instanceof Error ? err.message : String(err) }
      }
      // The panel may have closed while a scan chunk was being read.
      webview.postMessage(reply).then(undefined, () => {})
    })

    const nonce = Array.from({ length: 32 }, () => Math.random().toString(36)[2]).join(``)
    const asset = (name: string) => webview.asWebviewUri(vscode.Uri.joinPath(dist, name))
    const title = document.uri.path.split(`/`).pop() ?? ``
    webview.html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data: blob:; style-src ${webview.cspSource} 'unsafe-inline'; font-src ${webview.cspSource}; script-src 'nonce-${nonce}' ${webview.cspSource} 'wasm-unsafe-eval'; connect-src ${webview.cspSource}; worker-src ${webview.cspSource} blob:;">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="${asset(`index.css`)}">
</head>
<body data-title="${title.replace(/[&"<>]/g, ``)}">
  <div id="app"></div>
  <script type="module" nonce="${nonce}" src="${asset(`index.js`)}"></script>
</body>
</html>`
  }
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      `atompack.viewer`,
      new AtpEditorProvider(context),
      { webviewOptions: { retainContextWhenHidden: true } },
    ),
  )
}

export function deactivate() {}
