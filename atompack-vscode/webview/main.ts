import 'matterviz/app.css'
import './app.css'
import { apply_theme_to_dom } from 'matterviz/theme'
import { watch_theme } from 'matterviz/theme/embedded'
import { mount } from 'svelte'
import App from './App.svelte'

// VS Code marks <body> with vscode-dark / vscode-light; MatterViz follows it.
watch_theme(document.body, apply_theme_to_dom)
mount(App, { target: document.getElementById(`app`)! })
