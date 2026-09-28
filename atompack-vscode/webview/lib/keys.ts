/**
 * Records to move for a list navigation key: ↑/↓ or k/j by one, PageUp/PageDown or Ctrl-u/d by
 * a page, Home/End or g/G to the ends. Undefined for other keys, and in text fields and selects.
 */
export function step_key(event: KeyboardEvent, page: number): number | undefined {
  const target = event.target as Element | null
  if (target?.closest(`input:not([type=checkbox]), select, textarea, [contenteditable]`)) return undefined
  if (event.ctrlKey) return event.metaKey || event.altKey ? undefined : { d: page, u: -page }[event.key]
  if (event.metaKey || event.altKey) return undefined
  return {
    ArrowDown: 1,
    j: 1,
    ArrowUp: -1,
    k: -1,
    PageDown: page,
    PageUp: -page,
    Home: -Infinity,
    g: -Infinity,
    End: Infinity,
    G: Infinity,
  }[event.key]
}

/** Space toggles compare, unless typing in a field or on a plot (where it resets the view). */
export const is_toggle_key = (event: KeyboardEvent) =>
  event.key === ` ` && !(event.target as Element | null)?.closest(`input, select, textarea, svg, button, [contenteditable]`)
