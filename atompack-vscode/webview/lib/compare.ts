/**
 * One pane per record, in first-seen order, capped at `max`. A record listed more than once
 * (e.g. the same structure in two group roles) keeps every distinct label, joined with ", ".
 */
export function merge_compare(entries: { record: number; label?: string }[], max: number) {
  const labels = new Map<number, string[]>()
  for (const { record, label } of entries) {
    const seen = labels.get(record) ?? []
    if (!labels.has(record)) {
      if (labels.size >= max) continue
      labels.set(record, seen)
    }
    if (label && !seen.includes(label)) seen.push(label)
  }
  return {
    records: [...labels.keys()],
    labels: Object.fromEntries([...labels].filter(([, l]) => l.length).map(([r, l]) => [r, l.join(`, `)])),
  }
}
