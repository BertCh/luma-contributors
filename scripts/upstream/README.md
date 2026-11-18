# Upstream inventory tools

Run these maintained command-line tools from the repository root when checking a luma.gl upgrade:

```sh
node scripts/upstream/exports.mjs node_modules/@luma.gl /tmp/luma-exports.json
node scripts/upstream/usage.mjs /tmp/luma-exports.json /tmp/luma-usage.json /tmp/luma-report.json
node scripts/upstream/diff.mjs /tmp/previous-luma-exports.json /tmp/luma-exports.json
```

`exports.mjs` uses TypeScript declarations to list symbols from explicit public package entries;
wildcards and `_internal` entry paths are excluded. Conditional targets and string JavaScript
targets with declaration siblings resolve to the same entry. Unresolved public entries remain
explicit `null` inventory values and print as `unresolved`, rather than disappearing.
`usage.mjs` parses individual import declarations by repository area, distinguishing type imports
and aliases; Markdown is limited to fenced examples. It is an import census, not semantic reachability:
the report matches exported names across entry points, and namespace imports are recorded as `*`.
`diff.mjs` reports added/removed entries and symbol names, with kinds for additions. Changes between
resolved and unresolved declarations are labeled explicitly; an unresolved declaration is not a
removed public entry.

These tools maintain the workflow introduced in the
[2026-11-11 usage review](../../dev-docs/research/luma-usage-review-2026-11-11/README.md).
Original script snapshots and inventories remain with that dated evidence. New measurements should
record the dependency version, command and environment; do not overwrite historical baselines.
