# Upstream inventory tools

Run these maintained command-line tools from the repository root when checking a luma.gl upgrade:

```sh
node scripts/upstream/exports.mjs node_modules/@luma.gl /tmp/luma-exports.json
node scripts/upstream/usage.mjs /tmp/luma-exports.json /tmp/luma-usage.json /tmp/luma-report.json
node scripts/upstream/diff.mjs /tmp/previous-luma-exports.json /tmp/luma-exports.json
# also compare declaration text and string constants between two installed versions
node scripts/upstream/diff.mjs /tmp/previous-luma-exports.json /tmp/luma-exports.json \
  /path/to/old/node_modules/@luma.gl node_modules/@luma.gl
```

`exports.mjs` uses TypeScript declarations to list symbols from explicit public package entries;
wildcards and `_internal` entry paths are excluded. Conditional targets and string JavaScript
targets with declaration siblings resolve to the same entry. Unresolved public entries remain
explicit `null` inventory values and print as `unresolved`, rather than disappearing.
`usage.mjs` parses individual import declarations by repository area, distinguishing type imports
and aliases; Markdown is limited to fenced examples. It is an import census, not semantic reachability.
Usage is counted per exact entry point named in the import (entry plus name); namespace imports are
recorded as `*`. A name never imported through an entry, but imported through another entry that exports the same
declaration, is not credited to the first entry; the report lists it under `usedThroughOtherEntries`.
`diff.mjs` reports added/removed entries and symbol names, with kinds for additions. Changes between
resolved and unresolved declarations are labeled explicitly; an unresolved declaration is not a
removed public entry. Given two installed `@luma.gl` package roots (old, then new) it also reports, for
every name exported by an entry in both versions, `CHANGED DECLARATIONS` (the exported declaration text
differs, whitespace and comments ignored) and `CHANGED STRING VALUES` (an exported string constant, such
as WGSL source, or a string property of an exported plain object, differs; hashes print as 12 hex
digits). A name-only diff misses both. An entry that cannot be imported prints `UNLOADED ENTRY`.

These tools maintain the workflow introduced in the
[2026-11-11 usage review](../../dev-docs/research/luma-usage-review-2026-11-11/README.md).
Original script snapshots and inventories remain with that dated evidence. New measurements should
record the dependency version, command and environment; do not overwrite historical baselines.
