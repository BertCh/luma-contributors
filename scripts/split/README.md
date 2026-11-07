# Split scripts

These scripts produced this repository from the luma.gl `map-contributors` working tree. The source
snapshot is the local luma.gl branch `backup/map-contributors-pre-split-2026-11-07` (commit
`e0ed33791`), compared against the published `v10.0.0-alpha.10` tag (`52679fe4f`) and the master
merge base `704183953`.

| Script | Purpose |
| --- | --- |
| `closure.mjs` | Starts from the contributor code and follows relative and `@luma.gl/graph/*` imports through the graph source tree. Classifies each luma graph file it reaches as same as alpha.10, modified, or new. |
| `scaffold.mjs` | Copies the closure plus `graph-lenses` and `kepler` and rewrites `@luma.gl/graph*` and `@luma.gl/kepler` specifiers. |
| `manifests.mjs` | Generates package manifests and tsconfigs. Keeps only the graph exports whose source was forked. |
| `port-tests.mjs` | Copies tests, then prunes test files whose imports cannot resolve in the fork. |
| `fix-tests.py` | Fixes up test paths, package names and globs that the specifier rewrite cannot see. |
| `closure.json` | The closure used for the current fork. It covers 1,414 files. |

Reproduce the closure:

```bash
# in the luma.gl checkout
EXTRA_DIRS=geospatial,gpu-pointcloud,gpu-project FOLLOW_ALL=1 FOLLOW_SUBPATHS=1 OUT=closure.json \
  node closure.mjs "$PWD" v10.0.0-alpha.10 704183953 \
  modules/graph-lenses/src modules/kepler/src modules/graph-lenses/test modules/kepler/test
```

## Why the fork is a full closure

A partial fork does not work. The contributor code reaches 148 graph files that changed after
alpha.10 and 12 that were added after it. Most of them are in `gpu-core`, including
`GPUCommandGraph`. Unchanged graph files loaded from npm would use npm's `GPUCommandGraph` and
`GPUVector` classes while the forked files used their own. Every graph file the contributor code
reaches is therefore forked, including 213 that are unchanged from alpha.10.

## Manual changes after the scripts

- `modules/graph/src/utils/luma-core-compat.ts` (new) shims APIs that were missing from published
  core: `setBindings` options, `Device.waitForSubmittedWork()` and `OutOfMemoryError`.
- `execution/compute/kernel.ts` and `arrow/vectors/arrow-{matrix,temporal}-gpu-vector.ts` use those
  shims.
- The kepler `dataset-registry.ts` uses `isOutOfMemoryError`. Its two `DynamicBuffer` imports now come
  from `@luma.gl/core/dynamic` instead of `@luma.gl/engine`.
- The lens-example public-entry check in `lenses/src/computation-lenses/lens-example.ts` accepts
  `@luma-contributors/*`.
- `biome format --write` was run after the specifier rewrite.
