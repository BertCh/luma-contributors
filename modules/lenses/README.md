# @luma.gl/graph-lenses

Private workspace package (not published). It holds the documentation and visualization framework
for `@luma.gl/graph` computation records:

- `computation-lenses`: lens records, scenario runners, and recorded scenario data.
- `computation-views`: the view registry, encodings, and DOM hosts that render lens records.
- `gpu-*/computation-pack`: per-domain view packs that register with `computation-views`.

The package source imports `@luma.gl/graph` through its public subpaths only, and
`scripts/verify-package-layering.mjs` enforces that `@luma.gl/graph`, core and shaders never import
this package. Some tests still import graph test helpers, oracles and a few non-exported modules
(`@luma-contributors/graph/test/*`, `@luma-contributors/graph/utils/*`, `@luma-contributors/graph/gpu-dataframe/*`) through the
repository `tsconfig.json` aliases; these resolve only inside this repository. The runtime record types (`@luma-contributors/graph/computation-record`) stay in
`@luma.gl/graph` because graph execution emits them.

## Conventions

- Scenario modules register themselves on import (`registerLensScenario`, `registerLensScenarios`),
  so `package.json` lists them in `sideEffects`. Import `computation-lenses/scenarios/index` for
  the full registry.
- This package is private and exempt from the shipped-runtime validation guideline: registry
  errors use descriptive messages because they only surface in docs and test tooling.
- `"license"` points at `src/computation-lenses/data/README.md` because recorded datasets carry
  their own licences (for example CC BY-NC iNaturalist observations). Review it before removing
  `"private"`.
- `yarn tsc -p modules/lenses/tsconfig.test.json` type-checks sources and tests together.
  It also reports errors in the website and example code that tests import.
