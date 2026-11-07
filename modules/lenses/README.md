# @luma-contributors/lenses

Private workspace package (not published). It holds the documentation and visualization framework
for `@luma.gl/graph` computation records:

- `computation-lenses`: lens records, scenario runners, and recorded scenario data.
- `computation-views`: the view registry, encodings, and DOM hosts that render lens records.
- `gpu-*/computation-pack`: per-domain view packs that register with `computation-views`.

## Dependencies

The package depends on `@luma-contributors/analysis` (the computation-record types live in
`@luma-contributors/analysis/computation-record`) and on the published `@luma.gl/graph` public
subpaths only. It does not reach into non-exported luma.gl modules. Run `yarn check-upstream-first`
and `yarn verify-package-exports` to verify these boundaries.

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
