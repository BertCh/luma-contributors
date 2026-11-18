# luma.gl contributors

GPU analysis operations, computation lenses and kepler.gl adapters built on
[luma.gl](https://luma.gl). Tested with published luma.gl `10.0.0-alpha.14`.

## Run your first computation

Start with the [CPU variogram fit](docs/api/operation-forms-and-cpu-twins.md#variogram-fitting-without-a-graph):
20 generated bins, one public function, and an expected model; no dataset or GPU needed.
Then run [point density from device to readback](docs/api/operation-forms-and-cpu-twins.md#point-density-from-device-to-readback)
for a complete browser GPU computation, including allocation, submission and cleanup.
The [API index](docs/api/README.md) links the domain guides.

| Package | Contents |
| --- | --- |
| `@luma-contributors/analysis` | Spatial analysis, network, terrain, dataframe, raster and point-cloud operations |
| `@luma-contributors/lenses` | Computation lenses, computation views and scenario packs that document the operations |
| `@luma-contributors/kepler` | Explicit GPU compute and lifecycle adapters for kepler.gl |

Examples are in `examples/deck/spatial-analysis-showcase` and
`examples/deck/spatial-analysis-explorer`.

## Relationship to luma.gl

This repository holds only geospatial and analysis code. It builds on published `@luma.gl/core`,
`@luma.gl/shaders` and `@luma.gl/graph` through their public entry points, and carries no copies
of luma.gl code. See `dev-docs/upstream-first-plan.md`.

`apache-arrow >=17.0.0` is a required peer of analysis, including for native imports of the
authoring and network entry points. The accepted luma.gl peer range is `~10.0.0-alpha.14`,
which also permits stable 10.0.x versions. The tested version above and the accepted range are
separate: see the [alpha package contract](dev-docs/release-package.md) for installation and
packed-consumer verification.

Plans go in `dev-docs/roadmap.md`; see `dev-docs/README.md` for the dev-docs index.

## Repository map

| Path | Contents |
| --- | --- |
| `modules/analysis`, `modules/lenses`, `modules/kepler` | The three packages (see each package README) |
| `docs/api` | API documentation by domain; `docs/table-of-contents.json` orders it |
| `docs/lens-records` | Generated lens records used by the docs and lens tests |
| `dev-docs` | Roadmap, contributor contract, coverage matrix, upstream workarounds and proposals, research |
| `examples/deck` | Spatial analysis showcase and explorer (kepler-compute is parked) |
| `website` | Static docs site (TanStack Start, prerendered); `yarn website-dev`, `yarn website-build` |
| `scripts` | Build, export verification, upstream-first guard and the original split scripts |
| `CHANGELOG.md` | Release history |

## Development

```bash
nvm use
yarn install
yarn typecheck             # needs NODE_OPTIONS=--max-old-space-size=8192
yarn typecheck-tests       # staged checks; see runner and exclusion policy below
yarn build                 # ESM + CommonJS + declarations
yarn verify-package-exports
yarn lint
yarn test-node
yarn test-headless         # Playwright Chromium with WebGPU
```

`yarn typecheck-tests` runs ordinary tests sequentially by package and family, followed by
example-backed and website-backed projects, including eligible Kepler example specs.
The [runner](scripts/typecheck-tests.mjs) preserves all caller `NODE_OPTIONS`, including a lower
supplied heap limit, and adds an 8,192 MiB heap only when no heap limit is supplied. Each project
has its own incremental cache; `.typecheck/test-family-stages.json` records the ordinary test roots
and configurations. Imported source dependencies remain part of their test programs.
The [exclusion manifest](test/kepler-compute-example-specs.json) records `excludedFromTypecheck`;
`excludedFromRuns` separately controls runtime test exclusions. Consult the landed runner and
manifest rather than assuming those sets or memory policies are the same.

## Datasets and licensing

Analysis ships under `MIT AND Apache-2.0 AND BSD-3-Clause`; individual files retain their SPDX
expressions. See its [licence and notice inventory](modules/analysis/THIRD-PARTY-NOTICES.md)
and the MIT [LICENSE](LICENSE) for MIT components. Example and lens datasets have third-party licences recorded
in `THIRD-PARTY-DATA.md`. Some are non-commercial (CC BY-NC) or share-alike (ODbL), so
`@luma-contributors/lenses` and `@luma-contributors/kepler` stay `private` until that is resolved.
