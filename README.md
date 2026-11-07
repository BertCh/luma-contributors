# luma.gl contributors

GPU analysis operations, computation lenses and kepler.gl adapters built on
[luma.gl](https://luma.gl). They run against published luma.gl `10.0.0-alpha.10`.

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

`scripts/split/README.md` describes how this repository was produced and how to re-sync it.

## Development

```bash
nvm use
yarn install
yarn typecheck             # needs NODE_OPTIONS=--max-old-space-size=8192
yarn build                 # ESM + CommonJS + declarations
yarn verify-package-exports
yarn lint
yarn test-node
yarn test-headless         # Playwright Chromium with WebGPU
```

## Datasets and licensing

Code is MIT licensed (see `LICENSE`). Example and lens datasets have third-party licences recorded
in `THIRD-PARTY-DATA.md`. Some are non-commercial (CC BY-NC) or share-alike (ODbL), so
`@luma-contributors/lenses` and `@luma-contributors/kepler` stay `private` until that is resolved.
