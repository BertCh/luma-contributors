# luma.gl contributors

GPU analysis operations, computation lenses and kepler.gl adapters built on
[luma.gl](https://luma.gl). They run against published luma.gl `10.0.0-alpha.10`.

| Package | Contents |
| --- | --- |
| `@luma-contributors/graph` | Spatial analysis, network, terrain, dataframe, raster and point-cloud operations, `computation-record`, and the GPU graph execution layer they run on |
| `@luma-contributors/lenses` | Computation lenses, computation views and scenario packs that document the graph operations |
| `@luma-contributors/kepler` | Explicit GPU compute and lifecycle adapters for kepler.gl |

Examples are in `examples/deck/spatial-analysis-showcase` and
`examples/deck/spatial-analysis-explorer`.

## Relationship to luma.gl

This repository is independent of luma.gl's source tree. It installs `@luma.gl/core`,
`@luma.gl/shaders` and, for examples and tests, `@luma.gl/engine` from npm.

`@luma-contributors/graph` carries its own fork of the `@luma.gl/graph` execution layer, including
`gpu-core`, `gpu-data`, `gpu-tables` and `arrow`. The analysis operations were developed against
a newer graph than the published alpha. Always import graph APIs from `@luma-contributors/graph`.
Do not mix them with `@luma.gl/graph` objects, and do not pass forked tables into
`@luma.gl/engine`'s `GPUTableModel`.

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
