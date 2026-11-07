# AGENTS.md

## Setup commands
- Select Node version: `nvm use`
- Install deps: `yarn install`
- Typecheck sources: `yarn typecheck`
- Build packages (ESM + CommonJS + types): `yarn build`
- Verify every package entry point loads: `yarn verify-package-exports` (after `yarn build`)
- Lint and formatting: `yarn lint`, fix with `yarn lint-fix`
- Tests: `yarn test-node`, `yarn test-headless`, or `yarn test` for both
- Examples: `cd examples/deck/spatial-analysis-showcase && yarn start`

Large typechecks need `NODE_OPTIONS=--max-old-space-size=8192` or more.

## Relationship to luma.gl (upstream-first)
- This repository contains only geospatial and analysis code. Everything else comes from **published**
  `@luma.gl/core`, `@luma.gl/shaders`, `@luma.gl/graph` (and, for examples and tests only,
  `@luma.gl/engine`) at `10.0.0-alpha.10`, through their public package entry points.
- Never import `@luma.gl/*/_internal/*`, deep `dist` paths, or copied luma.gl source. Never fork,
  re-implement or re-export luma.gl APIs. See `dev-docs/upstream-first-plan.md`.
- When luma.gl lacks something, in order: use the public API even if more verbose; adapt to
  luma.gl's design; drop the feature; last resort, a small helper named for the analysis domain.
- luma.gl improvements go upstream as ordinary luma.gl pull requests. Do not carry them here.
- Source lives only in analysis-owned directories of `modules/analysis/src`: `analysis-core`,
  `computation-record`, `gpu-spatial-analysis`, `gpu-network`, `gpu-terrain`, `testing`,
  `dataframe-analysis`, `raster-analysis`, `pointcloud-analysis`, `geospatial-analysis`.

## Plans and docs
- Plans go in `dev-docs/roadmap.md`; see `dev-docs/README.md` for the dev-docs index.

## Packages
- `@luma-contributors/analysis`: GPU geospatial and analysis operations.
- `@luma-contributors/lenses`: computation lenses, views and scenario packs (private until the
  dataset licensing in `THIRD-PARTY-DATA.md` is resolved).
- `@luma-contributors/kepler`: GPU compute and lifecycle adapters for kepler.gl.
- Dependency direction is `kepler → lenses → analysis → @luma.gl/graph, @luma.gl/core, @luma.gl/shaders`.

## Before committing
- Always run `yarn lint-fix` after making changes.
- Run `yarn typecheck`, `yarn build`, `yarn verify-package-exports` and `yarn test` before merging.

## Code style
- TypeScript strict mode, semicolons, single quotes.
- Every first-party source file with `SPDX-License-Identifier`, including tests and examples, must also
  declare each copyright holder with `SPDX-FileCopyrightText`.
- Preserve existing license expressions and upstream attribution; do not infer MIT licensing or
  ownership for generated, vendored or third-party files.
- Never abbreviate variables. camelCase for variables and functions, PascalCase for types,
  CAPITAL_CASE for constants. Prefer verbNoun function names.
- Keep shipped runtime validation minimal. Put explanatory assertion messages in comments.
- Import individual symbols from `'apache-arrow'` in `modules/*/src`; prefer
  `import * as arrow from 'apache-arrow'` in tests, examples and docs.

## Datasets
- Example and lens datasets carry third-party licences recorded in `THIRD-PARTY-DATA.md`. Some are
  non-commercial (CC BY-NC) or share-alike (ODbL). Do not publish packages that bundle them.
