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

## Relationship to luma.gl
- This repository depends on **published** `@luma.gl/core`, `@luma.gl/shaders` and (examples and
  tests only) `@luma.gl/engine` at `10.0.0-alpha.10`. Never use workspace links or source aliases
  into a luma.gl checkout.
- `@luma-contributors/graph` carries its own fork of the `@luma.gl/graph` execution layer
  (`gpu-core`, `gpu-data`, `gpu-tables`, `arrow`, `gpu-simple-graph`, ...). Packages here must import
  graph code from `@luma-contributors/graph`, never from `@luma.gl/graph`. Mixing the two makes
  `GPUCommandGraph`, `GPUVector` and `GPUTable` instances incompatible.
- Do not pass forked graph objects into `@luma.gl/engine` APIs that consume graph types (for example
  `GPUTableModel`). Published engine binds through its own `@luma.gl/graph` copy.
- Core APIs that are not in every supported luma.gl release go through
  `modules/graph/src/utils/luma-core-compat.ts`, which prefers the native API when present.
- `scripts/split/` holds the scripts that produced this repository from the luma.gl
  `map-contributors` branch. See `scripts/split/README.md` before re-syncing.

## Packages
- `@luma-contributors/graph`: GPU analysis operations and the forked graph execution layer.
- `@luma-contributors/lenses`: computation lenses, views and scenario packs (private until the
  dataset licensing in `THIRD-PARTY-DATA.md` is resolved).
- `@luma-contributors/kepler`: GPU compute and lifecycle adapters for kepler.gl.
- Dependency direction is `kepler → lenses → graph → @luma.gl/core, @luma.gl/shaders`.

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
