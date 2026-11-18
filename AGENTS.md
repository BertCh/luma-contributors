# AGENTS.md

## Setup commands
- Select Node version: `nvm use`
- Install deps: `yarn install`
- Typecheck sources: `yarn typecheck`
- Typecheck tests: `yarn typecheck-tests` ([runner](scripts/typecheck-tests.mjs) preserves caller `NODE_OPTIONS`; defaults to an 8192 MiB heap when no heap limit is supplied; [typecheck exclusions](test/kepler-compute-example-specs.json) are separate from runtime exclusions)
- Build packages (ESM + CommonJS + types): `yarn build`
- Stage a clean release build: `yarn build:release` (see promotion rules below)
- Verify every package entry point loads: `yarn verify-package-exports` (after `yarn build`)
- Verify an isolated packed consumer: `yarn check-packed-consumer` (after `yarn build`)
- Run the sequential local gate: `yarn verify` (see resource limits below)
- Lint and formatting: `yarn lint`, fix with `yarn lint-fix`
- Upstream-first rules: `yarn check-upstream-first` (fails on a luma.gl import that is not a published public entry point and on source outside the analysis-owned directories)
- Contributor contract departures: `yarn check-contributor-contract` (fails on findings not in `scripts/contributor-contract-baseline.json`; `--all` lists everything, `--update-baseline` rewrites it)
- Tests: `yarn test-node`, `yarn test-headless`, or `yarn test` for both
- Benchmarks: `yarn test-benchmarks` (separate from ordinary correctness tests)
- Examples: `cd examples/deck/spatial-analysis-showcase && yarn start`

Large source typechecks normally use `NODE_OPTIONS=--max-old-space-size=8192`. The test-typecheck
runner prints the effective options for each sequential family project and the example/website
projects, preserves all caller options, and keeps a separate incremental cache for each project.
Ordinary test roots are grouped by package and the first two directory levels below its test root.
Analysis runs before lenses, then kepler; analysis-core runs first inside analysis. Imported source
dependencies remain part of each TypeScript program. `yarn typecheck` checks all source roots
separately; test projects no longer add every source file as an unrelated root.
`.typecheck/test-family-stages.json` records exact roots/configurations; caches/configurations live
in `.typecheck/test-families/`, with separate example/website caches under `.typecheck/`.
Family projects resolve the root configuration through TypeScript, including comments and inherited
ambient types, and add `vite/client`, matching the Vite runtime
used by Vitest for `import.meta.glob`, `import.meta.env` and raw asset imports. Generate configs
without launching a compiler with `node scripts/typecheck-tests.mjs --generate-only`.

## Local verification and resource limits
- `yarn verify` runs lint, upstream-first, contributor-contract, source typecheck, test typecheck,
  build, export verification, packed-consumer verification, Node tests, then headless tests. It
  stops at the first failure and prints timings and sampled peak RSS per command. Only a full run
  qualifies the complete gate; resumed and single-step summaries are labelled partial.
- `yarn verify --from typecheck-tests` resumes at that step; `yarn verify --only build` runs one
  step. `yarn verify --help` lists the exact step names.
- Heavy steps require at least **8 GiB free disk** on the checkout filesystem, checked before
  each step and headless chunk. `--minimum-free-disk-gib 12` raises the floor; it cannot lower it.
  The packed-consumer temporary filesystem must also have enough room for its tarball and extraction.
- The operational peak memory ceiling is **12 GiB RSS for the verify process and its descendants**,
  sampled with `ps` once per second; exceeding it kills the active command's process group and fails
  the gate. Brief peaks between samples and OS/driver GPU memory are outside this accounting. This
  is an enforced sampled limit, not a measured claim about a full run on this machine.
  Verify defaults to an **8192 MiB V8 heap** per Node process and rejects a larger explicit heap.
  Hyphen/underscore spellings and signed absolute values are recognized; percentage heap limits
  are rejected by verify because they override absolute limits. The typecheck runner preserves
  caller heap options using the same parser. A lower caller heap is preserved.
  RSS includes native allocations and workers, so it can exceed
  the V8 heap. Run one verify at a time with no other heavyweight checkout commands.
  Standalone `yarn build` preserves the caller environment, including higher `NODE_OPTIONS` heap
  settings; the 8192 MiB ceiling applies when verify or release supplies its validated environment.
- The isolated packed runtime clears module-loading options and NODE_PATH, but carries the
  validated absolute heap limit as a numeric Node argument (default 8192 MiB, maximum 8192 MiB).
  Caller command-line heap options take precedence over NODE_OPTIONS; loader options are not
  forwarded to the worker.
  Verify and release wrappers resolve this precedence at entry and forward the effective numeric
  limit to every child, including the compiler and packed checker.
- Node tests default to **one worker** in `vitest.config.ts`; verify also passes `--maxWorkers=1`.
  Headless tests run sequentially in directory chunks with a fresh browser command per chunk.
  `.typecheck/verify/headless-files.json` records the eligible files on disk; each JSON test report
  must contain exactly its chunk's files and a passing test in each file. Missing files, empty
  runs, wholly skipped files, missing devices and changed file membership cannot certify the gate.
- Optional jobs are excluded from both Vitest and the disk inventory unless enabled: Delaunay and
  overlay A/B suites require `A5_AB=1`, the Vecchia study requires a nonempty
  `VITE_VECCHIA_FLOAT32_STUDY`, and lens capture requires `VITE_CAPTURE_LENS_RECORDS=1`.
  Invoke the desired headless file with its flag to opt in. Lens capture writes records and stays
  outside the default gate. Only the A5_AB flag is forwarded into the browser process shim.
- Ordinary browser/headless runs exclude every `-bench.spec.ts/js` and `-benchmark.spec.ts/js` file.
  `yarn test-benchmarks` selects those suffixes, prints the complete inventory and enables their
  boolean `VITE_*BENCHMARK`/`VITE_*PROFILE` entry gates. Size and alternate-route settings remain
  caller options; verbose output names any skipped cases. Benchmarks are outside `yarn verify`.
- `yarn build` preserves composite TypeScript state and outputs, prunes outputs for removed sources,
  and transforms at most **two CommonJS modules concurrently**, skipping unchanged modules. It is
  an incremental developer build; a partial failed build is not a release artifact. CommonJS files
  are written to sibling temporary files and renamed only after successful writes. Missing native
  `.cts`/`.cjs` compiler outputs trigger repair just like missing ESM or declaration outputs.
- Successful package builds atomically write `dist/.build-record.json` with source/config/package
  input hashes and emitted-content hashes. Export verification uses this record so unchanged
  declarations may retain their compiler timestamps; changed inputs, missing outputs and corrupt
  output bytes still fail. Builds repair recorded output corruption before replacing the record.
  Changed input hashes whose mtimes could bypass the compiler invalidate the affected project
  cache. CommonJS reuse compares recorded JavaScript content hashes as well as transformer freshness.
  Timestamp comparisons use the compiler's millisecond Date precision. Establishing a first record
  over legacy outputs rebuilds the compiler and CommonJS outputs instead of trusting the old cache.
  Input hashes are captured before compilation. After all three CommonJS passes, every package's
  input snapshot is revalidated before any record is published. An edit during the build refuses
  publication and leaves prior records available for retry.
  Legacy output without a record retains the older timestamp-based freshness check.
- `yarn build:release` snapshots package sources, metadata, licences and compiler configuration
  under `.typecheck/release-builds/`, excluding prior outputs, compiler caches and test fixtures.
  It compiles the snapshot with the installed compiler, creates bounded CommonJS outputs, loads
  all snapshot package entry points, and checks the packed analysis consumer. Local package links
  resolve to snapshot packages; installed third-party dependencies resolve from the parent checkout.
  Existing developer outputs/caches stay in place. The checked tarball and snapshot are retained,
  with their paths in `.typecheck/release-build-latest.json`; Node/GPU qualification is still a
  separate full verify gate. Do not publish lenses while its dataset licensing is unresolved.
- `yarn build:release --promote` installs the three output trees only after those build/package
  gates pass and both original and staged source/config/package bytes still match the snapshot.
  Inputs are also checked immediately after copying. Previous output
  trees and developer compiler caches are retained under the stage's `previous/`; active developer
  caches are invalidated so a later source revert cannot reuse stale compiler state. Promotion
  failures restore both outputs and caches. Release records are prepared before installation,
  with a digest of each output tree retained from export verification. Installed output trees
  must match those digests inside the rollback window, before release records are committed;
  packed-gate mutations therefore cannot be promoted. Default staging checks the same digests.
  and the latest record is replaced atomically while rollback is still available. A record
  publication failure preserves the prior latest record and restores prior outputs/caches.
  Abrupt process/OS termination
  during promotion can require recovery from `promotion.json` and `previous/`: inspect the actual
  paths in that journal before moving anything. Multi-package replacement is not one atomic rename.
  Failed stages remain for diagnosis; only successful runs update the latest-build record. Run one
  release/developer build at a time, and remove retained stages when their artifacts are no longer
  needed. A stale `.typecheck/release-build.lock` requires checking `owner.json` and its process
  before removal. Clean builds need room for the source snapshot, new outputs and previous outputs;
  use the same 8192 MiB heap setting as other large compiler runs.

## Relationship to luma.gl (upstream-first)
- This repository contains only geospatial and analysis code. Everything else comes from **published**
  `@luma.gl/core`, `@luma.gl/shaders`, `@luma.gl/graph` (and, for examples and tests only,
  `@luma.gl/engine`) through their public package entry points. The integration target and accepted
  peer ranges are recorded separately in [the alpha package contract](dev-docs/release-package.md).
- Never import a `@luma.gl/*` path with an `_internal` segment (even one listed in the package's
  exports), deep `dist` paths, or copied luma.gl source. Never fork,
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
- Run the full `yarn verify` before merging; the local machine is the gate.

## Code style
- TypeScript strict mode, semicolons, single quotes.
- Every first-party source file with `SPDX-License-Identifier`, including tests and examples, must also
  declare each copyright holder with `SPDX-FileCopyrightText`.
- Preserve existing license expressions and upstream attribution; do not infer MIT licensing or
  ownership for generated, vendored or third-party files.
- Never abbreviate variables. camelCase for variables and functions, PascalCase for types,
  CAPITAL_CASE for constants. Prefer verbNoun function names.
- Contributors follow `dev-docs/contributor-contract.md`. Validate props once, at construction, with the
  named-limit helpers (`validateCapacity`, `validateMaximumCount`, `validateRequirement`, ...); messages
  start with the contributor ID and name the prop and the limit. Do not use bare `assert`. Do not
  validate per frame or per row on the CPU.
- Import individual symbols from `'apache-arrow'` in `modules/*/src`; prefer
  `import * as arrow from 'apache-arrow'` in tests, examples and docs.

## Datasets
- Example and lens datasets carry third-party licences recorded in `THIRD-PARTY-DATA.md`. Some are
  non-commercial (CC BY-NC) or share-alike (ODbL). Do not publish packages that bundle them.
