# @luma-contributors/lenses

This package stays private until its dataset licensing is resolved. The
[dataset licence inventory](https://github.com/visgl/luma-contributors/blob/master/modules/lenses/src/computation-lenses/data/README.md)
lists the terms and attribution for each extract. The
[third-party data policy](https://github.com/visgl/luma-contributors/blob/master/THIRD-PARTY-DATA.md)
records the publication restrictions.

## Public reader, host and author surfaces

The entry barrels retain consumer-used names and the surface documented before this tranche.
The five pack entries additionally expose the shared `createViewPack()` registration convention.
The sorted contract is pinned in
`test/computation-docs/public-surface.json`; adding an export requires reviewing that list. It pins the real export-map entries and separates runtime values from type-only names.
Type-only names use `export type` or inline `type` exports so isolated browser transforms
never request erased TypeScript declarations as JavaScript values. The public-surface Node
spec also links independently transformed modules to check this browser loading boundary.
See the [authoring guide](../../docs/api/computation-views.md) for records, views and scenarios.
Implementation helpers are imported internally from their defining files.
The example checker uses a concrete local entry inventory, verified against both real package
manifests and source targets by `public-entry-inventory.node.spec.ts`. Installed public
`@luma.gl/core`, `@luma.gl/engine` and `@luma.gl/graph` entries use the same concrete check; invented
deep paths and `_internal` entries are rejected. It distinguishes
ESM imports from CommonJS requires and rejects missing wildcard targets. Loading the inventory
does not read a sibling analysis manifest or depend on the package installation layout.

Replay captures retain their count readback: unavailable rows publish no distribution, and
valid rows publish only the count-reported prefix. Row-domain annotations live on captures.


- `.`: `DecodedLensFile`, `RecordedLensFile`, `captureLensScenarioRecord`, `fromRecordedLensFile`, `registerLensScenario`, `registerLensScenarios`, `toRecordedLensFile`.

- `./computation-lenses`: `DecodedLensFile`, `LensScenarioDescriptor`, `LensScenarioResult`, `LensUserInputRoleError`, `RecordedLensFile`, `applyLensUserInput`, `captureLensScenarioRecord`, `fromRecordedLensFile`, `getLensScenario`, `getLensUserInputDatasetId`, `listLensScenarios`, `listLensUserInputRoles`, `registerLensScenario`, `registerLensScenarios`, `runLensScenarioWithUserInput`, `toRecordedLensFile`.

- `./computation-views`: `LadderCoordinator`, `LensController`, `LensControllerOptions`, `buildDecodedLensTwin`, `buildLensTwin`, `createDefaultViewRegistry`, `defineLadderLensElement`, `mountLadderLens`.

- `./computation-lenses/registry`: `fromRecordedLensFile`, `registerLensScenario`, `toRecordedLensFile`.

- `./gpu-dataframe/computation-pack/index`: `createViewPack()` returns the dataframe registration.

- `./gpu-network/computation-pack/index`: `createViewPack()` returns the network registration.

- `./gpu-raster/computation-pack/index`: `createViewPack()` returns the raster registration.

- `./gpu-spatial-analysis/computation-pack/index`: `createViewPack()` returns the spatial registration.

- `./gpu-terrain/computation-pack/index`: `createViewPack()` returns the terrain registration.

Importing any pack entry still registers it with the default registry. Calling `createViewPack()`
returns its registration for a caller-owned registry; it does not repeat default registration.
Operator tables and author implementation helpers remain internal.

Private workspace package (not published). It holds the documentation and visualization framework
for `@luma.gl/graph` computation records:

- `computation-lenses`: lens records, scenario runners, and recorded scenario data.
- `computation-views`: the view registry, encodings, and DOM hosts that render lens records.
- `gpu-*/computation-pack`: per-domain view packs that register with `computation-views`.

## Dependencies

The package depends on `@luma-contributors/analysis` (the computation-record types live in
`@luma-contributors/analysis/computation-record`), `@math.gl/core` and `@math.gl/geospatial`.
Its luma.gl peers are `@luma.gl/core`, `@luma.gl/graph` and `@luma.gl/shaders`, accessed through
published public entry points. `apache-arrow` is also a required peer. The [alpha package contract](../../dev-docs/release-package.md#tested-dependencies-and-accepted-peers)
records the alpha.16 integration target (`10.0.0-alpha.16`) separately from the accepted
peer range (`~10.0.0-alpha.16`) and completed checks. Run
`yarn check-upstream-first` and `yarn verify-package-exports` to verify these boundaries.

## Conventions

New computation records have deterministic comparison/permalink `runId` and unique `executionId`.
Every capture and metrics sidecar copies both from its record. Scenario harnesses and
`readLensRunMetrics` preserve them, including when timestamps are unavailable. Never assign a new
execution id to an existing sidecar to make it match. When assembling a record around already
captured sidecars, supply their original execution id explicitly. Serialization, loading and pure
view contexts reject mismatched or missing execution identity; legacy `/1` pairs with both execution
ids absent remain supported with structural identity only. Reproducible fixtures supply an explicit
`executionId` and `createdAt`. Supply scalar `parameters`/choice values and advance
`implementationRevision` after correctness fixes; changing buffers also requires a dataset version.
Parameter maps and choice values are finite JSON scalars (strings, booleans or finite numbers),
validated before hashing and on load. S2 retains the full choice map even when a display template
omits a setting. Record assembly copies choices, alternatives and adapter fields before callers
reuse them. Website sidecars always require matching run ids; only execution ids may both be
absent for legacy pairs. Native Node generation imports the extractor's `.ts` dependencies directly.

Resource views preserve `bytesKind` (`logical`, `lower-bound`, `estimated`) and unknown ownership.
Memory aliasing excludes unknown ownership from imported and transient totals and reports incomplete
allocation evidence as unknown. A legacy `imported` resource without size precision is displayed as
unknown ownership because older unannotated descriptions asserted caller ownership incorrectly.
Website totals, lifetimes and dependency labels show `≥` for bounds, `~` for estimates, and
“precision unknown” when the legacy record/view model does not establish size precision. Equal sizes
with different ownership or precision remain separate rows. See [the record API](../../docs/api/computation-record.md).
Opaque sizes are `unavailable`, rather than a zero-byte estimate. Physical allocation totals are
exact only when allocation sizes/identities are known; otherwise known shared members give a
lower bound and unknown ownership/identity makes the total unspecified. Lower-bound labels use
exact byte counts to avoid overstating their evidence through rounding.

An omitted resource inventory is unavailable, distinct from an explicitly empty inventory's exact
zero. Website summaries retain that distinction for both counts and byte totals. Inspector entity
pins resolve their datum key space and only index captures in that domain, including `first:<key>`
windows; station and whole-run diagnostics stay empty when a cell is pinned.

S3 declared `metadata` enters presentation choices before the comparison ID is computed.
Readback-derived `observations` (conservation sums, island counts) live in graph attributes, not
identity-bearing choices. Spatial guards read those observations as well as declared choices.
Every declared choice template needs an explicit value in its scenario's choices map. Optional
features record `none` when omitted; operation-specific settings that the selected operation does
not use record `not applicable`, rather than an invented numeric setting. The recording boundary
reports the scenario ID and `choices.<name>` for a missing value before adding observation work.
The decoded lens-envelope reader applies the core record validator before accepting sidecars,
including finite scalar identity values; unmodified legacy recordings remain supported.

The kernel inspector attributes explicit override constants only from the selected node. A shared
shader source can retain another pipeline's overrides; absent node evidence leaves constants
unspecified. The building tessellation reference example calls
`addBuildingTessellationReferenceRecipe` directly, so extra `candidateRadius` fields on a supplied
variable cannot change its polygon-overlay output layout. The separate local tessellation example calls
`addLocalBuildingTessellationRecipe` and demonstrates the certified sample-cell-union layout.
The local scenario explicitly selects `output-cells` for its result headline; sampling and loss
certificates remain captured for inspection. An explicitly supplied `presentation: {}` selects
the last output and survives serialization; omitting presentation permits a scenario default.

- Scenario modules register themselves on import (`registerLensScenario`, `registerLensScenarios`),
  so `package.json` lists them in `sideEffects`. Import `computation-lenses/scenarios/index` for
  the full registry, including the core array-result view registration. Both source and built
  registration modules must survive consumer tree shaking.
- A named family scenario can demonstrate variants explicitly listed in the coverage spec's
  `familyScenarios` records. Each record names the scenario and explains the relationship.
  Pending captures require an entry in `test/computation-docs/pending-lens-records.ts`; existing
  twins always require their source captures. See the [authoring guide](../../docs/api/computation-views.md).
- Embedded datasets load lazily through literal JSON imports with native import attributes, shared
  by Node and Vite consumers. Attributes are supplied only in native Node: Vite serves browser JSON
  imports as JavaScript modules, which must load without `type: 'json'`, including in workers.
  Both branches retain literal import expressions and native options so esbuild can discover all
  assets when bundling a browser consumer; a variable options argument prevents that discovery.
  The CommonJS build preserves these attributes for native Node JSON
  loading. The package regression spec emits a small fixture with the production transform and
  loads all datasets in both ESM and CommonJS. After `yarn build`, run the lens-local qualification
  command `node modules/lenses/test/computation-docs/qualify-built-lenses.mjs`. It requires current
  build records for analysis and lenses, enables `LENS_VERIFY_BUILT=1`, and requires the exact
  case “checks emitted relative links and public role loading after a build” to pass. Missing or
  stale or unrecorded extra output, a missing/skipped case, or a failed ESM/CommonJS role load
  fails qualification. The emitted files, expected outputs and build records must agree exactly.
  This private-lens qualification remains separate from the analysis release gate and does not
  authorize publication of lenses or its datasets.
  The client-transform regression and headless dataset-loader spec separately check browser loading;
  Vite's Node/SSR module runner alone does not establish browser compatibility.
  The filename inventory in the data README is checked against the
  actual files. The private package still includes licensed extracts; publication requires the
  separate dataset and derived-artifact review in `THIRD-PARTY-DATA.md`.
- This package is private and exempt from the shipped-runtime validation guideline: registry
  errors use descriptive messages because they only surface in docs and test tooling.
- `yarn tsc -p modules/lenses/tsconfig.test.json` type-checks sources and tests together.
  It also reports errors in the website and example code that tests import.
