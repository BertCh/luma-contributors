# Script verification gates

These checks fail when required work was not performed. The shared-machine review permits named
Node specs and scoped typechecks; clean builds, wide gates and GPU execution remain the lead's work.

## Package artifacts and release qualification

After a clean build, `node scripts/verify-package-exports.mjs` reads each package's TypeScript project
inputs without typechecking. It checks emitted JavaScript/CommonJS/declarations and copied JSON,
including enabled JavaScript loaders and each source asset when other wildcard targets still exist.
Missing counterparts fail. Successful builds write `dist/.build-record.json`, covering source,
configuration and package input hashes plus emitted-content hashes. The gate rejects changed inputs
or output bytes while accepting unchanged declarations whose compiler timestamps are older than a
source edit. Legacy builds without a record retain the source/package/configuration timestamp check.
Backdated edits are checked by content: changed input hashes at or before a compiler cache's mtime
invalidate that cache before compilation. CommonJS reuse also checks the prior emitted JavaScript
hash so an unchanged timestamp cannot certify an old implementation.
The comparison uses the compiler's millisecond Date values. A build over unrecorded legacy output
rebuilds before publishing its first record. Pre-compilation input hashes must match the fresh
input inventory after all three CommonJS passes and before any record publication. An edit to an
earlier package during a later package's processing fails the build without replacing prior records.
Conditional
exports compare sorted ESM/CommonJS names by importing/requiring the public package specifier,
including equal-sized sets with different names. Node-invalid export targets fail. Wildcards expand
their actual built targets; source-input freshness checks independently detect missing members.
Empty patterns fail. String asset exports are checked for existing targets. This is a workspace
artifact check, with workspace dependencies available.

Then run `node scripts/check-packed-consumer.mjs --keep-artifact` as described in
[the release contract](./release-package.md). It packs once offline and checks the extracted bytes in
an external consumer. Retain/promote that checked tarball after the other release gates pass. Its
declaration check verifies targets, not a full NodeNext semantic typecheck; a clean build and GPU
qualification are separate requirements. An old internally consistent tarball is not proof that the
current source was built. Lenses remains private because of dataset licensing.

The packed runtime worker clears module-loading environment options while receiving the validated
absolute heap as a Node argument. Its default and maximum are 8192 MiB; a lower caller setting is
preserved, including the caller's command-line precedence over NODE_OPTIONS.
Verify and release wrappers forward that effective limit through their compiler and checker children.
Standalone builds preserve the caller environment, including heap settings above 8192 MiB; only
verify and release apply that ceiling before supplying the build environment.
Clean-release staging retains output-tree digests from export verification and checks the installed
trees before publishing release records, inside promotion's rollback scope.
Each GPU test must require its own device with `requireTestDevice`; a smoke test in another browser
chunk cannot validate a projection test that returns early without a device.
Unsupported fallback-device projection cases explicitly skip through their test context, so they
cannot appear as passing GPU assertions.

## Live shared typecheck

Start one supervisor in a terminal:

```sh
yarn typecheck-watch
```

Query it from another terminal, optionally limiting diagnostics to a path regular expression:

```sh
yarn typecheck-status
node scripts/typecheck-status.mjs 'modules/analysis/src/gpu-network/'
```

The supervisor owns `.typecheck/watch.lock`, forwards SIGINT/SIGTERM to the compiler and writes
atomic owner/cycle metadata alongside `watch.log`. Status requires both PIDs to be live and a new
heartbeat produced after its request. A completed cycle must match the current project input snapshot
at acceptance: enabled JavaScript, JSON, TypeScript and inherited configuration inputs participate.
JSON imports resolved from configured TypeScript and JavaScript roots participate even when config
expansion omits JSON roots. Changing only a lens dataset invalidates the previous completed cycle.
Every inventoried source and configuration input is content-hashed, including JavaScript,
TypeScript and JSON: a same-length edit with its original mtime restored still invalidates the cycle.
The snapshot captured at cycle start detects deletions and additions, and acceptance rescans after
the heartbeat to detect edits arriving during the wait. Old logs, dead owners, stale heartbeats and
outdated cycles cannot report success. A duplicate
start fails before starting another compiler. If an uncatchable termination leaves a lock, verify
both recorded processes are stopped before removing the empty lock directory and restarting.

Exit 0 means no selected errors; exit 1 means selected errors; exit 2 means invalid/unavailable/stale
owner state; exit 3 means a compilation-settlement timeout. Filtered success explicitly includes
`filtered matching` and the total error count: it does not mean the unfiltered program is clean.
The package watcher command starts the supervisor; its direct equivalent remains
`node scripts/typecheck-status.mjs --watch`.

## Oracle fixture execution

`scripts/oracle-fixtures.json` is the machine-readable execution contract. Each top-level generator
has a `kind` and interpreter `profile`. A `fixture` lists exact `outputs` and any `inputs`, relative
to `modules/analysis/test`, for example `oracles/fixtures/statistics-markov.json`. An `inventory` is
a source audit, not a numerical oracle. An `unconfigured` entry explains why execution is unavailable:
ambiguous/unmapped outputs, an unsupported C++ toolchain, or historical scratch/tool dependencies.
Neither kind is advertised by `--list` or executed by this runner. New scripts need explicit entries.
Review declarations against actual reads and writes, not mentions in prose. Ripley's positions and
round-eight's correlogram are retained inputs; the two Ripley files and ten round-eight files belong
to their actual generator. The column-analysis/reduction helper scripts do not own round-eight files
merely because that generator imports their writer functions.
Prepared-nearest is a retained input for buffer selection, line density, line length per polygon,
nearest weights and spherical lines; spherical lines also retains line length per polygon. The
cost-matrix-live and change-modes generators retain their synthetic-input and change-detection
fixtures respectively. Only each fixture's writer declares it as an output, so staging preserves
dependencies while requiring actual outputs to be recreated.
Cost-matrix retains its Chicago service-area input; topology modes retains its three graph fixtures
and owns the communities-showcase output. Contours retains isobands and stitched contours retains
contours. GWR/spatial regression retain regression scenes, and statistics upstream retains coverage.
Fixture provenance uses repository-relative paths to existing inputs and SHA-256 digests of their
exact bytes. Refresh a changed upstream record before its consumers: contours names the isobands
fixture under `modules/analysis/test`, and stitched contours hashes the refreshed contours bytes.
Embedded `generator` fields identify the runnable manifest owner. Topology modes stamps its noding
and communities helper outputs and all 24 neighborhood cases with its own entry-script name.

```sh
node scripts/run-oracle-fixture.mjs --list
node scripts/run-oracle-fixture.mjs statistics-markov --python /path/to/python3.14
node scripts/run-oracle-fixture.mjs rts04-cold-pyxccd --python /path/to/python3.12
```

The fixture census is generated from disk and the validated execution manifest, rather than the
historical mappings in `GENERATORS.md`:

```sh
node scripts/oracle-census.mjs --write
node scripts/oracle-census.mjs --check
node scripts/oracle-census.mjs --json
```

The Node census spec also checks drift in the generated Counts section and rejects reversed or
duplicate Counts/Reproducing headings. Declared ownership includes output paths attached to
unconfigured programs; runnable coverage includes only fixture producers. Distinct declared paths,
existing configured outputs, missing declared outputs and fixture JSONs with no configured producer
are separate populations; two producers sharing an output count that path once. Program totals
separate fixture producers, unconfigured programs and inventory programs. The literal/inferred/
unresolved split in the older mapping is historical.

Generator hash attribution uses explicit generator identity, exact recorded source paths, recorded
command script arguments and declared output ownership (including unconfigured programs). The two
round-eight composites/curvature command paths are resolved through their exact R100 moves in commit
`58dd86d8`; unrelated historical directories are not matched by basename. Conflicting or unresolved ownership stays unattributed; a matching basename is not
attribution. The JSON report lists each outcome. The census reads `sources[]` path/hash pairs and
additional top-level `*Sha256` fields with paired source paths or explicit hash-map keys. The trend
reference generator documents a fixture-relative base for its source hash keys; this format is
resolved relative to that fixture directory. Other missing paths remain unattributed. Hashes verify
exact recorded source bytes, not numerical validity or completed regeneration. A differing hash alone does not
establish that a fixture's results are stale.

To retain identified execution evidence, pass a new receipt path to a real fixture run, then render
a separate evidence report (repeat `--evidence` for multiple receipts):

```sh
node scripts/run-oracle-fixture.mjs statistics-markov --python /path/to/python3.14 --evidence /tmp/markov-run.json
node scripts/oracle-census.mjs --evidence /tmp/markov-run.json
```

Receipts identify the run, start/end times, generator digest, declared outputs, recreated outputs and
committed/regenerated byte digests. Incomplete generation records no comparisons and still exits 3.
A complete run records matching or differing comparisons before any `--write` acceptance. Existing
receipt files are never overwritten. Environment and generator failures do not produce comparison
receipts. Each receipt must contain exactly one row per declared output. A partially regenerated run
may contain no comparisons; a fully regenerated run with committed bytes must compare every such
row. The census rejects malformed/duplicate receipts and excludes receipts whose source or
compared fixture bytes no longer match either recorded snapshot. Reports name each supplied run,
separate missing/partial/differing/matching runs, and count distinct compared outputs. A differing
comparison remains visible if another supplied run matched the same output. Local receipts are
report-only: `--write` and `--check` use the reproducible census with no local run evidence. With no
receipts, the generated prose explicitly reports no regeneration/comparison evidence; configured
producers, installed prerequisites and matching provenance hashes never imply such evidence.

The default Python profile requires CPython 3.14 and the recorded packages in `requirements.txt`.
pyxccd has its own CPython 3.12 profile and two outputs. An explicit `--python` or `ORACLE_PYTHON`
selection never silently falls back to a different interpreter. Generators asserting NumPy 2.2.6 use
the separate `python-legacy` profile from the older environment section of requirements.txt, including
NumPy 2.2.6/SciPy 1.15.3 and each generator's extra pins. Explicit `--rscript`/`ORACLE_RSCRIPT` also
never fall back to a different Rscript. Node generators use the running Node executable. Missing selected interpreters, declared tool
settings, recognized Python/R packages or a wrong required Python version exit 2, without comparing
fixtures. Other generator failures exit 3. Additional per-generator tool arguments come from
the environment variables named in the manifest and are passed as separate process arguments.

Every declared input/output must exist inside the committed test tree. Outputs and inputs must be
distinct. The runner clones the test tree, removes staged output copies, runs the generator and
requires every output to be recreated inside staging before comparing bytes. An untouched recent
file cannot count as regeneration. A missing output exits 3 before any comparison or `--write`
copy-back. Fresh identical generation exits 0; differences exit 1 unless explicitly accepted with
`--write`; invalid declarations or inventory selection exit 64. `--keep` retains staging for inspection.
The clone protects normal committed fixture writes; it is not an OS security sandbox for arbitrary
generator code. Historical absolute scratch/tool dependencies remain disabled until ported.
This includes generators importing the historical round-eight GRASS helper, even if their own source
contains no absolute scratch path.

The visibility adapter uses `ORACLE_REFERENCE_PYTHON` (Python with GDAL bindings) and `ORACLE_GRASS`
(the GRASS executable), with NumPy/SciPy/rasterio in the selected generator interpreter. Direct use
also accepts `--input-directory`, `--output-directory`, and optional paired `--audit-log`/`--audit-output`
paths. Fixture generation no longer reads a historical GPU log or writes an audit automatically.
Its numerical references still require those external environments; CLI tests do not certify GDAL
or GRASS results.
Saddle reference commands execute with real temporary paths but serialize project-relative paths;
random temporary directory names cannot change recorded provenance bytes.

The source-only showcase inventory is invoked directly:

```sh
/usr/bin/python3 modules/analysis/test/oracles/parity-wp5-round8-showcase.py --output-directory /tmp/showcase-inventory
```

`--repository-root` and `--node` can select a different input checkout and Node executable. JSON and
Markdown are written only beneath the requested output directory.

## Architecture and contributor departures

`node scripts/check-upstream-first.mjs` independently enforces both package policy and public export
visibility. Installed packages outside core/shaders/graph/engine are rejected; engine is allowed only
under examples and tests, not production package source. `_internal`, non-public/deep paths and
production re-exports fail. Analysis-owned source directories are fixed; only `index.ts` may sit
directly in `modules/analysis/src`. The contributor checker separately derives public family roots
from every package export, including analysis-core and nested entries. Named infrastructure
exceptions are the package forwarding barrel, computation-record and gpu-spatial-analysis/ontology.
They are exact entry exceptions, not subtree exemptions for root-file coverage. Root directories
come from emitted file targets under `./dist/`: direct/non-index targets, native module suffixes
and concrete source matches for wildcard targets are supported. Unsupported target shapes and
unmatched patterns fail explicitly with exit 2. Immediate source implementations are scanned
independently of the other rules, including TypeScript/JavaScript JSX and native module extensions;
index barrels and declarations do not count as departures.
The six loose core helpers, ten point-cloud helpers and three geospatial implementations have moved
into family folders, preserving their public barrel names. Point-cloud helpers live beside their
contributors in `pointcloud-analysis/operations`; geospatial implementations live in
`geospatial-analysis/datum-grid-transform`, `geospatial-analysis/point-buffer` and
`geospatial-analysis/viewport-bounds`. Both entry roots contain only `index.ts`, and their thirteen
root-file baseline allowances have been removed. A comment, annotation or unrelated string does not
register a contributor: the actual catalog construction inputs must contain it.
Registration matches the public package entry point and exported name, following barrel aliases.
The spatial and network `addVisibilityGraphRecipe` exports need separate qualified catalog records;
one cannot satisfy the other's registration. Missing-registration diagnostics name the entry point.
Catalog copies use their final entry point and export name after spreads and assignments in property
order, including literal computed keys and nested literal spreads. A metadata-only spread preserves
identity fields; a nested literal is never an independent operator record. An overwritten name does
not register the original identity. Entry barrels are sorted before
shared public members receive site names, so directory enumeration order cannot change an allowance.

The contributor baseline is version 2:

```json
{"_version": 2, "rule": {"file.ts": [{"id": "signature:1", "description": "site", "reason": "reviewed departure"}]}}
```

Each allowance belongs to a file and individual site. Serial-kernel signatures include the enclosing
function/class context, stable kernel ID, normalized reachable shader text and transitively referenced
module constants/overrides/aliases that determine its bounds, ignoring whitespace/comments and line
shifts. Replacing a loop cannot spend a removed loop's allowance; changed shader text needs review
even when the count is unchanged. Every allowance needs a nonempty reason. Strict-zero rules remain
unbaselineable through ordinary updates. The final contract reconciliation explicitly retains three
pre-pass internal sites used outside their modules: study-window `slabs`/`packed` and the contributor
operation factory. These exact-site records additionally require an `owner` and
`legacy: 'count-baseline-2026-11-17'`, an opaque marker for migration from the old count-based
baseline; an update preserves them but cannot grant another strict-site allowance.
Only some allowances record an `owner`; many reasons name the departure generically (for example
a legacy departure retained for a separate review) and still await a stated bound. The weights-norm
merges in spatial error GM, spatial two-stage least squares and spatial weights traces enforce
`tileCount <= 4096` at construction, including explicit tile widths and the sum of per-chunk tile
counts, so each serial merge handles at most 4,096 partials.

```sh
node scripts/check-contributor-contract.mjs --check-stale
node scripts/check-contributor-contract.mjs --update-baseline --reason 'Reviewed bound and axis for the newly accepted sites'
```

`--check-stale` reports expired identities without rewriting. Review new signatures and remove/fix
departures before regenerating; an update prints every added signature. Existing site-specific
reasons survive. `--migrate-baseline` is only for one-time legacy conversion. The lead must reconcile
the final shared checkout after all streams finish; do not grow allowances merely to make it green.
