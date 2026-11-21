# Analysis alpha package contract

`@luma-contributors/analysis` is prepared as version `0.1.0-alpha.0`. Local metadata does not
establish that a version has been published; this work makes no registry request. Confirm publication
history before choosing a release version. Lenses remains private while its dataset licensing is
unresolved in [THIRD-PARTY-DATA.md](../THIRD-PARTY-DATA.md).

## Tested dependencies and accepted peers

Compatibility testing targets luma.gl `10.0.0-alpha.16` and Apache Arrow 17. Analysis, lenses
and kepler declare `~10.0.0-alpha.16` for `@luma.gl/core`, `@luma.gl/shaders` and
`@luma.gl/graph`, and `>=17.0.0` for `apache-arrow`. The luma.gl range permits later
prereleases on 10.0.0 and stable 10.0.x below 10.1.0. Root dev dependencies and resolutions keep
the installed luma.gl packages aligned with the integration target.

“Tested with” must identify checks actually run against that exact version; it does not follow
from a dependency pin or semver acceptance. On 2026-11-18 alpha.16 passed the source and test
typechecks, the build, the export and packed-consumer checks, the Node suite and the GPU suites,
each run separately; one continuation spec was intermittent (LR-22) and `yarn verify` has not
yet passed as one command on alpha.16. The audit found and adapted grid-coordinate drift and
compile-option forwarding, and the deck.gl package extension was aligned; see
[the alpha.16 audit](./upstream-workarounds.md#alpha16-public-api-audit-2026-11-18).

**Third-wave adoption check (2026-11-20, TWN-08):** the installed alpha.16 public
`@luma.gl/graph/gpu-raster` entry does not export the proposed
`createOriginPreservingIntegerRasterCalibration`. Its neighborhood shader still converts raw
integers to float32 before calibration. For `sint32` samples `[16777216,16777217]`, scale `1`
and offset `-16777216`, that expression produces `[0,0]` instead of `[0,1]`. This stream checked
the installed versions, public runtime export and shader source, and reproduced the arithmetic
on the host. It made no registry request and ran no build, packed consumer or GPU suite; it has
no evidence of a published replacement supplying the helper. The integration target and peer
floor therefore remain alpha.16, with this precision defect outstanding.

Adoption requires a published release exporting the helper through a public entry and passing
[proposal 16's acceptance controls](./upstream-proposals.md#16-origin-preserving-integer-raster-calibration-twn-08).
For the `2×1` acceptance raster, upload the original `Int32Array` samples `[16777216,16777217]`
as a `sint32` band and apply the qualified public calibration helper on the native GPU route.
Use integer origin `16777216`, scale `1` and calibrated offset `0`, equivalent to the original
scale `1` and offset `-16777216` without rebasing the uploaded samples.
Read back calibrated samples `[0,1]` and `parentPeak=[1,0xffffffff]` (`NONE`), with the nodata
and multiply/add cancellation controls also passing. Uploading a rebased `Float32Array` is
workaround coverage only; it bypasses the defective integer conversion and cannot qualify the
upstream fix. Record the exact release, export, adapter/backend and checks run before updating
the root pins, matching resolutions and required peer floor together with the lead's lockfile
regeneration. A proposal or an unpublished implementation does not satisfy this gate. Until
then, subtract the integer origin exactly on the host before upload, adjust the offset, and
rebase the raw nodata sentinel or preserve original validity as described in proposal 16.

The adoption handoff must also have the respective owners align every other exact integration
pin to that same qualified release: `.yarnrc.yml` pins graph in the deck.gl package extension;
`website/package.json`, `examples/deck/kepler-compute/package.json`,
`examples/deck/spatial-analysis-explorer/package.json` and
`examples/deck/spatial-analysis-showcase/package.json` each pin core, engine, graph and shaders.
Include those coordinated edits in the lead's lockfile regeneration; root resolutions alone do
not update these declarations.

`apache-arrow` is a **required peer**, with range `>=17.0.0`; the workspace currently installs
17.0.0. Native imports of `analysis-core`, `gpu-spatial-analysis`, `gpu-network` and
`dataframe-analysis` reach Arrow eagerly. Install the peer even when requesting only a validation
helper or network contributor. Existing entry points and Arrow adapter exports are preserved.

Analysis no longer declares a direct `probe.gl` dependency: its runtime uses the public
`@luma.gl/core` logging API, and upstream core declares its own runtime dependency. The root
development dependency remains. This manifest cleanup requires the lead to regenerate the
analysis workspace lock entry and qualify the packed consumer against that dependency set.
The manifest and regenerated lockfile must land in the same change: both CI jobs run
`yarn install --immutable`, which cannot accept the stale analysis dependency entry. Preserve
the root and upstream declarations and their shared lock records. The manifest edit alone does
not establish package qualification.

## Verify the packed bytes

After the lead's clean build, run:

```sh
node scripts/check-packed-consumer.mjs --keep-artifact
```

The checker runs `npm pack --offline --ignore-scripts`, extracts the tarball into a new consumer
under `os.tmpdir()` outside the checkout, and links only the package's declared dependencies and
peers to their installed workspace copies. It never installs, publishes or calls a registry.
Before packing it resolves the real temporary root and rejects one inside the workspace, including
symlink aliases, so native module resolution cannot walk up into the checkout. Before linking a
dependency or peer, it checks the installed version against every declared semver requirement.
Linked upstream packages use their installed transitive dependencies; this checks the analysis
tarball's native resolution and declared first-party dependencies, rather than simulating a new
dependency solver or testing every accepted peer version.

From that consumer it imports every explicit export in ESM and CommonJS, compares the exported
names, checks that each declaration target is a file inside the tarball, verifies README links,
and runs the synthetic CPU variogram fit from
[the first computation walkthrough](../docs/api/operation-forms-and-cpu-twins.md).
The exponential row must have status 0 and parameters within 1e-4 of `[0.2, 1.5, 2.2, 0]`;
nonfinite values fail. A native `.mts` consumer is also compiled with NodeNext and no workspace
path aliases. It reads the line-density schema's numeric `wordLength` and uses `@ts-expect-error`
on an invented property; an unresolved reference that erases the schema to `any` therefore fails
with TS2578. The schema uses an explicit imported type with a resolvable `.js` declaration reference.
The checker also resolves every relative module specifier, `reference path` directive and
`reference types` directive in every packed declaration under NodeNext, in the resolution mode
TypeScript uses for that reference. Each counts as an edge; failures name the declaration line and
reference. Every relative edge must resolve to a type-bearing TypeScript file inside the packed
package, with containment checked on its real path. Path references use the compiler's supported
extensions, including extensionless `.ts`/`.tsx`/`.d.ts` lookup. A JavaScript file alone cannot
substitute for its missing declaration sidecar. Type-directive resolution and the native consumer
program both use the isolated consumer as their current directory, so a local `./node` directive
cannot select the checkout's `@types/node`. An explicit directive `resolution-mode` is respected.
Legal in-package filenames beginning with two dots remain valid. An extensionless
`import("../../analysis-core")` type degrades a public type to `any` without a diagnostic under
`skipLibCheck`. A CommonJS `.cts` consumer is compiled beside the `.mts` one.
This is a native CPU/type consumer check, not GPU execution or an exhaustive check of every
declaration body. Empty exports, unsupported export patterns, missing files, unsatisfied peers and
unequal ESM/CommonJS names fail explicitly. For each expanded JavaScript entry, export verification
resolves and validates the physical ESM target and every applicable CommonJS target before either
loader executes initialization. A CommonJS target mismatch therefore cannot initialize that entry's
ESM module during preflight.

The isolated native worker also exercises objects across the ESM/CommonJS boundary in both
directions: a real two-chunk vector retains ordered IDs `two-chunk-0`, `two-chunk-1`
and passes owning-graph validation; the opposite-format public `GPUZonalStatistics` receives
those two float32 chunks, two feature rows, one feature and a sums destination, selects
`sumOrder: 'atomic'`, and declares accumulation nodes that read each value chunk and update the
supplied sums buffer on the owning graph. Preparation nodes alone do not qualify. An empty
vector remains a separate zero-chunk control. A recorder retrieves the opposite-format kernel's
actual node ID, WGSL and dispatch. Operation lowering receives a genuine opposite-format
operation with a selected float32 input and uint32 output; its factory requests both ports and
checks their formats and exact backing-view identities. The resolver returns real single-chunk
graph vectors, including format and length, and accepts either port lookup order. The context
receives its contributor once, and a type-string-only impostor is rejected. These declaration checks use a null device.
Native `.mts` and `.cts` consumers check the public family props/output types and reject unsupported
deep source and testing imports. These checks are covered by archive fixtures, including
reverse-only zonal and selected-port defects. They have not yet qualified a newly built package in
this wave. The lead must retain their passing output with the final dependency set before claiming
cross-format package qualification.

README links are extracted from parsed Markdown tokens, including nested reference links and HTML
assets. Fenced and indented code examples do not contribute destinations. The checker uses the root
development tools TypeScript, MarkdownIt and semver; these tools are not linked into the consumer
unless the packed package itself declares them.

The package README describes the required Arrow peer and accepted luma.gl range. Both README
licence summaries link the shipped notice inventory. The release-package node spec checks those
installer instructions and the current dependency statements in AGENTS, the roadmap and the
recording guide, while keeping historical benchmark versions outside that check. The changelog
check distinguishes the unreleased analysis alpha from published upstream dependencies and keeps
runtime/typecheck exclusion membership separate.

Package payloads come from `dist` and include ESM, CommonJS, declarations and JavaScript source
maps. ESM maps embed their original source text. CommonJS maps compose the transformation with
the compiler map so they point to the same original source. The packed consumer checks that both
root entry-point maps are present and embed the original source text. Package metadata, READMEs
and applicable licence files accompany the compiled output. Both archive gates reject paths under
`src/` and `dist/testing/`, plus the internal `dist/.build-record.json`.

The packed checker verifies the shipped licence texts and checks the shipped notice inventory
against the package's build sources. TypeScript files must carry an SPDX header. The inventory
must list each non-MIT source path, its exact SPDX expression and its copyright holders. The
analysis package licence is `MIT AND Apache-2.0 AND BSD-3-Clause`. See the
[shipped inventory](../modules/analysis/THIRD-PARTY-NOTICES.md).

`--keep-artifact` prints the checked tarball and consumer paths; promote those checked bytes after
all release gates pass. `build.json` and `release-build-latest.json` identify the checked `analysis.tgz`
by SHA-256 and byte length, and promotion refuses an archive whose bytes differ. Replacing or
changing the input archive after extraction makes the checker exit 1, with or without
`--keep-artifact`. It supplies the captured archive bytes on tar's standard input, checks the
retained pathname immediately after extraction before the first package check, and checks it
again after native runtime/type consumers complete. An early replacement must fail before package
checks; a replacement during native loading must fail at the final archive check. Regression
fixtures must distinguish those failure phases rather than accepting any exit 1. Without
`--keep-artifact`, temporary files are removed on success or failure. The optional
`--package-directory` and `--dependency-directory` switches support isolated regression fixtures.
The checker does not build: run it after a clean build so a stale `dist` cannot be mistaken for
the current source. It does not replace the lead's source/test typechecks, GPU tests or package
export checks across lenses and kepler.

## Release metadata still owned by the maintainer

Confirm the canonical public repository URL before publication. Analysis metadata currently names
`https://github.com/visgl/luma-contributors`; this stream deliberately preserves it. Package README
links must be absolute canonical URLs or resolve to files actually shipped in the tarball.

Root README, AGENTS, changelog and roadmap statements about the integration target and publication
status must agree with this contract. Historical benchmark versions remain historical evidence.
Do not copy an exclusion count into release prose: inspect `test/kepler-compute-example-specs.json`
and distinguish runtime exclusions (`excludedFromRuns`) from test-typecheck exclusions
(`excludedFromTypecheck`). Counts can be obtained without running the tests:

```sh
node --input-type=module -e "import {readFileSync} from 'node:fs'; const manifest = JSON.parse(readFileSync('test/kepler-compute-example-specs.json', 'utf8')); for (const key of ['excludedFromRuns', 'excludedFromTypecheck']) console.log(key, Object.keys(manifest[key]).length);"
```

## Typecheck measurement and package-reference experiment

`node scripts/measure-typecheck.mjs --project <tsconfig>` runs one project cold and then warm,
sequentially, with fresh private compiler caches under `.typecheck/measurements/run-*`. It extends
the original configuration, redirects every referenced project and all compiler outputs/caches into
that directory, and never deletes an existing cache or developer output. Both runs reuse the same
private directory. Cold means no compiler cache or generated output; it does not mean an empty OS
filesystem cache. A nonincremental project stays nonincremental in both runs, so the source baseline
is not silently given an incremental advantage.
Relocated configs explicitly retain the original parsed absolute root files, disable additional
include discovery and preserve the original root directory; implicit default includes are supported.
They also retain the original effective ambient type roots and pin the selected type declarations,
including app-local and directly typed packages, so relocation cannot select a hoisted alternative.
Package-source snapshots apply the same type-lookup preservation. Every `tsc -b` configuration gets
a private build-info file even when `incremental: false`; a no-emit nonincremental source check
retains its original cache policy.

The command prints project, compiler file count, cold/warm wall seconds and peak sampled child-tree
RSS. `report.json` retains separate cold/warm results, machine, compiler version and heap settings;
the two compiler logs and isolated configs remain alongside it. For a reference graph, the file
count sums the cold compiler programs, including repeated dependencies, rather than claiming a
deduplicated workspace count. A warm build with no dirty projects can report zero compiled files.
RSS uses the same `ps` descendant sampler as verify, once per second, rooted at the compiler child
(verify includes its wrapper). Short peaks and GPU memory are outside this accounting. The command
enforces the 8 GiB disk floor and 12 GiB sampled compiler-tree RSS ceiling and carries the effective
caller heap with the 8192 MiB default/cap. Failed compilers or failed sampling do not publish a
passing comparison. `--timing-only` explicitly records RSS as unavailable and retains the disk floor.
Each measured cold and warm interval must observe a finite positive child-tree RSS at least once;
a successful process listing without the compiler PID cannot certify zero RSS. Reports retain the
positive observation count as `rssSampleCount` alongside each interval's peak.
All modes fingerprint input contents and timestamps before cold and after cold/warm. Configs and
root lists are reparsed and imports, type references and standard-library references are resolved
without creating a semantic compiler program. This catches edits, new roots and changed imported
dependencies, including JSDoc-only `@import` and comment import types collected from JavaScript
syntax trees. As in TypeScript, JSDoc in typed sources and declaration files does not introduce
dependencies; editing a declaration mentioned only there preserves freshness. Implicit JSX runtime
declarations are also tracked using the compiler's
production/development runtime selection, `jsxImportSource` and per-file JSX pragmas, even when
there is no explicit runtime import. Input checks run outside the timed compiler intervals.
A changed fingerprint refuses publication and leaves compiler logs for diagnosis. Successful
reports retain `inputFingerprint`.

`--mode declarations` uses `tsc -b` with declaration-only outputs for checking a reference graph.
`--mode build` measures the TypeScript ESM/declaration phase. `--mode packages`, with
`--project tsconfig.build.json`, measures the existing `buildPackages` engine in a private source
snapshot, including TypeScript, output repair/pruning, two-at-a-time CommonJS transforms and build
records. It copies only root/package metadata/configs and package sources, links installed dependencies
and never promotes the snapshot. Preparation and the worker both validate every referenced project,
source root, resolved build/compiler cache and emit destination against the canonical private
workspace, including existing symlink ancestors. An absolute cache/output pointing back into the
developer tree is refused before the build engine can invalidate a cache or write output.
Original inputs must match before/after
copying; subsequent checks fingerprint the snapshot. Snapshot preparation and fingerprinting are
outside the timed intervals. A compiler wrapper adds `--extendedDiagnostics` to the engine's
compiler invocation; the engine's cache and CommonJS policies stay unchanged. Package mode defaults
to the installed patched `tspc`, including its configured JavaScript/declaration transformers, as
the developer build does. Other modes default to stock `tsc`; `--compiler` overrides either choice.
Neither mode performs export, packed-consumer, release or GPU qualification.

**Observation on 2026-11-18:** `roberts-MBP`, macOS/Darwin 23.1.0, arm64, Node 22.23.4,
TypeScript 6.0.3. Hardware-model and RAM queries were denied, so no chip/RAM specification is
inferred. The actual `ps -axo pid=,ppid=,rss=` call was denied by the sandbox. Available disk then
fell from approximately 11 GiB to 5.06 GiB and later 7.989 GiB during shared-checkout work. Those
attempts correctly refused the 8 GiB floor. During review fixes disk recovered to about 10 GiB;
the corrected runner then completed all timing-only trials sequentially with an 8192 MiB heap.
The normal command still refused `cannot monitor process-tree RSS: spawnSync ps EPERM`.
Neither restriction was bypassed. No actual RSS or idle-machine result is claimed.

During the 2026-11-18 review, the live checkout's aggregate source/config/tool digest changed
across the first measurement series. Therefore the final source, reference and package comparison
below uses **one private package source snapshot** of that day's analysis, lenses and kepler
sources, package metadata and compiler configuration. Its 3156
common source/metadata files hash to `598c8e33492537895a42f6aeadcab20bfcb252fd5ab48f8519c7291f1eecd58f`.
The source and reference checks point into that same snapshot; each cold/warm pair also passed its
input guard. CPU contention on the shared machine remains uncontrolled.

| Project / mode | Cold compiler files | Cold s | Warm s | Peak RSS |
| --- | ---: | ---: | ---: | --- |
| Snapshot source baseline / check | 4343 | 12.516 | 13.995 | unavailable: `ps` denied |
| Snapshot package-reference prototype / declarations | 8953 (sum) | 19.495 | 0.124 | unavailable: `ps` denied |
| Snapshot package pipeline / packages | 8953 (sum) | 33.558 | 5.329 | unavailable: `ps` denied |
| Separate live network-accessibility test stage / check | 1356 | 3.986 | 1.451 | unavailable: `ps` denied |
| Separate live kepler-utils test stage / check | 2351 | 6.967 | 1.713 | unavailable: `ps` denied |

The full package trial used the default patched compiler. Cold CommonJS writes were 1770 analysis,
1042 lenses and 295 kepler modules; the warm trial wrote zero in all three packages. These are
complete build-engine timing intervals, not release/export/GPU qualification. Both interval peaks
remain `null`, with zero RSS observations, in every timing-only report.

An initial continuation `--mode build` trial returned 39.365 / 17.355 seconds but compiled **8939
files cold and 8940 warm**. It is rejected as an unchanged-input comparison. That trial exposed the
missing input guard: four regressions initially returned exit 0 for changed root/imported source,
and now fail publication in every compiler mode. A real three-package fixture also verifies actual
CommonJS value 3, build records, cold writes, warm reuse and preserved developer outputs/caches.

The declaration-reference prototype used one config per package extending its current composite
build config and an empty-root solution config with references in dependency order
`analysis → lenses → kepler`. It emits declarations into a separate prototype tree; the measurement
runner further isolates that tree for every invocation. Before input guarding existed, its compiler
runs passed at 61.428 / 0.269 seconds versus the then-source baseline's 36.939 / 42.160 seconds.
These historical unguarded samples are retained, but cannot establish the split's performance cost
under concurrent edits. The corrected guarded same-snapshot retry is recorded above. A tiny real two-package fixture verifies
reference redirection and cold/warm reuse. Declaration emission is extra work relative to source
`--noEmit` and must be included in a future comparison.

The adoption threshold for this memory-focused change is **at least 20% lower sampled peak RSS**,
with **no more than 10% regression in either cold or warm wall time**, on an otherwise idle machine
at the same heap limit and unchanged source. The threshold was specified before measurements.
The earlier apparent 66.3% cold regression is a historical unguarded observation, not sufficient
evidence of architectural overhead. The corrected snapshot's observed cold reference time is 55.8%
above its source baseline, exceeding the 10% tolerance in that sample, while its unchanged warm run
is much faster. Actual RSS is still unavailable and idle execution is unverified, so the full
memory/time decision remains open. Keep current source configs; no root configuration/package
patch is proposed. Complete the experiment on an idle machine with actual process inspection.
The build already has package references, preserved incremental state and bounded CommonJS
concurrency, so adding package references to the build would duplicate existing structure.
No repository-wide verification, release, export or GPU gate ran. Package compilation and CommonJS
generation ran only in the private measurement snapshot; developer outputs/caches were not replaced.

The original local reports, prototype configs and snapshot archive are not repository fixtures.
The recorded snapshot archive SHA-256 was
`18d7d7dcf2d213ae0c3dcd1b22d221d2adc2940f96db32abdd4fffeda72dc416`, excluding dependencies,
compiler caches and generated outputs. These notes preserve the measurements and snapshot identity;
reproduce the experiment with a fresh snapshot and retain the runner's new reports alongside it.
R6 remains blocked on actual RSS/idle qualification; a verification log's bare PASS is not
benchmark evidence. Generated output trees and compiler caches from the original experiments
were removed afterward to conserve disk. The runner retains artifacts by default.

With at least 8 GiB free disk and process inspection available, run these commands one at a time:

```sh
node scripts/measure-typecheck.mjs --project tsconfig.check-src.json
node scripts/measure-typecheck.mjs --project .typecheck/test-families/analysis-gpu-network-network-accessibility.json
node scripts/measure-typecheck.mjs --project .typecheck/test-families/kepler-utils.json
node scripts/measure-typecheck.mjs --project tsconfig.build.json --mode build --compiler node_modules/ts-patch/bin/tspc.js
node scripts/measure-typecheck.mjs --project tsconfig.build.json --mode packages --compiler node_modules/ts-patch/bin/tspc.js
```

The first three compare source/test programs. To repeat the declaration-reference comparison,
create the prototype configs described above, with `emitDeclarationOnly: true`, separate
`outDir` and `tsBuildInfoFile` paths for each package, and pass the solution config to
`node scripts/measure-typecheck.mjs --project <solution-config> --mode declarations`.
The last two commands measure the
TypeScript phase and existing package build pipeline, respectively. Run them sequentially with room
for the package source snapshot and outputs above the 8 GiB floor; they do not authorize parallel
builds, promotion, publication or wider gates.
Retain fresh prototype configs and measurement reports with any experimental handoff.

## Associate qualification with the retained archive

Run the complete `yarn verify` after the final input edit, then `yarn build:release` (or
`--promote`) on matching inputs. Verification publishes a durable receipt as described in [testing
gates](./testing-gates.md#durable-qualification-evidence). Staging records the same
`qualificationIdentity`: revision, working-tree content digest, lockfile digest, actual resolved
luma.gl/Arrow/tool versions and Node/Yarn/npm versions, including the dev-tools configuration
package and adapter-resolved Playwright driver. These describe the versions used, independently of
accepted peer ranges. Package-local peer versions are retained as well as root resolutions. Staging
independently resolves peers from its snapshot packages and records
`stagedResolvedWorkspaceDependencies`; a split root/package-local installation cannot attach full
qualification when the snapshot uses different versions.

A matching full receipt can attach only when the packed checker also supplies the SHA-256 and byte
length of the archive it extracted. The handoff protocol is a `checked tarball identity: <JSON>`
line with `sha256` and `byteLength` alongside the existing `checked tarball: <path>` line. Staging
verifies those supplied bytes before retaining the archive, independently of its own later digest.
`archiveHandoff.kind` is `extracted-archive-digest` for that protocol; an older path-only checker
leaves the stage as `build-package`, even with a full matching test receipt. The checker must
hash the buffer supplied to extraction, revalidate the retained file immediately after extraction,
and check it again immediately before success. The current checker emits this evidence; the
retained copy's later digest alone does not establish which bytes were extracted.

A matching full receipt is copied into the stage as `qualification.json`; the release record binds
its digest and byte length to `qualification.tarballSha256`/`tarballByteLength`. The ordinary
`tarballSha256`/`tarballByteLength` still identify the packed-consumer-checked archive. Input and
receipt/archive identities are rechecked inside the existing publication/rollback window. An
unmatched, partial, failed, running or unavailable receipt leaves `qualification.kind` equal to
`build-package`. Such a stage remains useful build/export/packed-consumer evidence, and is not a
full Node/GPU qualification. A previous full receipt cannot qualify an edited source/test/lockfile
or different resolved dependency version. Promotion does not itself publish to a registry.

After installing all three output trees, promotion reruns the export/build-record verifier against
the live checkout, including native entry-point loading. This runs before either release record is
published, with rollback still available. A failure restores previous outputs and developer caches
and preserves the prior latest record. Input, archive and output digests are checked after loading.

The live export check compares the physical ESM and CommonJS targets selected by native
resolution with the expected installed package targets before loading them. Both paths are
canonicalized with `realpath`, so a workspace link pointing to a second checkout fails with a
target-mismatch error even if that checkout exports the same names. These checks run inside
the promotion rollback window described above.

Immediately before an explicitly tagged publication, the maintainer should inspect the retained
receipt (including skips and device/backend evidence), confirm the intended version and registry,
and check the exact retained bytes. From the original checkout, this read-only check refuses
mismatched inputs, receipt bytes or archive bytes:

```sh
node --input-type=module <<'NODE'
import {readFileSync} from 'node:fs';
import {captureQualificationIdentity, isFullQualificationReceipt} from './scripts/verification-policy.mjs';
import {verifyReleaseTarball} from './scripts/build-release.mjs';
const record = JSON.parse(readFileSync('.typecheck/release-build-latest.json', 'utf8'));
if (record.archiveHandoff?.kind !== 'extracted-archive-digest' || record.qualification?.kind !== 'full') throw new Error('stage has only build/package evidence');
verifyReleaseTarball(record.qualification.receipt);
const receipt = JSON.parse(readFileSync(record.qualification.receipt.path, 'utf8'));
const identity = captureQualificationIdentity(process.cwd());
if (!isFullQualificationReceipt(receipt, identity) ||
    record.qualificationIdentity.identitySha256 !== identity.identitySha256 ||
    record.qualification.identitySha256 !== identity.identitySha256 ||
    record.qualification.receiptId !== receipt.id ||
    record.archiveHandoff.sha256 !== record.tarballSha256 ||
    record.archiveHandoff.byteLength !== record.tarballByteLength ||
    record.qualification.tarballSha256 !== record.tarballSha256 ||
    record.qualification.tarballByteLength !== record.tarballByteLength) {
  throw new Error('stage and full qualification do not match');
}
verifyReleaseTarball({path: record.tarball, sha256: record.tarballSha256, byteLength: record.tarballByteLength});
console.log('Checked retained archive:', record.tarball);
NODE
```

The maintainer then explicitly publishes that printed archive, for example `npm publish
/absolute/stage/path/analysis.tgz --tag alpha`, after choosing the intended tag. Do not repack the
workspace for this handoff: publish the checked archive and repeat the check if its bytes, receipt
or checkout change before publication. No publication is performed by verification or release
staging. Keep the stage, full receipt and referenced command logs together.


## Build-record freshness and configured outputs

Incremental build/export records hash the resolved dependency declaration closure, not just package
versions and timestamps. Linked declarations remain freshness inputs even if an edit preserves their
version, byte length and timestamp. Dependency resolution also follows JS/JSDoc and transitive
declaration references. Compiler, patcher and transformer contents are fingerprinted: changing
TypeScript, the selected compiler executable, ts-patch, esbuild or build/transformer scripts
invalidates stale records and CommonJS reuse even without a declared version change.

Freshness inputs and emit roots have different roles. Dependency declarations can invalidate the
build without becoming independent compiler roots or acquiring planned first-party output files. The
parsed TypeScript project determines the emit roots; the shared output planner asks TypeScript for
the configured outputs, including native `.mts`/`.cts`, preserved JSX, JSON and enabled source and
declaration maps. Build repair and export verification use that same output inventory, so missing
configured maps cannot be certified by checking only `.js` and `.d.ts` siblings.

Version-1 records keep their shape and gain dependency/tool fingerprint entries. Records created
before these additions are stale and require a rebuild; retaining their old source timestamps does
not establish freshness. These build/package checks and their compiler-content hashes do not
substitute for a matching full qualification receipt or claim that the GPU gate passed.
