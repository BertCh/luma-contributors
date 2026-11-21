# Analysis test gates

The workspace-only helpers in `modules/analysis/src/testing` are not published package APIs.

CI builds all three workspace packages before any package-backed typecheck. Both the main test
job and the separate website job run `yarn build` after dependency installation, with an 8192 MiB
heap. The example checks cover showcase, explorer and kepler-compute; the website keeps its own
unchanged project configuration. A source-only checkout has no usable analysis package declarations
until that build completes. `ci-app-typechecks.node.spec.ts` guards the per-job build order and
the four application commands.

`website-view-model-parity.node.spec.ts` compares generated graph data and byte precision with
fresh record extraction, including workflow `totalBytesKind` and summary `bufferBytesKind`.
Its in-memory `lower_bound` mutations must fail parity so a misspelled lower bound cannot silently
be displayed as an unspecified size. The summary comparison folds the committed records' resource
precision using the same rules as the generator; the workflow comparison also checks what the page
actually receives.

Use `expectNumericArrayClose(actual, expected, options)` for numeric arrays. It checks equal
lengths and every element, including sparse or missing entries. For finite expectations the
allowance is `absoluteTolerance + relativeTolerance * abs(expected)`, with both terms defaulting
to zero. `absoluteTolerance` can be a function of the expected element and its index to preserve
an oracle's existing error budget. NaN matches only an explicitly expected NaN; infinities must
match sign. A finite expectation never accepts a nonfinite result. Supply `label` for context.

GPU correctness specs call `requireTestDevice` after acquiring their device. Missing devices fail
browser-backed runs and appear as skips in Node. An intentionally optional comparison profile
uses `context.skip` with the unavailable profile in the reason. The recording/profile source
guard in `require-test-device.node.spec.ts` rejects silent successful returns at these sites.

`getGPUCommandNodePipelineKeys` and `countDistinctPipelines` count requested pipeline
configurations using source, shader layout, entry point (default `main`) and serialized constants.
These are the installed luma.gl factory's cache inputs, including constant insertion order.
Layouts always participate; the old `includeShaderLayout` option is retained but has no effect.
Use `getGPUCommandNodeShaderModuleKeys` when measuring unique WGSL modules alone.
`recordComputePipelines` and `getGPUCommandNodeKernelSources` share one recorder, preserving
entry-point and layout metadata. Counts describe configuration identity, not observed GPU builds.
`getParsedKernelSources` retains the last source/constants by ID for source lookup. Resource
budgets instead iterate its `configurations` array, which preserves every distinct ID/configuration
pair, and pass each record's `source` and `constants` together. Use `getConfigurations(id)` to check
every configuration of a selected ID; a missing ID throws. Earlier configurations with the same ID
must never be discarded from a budget check.
`kernel-resource-configurations.node.spec.ts` checks analysis and lenses budget loops and direct
lookups for discarded configurations. Source-text assertions may use the ID map; resource
assertions use `configurations` or `getConfigurations(id)`. When a raw recorder is needed, retain
its configuration array for budgets and build a separate ID map only for source-text assertions.
The policy also covers direct `getParsedKernelSources(...)` loops, inline storage counts and
source-keyed deduplication of raw records. Identical WGSL with different overrides is still two
resource configurations. Storage counts include `var<storage>`'s default read access and share
the module audit's declaration rule.

`getPrivateMemoryBytes(source, constants)` and `getWorkgroupMemoryBytes(source, constants)`
share layout accounting with the kernel audit. Pass the recorded pipeline's `constants` when
checking a compiled kernel: override values can change array sizes. Named constants, expressions,
aliases, struct padding and explicit member size/alignment are resolved. Unresolved memory
layouts throw rather than certifying a small bound. The detailed `getWGSLModuleUsage` result
reports unresolved lengths and types explicitly. Private storage here means module-private
declarations plus typed or inferred local arrays and structs, summed over functions. Both `var`
and immutable `let` aggregate copies count; scalar/vector `let` bindings remain registers.
`localArrayBytes` reports arrays and structs containing arrays; `localAggregateBytes` also includes
structs without arrays (including enlarged `@size` members), and is used by the private helper.
It is a declaration estimate, not compiler/device memory allocation. Function-array declaration
fragments are also accepted by the private-storage helper.

Workgroup budget checks use `workgroupStorageBytes`: WebGPU rounds each workgroup variable's
store size up to 16 bytes before summing against `maxComputeWorkgroupStorageSize` (16,384 bytes
for the core gate). `getWorkgroupMemoryBytes` returns this rounded limit usage. The detailed
`workgroupBytes` field retains the unrounded declared payload. For example, an `array<u32,4093>`
and a separate `u32` have 16,376 payload bytes but consume 16,400 bytes of the limit; a single
`array<u32,4096>` fits exactly. This calculation does not predict driver allocation.

The shared scanner recognizes whitespace before an address space, explicit `var<function>`,
and constructor-derived array types. Module constants cannot be overwritten by function-local
names; local array lengths follow their own block scopes, while named struct layouts stay at
module scope. An override with `@id` uses that numeric pipeline key. Supported integer expressions
use integer division at each operation; unsupported expressions remain unresolved rather than
being evaluated with JavaScript floating-point arithmetic.
For-loop initializer names stay in scope through that loop's body only. Block comments retain a
token separator, so inserting a comment between `let` and its name cannot hide a declaration.
Declaration and expression identifiers use Unicode XID start/continue characters (plus the
WGSL underscore spelling), including function, parameter, alias, member and constant names.
Renaming an array copy from `copy` to `τ` therefore preserves its memory accounting.
Vector and matrix component types resolve scalar aliases, including `f16`; unknown components
remain unresolved instead of receiving a guessed four-byte component layout.
Identifier initializers follow their declarations, including local `let` chains and typed function
parameters. Token-based expression inference resolves literals, constructors, parenthesized pointer
dereferences, arithmetic, shifts, comparisons, struct members, array elements, and declared function
returns. Builtins have their WGSL result types: for example `dot`/`length` return a component scalar,
`arrayLength` returns `u32`, `frexp` returns its fraction/exponent struct, and atomic/workgroup loads
return their pointee data type. An array or struct copy retains its declared layout and constant
scope. The helper reports an unresolved layout only when it cannot establish the copied type;
it never guesses a four-byte scalar for an unknown initializer.
Explicitly typed immutable bindings also require a recognized layout before they can be treated
as registers. Unknown data types and aliases to them report an unresolved layout; pointers and
texture/sampler handles remain register bindings without a data-byte estimate of their own.
Module-level texture and sampler variables use WGSL's implicit handle address space. They count
toward sampled-texture, storage-texture and sampler binding limits, including aliased handle
types, and do not occupy declared private/workgroup data memory. Unknown data layouts still fail.
Boundary fixtures must account for scratch shape as well as input/output rows: Huff normalization
uses sixteen bytes per demand row, while facility offsets and probability slots retain their
independent four-byte limits. Test the actual boundary and the first refused row above it.
Geometry audit fixtures also include derived storage: Voronoi's two triangles per site require
96 candidate-endpoint bytes per site, make-valid adds one closing vertex per ring, and closed
multi-iteration smoothing needs one output look-ahead row. Rejection fixtures name the earliest
actual scratch or row limit, including the limit's numeric value.

`kernel-audit-memory-layouts.node.spec.ts` enumerates every case in `KERNEL_AUDIT_CASES` directly.
Every accepted case must record at least one shader module and have no unresolved memory layout;
construction or compilation failures fail the check. Expected construction refusals are checked
against the audit rejection table.

The layering gate reads family pages from `ANALYSIS_DOC_PAGES` and verifies named contributor
and recipe headings against their entry exports. A guide can explicitly name another analysis
entry for cross-family contributors. Missing pages, empty inventories and zero checked headings
fail. The runtime gate walks runtime imports and re-exports, including dynamic imports and
`require`, and rejects visualization, host-package and private testing dependencies with an import
chain, including `@kepler.gl`, the analysis-test alias and literal backtick dynamic imports.
Type-only dependencies do not count. Kernel contributor discovery includes `analysis-core`
and inherited producer methods; the existing fitting cases serve both the focused suite and
library-wide coverage. Non-contributor core classes have explicit helper classifications.

`documented-page-headings.ts` classifies each discovered API page independently of its current
contents. Empty requirements identify narrative pages; contributor pages must retain every
required heading, including the computation-record guide's `GPUTransferMask`. Contributor and
recipe headings must reach runtime value exports. Type-only named and star re-exports cannot
satisfy that requirement, even when their underlying declaration is a class.

Whole-family barrel checks compare exported symbol identities and runtime availability. A
compatibility module may forward an entire family without a direct `export *` statement in the
package entry. Such bridges pass when every family export survives; partial bridges, replacement
constructors with the same name, and constructors erased by type-only exports fail.

## Durable qualification evidence

`yarn verify` writes `.typecheck/verify/run-<id>.json` and atomically replaces
`.typecheck/verify/latest.json` as it progresses. `qualification` is `running`, `failed`, `partial`
or `full`. A process killed before final publication leaves running evidence; it cannot qualify a
release. Single-step runs and resumes after the first step remain partial. `--from lint` is the
first-step exception: it requests all ten steps and can produce a full receipt. A full receipt
requires all ten ordered steps, successful commands, reconciled Node/headless inventories and
identical starting/finishing input identities plus matching input checkpoints before and after every
command. Full runs fail at the first changed checkpoint, so an edit between Node and headless cannot
be hidden by restoring it later. These checkpoints do not claim to detect an edit and restoration
wholly inside one command; run on a quiescent checkout. Successful partial capture runs retain
changed-input evidence in `inputChanges` and remain partial without failing solely because their
requested outputs changed inputs.

Node uses a fresh JSON report with one worker and reconciles it against `listTestFiles(ROOT, 'node',
environment)`. Missing, unexpected, duplicate, failed and empty suites fail; failed or unknown
assertion statuses fail. Explicit `skipped` and `todo` assertions remain visible and are allowed in
Node, including an explicitly skipped file. The installed reporter maps queued/runnable tests to
`pending`; pending is rejected as unfinished evidence. Headless still requires a passing assertion
in every selected file. Both inventories are checked again after execution. Existing runtime
exclusions and their explicit opt-ins still apply; inventory reconciliation does not silently enable
them.

Selections are persisted before launch. `attemptedSteps` includes a launched failing step;
`executedSteps` retains successfully completed steps. Fresh Node and headless JSON reports have
run-specific filenames, digests and byte lengths, including failed reports, so later runs cannot
overwrite earlier evidence. Each headless command must account for its exact directory chunk and
file arguments. Malformed or incomplete receipt evidence cannot qualify a stage.

The receipt retains selected files, per-suite passing counts and explicit skip identities/statuses,
commands and arguments, effective Node options, one-worker policy, opt-ins, disk/RSS ceilings,
command timings and sampled peak process-tree RSS. Command output is retained beside the receipt; up
to 200 matching device/backend lines per command are also copied into the receipt (2,000 characters
per line). These lines are observations, not a parsed device certification. Optional runner-supplied
`deviceEvidence`/`backendEvidence` in JSON are retained. The receipt separately records the
configured Chromium/Playwright launch request: `--enable-unsafe-webgpu`, `--ignore-gpu-blocklist`
and, on Linux with nonempty `CI`, `--use-angle=swiftshader` plus `--enable-unsafe-swiftshader`. A
configured software-renderer request is visible even without browser console output. It does not
establish the observed adapter. Absent device metadata is explicitly unknown; a Chromium label alone
does not establish a hardware adapter or Metal backend. RSS is sampled once per second, including
the wrapper and descendants, and excludes OS/driver GPU allocations. The existing 8 GiB disk floor,
12 GiB sampled RSS ceiling and 8192 MiB heap ceiling apply.

Verification and release staging use the same identity helper. It records HEAD, a SHA-256 over
Git-listed working-tree file names and bytes plus recursively inventoried modules, examples, test
helpers, scripts and website inputs regardless of Git ignore rules, the independent `yarn.lock`
SHA-256, resolved dependency versions and Node/Yarn/npm versions/platform/architecture. Ignored
optional cached Kepler Arrow/config fixtures are included when present, and their addition/removal
changes the identity. Generated `dist`, `.typecheck`, compiler caches, `.git` and `node_modules` are
excluded from the source digest; installed versions are recorded separately, including package-local
peer resolution for analysis/lenses/kepler. Same-name source edits and test edits change the
identity. The tool list includes `@vis.gl/dev-tools` and the Playwright/playwright-core versions
resolved from the browser adapter, with available Chromium build metadata. Missing installed
versions are explicit nulls, never inferred from a peer range. An exported source tree without Git
uses a recursive file inventory and records a null revision. The identity is evidence of selected
bytes and resolved versions, not a hash of every installed dependency byte or a signed attestation.
Keep receipts and referenced output logs with the release evidence.

## November historical failure evidence (TOOL-113)

Reviewed on 2026-11-19. Contemporary evidence, a later counterfactual and an unavailable run are
different claims. The November implementation account and changelog do not by themselves establish
these four particular pre-fix failures. The following traces identify what is recoverable; no
historical test or device replay was executed in this R19 stream.

| Case | Parent and exact case | Recoverable execution evidence | Limit and replay command |
| --- | --- | --- | --- |
| Final-flow rerouting | Introduction commit `02c39a00972c25e9affb588a003907ec82c03087`, parent `bdc715ea17b473973af8ece05994a41d276fc86c`; `network-flow-paths.node.spec.ts`, `decomposes the final rerouted flow instead of returning augmentation history` | The retained `tranche-2/state/net-flow-paths.status.md` records initial Node failures for three different cases: diamond paths/counts expected `[2]`, one-augmentation `[1]`, zero-demand `[0]`, received `undefined`. It separately lists reverse cancellation coverage. No exact historical command or rerouting failure excerpt was recovered. | The current fixture cancels edge 2 to zero and expects final paths `[0,3,7]` and `[1,4,6]`, one unit each. Missing-path reds do not establish rejection of augmentation history. A proposed later counterfactual uses `yarn vitest run --project node modules/analysis/test/gpu-network/network-optimization/network-flow-paths.node.spec.ts -t 'decomposes the final rerouted flow'` with a documented history-emitting mutant. Not run; retain its actual future date and mutant identity. |
| Old grid reader versus alpha.16 builder | Integration commit `34179d4c9df2e0e4d3fc4ca28ae626613f8f815f`, parent `0fae223e89f4087a0d3590f60eabe05a68e51b57`; `grid-coordinate-bump.spec.ts`, five reader families × six inputs | `tranche-3/grid-before.log` identifies Vitest 5.0.2, Node project, start 11:57:37; 6 files failed, 9 assertions failed/28 passed. Exact excerpts: flow `CPU reference finds the published builder's point: upload rounds onto edge`: `expected 2 to be 3`; density counterpart: `expected +0 to be 1`. Other failures reject old generated source or concern compile forwarding/a fixture. The original shell command is not in the log. | This is host/source evidence. No old-reader device run of the current 30 cases was recovered. A proposed headless replay command is `yarn vitest run --project headless modules/analysis/test/analysis-core/grid-coordinate-bump.spec.ts`. Keep the corrected probe, alpha.16 builder and old readers: the parent alone also changes upstream pins and is not the specified mixed counterfactual. Not run here. |
| Color-scale counts without colors | Fix `73e8f6382096f77d71c45fa986ff213413ff60df`, parent `4e124a615b6aff39964856a0dfda384dee692fbf`; historical case `GPUColorScale writes subsets of outputs and is bitwise reproducible` | Contemporary commit testimony says unused bindings caused WebGPU validation failure and zero classCounts; it says both new assertions fail without the fix. The diff adds cross-subset comparisons and literal total 2,000. No parent command, device/backend identity or raw failure excerpt was recovered. | Proposed later device counterfactual: retain that commit's strengthened assertions with parent production source, run `yarn vitest run --project headless modules/analysis/test/gpu-dataframe/column-classification/gpu-color-scale.spec.ts -t 'GPUColorScale writes subsets'`. Parent assertions alone compared a run with itself. Current R19 specs have since changed; pin test bytes as well as source. Not run here. |
| Bivariate counts without colors | Same fix and parent; historical case `GPUBivariateClassification without a colors output writes the same class ids and counts` | Same commit testimony, with no located raw execution record. The diff adds the no-colors fixture, compares IDs/counts to the full run and checks a positive full-run count. The case did not exist in the parent. | Proposed later device counterfactual: retain the fix commit's new case with parent production source, run `yarn vitest run --project headless modules/analysis/test/gpu-dataframe/column-classification/gpu-bivariate-classification.spec.ts -t 'without a colors output'`. Pin both trees and save actual date, exact command, execution mode/device and failure output. Not run here. |

Evidence paths above are relative to the retained local directory
`.typecheck/library-review-2026-11-17-evidence/`. Its logs are evidence references, not inputs to
portable tests. Any newly performed counterfactual must be dated when executed; it cannot be
presented as a recovered November 17–18 run. Device replay remains with the lead.


## Oracle census and comparison receipts

Check the generated disk/manifest census without regenerating fixtures:

```sh
node scripts/oracle-census.mjs --check
```

Source hash attribution, configured generator ownership and a completed regeneration comparison are
separate evidence. An exact generator-source hash can identify recorded source bytes; it cannot
establish numerical validity or prove that the configured generator recreated its outputs.
Configured ownership names the declared producer of an output and is not an execution claim. A
differing source hash alone does not establish a stale numerical result.

To retain actual comparison evidence, run the selected generator with a **new** receipt path, then
ask the census to inspect it (repeat `--evidence` for multiple receipts):

```sh
node scripts/run-oracle-fixture.mjs statistics-markov --python /path/to/python3.14 --evidence /tmp/markov-run.json
node scripts/oracle-census.mjs --evidence /tmp/markov-run.json
```

The optional receipt identifies the generator/source digest, run times, declared/recreated outputs
and committed/regenerated byte digests. Incomplete generation exits 3 and records no comparisons;
complete runs record matching or differing comparisons before any `--write` acceptance. Existing
receipt files are never overwritten. Environment/generator failures do not supply comparison
receipts. The census validates receipts and source/output digests rather than treating a configured
manifest entry as a completed run. This guide does not claim corpus regeneration or device replay.
