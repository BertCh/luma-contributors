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
