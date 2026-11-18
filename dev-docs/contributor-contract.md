# Contributor contract

Status date: 2026-11-09. The rules every contributor in `@luma-contributors/analysis` follows. They come from the [library review of 2026-11-09](./research/library-review-2026-11-09/README.md); open work to bring existing code up to them is in [roadmap.md](./roadmap.md) under "Contributor contracts and GPU-scale designs" (STD, GPU, DUP and GAP items). `node scripts/check-contributor-contract.mjs` reports departures.

Import the contract from `@luma-contributors/analysis/analysis-core`: the contract types, `validateCapacity`, `validateMaximumCount`, `validateRequirement`, `validateFiniteNumber`, `validateViewLength`, `validateCommandNodeBudget`, `validateCompactOutput`, `validateGPUStorageBindingByteLength`, `createWGSLKernelNode`, `getWGSLKernelBinding`, `createPublishNode`, `createFillNode`, `createWGSLDispatchGate`, `GPUParameterBuffer`.

Checker allowances use a version-2 baseline with one file/site signature, stable description and
nonempty reason per departure. Removing a loop does not permit a replacement loop in that file.
Run `node scripts/check-contributor-contract.mjs --check-stale` for a read-only expired-site audit.
New allowances require review and `--update-baseline --reason '<reviewed justification>'`; strict-zero
rules cannot be baselined. See [script gates](./script-gates.md) for signature stability, migration,
shared typecheck ownership and oracle execution requirements.

## 1. Shape

- A contributor is a class that declares `implements GPUCommandNodeProducer`. One class per file; the file is named after the class (`GPUTerrainBreach` in `gpu-terrain-breach.ts`). Props are a `GPU<Name>Props` type; outputs are a `GPU<Name>Output` type.
- A recipe is an `add<Name>Recipe` function that composes contributors through `RecipeBuilder`, takes caller-owned outputs and returns a `GPURecipeResult` with per-stage status. A function made of raw kernels is a contributor and is written as a class.
- Package roots hold only `index.ts`. Helpers shared inside a family live in that family's directory; helpers shared across families live in `analysis-core`.

The root-file checker derives its directories from every published analysis entry in
`modules/analysis/package.json`, including `analysis-core` and nested entries. Named infrastructure
exceptions are the package forwarding barrel, `computation-record` and
`gpu-spatial-analysis/ontology`; a new family entry is checked automatically. Core device-request
helpers live in `analysis-core/graph-support`, shared network/raster contracts in
`analysis-core/contracts`, and cell/ring/area shader helpers in `analysis-core/kernels`.
Point-cloud contributors and their shared helpers live in `pointcloud-analysis/operations`.
Geospatial implementations live in `geospatial-analysis/datum-grid-transform`, `point-buffer`
and `viewport-bounds`. The entry barrels re-export their family indexes whole; public names are
unchanged.

Export resolution accepts emitted source files under `./dist/`, including non-index entry files
and native ESM/CommonJS suffixes, and expands wildcard targets against concrete source files.
Unsupported targets and patterns with no source matches fail with an explicit diagnostic (exit 2).
An independent immediate-file scan checks `.ts`, `.tsx`, `.mts`, `.cts`, `.js`, `.jsx`, `.mjs` and
`.cjs` implementations in each resolved directory; index barrels and declaration files are exempt.
Infrastructure exceptions apply only to the exact entry, so a published child entry under
`computation-record` is still checked. The other contract rules retain their existing traversal.

Recipe implementation form and result compatibility are separate acceptance checks. Moving raw
kernels into a contributor or adopting `RecipeBuilder` completes only the implementation check.
Before declaring result compatibility complete, verify every overload and optional-stage branch
returns typed `contributors`, `outputs`, `intermediates` and a `status` with named `stages`.
Retain any legacy packed status as an explicitly named view when adapting a result.
`addPopulationSupportAlignmentRecipe`, for example, supplies `intermediates: {}`, named diagnostic
stages under `status.stages`, and its original four words under `packedStatus`. The scalar stage
views alias those words, so callers can inspect the canonical contract without extra GPU commands.

## 2. Parallel by construction

- No kernel loops over a data-sized range (rows, cells, pairs, points) in a single invocation. Totals, counts, minima, argmax, convergence tests and compaction use the published primitives (`GPUReduction`, `GPUScan`, `GPUCompaction`, `GPUSort`) or the core helpers built on them (`scan-total`, `sorted-segment-sums`, `load-balanced-for-each`).
- A loop inside one invocation may run over a bounded neighbourhood, a capacity, or a small dense system.
- A contributor that cannot meet this rule (an exact exhaustive search, a small dense factorization) is registered with `tier: 'reference'` and a documented maximum size, says so in its class documentation, and enforces the maximum at construction.

The `single-invocation-data-loop` checker is a heuristic: its count does not distinguish a small
coefficient solve from a row scan or tile merge. Review the loop axis, its construction-time bound,
and the selected execution route; the [regression loop inventory](./regression-loop-inventory.md)
records examples. The checker follows one local function/arrow layer forwarding invocation count
and shader body, resolving each call's arguments separately. Put a reviewed
`contract-allow: single-invocation` comment immediately before the `invocationCount` property of a
direct kernel, or immediately before a wrapper call, with the axis and enforced bound in the reason.
Multiple invocations can still each perform a serial row scan, so this check is not a complete
inventory of data-dependent work.

## 3. Candidates through an index

- Pair and neighbour candidates come from the published grid index, the BVH, or a sort plus search. Brute force over all pairs is an explicit option with a named cap. It is never the silent default above that cap.

## 4. Iteration with a budget

- Rounds are gated on the GPU (`createWGSLDispatchGate`), so finished rounds dispatch no work.
- A contributor whose command node count grows with a prop computes the count at construction, exposes it as `estimatedCommandNodeCount`, and calls `validateCommandNodeBudget`. When the count includes published primitives that do not expose theirs (`GPUSort`, `GPUScan`, a BVH build), the estimate is an upper bound with a named allowance, and a test checks that the real count does not exceed it. A contributor whose legitimate configurations exceed the shared budget passes its own named maximum with the measured justification (`GPUDelaunayTessellation` does).
- An iteration that stops at its limit publishes what it has and reports `converged = 0` and `iterationLimitReached = 1`. It does not withhold the result.

## 5. Per-frame parameters

- A numeric prop that does not change a buffer shape or a loop bound is read from a parameters view filled by a `getGPU<Name>ParameterValues` packer (hosted in a `GPUParameterBuffer`), so changing it does not recompile.
- Row counts, capacities, slice counts, and which optional views exist are compile-time. The class documentation lists each prop under "per-frame" or "compile-time".
- Snapshot compile-time scalars, options and nested port maps before validating them at construction. Keep the identities of GPU views and buffers so their contents remain per-frame inputs; later mutation of the caller's props must not change validated shapes or bindings. The chunk-contributor authoring fixture in `modules/analysis/test/analysis-core/graph-chunk-dispatch.test-utils.ts` demonstrates this for options.

## 6. Declared context

- A contributor that reads coordinates takes `spatialContext` and resolves it with `resolveGPUSpatialContext`. A contributor that only supports one coordinate space says which and rejects the others at construction.
- Raster and terrain contributors take the published `GPURasterBand` and the raster grid, honour `cellSizeMode` wherever a ground distance, slope or area is computed, and read the affine from parameters, not from shader literals.
- Time follows `GPUTemporalContext`, resolved with `resolveGPUTemporalContext` and a
  `GPUTemporalContextSupport` declaration. The [temporal-context guide](../docs/api/gpu-spatial-analysis/temporal-context.md#resolving-and-checking)
  describes the supported encodings, units and axis checks.

## 7. One no-data rule

- An input row filter is `mask` (`uint32`, nonzero keeps the row). An output flag is `validity` (`uint32`, nonzero is valid).
- Rasters carry no-data in the band's validity. NaN marks a missing float value inside a kernel; it is not the only way a caller can mark no-data.
- Missing keys and categories are `0xffffffff`.

## 8. One output layout

- Variable-length output is a `GPUCompactOutput`: `count` is clamped to capacity, `status.requiredCount` holds the unclamped requirement, `status.overflow` is one when rows were dropped.
- `count` is the number of output rows of the primary output (features for geometry, pairs for joins).
- Geometry output carries the source row of each output feature. Output derived from two inputs carries both.

## 9. Named status

- Scalar status uses the port types in `analysis-core/contracts/status.ts`. Use the existing field names; do not invent synonyms (`candidateCount`, `truncated`, `complete`).
- An intermediate capacity that was too small is `candidateOverflow`. It is not folded into `overflow`.
- A consumer of a supplied incomplete-result status propagates it in its own status, suppresses dependent results, or refuses/falls back; it never ignores the flag. Compact `requiredCount` counts accepted rows reaching the final stage, before output truncation; it is a lower bound after candidate loss or uint32 saturation.
- A deliberate approximation sets `approximation`.
- Per-row status words document every bit in the output type.
- Status is never read back inside a contributor.

## 10. Named limits

- Construction-time checks throw an `Error` whose message starts with the contributor ID and names the prop and the accepted range. Use `validateCapacity`, `validateMaximumCount`, `validateFiniteNumber`, `validateViewLength`, `validateRequirement`. Do not use bare `assert`.
- Limits are named constants with a comment that says why the limit exists (a binding size, exact float32 integers, a per-thread array, a reference-tier algorithm).
- Row-proportional scratch is checked against the storage binding size with `validateGPUStorageBindingByteLength` or `getStorageBindingElementLimit`. These need the device, so they run at the start of `getCommandNodes`, before any node is created; every other check runs in the constructor.
- Inputs are single packed views unless the class documents chunked input.

## 11. Shared numerics

- Random numbers come from the Philox stream with a registered purpose.
- Sums that must not depend on thread order use the fixed-point accumulator or a fixed-order tree.
- Do not rely on float compensated summation or float error-free transformations (`two-sum`,
  `two-product`, `fma` residuals): shader compilers can reassociate away the correction. Use the
  fixed-point accumulator or `sorted-segment-sums.ts` for data-sized sums. For a bounded serial
  walk, use `getAdditionRoundingError(first, second, sum)` from `ADDITION_ROUNDING_ERROR_WGSL`
  in `analysis-core/kernels/addition-rounding-error-wgsl.ts`, or a direct formula that avoids
  differencing large sums. Do not promise bitwise CPU/GPU loop-sum parity without a fixed order
  enforced by integer or bit operations.
- Do not fill a local integer lookup table solely with a constant-trip-count loop before a
  dynamic-index read. This includes module-scope `var<private>` arrays. Prefer a literal initializer
  or module-scope `const`; otherwise use a runtime-bounded fill, or eliminate the table when the
  index is the value. Do not assume another array size or a float/vector table is a general cure.
  The compiler investigations, reproductions and measured exceptions are retained in
  [shader-compiler-findings.md](./research/math-correctness-review-2026-11-10/shader-compiler-findings.md);
  [the workaround register](./upstream-workarounds.md#6-shader-compiler-behaviour-that-kernels-work-around)
  names the focused regression specs to rerun after browser, OS or luma.gl upgrades.
- Orientation and intersection tests on lattice coordinates use the exact WGSL in `analysis-core/kernels`.
- An `atan2` whose second argument can be exactly zero (a negation, difference or product can be -0.0) calls `guardedAtan2` (`GUARDED_ATAN2_WGSL` in `analysis-core/kernels/wgsl-common.ts`), because some GPUs return `-sign(y)·π/2` for `atan2(y, -0.0)` and NaN for `atan2(0, 0)`.
- Do not copy a small dense solver, a float test, a binary search or a distance function into a contributor when `analysis-core` has one. Add the missing one to `analysis-core`.

## 12. Registry

- Every exported contributor has an entry in the operator catalog (`gpu-spatial-analysis/ontology`) with its package, tier, limits, coordinate model and reference.
- Catalog registration identifies both the public package entry point and export name, including barrel aliases. Same-name exports in different entry points require separate records.
- A new contributor adds its catalog entry, a docs heading and a kernel-audit case in the same change.

## 13. Public surface

- `analysis-core` is the public home of the contract. Domain entry points export contributors.
- Symbols marked `@internal` are not exported from an entry barrel.
- The `internal-export` checker also follows public props aliases, inheritance and nested object
  members, public instance members of class-typed props, callback return types, and visible constructor
  and method signatures. Static members are excluded when following an instance type. Imported
  aliases, namespace imports,
  generic arguments and literal-key `Pick`/`Omit` are resolved without typechecking. Private and
  protected members and implementation signatures hidden by overloads are excluded. Keep composition
  switches in non-barrel factories; an `@internal` comment on a public prop does not hide it.
- Never re-export a luma.gl symbol (AGENTS.md, upstream-first).

## 14. Operation form and CPU twins

Contributors can run in a published luma.gl `GPUGraph` beside CPU stages (`GPUCPUOperation`), and a contributor can have a CPU twin that takes its place in the graph. The survey behind this section, the measurements and the list of candidates are in [research/cpu-suitability-2026-11-11](./research/cpu-suitability-2026-11-11/README.md); open work is in the roadmap (CPU items). Import `GPUContributorOperation`, `GPU_CONTRIBUTOR_OPERATION_LOWERING` and the `getCPUOperation<Type>Rows` helpers from `@luma-contributors/analysis/analysis-core`.

- **Operation form.** `GPU<Name>Operation extends GPUContributorOperation`, in `gpu-<name>-operation.ts` beside the contributor. Where `GPU<Name>Operation` is already an exported name of something else, the class is `GPU<Name>GraphOperation` (`GPULocationAllocationGraphOperation`). Its props type is `<Name>OperationProps`: the contributor's props with the name of a graph value wherever the contributor takes a view. The class declares the ports and builds the contributor; it adds no behaviour and no validation of its own. Compile the graph with `GPU_CONTRIBUTOR_OPERATION_LOWERING`. When the contributor still sits in a package root, the two files go in the family directory that fits it (section 1: a root holds only `index.ts`). The operation keeps a contributor that owns GPU resources and destroys it when the next run replaces it; call the operation's `destroy()` after the compiled graph is destroyed.
- **CPU twin.** `CPU<Name>Operation extends GPUCPUOperation`, in `cpu-<name>-operation.ts`, built from the same `<Name>OperationProps`. It reads and writes the same graph values as the operation form: the same formats, lengths and layouts, including status words, counts and missing-value codes. A props object that builds one builds the other.
- **The computation is a function.** The twin's `execute` calls an exported pure function `compute<Name>OnCPU` in the same file, which takes and returns typed arrays and can be used without a graph. An existing public CPU function (such as `fitVariogramModel`) is called, not copied.
- **When to write a twin.** When the contributor is serial by nature, capped at a few thousand rows, unrolls rounds into command nodes, or needs precision the shader lacks, and its inputs and outputs are plain vectors. Not for raster or texture work, and not for large data-parallel work: a CPU stage pays a readback and an upload (measured on the test GPU through luma.gl 10.0.0-alpha.12: 0.7 ms for 1,000 rows of `float32x2`, 11 ms for 100,000, 90 ms for one million).
- **Float64 inside, one rounding at the output.** The twin computes in float64 and rounds to the output format once. It keeps the output contract of sections 7 to 9: `count` clamped to the output length, `requiredCount`, `overflow`, `validity`, `0xffffffff` for a missing key.
- **Limits and input checks.** A twin validates its props at construction with the helpers of section 10. `compute<Name>OnCPU` also checks the arrays it is given as it reads them (lengths that must agree, offsets that must be nondecreasing and in range) and throws an `Error` that names the argument. This is not the per-row CPU validation that AGENTS.md rules out for contributors: the rows are already on the CPU. Where the GPU contributor defines a result for such input (a clamp, a status), the twin returns that result instead of throwing. It keeps a limit of the contributor only where the output layout needs it; a limit that existed for the GPU's sake (a per-thread array, a command node budget) is dropped, and the class documentation says so.
- **Differences are documented.** The documentation of `compute<Name>OnCPU` has a list headed "What differs from the GPU contributor": tolerance, tie-breaking, statuses the twin never sets, inputs the GPU refuses and the twin accepts. An empty list is stated as "Nothing".
- **Two specs.** A `.node.spec.ts` checks `compute<Name>OnCPU` against an oracle that is not the same code: a committed fixture, a closed form, a hand-computed case or a different algorithm. A test reference that was promoted into the twin no longer counts as its oracle. A headless spec runs both operations on the same inputs through `runOperationGraph` (`test/utils/mixed-execution-test-utils.ts`) and compares every output with a stated tolerance, on ordinary input and on one edge case (empty, overflow or invalid input).
- **Current execution support.** The workspace integration target is luma.gl **10.0.0-alpha.14**;
  see [the package contract](./release-package.md#tested-dependencies-and-accepted-peers) for the
  accepted peer range. Mixed execution and the local adapters have different boundaries:

  | Layer | Supported values and layout | Copies and submission boundaries |
  | --- | --- | --- |
  | Published mixed CPU/GPU executor | Numeric `float32`, `sint32` and `uint32` vectors with one to four components, including multiple chunks and declared `chunkLengths`; scalar inputs are supported. CPU callback results may contain multiple chunks matching the declared shape. Textures, state vectors, list offsets and validity bitmaps are unsupported. | GPU-to-CPU transitions read back; CPU-to-GPU transitions upload. CPU-only stages also materialize callback chunk snapshots from caller-owned CPU buffers; one packed CPU input therefore still incurs this executor copy. Each GPU operation is a separate stage/submission. |
  | Local `GPU_CONTRIBUTOR_OPERATION_LOWERING` | Every resolved contributor port must be exactly one packed chunk (`gpu-contributor-operation.ts`). This adapter restriction applies even when the underlying contributor accepts chunked views. | Rejects multiple chunks instead of packing them. A chunk-capable contributor can still be used directly through `getCommandNodes(graph)`. |
  | Local `getCPUOperation<Type>Rows` helpers | Numeric vector inputs of the requested format; scalars are rejected by these row helpers. | Borrows the callback snapshot for one chunk; calls `input.pack()` for multiple chunks, allocating an additional packed copy after the executor's chunk materialization. Borrowing here does not promise the caller's original buffer identity. Treat the returned rows as read-only. |

  These are source-verified boundaries, not a new GPU measurement. The transfer timings above remain
  dated measurements of the older release, not performance claims for the integration target.
- **Reference tier.** A contributor registered with `tier: 'reference'` (section 2) should have a twin, and its documentation should name the twin as the path to prefer.
