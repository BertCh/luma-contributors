# Contributor contract

Status date: 2026-11-09. The rules every contributor in `@luma-contributors/analysis` follows. They come from the [library review of 2026-11-09](./research/library-review-2026-11-09/README.md); open work to bring existing code up to them is in [roadmap.md](./roadmap.md) under "Contributor contracts and GPU-scale designs" (STD, GPU, DUP and GAP items). `node scripts/check-contributor-contract.mjs` reports departures.

Import the contract from `@luma-contributors/analysis/analysis-core`: the contract types, `validateCapacity`, `validateMaximumCount`, `validateRequirement`, `validateFiniteNumber`, `validateViewLength`, `validateCommandNodeBudget`, `validateCompactOutput`, `validateGPUStorageBindingByteLength`, `createWGSLKernelNode`, `getWGSLKernelBinding`, `createPublishNode`, `createFillNode`, `createWGSLDispatchGate`, `GPUParameterBuffer`.

## 1. Shape

- A contributor is a class that declares `implements GPUCommandNodeProducer`. One class per file; the file is named after the class (`GPUTerrainBreach` in `gpu-terrain-breach.ts`). Props are a `GPU<Name>Props` type; outputs are a `GPU<Name>Output` type.
- A recipe is an `add<Name>Recipe` function that composes contributors through `RecipeBuilder`, takes caller-owned outputs and returns a `GPURecipeResult` with per-stage status. A function made of raw kernels is a contributor and is written as a class.
- Package roots hold only `index.ts`. Helpers shared inside a family live in that family's directory; helpers shared across families live in `analysis-core`.

## 2. Parallel by construction

- No kernel loops over a data-sized range (rows, cells, pairs, points) in a single invocation. Totals, counts, minima, argmax, convergence tests and compaction use the published primitives (`GPUReduction`, `GPUScan`, `GPUCompaction`, `GPUSort`) or the core helpers built on them (`scan-total`, `sorted-segment-sums`, `load-balanced-for-each`).
- A loop inside one invocation may run over a bounded neighbourhood, a capacity, or a small dense system.
- A contributor that cannot meet this rule (an exact exhaustive search, a small dense factorization) is registered with `tier: 'reference'` and a documented maximum size, says so in its class documentation, and enforces the maximum at construction.

## 3. Candidates through an index

- Pair and neighbour candidates come from the published grid index, the BVH, or a sort plus search. Brute force over all pairs is an explicit option with a named cap. It is never the silent default above that cap.

## 4. Iteration with a budget

- Rounds are gated on the GPU (`createWGSLDispatchGate`), so finished rounds dispatch no work.
- A contributor whose command node count grows with a prop computes the count at construction, exposes it as `estimatedCommandNodeCount`, and calls `validateCommandNodeBudget`. When the count includes published primitives that do not expose theirs (`GPUSort`, `GPUScan`, a BVH build), the estimate is an upper bound with a named allowance, and a test checks that the real count does not exceed it. A contributor whose legitimate configurations exceed the shared budget passes its own named maximum with the measured justification (`GPUDelaunayTessellation` does).
- An iteration that stops at its limit publishes what it has and reports `converged = 0` and `iterationLimitReached = 1`. It does not withhold the result.

## 5. Per-frame parameters

- A numeric prop that does not change a buffer shape or a loop bound is read from a parameters view filled by a `getGPU<Name>ParameterValues` packer (hosted in a `GPUParameterBuffer`), so changing it does not recompile.
- Row counts, capacities, slice counts, and which optional views exist are compile-time. The class documentation lists each prop under "per-frame" or "compile-time".

## 6. Declared context

- A contributor that reads coordinates takes `spatialContext` and resolves it with `resolveGPUSpatialContext`. A contributor that only supports one coordinate space says which and rejects the others at construction.
- Raster and terrain contributors take the published `GPURasterBand` and the raster grid, honour `cellSizeMode` wherever a ground distance, slope or area is computed, and read the affine from parameters, not from shader literals.
- Time follows the temporal context (roadmap TC-01).

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
- Float compensated summation does not survive the shader compiler. Kahan's `(next - sum) - term`, Neumaier's `(sum - next) + term` and the float two-sum are algebraically zero, and a compiler that reassociates floating point (Metal fast math) removes them: on the test GPU such a sum of a million terms is bit for bit the plain float32 sum, and a plain running sum is not even the IEEE serial sum (`test/utils/addition-rounding-error-wgsl.spec.ts`). The accepted cures: the fixed-point accumulator or `sorted-segment-sums.ts` for a sum over a data-sized range; for a bounded serial walk (a prefix along one line), `getAdditionRoundingError(first, second, sum)` from `ADDITION_ROUNDING_ERROR_WGSL` in `analysis-core/kernels/addition-rounding-error-wgsl.ts`, which reads the exact error of an f32 addition off the integer significands (finite operands of either sign); or the direct form of the quantity, so that no large sums are differenced.
- The same holds for any float error-free transformation (two-product, `fma` residuals): use the exact integer WGSL, not a float identity the compiler can simplify. Do not expect a GPU float32 loop sum to equal a CPU twin bit for bit unless the kernel fixes the order with integer or bit operations. Fast math is the measured cause of all of this and of the inexact division: the Metal source the browser generates gives the IEEE result when compiled with fast math off, and Chromium 151 has no switch for it (`test/analysis-core/shader-compiler-fast-math.spec.ts`).
- A local `var table: array<u32, N>` or `array<i32, N>` can read back as zeros on the test GPU when its only stores are those of loops with a constant trip count and it is read at an index the compiler cannot resolve. Local means function scope and also `var<private>` at module scope, which the browser emits as the same zero-initialised local. Any stored value is lost (the index, a constant, a sentinel, values copied from a buffer), not only an identity fill. The cause is the Apple GPU back end, which loses the stores of an unrolled constant-count loop (3 to 96 iterations) into a zero-initialised integer array; fast math plays no part (`dev-docs/research/math-correctness-review-2026-11-10/shader-compiler-findings.md`). No size is safe: a kernel that does nothing else loses the table at 33 to about 250 entries, and the Louvain kernel as it was lost a table from 12 entries up. The cures, in this order: give the table a literal initializer or make it a module-scope `const` when its contents are known when the shader is built; otherwise bound the fill loop by a value the compiler cannot know (a kernel without a uniform can write `N * min(arrayLength(&binding), 1u)`, which equals `N`); or do without the table when the index is the value. `var<private>` with a fill loop is not a cure. Read right at every size tried, for a reason that is not known, so do not build new code on it: float and vector tables; a table that later receives a store at a data-dependent index (an insertion, a hash table, a stack); a fill loop that also writes storage or stores under a data condition. Pinned by `test/analysis-core/shader-compiler-table-fill.spec.ts` (plain WebGPU), `test/analysis-core/shader-compiler-table-shapes.spec.ts` (the shapes kernels use, and the old Louvain kernel with its cures), `test/utils/function-scope-table-readback.spec.ts` and `test/gpu-network/network-optimization/gpu-network-louvain-visit-order.spec.ts`. The library audit of 2026-11-11 (the specs in `test/analysis-core/table-audit/`; findings report, "Library audit") ran the kernels that hold such a table and found none that loses it today.
- Orientation and intersection tests on lattice coordinates use the exact WGSL in `analysis-core/kernels`.
- An `atan2` whose second argument can be exactly zero (a negation, difference or product can be -0.0) calls `guardedAtan2` (`GUARDED_ATAN2_WGSL` in `analysis-core/kernels/wgsl-common.ts`), because some GPUs return `-sign(y)·π/2` for `atan2(y, -0.0)` and NaN for `atan2(0, 0)`.
- Do not copy a small dense solver, a float test, a binary search or a distance function into a contributor when `analysis-core` has one. Add the missing one to `analysis-core`.

## 12. Registry

- Every exported contributor has an entry in the operator catalog (`gpu-spatial-analysis/ontology`) with its package, tier, limits, coordinate model and reference.
- A new contributor adds its catalog entry, a docs heading and a kernel-audit case in the same change.

## 13. Public surface

- `analysis-core` is the public home of the contract. Domain entry points export contributors.
- Symbols marked `@internal` are not exported from an entry barrel.
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
- **Executor limits (luma.gl 10.0.0-alpha.12).** In a graph that has a CPU stage or a CPU output: values are `float32`, `sint32` or `uint32` vectors of one to four components, one packed chunk each; no textures; no state vectors; every GPU operation is its own submission. A contributor that needs anything else has no operation form yet.
- **Reference tier.** A contributor registered with `tier: 'reference'` (section 2) should have a twin, and its documentation should name the twin as the path to prefer.
