# Contributor contract

Status date: 2026-11-09. The rules every contributor in `@luma-contributors/analysis` follows. They come from the [library review of 2026-11-09](./research/library-review-2026-11-09/README.md); open work to bring existing code up to them is in [roadmap.md](./roadmap.md) section 2.16. `node scripts/check-contributor-contract.mjs` reports departures.

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
- Orientation and intersection tests on lattice coordinates use the exact WGSL in `analysis-core/kernels`.
- Do not copy a small dense solver, a float test, a binary search or a distance function into a contributor when `analysis-core` has one. Add the missing one to `analysis-core`.

## 12. Registry

- Every exported contributor has an entry in the operator catalog (`gpu-spatial-analysis/ontology`) with its package, tier, limits, coordinate model and reference.
- A new contributor adds its catalog entry, a docs heading and a kernel-audit case in the same change.

## 13. Public surface

- `analysis-core` is the public home of the contract. Domain entry points export contributors.
- Symbols marked `@internal` are not exported from an entry barrel.
- Never re-export a luma.gl symbol (AGENTS.md, upstream-first).
