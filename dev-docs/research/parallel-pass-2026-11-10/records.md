# Stream `records`: LN-REC (roadmap 2.12), unblocks part of WEB-HIW-01 (9.6)

## Design (ten lines)

1. No `recorder?` prop on contributors. The one capture point is `createWGSLKernelNode`.
2. `recordWGSLKernelNodes(graph)` (analysis-core) opts in one graph. `recorder.recordKernels(graph)` does the same and remembers the graph.
3. While a graph is opted in, the helper stores `{props, source, workgroupSize, layout}` under the node id in a module `WeakMap<graph, Map<nodeId, entry>>`. The entry only references things the node's compile and encode closures already hold.
4. A graph that is not opted in costs one `WeakMap.get` per kernel. Nothing is stored, no strings are built, there is no GPU work.
5. Entries are keyed by node id, not by node object. The compiled graph publishes only ids, and `graph.add` / `addGPUCommandNode` copy node objects (`{...node}`), so node identity is lost.
6. `getWGSLKernelNodeRecord(graph, id)` is the public, documented accessor. `describeGPUCommandGraph(compiled, {graph | recorder, sourceText})` turns each entry into a kernel annotation: op, `wgsl:<hash>` source with moduleId `<operation>/<variant>`, `bodyHash`, entry point, optional text, `workgroupSize`, `dispatch` (with `indirect` for GPU-gated nodes), per-node `constants`, and buffer and texture `bindings`.
7. Resources bound by those kernels get their real ownership (`transient` or `imported`). The graph gets the published transient allocation totals as `attrs`.
8. Explicit `recordNode` annotations win field by field. Formulas, `derivedFrom`, loop coordinates and datum stay at pack level.
9. Timings: `submitGPUCommandGraphWithTimings` sets the public `timeProfilingQuerySet` when the device has `timestamp-query`. The lens `readLensRunMetrics` feeds the published report to `createGPUComputationMetrics`. With no timestamps the node table is empty, never zero.
10. The lens graphs opt in in their constructors (`LensCommandGraph`, `NetworkLensGraph`, `ElementLensGraph`). Every lens `describeGPUCommandGraph` call passes `graph` and `sourceText: LENS_RECORD_SOURCE_TEXT` (true).

## 1. Done

- `modules/analysis/src/analysis-core/kernels/wgsl-kernel-nodes.ts`: additive only, in my zone. One line before `return node` in `createWGSLKernelNode`, then directly after it: the `WGSLKernelNodeRecord` type, the private table, and `recordWGSLKernelNodes`, `isRecordingWGSLKernelNodes` and `getWGSLKernelNodeRecord`.
- `modules/analysis/src/analysis-core/index.ts` (barrel, small edit): exports those three functions, plus the types `WGSLKernelNodeRecord`, `WGSLKernelTextureBinding` and `WGSLKernelTextureSampleType`.
- `computation-record/describe-command-graph.ts`:
  - new options `graph` and `sourceText`;
  - kernel facts merged under recorder annotations;
  - ownership taken from bound handles;
  - graph `attrs` with transient allocation totals, only when a graph is given;
  - new export `createWGSLKernelNodeAnnotation`.
- `computation-record/computation-recorder.ts`: `recordKernels(graph)` and the `graph` getter.
- `computation-record/submit-timed-command-graph.ts` (new): `submitGPUCommandGraphWithTimings` and the `GPUCommandGraphTimedSubmission` type. Both are exported from `index.ts`.
- `computation-record/computation-record-types.ts`: optional `constants` on `GPUComputationNode` and on `GPUComputationNodeAnnotation`. Overrides differ per pipeline while the source ref is shared by text hash, so they live on the node.
- `computation-record/validate-computation-record.ts`: optional checks for `constants` (finite numbers), `workgroupSize` (3 positive integers) and `dispatch.workgroups` (3 non-negative integers). Records without these fields still validate.
- `modules/lenses/src/computation-lenses/lens-view-snapshot.ts`:
  - `LensCommandGraph` constructor opts in;
  - `LENS_RECORD_SOURCE_TEXT`;
  - `submitTimedLensGraph`;
  - `readLensRunMetrics`.
- Lens harnesses:
  - Timed and with kernel facts: `raster-lens-harness.ts` (raster and terrain), `network-element-lens.ts`, `network-reachability.lens.ts`, `network-lens-support.ts` (graph opt-in), and `spatial-scenario-utils.ts`. `runSpatialScenario` now also returns a `metrics` sidecar; before, it had none.
  - Kernel facts only: `dataframe-lens-support.ts` and `s2-lens-run.ts`.
- Specs:
  - New `modules/analysis/test/computation-record/wgsl-kernel-records.node.spec.ts` (4 tests): a non-recording graph keeps nothing; a recording graph gets full facts and ownership and validates; the recorder path, where annotations win; validation accept and reject cases.
  - New headless `modules/lenses/test/computation-lenses/kernel-records.spec.ts` (8 tests). It runs one real scenario per family plus a point-cloud graph. Every compute node must carry text, bindings, workgroup size and dispatch, or match `KERNELS_WITHOUT_FACTS`, which is the list of kernels still lacking facts, with reasons. It also checks timings present with timestamps, and absent (empty node table) with `timestamps: false`.
  - Network `{structure,construction,search}-scenarios.node.spec.ts`: the dropped `wgsl:` source and binding assertions are restored on the null device.
- Docs: `docs/api/computation-record.md` gains a "Kernel facts without a recorder prop" section and a paragraph on timed submission. `dev-docs/upstream-first-recording.md` gains a section with this decision.

### What a record holds now, against before (real scenarios, headless Chromium WebGPU with timestamp-query)

Before: none of the 410 committed `docs/lens-records` has a source, a binding, a workgroupSize or a timed node (checked by grep: 0, 0, 0, 0).

| Family | Scenario | Compute nodes | With source text and bindings | Distinct sources | Text chars | Timed nodes | File bytes before → now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spatial | local-moran | 175 | 32 | 30 | 66,800 | 175 | 268,208 → 342,438 |
| network | network-centrality | 39 | 39 | 8 | 18,831 | 39 | 74,354 → 100,008 |
| raster | raster-rank-filter | 2 | 1 | 1 | 2,147 | 2 | 3,851 → 7,323 |
| terrain | geomorphons | 2 | 1 | 1 | 8,772 | 2 | 49,312 → 59,853 |
| dataframe | class-breaks | 16 | 16 | 15 | 65,144 | not wired | 13,236 → 42,370 |
| point cloud | (no lens; graph of Neighborhoods, Features, StatisticalFilter) | 14 | 8 | 8 | 21,934 | 14 | n/a |

Kernels still without facts are all nodes of published `@luma.gl/graph` primitives:

- `GPUScan`;
- `GPUSort` radix passes, which carry no operation name;
- `GPUReduction` stages;
- `GPUGroupAggregation`;
- the `GPURasterNeighborhood` / `GPURasterTextureToBuffer` band canonicalization in `raster-band-utils.ts`.

In local-moran, 143 of the 175 kernels are GPUSort and GPUScan nodes inside contiguity weights.

### WEB-HIW-01 claims

- **Durations: possible.** Raster, terrain, network and spatial lens runs write per-node `gpuBegin` and `gpuEnd` when the adapter has timestamps. Dataframe and S2 records are not timed yet.
- **Shader source from a record: possible** for every helper-built kernel, and in every family. It is not possible for published-primitive nodes.
- **Shared allocations: partly.**
  - Real per-resource `transient` / `imported` ownership is now set for every resource bound by a helper kernel. This replaces the blanket `imported` at `describe-command-graph.ts:191`.
  - Graph-level totals are recorded: physical against logical transient counts and bytes, and reused bytes.
  - Which transients share one physical buffer (`allocation`) is **not** possible: published luma.gl does not expose its allocation plan.

## 2. Verified

- `npx vitest run --project node modules/analysis/test/computation-record/ modules/lenses/test/computation-lenses/gpu-network/`: 11 files, 209 tests passed.
- `npx vitest run --project node modules/lenses/test/computation-lenses modules/lenses/test/computation-views`: 73 files, 756 passed, 374 skipped.
- `npx vitest run --project node modules/lenses/test/computation-docs` plus record-budget, record-structure, record-encoding and define-lens-scenarios: 6 of 7 files passed. lens-twins passed.
  - **lens-coverage failed 2 tests** on `addRasterResistanceConnectivityRecipe` (missing operator, formula, scenario and marker). That recipe was added by another stream; it is not my change.
- `npx vitest run --project headless modules/lenses/test/computation-lenses/kernel-records.spec.ts`: 8 of 8 passed.
- Headless: record-structure, remote-sensing, network-movement-coverage and statistics-coverage gave 4 files, 75 passed, 372 skipped.
- Headless: global-coordinates, interpolation-coverage, geometry-refinement, line-length-per-polygon and urban-ecology gave 5 files, 40 passed.
- Headless: coverage-tranche-one, terrain-processing, point-process-lenses and nested-interpolation-validation gave 4 files, 30 passed.
- Headless: `modules/analysis/test/computation-record/computation-record-capture.spec.ts` gave 1 of 1 passed.
- Node: analysis-core layering, runtime-boundaries and kernel-audit-dataframe passed.
  - **analysis-vocabulary failed** on six `status` names in gpu-terrain, OPTICS and network kernel density. Those are other streams' files.
- `node scripts/check-upstream-first.mjs`: no violations.
- `node scripts/typecheck-status.mjs 'computation-record|wgsl-kernel-nodes|analysis-core/index|modules/lenses'`: 0 matching errors. Test files are not covered by the watcher; `yarn typecheck-tests` was not run.
- `npx biome check --write` on my files: clean apart from `noExplicitAny` warnings on `GPUCommandGraph<any>`, the same pattern the recorder already uses.
  - `wgsl-kernel-nodes.ts` has one formatter error at its import block. It comes from the core-wgsl stream's `assert` removal; I did not reformat that shared file.
- **Rule breach:** I once ran `npx tsc --noEmit -p tsconfig.json` over the repository by mistake. I stopped it within seconds (TaskStop), no process remains, and it wrote nothing (`--noEmit`).

## 3. Not done

- Timings for dataframe and S2 lens records: their harnesses build records in two places. The pattern is the same: `submitTimedLensGraph` plus `readLensRunMetrics`. Left for LN-REC2.
- Formulas, loop provenance, `derivedFrom`: stay at pack level, as by default.
  - The one cheap generic source found is `createWGSLDispatchGate`, a shared helper for unrolled rounds. It knows neither a loop id nor an iteration. An optional `loop` prop there would need adoption family by family. This is not a per-contributor prop on 430 classes, so I did not do it in this wave.
  - Gated rounds already show up as `dispatch.indirect: true`.
- Kernels of published luma.gl primitives: they need a luma.gl hook (see 4).
- Records and website JSON not regenerated, as instructed.

## 4. Wave 2 requests

- **LN-REC2, regenerate records:**
  - Records: `VITE_CAPTURE_LENS_RECORDS=1 npx vitest run --project headless modules/lenses/test/computation-lenses/capture-records.spec.ts`
  - Then twins: `UPDATE_LENS_TWINS=1 npx vitest run --project node modules/lenses/test/computation-docs/lens-twins.node.spec.ts`
  - Then website view models: `node --experimental-strip-types src/lens/generate-view-models.ts` (from `website/`), and the inventory through `website/scripts/extract-inventory.ts`.
  - What changes:
    - every record gains `refs.sources` with text (`encoding.sourceLines`), plus `bindings`, `workgroupSize`, `dispatch` and `constants` on helper-built nodes;
    - kernel nodes may now carry an `op` they lacked;
    - resource ownership flips from `imported` to `transient` for graph-owned scratch;
    - compiled graphs gain allocation `attrs`;
    - spatial records gain a `metrics` sidecar;
    - raster, terrain, network and spatial metrics gain per-node GPU begin and end, with `timestamped: true`. Timestamps disable coalescing, so `coalesced` becomes false where it was true.
  - Measured growth is 1.3x to 3.2x per record. Run `record-budget.node.spec.ts`: the largest records (389 KB now) may pass the 512 KB per-record budget. If they do, set `LENS_RECORD_SOURCE_TEXT` false for those, or accept the overage in `OVER_BUDGET` with a reason. The 64 MB total needs watching too.
- The `build` must happen before the website sees `getWGSLKernelNodeRecord` and the new describe options (dist is stale).
- No renames.
- **luma.gl (upstream proposal; not sent):**
  - publish a per-node shader description (source, entry point, workgroup size, bindings) on `compiled.preflight.nodes[]` or on command nodes, to cover GPUScan, GPUSort, GPUReduction, GPUGroupAggregation and the raster primitives;
  - publish the transient allocation plan (logical resource to physical allocation) for `allocation` and `allocationBytes`;
  - give GPUSort radix passes a `workload.operation`.

## 5. Roadmap

- **LN-REC (2.12):** replace the row with: "Done 2026-11-10 through `createWGSLKernelNode` (no per-contributor prop; see upstream-first-recording.md). Remaining:
  - (a) LN-REC2: regenerate records and twins, and check the record budget;
  - (b) timings in the dataframe and S2 harnesses;
  - (c) kernels of published luma.gl primitives need the luma.gl per-node shader description;
  - (d) formula, loop and `derivedFrom` stay pack-level; optional `loop` on `createWGSLDispatchGate`."
- **WEB-HIW-01 (9.6):** "Durations and shader source are available in records after LN-REC2 (dataframe and S2 are untimed). Per-resource shared allocation needs luma.gl to publish its allocation plan; ownership and graph totals are recorded."
- **WEB-HIW-04:** "GPUSort, GPUReduction and GPUScan kernels still carry no operator name; GPUGridIndex is unchecked."

## 6. Noticed

- `lens-coverage.node.spec.ts` fails on `addRasterResistanceConnectivityRecipe`, which has no lens slot (another stream's new recipe).
- `analysis-vocabulary.node.spec.ts` fails on six non-canonical `status` names (GPUTerrainDepressionHierarchy, GPUTerrainSpikeRepair, GPUOPTICS, GPUNetworkKernelDensity).
- `wgsl-kernel-nodes.ts` imports are not biome-formatted after the `assert` removal (core-wgsl).
- The structure specs' comment said three lens specs already used the recorder. They did not: they had no recorder at all, only a comment about dropped coverage.
