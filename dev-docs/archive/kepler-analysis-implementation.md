> **Archived 2026-11-07.** Copied from the luma.gl `map-contributors` branch (`dev-docs/roadmaps/kepler-analysis-implementation.md`). Paths such as `modules/graph/src/...` and `@luma.gl/graph/...` refer to that branch, not this repository. Current plans: [roadmap](../roadmap.md). Superseded by the roadmap and the kepler sandbox research; kept for history.

# Kepler analysis implementation record

Date: October 25, 2026. This records the working-tree KG01–KG06 implementation, not a release
or an integrated Kepler frontend. Scope and original acceptance criteria remain in the
[coverage backlog](./geospatial-coverage-backlog.md#kepler-analysis-coverage-2026-10-25);
[function coverage](./geospatial-function-coverage.md#kepler-analysis-coverage-2026-10-25)
tracks the bounded delivered operations. Implementers used GPT-6.1 Sol as requested.

## Delivered operations

| ID | API and contract | Reference and numerical evidence | Remaining boundary |
| --- | --- | --- | --- |
| KG01 | [GPUQuantileLISA](../../docs/api-reference/experimental/gpu-spatial-analysis/quantile-lisa.md): resident selected quantile, type-5 breaks, binary local join counts, conditional permutations, labels and explicit status. | Executed PyGeoDa 0.1.3 fixtures; exact observed counts; independent sorted-boundary and uniform Philox oracles. Pinned libgeoda `71238cb58979032048081434ce6e36a53e254280`. | Single variable only; packed input, at most 2^24 rows and 64 neighbors. Corrects upstream ignored missing masks and biased endpoint sampling. Retains the documented legacy strict-complement directed p-value, which is anti-conservative even for all ties. Cross-library p-values are not same-seed parity. |
| KG02 | [GPUDorlingCartogram](../../docs/api-reference/experimental/gpu-spatial-analysis/gpu-dorling-cartogram.md): area-derived radii, planar positions, source IDs, all-pairs circle relaxation, origin attraction, residual overlap/displacement and GPU convergence/budget status. | GeoDa workbook pinned by SHA256 `dc3f15071e278cc824c4f3de74ad6dbb7463ff11c79cbef76bb9e92f1598f635`; independently derived analytic/invariant fixtures. Genuine GPU capacity test: 1024 rows × 256 unresolved passes. | Bounded heuristic variant, no coordinate parity or guaranteed convergence; larger interacting layouts require another algorithm. Zero-area circles stay stationary; invalid values are excluded. |
| KG03 | [GPUClassBreaks](../../docs/api-reference/experimental/gpu-dataframe-analysis.md) with `naturalBreaksMode: 'exact'`; approximate default preserved. Full dynamic programming, deterministic optimal split ties and lower-inclusive upper-class edges. | Independent exhaustive partition oracle, pinned jenkspy 0.4.1 comparisons and GeoDa assignment source audit. Two cases where approximate refinement misses the optimum; small/large/offset/extreme floats, replay and maximum-capacity GPU execution. A distinguishing 1e37-scaled fixture exposed flushed reciprocal division; bounded mantissa division with frexp/ldexp fixes it. | At most 1024 source rows and 64 classes. Global optimum for the documented normalized float32 interval-cost recurrence, not a float64 or real-arithmetic guarantee. |
| KG04 | [GPUVariableStandardization](../../docs/api-reference/graph/gpu-variable-standardization.md): deviation from mean, sample/population z-score, mean absolute deviation, range adjust and range standardize; shared reductions, masks and mismatched chunk boundaries. | GeoDa `37510860bd9f7b8ae145c15759f7abc5ab217cf3` formula audit and independent numerical oracle; GPU ±3e38 and large-offset fixtures. | Float32 rounding, underflow and unrepresentable-output rules are explicit. Corrects GeoDa range-extrema implementation defects; MAD means mean absolute deviation. |
| KG05 | [addExcessRiskRecipe and addSpatialRateRecipe](../../docs/api-reference/experimental/gpu-spatial-analysis/rate-mapping.md); optional `GPUEmpiricalBayesRates.output.excessRisk`, existing EB outputs preserved. | Same pinned GeoDa `RateSmoothing.cpp` source audit; independent hand-arithmetic GPU fixtures and existing EB regressions. | Binary CSR neighbors, ratio of pooled counts/populations, explicit self-edge/island/mask/scale policies. No direct upstream rate executable was run. Global rates require the complete intended population. |
| KG06 | LuSQLContext query ledger (local research log, not checked in): quoted identifiers and single-key JOIN USING over existing GPU joins; October 26 adds bounded CASE/COALESCE and nullable column counts. | Pinned Kepler/OpenAssistant source generators/prompts; directly executed DuckDB 1.4.1 + PyArrow 22.0.0 Arrow reference. Node/GPU tests for escaping, aliases, projections, anti-join and duplicate-key diagnostics. | Partial. Explicit left projections and existing uint32/unique-right join contract. DOUBLE, WKB, arbitrary text/functions, right projection, UNION and broader composition remain scoped follow-ups. This does not replace DuckDB. |

New contributors have public exports, operator metadata, executable documentation lenses and
kernel-audit coverage. Scratch allocation and graph execution use existing infrastructure;
callers explicitly submit work and read results. No GeoDa/Kepler runtime dependency, CPU fallback,
implicit batch packing or hidden GPU readback was added. Arrow remains an adapter boundary.

## Verification

Focused implementer evidence (actual WebGPU numerical assertions, not presence-only checks):

- Quantile LISA: 2 GPU tests looping over reference and adversarial fixtures; node/audit coverage.
- Dorling cartogram: 6 GPU tests; 18 node/audit tests.
- Exact natural breaks: 3 final GPU tests and 14 node tests, in addition to classification regressions.
- Standardization: 5 GPU tests and 9 node tests.
- Rates: 10 GPU tests across 3 files; 12 node tests across 4 files.
- SQL: 7 GPU and 10 node tests; executed pinned DuckDB reference.
- Shared integration: 52 tests across 5 files, including metadata/lens coverage and kernel audits.

These counts overlap some integrated checks and are not an additive suite total. The final
integration gates and captured lens records are recorded below. Late test-only
quantile/standardization assertions and the cartogram input rename also passed a combined
3-file/13-test WebGPU run.

| Required check | Integration result |
| --- | --- |
| `nvm use` | Passed, Node 22.23.4. |
| `yarn install` | Passed; existing peer-dependency warnings. |
| `yarn lint fix` | Passed after integration fixes; 8068 files, no fixes. |
| `yarn build` | Passed after integration fixes; all modules and CommonJS identity checks. |
| `yarn test-node` | Passed as the full node phase of `yarn test`: 1246 files / 12592 tests; 2 files / 439 tests skipped. |
| `yarn test` | Full command ran node and all 8 headless shards. Node and shards 1–6/8 passed; shard 7 lost its browser connection during a terrain-shadow test, so this command exited 1. The complete shard retry below resolved the incomplete run. |
| `yarn test-headless --shard=7/8` | Passed in a fresh browser: 140 files / 1662 tests; 1 file / 344 tests skipped. No assertion failures or unhandled errors. |
| `yarn website:build` | Passed; static site, search index, llms.txt and 1210 raw documentation pages validated. |
| `(cd website && yarn build)` | Passed with the same documentation validation. |
| Five new executable lens record captures | Passed, 5/5 on WebGPU; records in `docs/lens-records/` for quantile LISA, Dorling, standardization, excess risk and spatial rate. |

Across the eight completed headless shards (using the complete retry for shard 7):
**1106 files and 12616 tests passed; 8 files and 846 tests skipped**. The full node phase
passed independently. The initial browser disconnection is retained above rather than reporting
the original `yarn test` command as exit-zero. Both website commands were also rerun successfully
after the final numeric fix. No performance claim follows from these correctness checks.

## Follow-up work

- Connect these APIs to Kepler through an explicit adapter for datasets, columns, spatial weights,
  parameter changes, diagnostic display and result publication. This implementation does not embed
  Kepler or replace its upload/UI state management.
- Retain KG06.3–KG06.7 in the query ledger (local research log, not checked in).
  Select further SQL work from demonstrated consumer requirements, including precision and ownership.
- Audit multivariate quantile LISA separately; empty joint classes change upstream statistic semantics.
- Benchmark larger consumer datasets before widening capacity or performance claims. The cartogram
  bound and float32 exact-break objective are intentional contracts, not full upstream parity.
