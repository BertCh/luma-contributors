# GPUAnalysisContiguityStatistics

See also the [analysis experience guide](analysis-experience.md#gpuanalysiscontiguitystatistics).

Composes `GPUContiguityWeights`, `GPUSpatialWeightsSummary` and `GPUColumnQuantiles` for the polygon
part of AX-85. Select `criterion: 'queen'` or `'rook'`; use two instances for both profiles.
Caller-owned outputs include the binary CSR for reuse, one `cardinality` per feature, `counts`,
one float32 `median`, and one uint32 `validity`.

`counts` uses `GPU_ANALYSIS_CONTIGUITY_STATISTICS_COUNTS`: slots, asymmetric slots, isolates,
minimum cardinality, maximum cardinality, empty rows. Median includes islands. An island is a
feature with no active neighbor. Multipart features count once; every ring, including holes,
contributes its vertices and edges. Self-neighbors and duplicate contacts are removed by the
existing contiguity contributor. This is the Queen/Rook feature-key definition, as in PySAL,
not a coverage-validity or connected-components test. Rook requires matching edge endpoints;
an edge subdivided differently on either side is not matched.

The profile is exact for the supplied float32 topology when `validity[0] === 1` and both
`output.weights.status.overflow` and `candidateOverflow` are zero. `requiredCount` is also
required, so a caller can retry with `getWeightsCapacityPlan({requiredCount, candidateOverflow})`.
That method delegates the existing topology grow plan; its memory estimate excludes the
summary and quantile scratch and the published primitives' internal scratch.

`overflow` reports final CSR slot truncation only; `candidateOverflow` independently reports
intermediate pair truncation. With candidate overflow, `requiredCount` is a lower bound on the
full requirement, and `overflow` reports whether those materialized distinct slots exceeded
the final capacity. A zero `overflow` alone cannot establish a complete topology.

If either capacity overflows, the contributor writes `validity = 0`, all defined counts and
cardinalities as `0xffffffff`, and median as NaN. The CSR and its statuses remain available for
recovery. `decodeGPUAnalysisContiguityStatistics` returns `{exact: false, ...}` without any
summary fields in that case, including when a stale validity word incorrectly claims success.
Do not infer islands from truncated CSR rows.

Per-frame inputs are the polygon buffer contents. Criterion, snap tolerance, coordinate extent,
view lengths and capacities require a rebuild. A positive snap tolerance requires a finite
`coordinateExtent`, enforcing the inherited maximum of 2^24 local snap cells. Exactness refers
to these float32 keys, not original float64 coordinates; use the existing
`preparePolygonContiguityTopology` equality-token preparation when source equality matters.
No metric distances are calculated, so no coordinate model conversion or spatial context is
needed. Non-finite vertices follow the existing contiguity rule and are ignored.

At least one feature and one position are required. Empty features and all-island datasets
are supported. The maximum is 2^24 features for the existing quantile rank path; median is
float32, so large half-integer medians can round. All inputs/outputs must be packed, belong
to the target graph and have separate buffers. Scratch is graph-owned; the composition owns
no persistent resources and performs no GPU readback.
Input/output bindings and intermediate pair/vertex scratch are checked before declaring graph
resources, so a rejected oversized pair plan can be retried with the same ID on the same graph.

Node specifications cover construction, capacity recovery, decoding and kernel declaration.
`modules/analysis/test/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.spec.ts`
is the WebGPU numerical gate. Its existence is not evidence that it has run.
