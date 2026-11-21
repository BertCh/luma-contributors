# GPUAnalysisContiguityStatistics

Import from `@luma-contributors/analysis/gpu-spatial-analysis`.

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

Decode each port with its own scalar type. The decoder accepts numeric typed arrays, byte
reads (`Uint8Array` or `ArrayBuffer`, including logical slices), and the producer-shaped nested
status map. It rejects missing/short ports or byte lengths that are not whole 32-bit words;
byte offsets and lengths are respected. For example, these are readbacks of the ports, not
GPU views or a concatenated buffer:

```ts
import {decodeGPUAnalysisContiguityStatistics} from '@luma-contributors/analysis/gpu-spatial-analysis';

function readContiguityProfile(reads: {
  counts: Uint8Array;
  median: Uint8Array;
  validity: Uint8Array;
  overflow: Uint8Array;
  candidateOverflow: Uint8Array;
}) {
  return decodeGPUAnalysisContiguityStatistics({
    counts: {bytes: reads.counts},
    median: {bytes: reads.median},
    validity: {bytes: reads.validity},
    weights: {status: {
      overflow: {bytes: reads.overflow},
      candidateOverflow: {bytes: reads.candidateOverflow}
    }}
  });
}
```

Only `result.exact === true` exposes `slots`, `islands`, `minimum`, `median` and `maximum`.
It requires validity exactly 1 **and** both loss flags zero. An exact result describes the
supplied float32 equality/snap topology under the geometry conditions below, not the source
float64 geometry or geographic coverage validity.

For a hand-counted example, place four unit squares at lower-left coordinates `(0, 0)`,
`(1, 0)`, `(2, 0)` and `(10, 0)`. Rook adjacency is the chain `0—1—2` plus island `3`.
With final slot capacity 4 and pair capacity 64, cardinalities are `[1, 2, 1, 0]`,
`counts = [4, 0, 1, 0, 2, 1]` and median is 1 (the middle of `[0, 1, 1, 2]`).
Final slot capacity 1 with pair capacity 64 reports `[overflow, candidateOverflow] = [1, 0]`;
final slot capacity 4 with pair capacity 1 reports `[0, 1]`. Both lossy profiles publish
`validity = 0`, sentinel counts/cardinalities and NaN median, even if their retained rows
look like plausible islands.

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
