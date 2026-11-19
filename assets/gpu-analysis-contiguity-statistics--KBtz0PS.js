var e={slug:`api/gpu-spatial-analysis/gpu-analysis-contiguity-statistics`,title:`GPUAnalysisContiguityStatistics`,html:`<h1 id="gpuanalysiscontiguitystatistics">GPUAnalysisContiguityStatistics</h1>
<p>See also the <a href="/luma-contributors/docs/api/gpu-spatial-analysis/analysis-experience#gpuanalysiscontiguitystatistics">analysis experience guide</a>.</p>
<p>Composes <code>GPUContiguityWeights</code>, <code>GPUSpatialWeightsSummary</code> and <code>GPUColumnQuantiles</code> for the polygon
part of AX-85. Select <code>criterion: 'queen'</code> or <code>'rook'</code>; use two instances for both profiles.
Caller-owned outputs include the binary CSR for reuse, one <code>cardinality</code> per feature, <code>counts</code>,
one float32 <code>median</code>, and one uint32 <code>validity</code>.</p>
<p><code>counts</code> uses <code>GPU_ANALYSIS_CONTIGUITY_STATISTICS_COUNTS</code>: slots, asymmetric slots, isolates,
minimum cardinality, maximum cardinality, empty rows. Median includes islands. An island is a
feature with no active neighbor. Multipart features count once; every ring, including holes,
contributes its vertices and edges. Self-neighbors and duplicate contacts are removed by the
existing contiguity contributor. This is the Queen/Rook feature-key definition, as in PySAL,
not a coverage-validity or connected-components test. Rook requires matching edge endpoints;
an edge subdivided differently on either side is not matched.</p>
<p>The profile is exact for the supplied float32 topology when <code>validity[0] === 1</code> and both
<code>output.weights.status.overflow</code> and <code>candidateOverflow</code> are zero. <code>requiredCount</code> is also
required, so a caller can retry with <code>getWeightsCapacityPlan({requiredCount, candidateOverflow})</code>.
That method delegates the existing topology grow plan; its memory estimate excludes the
summary and quantile scratch and the published primitives' internal scratch.</p>
<p><code>overflow</code> reports final CSR slot truncation only; <code>candidateOverflow</code> independently reports
intermediate pair truncation. With candidate overflow, <code>requiredCount</code> is a lower bound on the
full requirement, and <code>overflow</code> reports whether those materialized distinct slots exceeded
the final capacity. A zero <code>overflow</code> alone cannot establish a complete topology.</p>
<p>If either capacity overflows, the contributor writes <code>validity = 0</code>, all defined counts and
cardinalities as <code>0xffffffff</code>, and median as NaN. The CSR and its statuses remain available for
recovery. <code>decodeGPUAnalysisContiguityStatistics</code> returns <code>{exact: false, ...}</code> without any
summary fields in that case, including when a stale validity word incorrectly claims success.
Do not infer islands from truncated CSR rows.</p>
<p>Per-frame inputs are the polygon buffer contents. Criterion, snap tolerance, coordinate extent,
view lengths and capacities require a rebuild. A positive snap tolerance requires a finite
<code>coordinateExtent</code>, enforcing the inherited maximum of 2^24 local snap cells. Exactness refers
to these float32 keys, not original float64 coordinates; use the existing
<code>preparePolygonContiguityTopology</code> equality-token preparation when source equality matters.
No metric distances are calculated, so no coordinate model conversion or spatial context is
needed. Non-finite vertices follow the existing contiguity rule and are ignored.</p>
<p>At least one feature and one position are required. Empty features and all-island datasets
are supported. The maximum is 2^24 features for the existing quantile rank path; median is
float32, so large half-integer medians can round. All inputs/outputs must be packed, belong
to the target graph and have separate buffers. Scratch is graph-owned; the composition owns
no persistent resources and performs no GPU readback.
Input/output bindings and intermediate pair/vertex scratch are checked before declaring graph
resources, so a rejected oversized pair plan can be retried with the same ID on the same graph.</p>
<p>Node specifications cover construction, capacity recovery, decoding and kernel declaration.
<code>modules/analysis/test/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics.spec.ts</code>
is the WebGPU numerical gate. Its existence is not evidence that it has run.</p>`};export{e as default};