var e={slug:`api/gpu-spatial-analysis/gpu-dorling-cartogram`,title:`GPUDorlingCartogram`,html:`<h1 id="gpudorlingcartogram">GPUDorlingCartogram</h1>
<p>{/* lens:GPUDorlingCartogram */}</p>
<p>::lens{scenario=gpu-spatial-analysis.dorling-cartogram rung=graph}</p>
<p>Constructs a bounded planar circle cartogram entirely in a caller-encoded GPU graph. Input
<code>positions</code> are initial centers in equal units on both axes; project geographic coordinates first.
<code>values</code> determine circle area: <code>pi * radius² = value * areaScale</code> (<code>areaScale</code> defaults to 1).
Rows and source IDs are preserved. Optional <code>sourceIds</code> default to zero-based input rows.</p>
<p>Each synchronous iteration sums half-penetration forces over exact spatial neighbors (all other
circles), averages by overlapping neighbor count, and attracts the circle toward its original
center by <code>attraction * (origin - center)</code>. <code>attraction</code> defaults to 0.01 and accepts 0..1.
Coincident circles separate along the x axis using row order. This is a deterministic, geography
anchored Dorling variant; it does not reproduce GeoDa's neighborhood forces or coordinates.
Attraction and competing forces may leave unresolved overlap. No convergence guarantee is made.</p>
<p><code>maximumIterations</code> is a static budget from 0 to 256 (default 100), with at most 1024 input rows. GPU
checks after initialization and every pass freeze the centers when maximum penetration is at
most <code>overlapTolerance</code> (default 0.001 position units). A stationary overlapping layout remains
unconverged. There is no CPU iteration decision, submission, readback, or fallback.</p>
<p>Caller-owned <code>output</code> views are:</p>
<table>
<thead>
<tr>
<th>View</th>
<th>Format and length</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>positions</code></td>
<td>float32x2, input rows</td>
<td>Final centers</td>
</tr>
<tr>
<td><code>radii</code></td>
<td>float32, input rows</td>
<td>Area-derived radii</td>
</tr>
<tr>
<td><code>sourceIds</code></td>
<td>uint32, input rows</td>
<td>Input IDs, including excluded rows</td>
</tr>
<tr>
<td><code>metrics</code></td>
<td>float32x2, one row</td>
<td>Maximum residual penetration; maximum distance from original center</td>
</tr>
<tr>
<td><code>status</code></td>
<td>uint32, four words</td>
<td>Iterations used; converged flag; budget exhausted flag; excluded rows</td>
</tr>
</tbody>
</table>
<p>Zero values produce stationary zero-radius points that exert no force. Negative or nonfinite
values, nonfinite coordinates, coordinates outside ±1e15, and derived radii above 1e15 are
excluded and produce NaN centers/radii. A zero area scale produces zero radii. Empty/all-excluded
inputs converge vacuously with zero metrics and zero iterations. Nonempty inputs require packed
views, matching row counts, and separate output buffers. Borrowed input ownership is preserved;
chunked vectors require explicit caller packing. Float32 precision can prevent small corrections
at large coordinate magnitudes; inspect residuals and subtract a common coordinate origin.</p>
<p>Each pass has O(n²) exact pair queries distributed over rows; the scalar diagnostic reduction is
O(n). Temporary storage is O(n). Kernels use at most seven storage buffers, within default WebGPU
limits. Larger interacting layouts require a future indexed implementation. Independent partitions
do not resolve overlaps across partition boundaries and are not equivalent to one global layout.</p>
<p>The pinned behavioral reference is Luc Anselin's <em>Exploring Spatial Data with GeoDa: A Workbook</em>,
2005, section 11.3.2, printed pages 83–85,
<a href="https://geodacenter.github.io/docs/geodaworkbook.pdf">GeoDa workbook</a>, SHA256
<code>dc3f15071e278cc824c4f3de74ad6dbb7463ff11c79cbef76bb9e92f1598f635</code> (retrieved 2026-10-25).
It describes iterative nonlinear placement and optional additional improvement iterations. This
implementation compares those behaviors and circle invariants; the analytic pair fixtures are
independently derived and are not GeoDa output coordinates. GeoDa outlier styling, adjacency
attraction, polygon centroid extraction, automatic area scaling and geographic projection are
outside this contract.</p>`};export{e as default};