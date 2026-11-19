var e={slug:`api/gpu-raster/lagrangian-diagnostics-mob06`,title:`Lagrangian diagnostics (time-varying fields, FTLE, connectivity, residence time)`,html:`<h1 id="lagrangian-diagnostics-time-varying-fields-ftle-connectivity-residence-time">Lagrangian diagnostics (time-varying fields, FTLE, connectivity, residence time)</h1>
<p>Particles integrated through a time-varying vector field, with FTLE, raster ridge extraction,
connectivity and residence time. Planar inputs use the field's own distance units. Geographic
inputs are projected internally through published luma.gl projection programs. Tests include
NumPy-generated ridge and geographic-flow fixtures; OceanParcels was not run.</p>
<h2 id="gpulagrangianflowmap"><code>GPULagrangianFlowMap</code></h2>
<p>Takes one second-order Runge-Kutta (midpoint) step <code>p' = p + h v(p + h/2 v(p, t), t + h/2)</code>,
<code>stepCount</code> times, for every seed. The field has two slices, <code>velocitiesStart</code> valid at
<code>sliceStartTime</code> and <code>velocitiesEnd</code> at <code>sliceEndTime</code>; the velocity at time <code>t</code> is their linear
blend with weight <code>(t - sliceStartTime) / (sliceEndTime - sliceStartTime)</code> clamped to <code>[0, 1]</code>, so
the field is held constant outside the interval. To integrate a longer record, swap the slices and
slice times between encodings. A negative <code>timeStep</code> integrates backward. Spatial sampling is the
bilinear sampler of <code>GPUParticleAdvection</code>.</p>
<p>There is no respawn, no randomness and no atomic; a run is bitwise reproducible.</p>
<p>Set <code>spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}</code> for geographic
fields. Seeds, <code>fieldExtent</code> and final <code>output.positions</code> are longitude/latitude <strong>degrees</strong>.
The two velocity components are <strong>Web Mercator projected metres per time unit</strong>. The contributor
projects each seed into EPSG:3857, integrates there, and inversely projects each RK2 sampling
position before sampling the degree grid. Ground east/north velocities must be transformed to
that projected frame by the caller. There is no implicit geodesic motion, longitude wrapping or
pole clamping: projection-rejected positions are lost. Continuous longitudes must stay inside
[-180, 180] and latitudes inside the Web Mercator square-world domain (about ±85.05113°).
Absolute projected float32 positions lose metre-scale detail at continental offsets; this mode
is unsuitable for sub-metre movement there.</p>
<p><code>spatialContext</code>, field dimensions, step count and optional outputs are compile-time. Input values
and the parameter view are per-frame. <code>output.validity</code> optionally writes one flag per seed,
including zero for projection rejection. Lost particles also retain the existing NaN positions.</p>
<p>Row direction: the field slices place row 0 at the smallest y of <code>fieldExtent</code> and <code>v</code> is the
velocity toward larger y. No grid affine is read and nothing is checked at construction, so a
north-up field (row 0 at the top) must have its row order reversed before upload. Following a grid
affine is deferred (roadmap 12.4). The seed grid that <code>GPUFiniteTimeLyapunovExponent</code> reads needs
no reordering: output row <code>r</code> belongs to seed row <code>r</code>, and the exponent is the same whether seed
rows advance toward larger or smaller y.</p>
<table>
<thead>
<tr>
<th>Prop</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>seeds</code></td>
<td><code>float32x2</code> start positions</td>
</tr>
<tr>
<td><code>velocitiesStart</code>, <code>velocitiesEnd</code></td>
<td>Packed <code>(u, v)</code> rows, <code>fieldWidth * fieldHeight</code>, row 0 at the smallest y</td>
</tr>
<tr>
<td><code>parameters</code></td>
<td>Per-frame, <code>getGPULagrangianFlowMapParameterValues</code>: field extent, start time, signed time step, slice times</td>
</tr>
<tr>
<td><code>stepCount</code></td>
<td>Compile-time, 1 to 8192: the bounded per-thread loop</td>
</tr>
<tr>
<td><code>regions</code></td>
<td>Optional <code>uint32</code> region of each field cell, <code>GPU_LAGRANGIAN_NO_REGION</code> outside every region</td>
</tr>
<tr>
<td><code>spatialContext</code></td>
<td>Planar by default, or longitude-latitude with metric <code>none</code></td>
</tr>
<tr>
<td><code>output.validity</code></td>
<td>Optional explicit flag: 1 for a valid final position, otherwise 0</td>
</tr>
<tr>
<td><code>output.positions</code></td>
<td>Final positions. NaN for a particle that left the field or reached a NaN cell</td>
</tr>
<tr>
<td><code>output.regionClasses</code></td>
<td>Start regions then end regions, period-major (needs <code>regions</code>)</td>
</tr>
<tr>
<td><code>output.residenceTimes</code></td>
<td>Time in the start region before its first exit (needs <code>regions</code>)</td>
</tr>
</tbody>
</table>
<p>Residence time is <code>|timeStep|</code> times the number of steps that began inside the start region, so its
resolution is one step. A particle that never leaves reports <code>stepCount * |timeStep|</code> (censored); a
particle that leaves the field counts as having exited; a seed outside every region reports 0.</p>
<p>{/* parameters: GPULagrangianFlowMap generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpulagrangianflowmap-parameters"><code>GPULagrangianFlowMap</code> parameters</h3>
<ul>
<li>Packer: <code>getGPULagrangianFlowMapParameterValues(settings)</code>, writes <code>GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH</code> (8) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPULagrangianFlowMapParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>fieldExtent</code></td>
<td><code>readonly [number, number, number, number]</code></td>
<td>Field raster placement <code>[originX, originY, cellWidth, cellHeight]</code>, the placement of <code>GPUParticleAdvection</code>: row 0 has the smallest y and cell <code>(i, j)</code> is centred at <code>origin + (i + 0.5, j + 0.5) * cellSize</code>.</td>
</tr>
<tr>
<td><code>startTime</code></td>
<td><code>number</code></td>
<td>Time at which the particles are released, in the time unit of the field velocities.</td>
</tr>
<tr>
<td><code>timeStep</code></td>
<td><code>number</code></td>
<td>Signed time step. A negative step integrates backward in time, which is what backward FTLE and backward connectivity need. The integration time is <code>stepCount * |timeStep|</code>.</td>
</tr>
<tr>
<td><code>sliceStartTime</code></td>
<td><code>number</code></td>
<td>Time at which the field <code>velocitiesStart</code> is valid.</td>
</tr>
<tr>
<td><code>sliceEndTime</code></td>
<td><code>number</code></td>
<td>Time at which the field <code>velocitiesEnd</code> is valid.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpufinitetimelyapunovexponent"><code>GPUFiniteTimeLyapunovExponent</code></h2>
<p><code>FTLE = ln(sqrt(lambdaMax(J^T J))) / |T|</code>, where <code>J</code> is the gradient of the final positions with
respect to the seeds on a regular seed grid. Central differences are used in the interior and
one-sided differences on the border; <code>lambdaMax</code> is the closed-form larger eigenvalue of the 2 by 2
matrix. The result is NaN where a particle in the stencil is lost. Integrate with a negative time
step for the backward exponent.</p>
<p>For geographic final positions, pass the same longitude-latitude spatial context. The stencil
projects them before differencing, so the original seeds must form a <strong>regular grid in projected
metres</strong>, with <code>seedSpacingX/Y</code> in those metres. A regular degree grid does not meet that
requirement. Optional input <code>validity</code> rejects the center and every stencil that uses it;
optional <code>output.validity</code> identifies finite FTLE results.</p>
<p>Verified on a linear strain field <code>u = e x, v = -e y</code>, where the exponent is <code>e</code> everywhere, on a
one-directional stretch (forward <code>e</code>, backward 0), and on a strain rate that ramps linearly across
the two slices, where the exponent is the time mean of the rate (a frozen first slice gives 0).</p>
<p>{/* parameters: GPUFiniteTimeLyapunovExponent generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpufinitetimelyapunovexponent-parameters"><code>GPUFiniteTimeLyapunovExponent</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUFiniteTimeLyapunovExponentParameterValues(settings)</code>, writes <code>GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH</code> (4) <code>float32</code> values.</li>
</ul>
<p>Settings of <code>getGPUFiniteTimeLyapunovExponentParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>seedSpacingX</code></td>
<td><code>number</code></td>
<td>Distance between neighbouring seeds along x, in field units.</td>
</tr>
<tr>
<td><code>seedSpacingY</code></td>
<td><code>number</code></td>
<td>Distance between neighbouring seeds along y, in field units.</td>
</tr>
<tr>
<td><code>integrationTime</code></td>
<td><code>number</code></td>
<td>Integration time <code>T</code> of the flow map. The exponent is divided by <code>|T|</code>, so a backward flow map (negative time step) gives the backward FTLE, whose ridges are attracting structures.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuscalarfieldridges"><code>GPUScalarFieldRidges</code></h2>
<p>{/* lens:GPUScalarFieldRidges */}</p>
<p>::lens{scenario=gpu-raster.scalar-field-ridges rung=result caption="Quadratic field: the half-cell tie selects column 3; missing samples invalidate their stencils"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=formula caption="Negative Hessian curvature and a normal Newton step inside the half-open pixel box"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=datum datum=24 caption="Cell 24: column 3 owns the exact half-cell ridge tie"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=kernel caption="The central difference stencil, smallest eigenpair and subcell ridge test"}
::lens{scenario=gpu-raster.scalar-field-ridges rung=graph caption="A fixed-size ridge mask with stencil validity and optional curvature strength"}</p>
<p>The synthetic lens uses <code>f = 20 - (column - 2.5)^2</code> on a 7 × 7 metre grid. Its normal curvature
is 2. The half-open pixel rule selects column 3 in interior rows 1 through 5. Missing cell 40
(column 5, row 5) invalidates every stencil touching it; border outputs are also zero.</p>
<p>Extracts a fixed-sized ridge mask from a scalar <code>GPURasterBand</code>, including Brownian-bridge
utilization densities, cumulative utilization volumes and FTLE. The grid must have orthogonal
pixel axes and at least 3 × 3 cells. Rotated uniform grids work; geographic and Web Mercator
ground-distance modes require axis-aligned grids. The parameter packer reads the grid affine,
so update the parameter view when the affine changes. Dimensions and <code>cellSizeMode</code> are
compile-time. Geographic grids require a longitude-latitude spatial context with metric <code>none</code>
and <code>cellSizeMode: 'geographic'</code>; otherwise the context is planar.</p>
<p>A central 3 × 3 difference stencil gives the gradient and Hessian in the orthogonal pixel basis.
The smallest Hessian eigenvalue defines downward curvature and its eigenvector the ridge normal.
Spacing quotients combine mantissas and integer exponents without forming a spacing product,
so a representable curvature survives even when the squared cell spacing would overflow.
The shared 2 × 2 eigenpair helper retains the smaller diagonal in its original units and subtracts
a rationalized Jacobi correction; only its bounded direction calculation is scaled. This avoids
half-trace cancellation and preserves a small normal negative eigenvalue beside a much larger
positive diagonal. Subnormal inputs and results may still be flushed to zero by the GPU.
A cell is selected when curvature is strictly positive and meets <code>minimumCurvature</code>, its calibrated
value meets <code>minimumValue</code>, and the normal Newton displacement falls inside its pixel box
[-0.5, 0.5) along both axes. Exact half-cell ties go to the larger column/row; equal Hessian
eigenvalues choose the column normal. This is a local raster approximation of the
<a href="https://cgl.ethz.ch/Downloads/Publications/Papers/2008/Pei08/Pei08.pdf">height-ridge criterion of Peikert and Sadlo (2008)</a>,
with no curve tracing, smoothing, or claim that every detected ridge is a material LCS.</p>
<p><code>output.ridges</code> and <code>output.validity</code> are required <code>uint32</code> views, one row per cell.
<code>output.strength</code> optionally holds <code>max(0, -smallestEigenvalue)</code> at every valid stencil.
Borders and stencils touching missing or nonfinite samples write zero to every output.
Band calibration, no-data and validity are honoured. Flat plateaus and valleys have no ridges.
Geographic and Web Mercator modes use the existing row-local WGS84 ground scale; their Hessian
is a local tangent approximation. Outputs preserve the source cell index and never truncate,
so no compact-output capacity or overflow status is needed. There is no CPU operation twin for
these raster contributors.</p>
<p>{/* parameters: GPUScalarFieldRidges generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuscalarfieldridges-parameters"><code>GPUScalarFieldRidges</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUScalarFieldRidgesParameterValues(settings, grid, target?)</code>, writes <code>GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH</code> (8) <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): Grid affine; update these parameters when the affine changes.</li>
<li><code>target</code> (<code>unknown</code>): Optional destination of at least 8 float32 elements.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUScalarFieldRidgesParameterValues</code>:</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Type</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>minimumValue</code> (optional)</td>
<td><code>number</code></td>
<td>Lowest accepted calibrated field value. Defaults to 0.</td>
</tr>
<tr>
<td><code>minimumCurvature</code> (optional)</td>
<td><code>number</code></td>
<td>Lowest accepted downward curvature, in field units per squared distance. Defaults to 0; flat cells are always rejected.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="connectivity-matrix">Connectivity matrix</h2>
<p>Compose <code>GPULagrangianFlowMap</code> (with <code>regions</code> and <code>output.regionClasses</code>) with
<code>GPUTransitionMatrix</code> using <code>periodCount: 2</code> and <code>classCount</code> equal to the number of regions
(at most 64). Its <code>counts</code> are the connectivity matrix <code>counts[from * K + to]</code> and its
<code>probabilities</code> are the row-normalized matrix. Particles that end outside every region, including
lost ones, are not counted; <code>output.ignored</code> of the transition matrix counts them.</p>`};export{e as default};