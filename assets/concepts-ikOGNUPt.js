var e={slug:`api/gpu-terrain/concepts`,title:`GPU Terrain concepts`,html:`<h1 id="gpu-terrain-concepts">GPU Terrain concepts</h1>
<h2 id="conventions">Conventions</h2>
<ul>
<li><strong>One grid.</strong> Every contributor takes <code>grid: GPURasterMetadata</code> (<code>makeGPURasterMetadata</code> builds
an axis-aligned one); its width and height size the kernels. Cell sizes and the latitude band
are never typed twice: each packer that needs them takes the frame's grid as its second
argument, <code>getGPU&#x3C;Class>ParameterValues(settings, grid)</code>, and reads them from <code>grid.affine</code>
in world units (meters for <code>'uniform'</code>, EPSG:3857 meters for <code>'web-mercator'</code>, degrees for
<code>'geographic'</code>, see <code>GPURasterCellSizeMode</code>).</li>
<li><strong><code>parameters</code> in, <code>output</code> out.</strong> The packed per-frame view is <code>parameters</code> (a second view of
another format is <code>uint32Parameters</code>). Outputs live in one <code>output</code> bag; the validity output is
<code>output.outputValidity</code> (a <code>validity</code> prop is always an input mask) and one-row status words are
in <code>output.status</code> (<code>converged</code>, <code>iterationCount</code>, <code>overflow</code>). <code>makeGPURasterBand(view, grid)</code>
turns one contributor's output view into the next contributor's input band.</li>
<li><strong>Compile-time limits say so.</strong> Values that size kernels are named <code>maximum*</code>
(<code>maximumRadius</code>, <code>maximumFillIterations</code>); the value actually used each frame (<code>radius</code>, a
distance) is in <code>parameters</code>.</li>
<li><strong>Graph views in, graph views out.</strong> Inputs and outputs are <code>GraphDataView</code>, <code>GraphVectorView</code>, or
<code>GraphTextureView</code> objects created on the target graph. Outputs are always caller-owned; scratch
storage is graph transients that die with <code>compiled.destroy()</code>. Contributors never compile, encode,
submit, or read back.</li>
<li><strong>Per-frame values never recompile.</strong> Viewports, thresholds, radii, observer positions, budgets,
and region shapes are read from storage views, usually a <code>GPUParameterBuffer</code> that the
application rewrites with <code>write()</code>. Lengths, capacities, grid sizes, and which optional views
exist are compile-time topology; each prop's TSDoc says which category it belongs to.</li>
<li><strong>Bounded results report overflow on the GPU.</strong> Compact ID lists use <code>GPUCompactOutput</code>
(<code>ids</code> and a <code>status</code> with <code>count</code>, <code>overflow</code> and optional <code>requiredCount</code>). <code>status.count</code> is clamped
to <code>ids.length</code> and can be an indirect draw instance count; <code>status.overflow</code> is rewritten every
encoding.</li>
<li><strong>Stable IDs.</strong> Result IDs are the caller's <code>sourceIds[row]</code> (or tile IDs) when given and zero-based
rows otherwise. Node and transient IDs are <code>\${id}-&#x3C;step></code>, so two instances in one graph need
different <code>id</code> props.</li>
<li><strong>One import per buffer.</strong> Import a buffer once (for example with <code>graph.importGPUData()</code> or
<code>GPUParameterBuffer.importToGraph()</code>) and pass the returned view to every contributor that reads
it, so the graph tracks hazards on one logical handle.</li>
<li><strong>Shared contracts.</strong> Four shapes mean the same on every analysis page:
<ul>
<li><strong>Status.</strong> Every GPU-written status word sits in the <code>status</code> member of the bag that owns it:
<code>output.status.overflow</code>, <code>output.status.converged</code>, and <code>output.status.code</code> for model fits.</li>
<li><strong>Grid.</strong> Raster and terrain contributors take <code>grid: GPURasterMetadata</code>, and their packers read
the cell size from it (see <a href="#conventions">GPU Terrain conventions</a>).</li>
<li><strong>Network.</strong> Network contributors take <code>topology: GPUNetworkTopology</code> (<code>offsets</code>, <code>neighbors</code>,
<code>weights</code>, optional <code>nodePositions</code>). Its CSR part can be passed as <code>GPUSpatialWeights</code> (see
<a href="/luma-contributors/docs/api/gpu-network/concepts#conventions">GPU Network conventions</a>).</li>
<li><strong>Polygons.</strong> Polygons use the four-level <code>GPUPolygonGeometryPort</code>: <code>kind: 'polygons'</code>,
<code>positions</code>, <code>featureOffsets</code>, <code>polygonOffsets</code> and <code>ringOffsets</code>. Contributors with a single
geometry take the same fields flat, without <code>kind</code> (<code>GPUPolygonGeometryFields</code>). See
<a href="/luma-contributors/docs/api/gpu-spatial-analysis/concepts#shared-ports-status-and-planning">Shared ports</a>.</li>
</ul>
</li>
</ul>
<h2 id="heights-and-datums">Heights and datums</h2>
<p>Declare height datums when combining DEM elevations with absolute observer or catalogue heights.</p>
<p>Absolute heights and DEM heights must share one vertical datum, and observer positions must share the DEM grid's horizontal datum. Declare both (<code>verticalDatum</code> on the band or <code>elevationVerticalDatum</code> on the operator, the operator prop taking precedence; <code>observerVerticalDatum</code> on the operator, <code>datum</code> on contexts). The library checks the declarations and never converts. Heights above the ground (<code>heightReference: 'ground'</code>) need no conversion. For a few observers, convert them on the CPU in binary64: h_orthometric = h_ellipsoidal − N(lat, lon), with N taken from the DEM's own geoid model (PROJ, pyproj or GeographicLib). For many observers or targets, convert the DEM on the GPU: resample N onto the DEM grid, add it with <code>GPURasterBandMath('add')</code>, and declare <code>'ellipsoidal'</code>. Convert horizontal datums (NAD27, GCJ-02) on the CPU before upload.</p>
<table>
<thead>
<tr>
<th>Source</th>
<th>Vertical datum to declare</th>
</tr>
</thead>
<tbody>
<tr>
<td>SRTM/NASADEM/AW3D30</td>
<td>EGM96 (<code>'EPSG:5773'</code>)</td>
</tr>
<tr>
<td>Copernicus GLO-30</td>
<td>EGM2008 (<code>'EPSG:3855'</code>)</td>
</tr>
<tr>
<td>USGS 3DEP</td>
<td>NAVD88 (<code>'EPSG:5703'</code>)</td>
</tr>
<tr>
<td>Terrarium / Mapbox Terrain-RGB</td>
<td>"above sea level", unspecified geoid — leave <code>verticalDatum</code> undeclared</td>
</tr>
<tr>
<td>GNSS receivers</td>
<td><code>'ellipsoidal'</code></td>
</tr>
<tr>
<td>drone metadata</td>
<td>vendor-dependent, verify before declaring</td>
</tr>
</tbody>
</table>
<ul>
<li><strong>Absolute inputs.</strong> <code>GPUPointHorizonProfile</code>, <code>GPUPointHorizonVisibility</code>, <code>GPUTerrainViewshed</code>,
<code>GPUTerrainLineOfSight</code>, and <code>GPUTerrainCumulativeViewshed</code> accept <code>heightReference: 'absolute'</code>
(default: <code>'ground'</code>). Their <code>observerVerticalDatum</code> is checked against the elevation band's
<code>verticalDatum</code>; declaring it in ground mode throws. <code>GPUTerrainPeakSnap</code> checks
<code>candidateVerticalDatum</code> against the band's <code>verticalDatum</code> for absolute <code>candidateHeights</code>;
declaring it without <code>candidateHeights</code> throws.</li>
<li><strong>Compatibility.</strong> A mismatch between two declared vertical datums throws; either undeclared
passes. Comparison trims whitespace and is case-insensitive, with no aliases.</li>
<li><strong>Absolute observers.</strong> Set the observer height explicitly: the <code>observerHeight</code> default of 1.7
is meaningless in absolute mode. The observer component of LOS <code>pairHeights</code> and cumulative
<code>observerHeights</code> also becomes absolute. Observers must be inside the grid but may sit over
no-data ground. Targets stay ground-relative; an eye below the ground hides most targets.</li>
<li><strong>Conversion.</strong> There is no geoid-undulation input: the contract performs no CRS conversion,
GPU Project rejects geoid grids, and a constant undulation can be silently wrong across a DEM.
Use <code>GPURasterBandMath</code> with <code>'add'</code> or <code>'subtract'</code> and an undulation raster on the same grid;
a band <code>offset</code> is appropriate only when the correction really is constant.</li>
</ul>
<h2 id="gpu-execution-model">GPU execution model</h2>
<p>For tiled terrain, a halo stage must bound every source read and preserve the sampling geometry
when the window moves. <code>GPUTerrainHorizon.getHaloStage()</code> provides the march bound
<code>maximumRadius + 1</code>. For sweep it provides <code>maximumRadius</code> with explicit <code>parameterLayout: 'sweep'</code>.
That layout packs a common <code>sweepSourceGrid</code>, <code>sweepReferenceCellSize</code> and per-frame <code>sweepOrigin</code>
in 24 parameters. Both algorithms require this shared source-row layout for latitude-dependent
halos, including distance cutoffs. View padding never selects a layout. The origin is measured in
overview pixels relative to the shared source grid; do not infer downsampling from a level number.
Placement updates reuse the compiled shader. March adds only integer displacements to the pixel
origin, keeping fractions separate so crop offsets cannot change validity corners by rounding.
Horizon construction captures declared options by property access, including inherited and
non-enumerable getters. In particular, <code>cellSizeMode: 'geographic'</code> requires the shared source-row
layout for a halo regardless of how the property is defined; changing the caller's option later
does not change that validated requirement.
<code>GPUTerrainCastShadow</code> uses the same shared-source model with its own 32-value sweep parameter
layout. Its <code>getHaloStage()</code> returns the radius only with explicit <code>parameterLayout: 'sweep'</code>;
the standard 16-value window-local model refuses halo assembly. Use the same source grid,
reference cells and fixed radius across windows, and pack each window's origin per frame.
<code>GPUTerrainWeissLandforms.getHaloStage()</code> provides <code>localRadius</code> only with local
standardization. Global Weiss statistics and geomorphons with per-frame anisotropic cell sizes
have no complete fixed local bound. Their advisory <code>localRadius</code> (Weiss) and <code>searchRadius</code>
(geomorphons) are not halo stages. Keep only the core outputs of a correctly expanded tile.</p>
<p>How terrain contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.</p>
<p>{/* lens:execution-model */}</p>
<p>A local operator is one kernel over every cell; a propagating one (fill, flats, accumulation) is an
unrolled loop of gated rounds whose indirect dispatch drops to zero once a round changes nothing.</p>
<p>::lens{scenario=gpu-terrain.terrain-derivatives rung=graph caption="A local operator: two Sobel passes feed one fused shade kernel, each one invocation per cell."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=kernel caption="One shade invocation per cell; the dispatch is the grid, not a loop."}</p>`};export{e as default};