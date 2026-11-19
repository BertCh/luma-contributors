var e={slug:`api/gpu-raster/expansion-lenses`,title:`Expansion computation lenses`,html:`<h1 id="expansion-computation-lenses">Expansion computation lenses</h1>
<p>These executable examples use an explicit synthetic 3 × 3 grid. Each graph runs the named public contributor and records real GPU inputs and outputs. The graph rung exposes procedural stages and completeness evidence.</p>
<h2 id="gpurasterfillnodata">GPURasterFillNoData</h2>
<p>{/* lens:GPURasterFillNoData */}</p>
<p>Disk search uses original valid donors only; nearest or inverse-distance weighted interpolation preserves observed samples.</p>
<p>::lens{scenario=gpu-raster.raster-fill-no-data rung=graph}</p>
<h2 id="gpurasterrankfilter">GPURasterRankFilter</h2>
<p>{/* lens:GPURasterRankFilter */}</p>
<p>Select sorted[min(floor(q*n),n-1)] from the finite square neighborhood; invalid centers remain invalid.</p>
<p>::lens{scenario=gpu-raster.raster-rank-filter rung=graph}</p>
<h2 id="gpurasterfractionalcoverage">GPURasterFractionalCoverage</h2>
<p>{/* lens:GPURasterFractionalCoverage */}</p>
<p>Coverage is polygon-cell intersection area divided by affine cell area, assembled from clipped triangles.</p>
<p>::lens{scenario=gpu-raster.raster-fractional-coverage rung=graph}</p>
<h2 id="gpurasterfractionalzonalstatistics">GPURasterFractionalZonalStatistics</h2>
<p>{/* lens:GPURasterFractionalZonalStatistics */}</p>
<p>Extensive totals sum value times fraction; density integrals additionally multiply cell area. Means divide weighted values by observed fractions.</p>
<p>::lens{scenario=gpu-raster.raster-fractional-zonal-statistics rung=graph}</p>
<h2 id="gpurasterreconstruction">GPURasterReconstruction</h2>
<p>Reconstruction accepts raster bands through the CPU/GPU operation wrappers; caller-owned geometry
and raster metadata remain explicit.</p>
<p>{/* lens:GPURasterReconstruction */}</p>
<p>Geodesic dilation iterates min(mask, neighborhood maximum) until a verified fixed point; erosion is the dual.</p>
<p>::lens{scenario=gpu-raster.raster-reconstruction rung=graph}</p>
<h2 id="gpurasterfillholes">GPURasterFillHoles</h2>
<p>{/* lens:GPURasterFillHoles */}</p>
<p>Reconstruct boundary-connected background and complement it to fill enclosed background components.</p>
<p>::lens{scenario=gpu-raster.raster-fill-holes rung=graph}</p>
<h2 id="gpurasterreprojection">GPURasterReprojection</h2>
<p>{/* lens:GPURasterReprojection */}</p>
<p>Map destination centers through the inverse projection and source affine, then sample the supplied source support.</p>
<p>::lens{scenario=gpu-raster.raster-reprojection rung=graph}</p>
<h2 id="gpurasterwalkingcost">GPURasterWalkingCost</h2>
<p>{/* lens:GPURasterWalkingCost */}</p>
<p>Directed least costs obey D(v)=min(D(v),min_u(D(u)+cost(u,v))); Tobler speed is 6 exp(-3.5 abs(slope+0.05)) km/hour.</p>
<p>::lens{scenario=gpu-raster.raster-walking-cost rung=graph}</p>
<h2 id="gpurasterwatershedsegmentation">GPURasterWatershedSegmentation</h2>
<p>{/* lens:GPURasterWatershedSegmentation */}</p>
<p>Priority flood selects the smallest path maximum elevation, breaking ties by FIFO discovery age.</p>
<p>::lens{scenario=gpu-raster.raster-watershed-segmentation rung=graph}</p>
<h2 id="gpurasterskeleton">GPURasterSkeleton</h2>
<p>{/* lens:GPURasterSkeleton */}</p>
<p>Zhang-Suen alternates two deletion phases preserving connectivity until no foreground cell changes.</p>
<p>::lens{scenario=gpu-raster.raster-skeleton rung=graph}</p>
<h2 id="gpurastermedialaxis">GPURasterMedialAxis</h2>
<p>{/* lens:GPURasterMedialAxis */}</p>
<p>Retain Euclidean foreground disks that are not contained in another foreground disk, including exterior background.</p>
<p>::lens{scenario=gpu-raster.raster-medial-axis rung=graph}</p>
<h2 id="gpurastertilereprojection"><code>GPURasterTileReprojection</code></h2>
<p>Inspect explicit bounded source mosaic assembly and bilinear sampling. Source tiles share a lattice; the caller supplies all tiles and interpolation support.</p>
<p>{/* lens:GPURasterTileReprojection */}</p>
<p>::lens{scenario=gpu-raster.raster-tile-reprojection rung=result caption="Inspect synthetic outputs and validity."}</p>
<p>::lens{scenario=gpu-raster.raster-tile-reprojection rung=graph caption="Inspect the actual GPU command graph."}</p>`};export{e as default};