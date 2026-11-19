var e={slug:`api/gpu-raster/operations`,title:`Raster analysis operations`,html:`<h1 id="raster-analysis-reference">Raster analysis reference</h1>
<p><code>@luma-contributors/analysis/raster-analysis</code> holds the raster analysis contributors that luma.gl
does not provide. Raster storage and residency, analytical overviews, pixel operations, statistics,
filters and morphology, connected components, region measurements and contours are owned by
luma.gl: see <a href="https://luma.gl/docs/api-reference/experimental/gpu-raster"><code>@luma.gl/graph/gpu-raster</code></a>.
This entry composes with those operations through <code>GPUCommandGraph</code>; it does not wrap or replace them.</p>
<p>Each contributor declares work into a caller-owned graph and never compiles, submits or reads back.</p>
<h2 id="choose-a-page">Choose a page</h2>
<table>
<thead>
<tr>
<th>Page</th>
<th>Use it for</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/operations-analysis">Analysis contributors</a></td>
<td>Zonal statistics, stretches, isolines, distance and cost surfaces, rasterization, change detection, and flow textures.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/raster-patches">Patch metrics and sieving</a></td>
<td>Per-patch area and perimeter, and small-patch removal over dense labels.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/expansion">Raster, terrain and geometry expansion</a></td>
<td>Continuous-surface preparation, reprojection, directed walking cost and bounded topology.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/fill-nodata">Fill nodata</a></td>
<td>Four-quadrant nodata filling.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/tile-reprojection">Tile reprojection</a></td>
<td>Explicit tile reprojection.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/fractional-zonal-distribution">Fractional zonal distribution</a> and <a href="/luma-contributors/docs/api/gpu-raster/fractional-categorical-statistics">categorical statistics</a></td>
<td>Weighted distributions and composition with observed support.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/focal-pairs">Focal pairs</a></td>
<td>Paired-valid covariance, correlation and local regression.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/spectral-indices">Spectral indices</a></td>
<td>Calibrated NDWI, NBR, EVI and SAVI.</td>
</tr>
<tr>
<td><a href="/luma-contributors/docs/api/gpu-raster/remote-sensing">Remote sensing</a></td>
<td>Remote-sensing contributors.</td>
</tr>
</tbody>
</table>
<h2 id="related-pages">Related pages</h2>
<ul>
<li><a href="/luma-contributors/docs/api/gpu-raster">Raster analysis overview</a></li>
<li><a href="/luma-contributors/docs/api/gpu-terrain">GPU Terrain</a></li>
<li><a href="https://luma.gl/docs/api-reference/experimental/gpu-core">luma.gl GPU Core</a></li>
</ul>`};export{e as default};