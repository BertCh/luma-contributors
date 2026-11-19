var e={slug:`api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density`,title:`Colocation quotient, local category dependence and adaptive-bandwidth density`,html:`<h1 id="colocation-quotient-local-category-dependence-and-adaptive-bandwidth-density">Colocation quotient, local category dependence and adaptive-bandwidth density</h1>
<p>STX-08, second group. Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>. Weights are the published <code>GPUSpatialWeights</code> CSR; the density reuses the kernels, parameter packer and grid-index gather of <code>GPUKernelDensity</code>.</p>
<h2 id="gpucolocationquotient"><code>GPUColocationQuotient</code></h2>
<p>Reference: Leslie and Kronenfeld (2011), "The colocation quotient: a new measure of spatial association between categorical subsets of points", Geographical Analysis 43(3). No pinned package implements it, so the specs compare with an independent binary64 evaluation of the definition.</p>
<p>For a row <code>i</code> of category <code>a</code>, the neighbour share <code>f(i, b)</code> is the weight on neighbours of category <code>b</code> divided by the weight on all neighbours. The chance share is <code>ratio(a, b) = (N_b - [a = b]) / (N - 1)</code>. The local quotient is <code>f(i, b) / ratio(a, b)</code>; the global quotient <code>CLQ(a -> b)</code> is the mean of <code>f(i, b)</code> over the rows of <code>a</code> that have a neighbour, divided by the same ratio. With one neighbour per row it is Leslie and Kronenfeld's <code>(N_{a->b} / N_a) / (N_b / (N - 1))</code>; Gaussian-kernel weights give the adaptive form of Wang et al. (2017). 1 is independence, above 1 attraction, below 1 avoidance.</p>
<ul>
<li>Neighbours are listed slots with a finite positive weight, to another row with a valid category. Rows with category <code>0xffffffff</code> (or at or above <code>categoryCount</code>) are left out. Results are NaN where a row has no neighbour or the chance share is 0.</li>
<li>Outputs: <code>globalQuotients</code> (<code>categoryCount^2</code>, entry <code>a * categoryCount + b</code>), optional <code>localQuotients</code> (<code>rows * categoryCount</code>) and <code>categoryRowCounts</code>.</li>
<li>At most 32 categories. The shares are summed in the shared fixed-point accumulator, so the global values do not depend on thread order.</li>
<li>Not implemented: the permutation test of the paper.</li>
</ul>
<p>{/* lens:GPUColocationQuotient */}</p>
<h2 id="gpulocalcategorydependence"><code>GPULocalCategoryDependence</code></h2>
<p>Reference: Boots (2003), "Developing local measures of spatial association for categorical data", Journal of Geographical Systems 5, which defines the LICD family. Implemented: a local composition measure. Per row (<code>GPU_LOCAL_CATEGORY_DEPENDENCE_COLUMNS</code>), with the row itself never counted: own-category neighbours (<code>likeCount</code>), neighbours with a category (<code>neighborCount</code>, <code>k</code>), the local share <code>likeCount / k</code> (<code>proportion</code>), the global share <code>N_c / N</code> (<code>expectedProportion</code>), the binomial upper tail <code>P(X >= likeCount)</code> for <code>Binomial(k, N_c / N)</code> and the z score of the same binomial. A neighbour is an active slot of the shared rule (positive weight, another row); the size of a positive weight is not used.</p>
<p>This is not the output of <code>licd_multi</code> of the R package spdep, and no parity with it is claimed (R is not installed, so it cannot be checked here). spdep counts the focal row among its own neighbours (<code>k + 1</code> trials, the focal row always a like), reports the global share as <code>prop_i</code>, and its <code>pbinom_like_BW</code> is the lower tail <code>P(X &#x3C;= like)</code>. Example: 4 neighbours, 3 of them like, share 0.5. This class gives <code>P(X >= 3) = 0.3125</code> on <code>Binomial(4, 0.5)</code>; spdep's <code>pbinom_like_BW</code> is <code>P(X &#x3C;= 4) = 0.96875</code> on <code>Binomial(5, 0.5)</code>.</p>
<p>Not implemented: the local configuration part (local join counts and the chi-square), the permutation p-values and the unlike-neighbour tails. The binomial test is checked against hand-computed tails (0.3125, 0.648) and a binary64 evaluation.</p>
<p>{/* lens:GPULocalCategoryDependence */}</p>
<h2 id="gpuadaptivebandwidthkerneldensity"><code>GPUAdaptiveBandwidthKernelDensity</code></h2>
<p>Reference: Abramson (1982), "On bandwidth variation in kernel estimates: a square root law", Annals of Statistics 10(4), with the two-stage form of Silverman (1986), Density Estimation, section 5.3.1. A fixed-bandwidth pilot <code>f</code> at every point (including itself) sets <code>h_i = h * clamp((f(x_i) / g)^(-alpha), 1/L, L)</code>, where <code>g</code> is the weighted geometric mean of <code>f</code> and <code>alpha</code> the sensitivity (default 1/2; 0 is the fixed bandwidth). The density is <code>sum_i w_i K_{h_i}(|u - x_i|)</code> with the gaussian (truncated at 5 bandwidths), quartic or epanechnikov kernel of <code>GPUKernelDensity</code>.</p>
<ul>
<li>Per-frame: <code>parameters</code> from <code>getGPUAdaptiveBandwidthKernelDensityParameterValues</code> (the layout of <code>getGPUKernelDensityParameterValues</code>, plus the sensitivity in slot 9). Compile-time: grid, index, kernel, normalization, <code>bandwidthFactorLimit</code> (1 to 16, default 4; it bounds the gather radius, so it is a deliberate approximation) and the optional <code>pointBandwidths</code> and <code>pilotDensities</code> outputs.</li>
<li><code>GPUKernelDensity</code> is not modified, so its results are unchanged. The new class is separate because the existing gather, edge-correction mass and binned path all assume one bandwidth.</li>
<li>With <code>alpha = 0</code> it matches <code>GPUKernelDensity</code> with <code>edgeCorrection: 'none'</code> to 1e-4 of the peak (tested).</li>
<li>Not implemented: edge correction, the binned method, polygon windows, bandwidth selection.</li>
<li>Accuracy against binary64 all-pairs (weighted clustered data, 24 by 20 grid): point bandwidths and pilots within 1e-3 relative, cell values within 2e-3 of the peak, for the three kernels. The pilot is evaluated in float32.</li>
</ul>
<p>{/* lens:GPUAdaptiveBandwidthKernelDensity */}</p>
<p>{/* parameters: GPUAdaptiveBandwidthKernelDensity generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuadaptivebandwidthkerneldensity-parameters"><code>GPUAdaptiveBandwidthKernelDensity</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUAdaptiveBandwidthKernelDensityParameterValues(settings, grid)</code>, writes <code>GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH</code> <code>float32</code> values.
<ul>
<li><code>grid</code> (<code>GPURasterMetadata</code>): The output grid: axis-aligned with a nonzero finite cell size.</li>
</ul>
</li>
</ul>
<p>Settings of <code>getGPUAdaptiveBandwidthKernelDensityParameterValues</code>:</p>
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
<td><code>bandwidth</code></td>
<td><code>number</code></td>
<td>Pilot and global bandwidth <code>h</code> in position units. Positive and finite.</td>
</tr>
<tr>
<td><code>sensitivity</code> (optional)</td>
<td><code>number</code></td>
<td>Abramson exponent <code>alpha</code> in <code>[0, 1]</code>; 0 is the fixed bandwidth. Defaults to 1/2.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>`};export{e as default};