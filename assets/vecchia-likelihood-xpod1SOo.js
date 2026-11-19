var e={slug:`api/gpu-spatial-analysis/vecchia-likelihood`,title:`Vecchia likelihood (STM-07)`,html:`<h1 id="vecchia-likelihood-stm-07">Vecchia likelihood (STM-07)</h1>
<p>Nearest-neighbour Gaussian process likelihood, fitting and prediction. The likelihood of ordered points factors into one conditional per point, <code>log L = -1/2 Σ [log 2π + log d_i + r_i² / d_i]</code>, and each conditional is a small kriging solve on earlier points, which is what <code>GPULocalKriging</code> already does.</p>
<p>Import from <code>@luma-contributors/analysis/gpu-spatial-analysis</code>.</p>
<pre class="shiki shiki-themes github-light github-dark" style="--shiki-light:#24292e;--shiki-dark:#e1e4e8;--shiki-light-bg:#fff;--shiki-dark-bg:#24292e" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">import</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8"> {</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUVecchiaLogLikelihood,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUVecchiaFisherScoring,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  GPUVecchiaSpaceTimeLogLikelihood,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  buildVecchiaConditioningSets,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getVecchiaMaxMinOrdering,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  buildVecchiaPredictionNeighborhoods,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  fitVecchiaCovarianceFisherScoring,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  maximizeVecchiaLogLikelihood,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUVecchiaLogLikelihoodParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getVecchiaLogLikelihoodResult,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getVecchiaFisherEvaluation,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">  getVecchiaNuggetFloor</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583">from</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF"> '@luma-contributors/analysis/gpu-spatial-analysis'</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8">;</span></span></code></pre>
<h2 id="conditioning-sets-cpu-setup">Conditioning sets (CPU setup)</h2>
<p><code>buildVecchiaConditioningSets({positions, neighborCount, order, validity, times, timeScale})</code> returns CSR <code>{offsets, neighbors}</code> over the input row order: for each row, the <code>neighborCount</code> (at most 32) rows earlier in the Vecchia order that are nearest. <code>order</code> is any permutation; <code>getVecchiaMaxMinOrdering(positions)</code> gives the exact max-min order (Guinness 2018). Pass <code>times</code> and <code>timeScale</code> (spatial units per unit of time) to use the space-time metric <code>sqrt(h² + (timeScale·u)²)</code>. Build once per ordering; the sets do not depend on the covariance. <code>buildVecchiaPredictionNeighborhoods({positions, queries, neighborCount})</code> is the same search from arbitrary query points over the data.</p>
<h2 id="gpuvecchialoglikelihood"><code>GPUVecchiaLogLikelihood</code></h2>
<p><code>neighborhoods</code> and <code>neighborhoodOffsets</code> are the CSR above; <code>positions</code> and <code>observations</code> have one row per site; <code>parameters</code> is written by <code>getGPUVecchiaLogLikelihoodParameterValues({model, range, sill, nugget})</code>, per frame, with no recompile.</p>
<p>Each site's solve is the <code>simple</code> mode of <code>GPULocalKriging</code> over its conditioning set: <code>d_i</code> is <code>values.y</code>, and the observation error variance of the site is added to it. A site with an empty conditioning set, or whose conditioning rows are all unusable (validity 0, non-finite value, error variance or position), contributes its marginal term with variance <code>sill + nugget</code>, in this class, in the space-time class and in <code>GPUVecchiaFisherScoring</code> alike; rebuilding the sets after changing <code>validity</code> is therefore not required for a finite likelihood. Observations are residuals from any known mean; subtract an estimated trend first.</p>
<p><strong>Outputs</strong> (<code>output</code>):</p>
<table>
<thead>
<tr>
<th>Field</th>
<th>Rows</th>
<th>Content</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>conditionals</code></td>
<td>one per site</td>
<td><code>[mean, variance d_i, pivot ratio, retained]</code> from the kriging solve</td>
</tr>
<tr>
<td><code>conditionalStatus</code></td>
<td>one per site</td>
<td>the solve status</td>
</tr>
<tr>
<td><code>siteTerms</code></td>
<td>one per site</td>
<td><code>-1/2 (log 2π + log d_i + r_i² / d_i)</code>; NaN if excluded or failed</td>
</tr>
<tr>
<td><code>siteStatus</code></td>
<td>one per site</td>
<td>0 used, 1 excluded, 2 solve failed, 3 variance non-positive</td>
</tr>
<tr>
<td><code>logLikelihood</code></td>
<td>2</td>
<td><code>[high, low]</code> double-single total: <code>high + low</code></td>
</tr>
<tr>
<td><code>status</code></td>
<td>3 words</td>
<td><code>[failedSiteCount, excludedSiteCount, usedSiteCount]</code></td>
</tr>
</tbody>
</table>
<p><code>getVecchiaLogLikelihoodResult(total, words)</code> reads these as a number. The total is a fixed-point sum of the float32 terms, so it does not depend on the order or on a float32 accumulation.</p>
<p><strong>Failure.</strong> Any site that does not solve, or has a non-positive conditional variance, is counted in <code>status[0]</code> and the whole likelihood is NaN. It is a failed likelihood, not a partial one.</p>
<p><strong>Limits and floors.</strong> Exponential and Matérn 3/2 only. <code>getGPUVecchiaLogLikelihoodParameterValues</code> throws if a Matérn 3/2 nugget is below <code>getVecchiaNuggetFloor(covariance)</code>, which is <code>1e-3</code> of the sill (0 for the exponential). Prediction does not use this floor. Row and conditioning-set sizes are compile-time; the conditioning set is capped at 32 by the kriging private arrays.</p>
<p>{/* lens:GPUVecchiaLogLikelihood */}</p>
<p>{/* parameters: GPUVecchiaLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuvecchialoglikelihood-parameters"><code>GPUVecchiaLogLikelihood</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUVecchiaLogLikelihoodParameterValues(covariance)</code>, writes the <code>float32</code> parameter view.</li>
</ul>
<p>Settings of <code>getGPUVecchiaLogLikelihoodParameterValues</code>:</p>
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
<td><code>model</code></td>
<td><code>GPUVecchiaCovarianceModel</code></td>
<td>-</td>
</tr>
<tr>
<td><code>range</code></td>
<td><code>number</code></td>
<td>Effective range, positive.</td>
</tr>
<tr>
<td><code>sill</code></td>
<td><code>number</code></td>
<td>Partial sill, at least 0.</td>
</tr>
<tr>
<td><code>nugget</code> (optional)</td>
<td><code>number</code></td>
<td>Process microscale variance. Matern 3/2 needs at least <code>VECCHIA_NUGGET_FLOOR_FRACTION * sill</code>.</td>
</tr>
</tbody>
</table>
<p>{/* /parameters */}</p>
<h2 id="gpuvecchiafisherscoring"><code>GPUVecchiaFisherScoring</code></h2>
<p>One evaluation of the likelihood, the score and the expected information in <code>(log sill, log range, log nugget)</code>, per Guinness 2019. Fixed-point totals, one <code>[high, low]</code> pair per cell: read them with <code>getVecchiaFisherEvaluation(totals, status)</code>. Rows of its <code>siteTerms</code> are <code>[logLik, score(3), information(6)]</code>.</p>
<p><code>fitVecchiaCovarianceFisherScoring({initial, evaluate, ...})</code> runs the CPU outer loop: Fisher steps in log parameters, halving line search, at most 30 iterations by default, with <code>converged</code> and <code>iterationLimitReached</code> in the result. A nugget below the floor is pinned at the floor; while it is pinned the nugget moves with the sill.</p>
<p>{/* lens:GPUVecchiaFisherScoring */}</p>
<h2 id="gpuvecchiaspacetimeloglikelihood"><code>GPUVecchiaSpaceTimeLogLikelihood</code></h2>
<p>Same outputs as the spatial form, over the per-frame <code>SpaceTimeVariogramModel</code> covariance of <code>GPUSpaceTimeKriging</code> (separable, product-sum, metric, sum-metric, Gneiting). <code>getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, target, searchAnisotropy)</code> packs the layout of <code>getGPUSpaceTimeKrigingParameterValues</code> and enforces the floor for smooth components (the variogram at a tiny lag must be at least <code>1e-3</code> of <code>C(0, 0)</code>). Conditioning sets come from the CSR; <code>GPUSpaceTimeKriging</code> itself has no CSR path and is unchanged, so that contributor stays within the core eight storage buffers.</p>
<p>There is no space-time Fisher kernel: families have different parameter sets. Use <code>maximizeVecchiaLogLikelihood({initial, evaluate})</code>, a CPU Nelder-Mead over the free per-frame parameters (usually log-transformed).</p>
<p>{/* lens:GPUVecchiaSpaceTimeLogLikelihood */}</p>
<p>{/* parameters: GPUVecchiaSpaceTimeLogLikelihood generated by modules/analysis/test/analysis-core/doc-parameters.ts from the TSDoc; do not edit */}</p>
<h3 id="gpuvecchiaspacetimeloglikelihood-parameters"><code>GPUVecchiaSpaceTimeLogLikelihood</code> parameters</h3>
<ul>
<li>Packer: <code>getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(model, searchAnisotropy?)</code>, writes the <code>float32</code> parameter view.
<ul>
<li><code>model</code> (<code>SpaceTimeVariogramModel</code>): Space-time covariance.</li>
<li><code>searchAnisotropy</code> (<code>number</code>): Space units per unit of time, needed when the model has none.</li>
</ul>
</li>
</ul>
<p>{/* /parameters */}</p>
<h2 id="prediction">Prediction</h2>
<p>Prediction does not get a new contributor: use <code>GPULocalKriging</code> in <code>simple</code> mode with <code>buildVecchiaPredictionNeighborhoods</code> and the same parameters. The existing kriging rules apply, including no nugget floor, so a smooth covariance with no nugget can report status 2 at some queries. For space-time, use <code>GPUSpaceTimeKriging</code> with its own neighbour search.</p>
<h2 id="float32-on-the-gpu">Float32 on the GPU</h2>
<p>Repeat of the float32 experiment in <a href="/luma-contributors/docs/../dev-docs/research/spatiotemporal-review-2026-11-09">the spatiotemporal review</a>, same points, conditioning sets, ordering (<code>random</code>), grid and kernels, with <code>GPUVecchiaLogLikelihood</code> on an Apple GPU (Metal, headless Chrome) compared with a binary64 CPU reference of the same sets. Run: <code>VITE_VECCHIA_FLOAT32_STUDY=1 VITE_VECCHIA_FLOAT32_STUDY_SIZES=10000 npx vitest run --project headless --silent=false modules/analysis/test/gpu-spatial-analysis/spatial-interpolation/vecchia-float32-gpu-study.spec.ts</code>. The binary64 reference reproduces the emulated experiment's <code>ll64 at best</code> and gaps to the printed digits. n = 10,000, m = 30.</p>
<table>
<thead>
<tr>
<th>Kernel</th>
<th>Nugget / sill</th>
<th>Failed sites</th>
<th>Error</th>
<th>Gap</th>
<th>Same best range</th>
<th>ms per evaluation</th>
</tr>
</thead>
<tbody>
<tr>
<td>Exponential</td>
<td>0</td>
<td>0</td>
<td>0.0021</td>
<td>4.2</td>
<td>yes</td>
<td>15</td>
</tr>
<tr>
<td>Exponential</td>
<td>1e-3</td>
<td>0</td>
<td>0.0095</td>
<td>3.1</td>
<td>yes</td>
<td>15</td>
</tr>
<tr>
<td>Exponential</td>
<td>0.1</td>
<td>0</td>
<td>0.00054</td>
<td>0.07</td>
<td>yes</td>
<td>16</td>
</tr>
<tr>
<td>Matérn 3/2</td>
<td>0</td>
<td>6,578</td>
<td>n/a</td>
<td>43</td>
<td>n/a</td>
<td>15</td>
</tr>
<tr>
<td>Matérn 3/2</td>
<td>1e-4</td>
<td>48</td>
<td>n/a</td>
<td>25</td>
<td>n/a</td>
<td>15</td>
</tr>
<tr>
<td>Matérn 3/2</td>
<td>1e-3</td>
<td>0</td>
<td>0.21</td>
<td>16</td>
<td>yes</td>
<td>16</td>
</tr>
<tr>
<td>Matérn 5/2</td>
<td>0</td>
<td>9,411</td>
<td>n/a</td>
<td>136</td>
<td>n/a</td>
<td>16</td>
</tr>
<tr>
<td>Matérn 5/2</td>
<td>1e-4</td>
<td>1,211</td>
<td>n/a</td>
<td>24</td>
<td>n/a</td>
<td>16</td>
</tr>
<tr>
<td>Matérn 5/2</td>
<td>1e-3</td>
<td>0</td>
<td>0.092</td>
<td>12</td>
<td>yes</td>
<td>16</td>
</tr>
<tr>
<td>Matérn 5/2</td>
<td>1e-2</td>
<td>0</td>
<td>0.016</td>
<td>5.9</td>
<td>yes</td>
<td>15</td>
</tr>
<tr>
<td>Gaussian</td>
<td>0</td>
<td>9,046</td>
<td>n/a</td>
<td>n/a</td>
<td>n/a</td>
<td>15</td>
</tr>
<tr>
<td>Gaussian</td>
<td>1e-4</td>
<td>3,055</td>
<td>n/a</td>
<td>40</td>
<td>n/a</td>
<td>15</td>
</tr>
<tr>
<td>Gaussian</td>
<td>1e-3</td>
<td>0</td>
<td>0.46</td>
<td>16</td>
<td>yes</td>
<td>17</td>
</tr>
<tr>
<td>Gaussian</td>
<td>1e-2</td>
<td>0</td>
<td>0.0050</td>
<td>7.1</td>
<td>yes</td>
<td>15</td>
</tr>
</tbody>
</table>
<p>"Error" is the largest absolute difference from the binary64 log-likelihood over the ten-point range grid and "gap" is how far the best grid point sits above the next. Where sites failed the likelihood is NaN by design, so no error is defined. The failures are stricter than the emulated Cholesky failures because <code>GPULocalKriging</code> also reports status 2 at a pivot ratio below <code>1e-4</code>.</p>
<ul>
<li>At n = 100,000, m = 30 (about 100 ms per evaluation on the same GPU): exponential 1e-3 error 0.26 against a gap of 45, Matérn 3/2 1e-3 error 0.51 against a gap of 42, Matérn 3/2 1e-2 error 0.016 against 13. Exponential with <strong>no</strong> nugget now reports 18 failed sites, from the kriging pivot-ratio check on closely spaced neighbours, so the conclusion that the exponential is safe at any nugget does not extend past n = 10,000 under this solver. Use a small nugget.</li>
<li>At n = 500,000 one evaluation takes about 570 ms; the single-threaded setup (conditioning sets by grid) about 2 s.</li>
<li>With max-min ordering at n = 10,000 the same pattern holds: exponential 1e-3 error 0.0096, Matérn 3/2 1e-3 error 0.19, Gaussian 1e-3 error 0.73, same best range.</li>
</ul>
<p>Fisher scoring in float32 was within binary64 precision for six datasets at n = 2,500 and two at n = 10,000 (exponential and Matérn 3/2, four to six iterations, parameters agreeing to about 5 digits, log-likelihood within 0.001). When the data has no nugget and Matérn 3/2 is held at the floor, Fisher scoring zigzags in binary64 as well as float32, so <code>converged</code> is not reached in 30 iterations: that limit is not an arithmetic one.</p>
<p>GpGp, the intended external oracle, is R-only; the reference here is a binary64 CPU implementation of the same conditioning sets and the same ordering.</p>`};export{e as default};