var e={slug:`api/gpu-spatial-analysis/interpolation-lenses`,title:`Executable interpolation lenses`,html:`<h1 id="executable-interpolation-lenses">Executable interpolation lenses</h1>
<p>These examples use six explicitly synthetic planar sites with response <code>2 + 0.5x - 0.25y</code> and four interior query sites. Every result is produced by the GPU contributor and captured with its graph, input views, settings and status outputs. The <a href="/luma-contributors/docs/api/gpu-spatial-analysis/interpolation-expansion">interpolation contracts</a> describe model assumptions, bounds and numerical failure policies.</p>
<h2 id="local-universal-kriging">Local universal kriging</h2>
<p>{/* lens:GPULocalKriging */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-kriging rung=result caption="Affine universal kriging with six neighbors, a supplied exponential covariance and separate measurement-error variance."}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-kriging rung=graph caption="The GPU neighborhood solve and scalar prediction capture; inspect results for variance, pivot ratio and retained count."}</p>
<p>The sample observation rows contain response, measurement-error variance <code>0.1</code>, variable index and validity. The result includes the latent point variance and numerical conditioning evidence.</p>
<h2 id="local-thin-plate-interpolation">Local thin-plate interpolation</h2>
<p>{/* lens:GPULocalRBF */}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-rbf rung=result caption="The thin-plate kernel with an affine polynomial and smoothing 0.02 reproduces the synthetic affine field."}</p>
<p>::lens{scenario=gpu-spatial-analysis.local-rbf rung=graph}</p>
<p>The graph retains six neighbors per query and captures prediction, pivot ratio, retained count and status. These are point predictions with explicitly permitted extrapolation in the general API; this dataset uses interior queries.</p>
<h2 id="sibson-natural-neighbors">Sibson natural neighbors</h2>
<p>{/* lens:GPUNaturalNeighbor */}</p>
<p>::lens{scenario=gpu-spatial-analysis.natural-neighbor rung=result caption="Sibson interpolation from actual stolen Voronoi areas, with captured query-major sample weights."}</p>
<p>::lens{scenario=gpu-spatial-analysis.natural-neighbor rung=graph}</p>
<p>The example captures all 24 query/sample weights together with cell area, contributing count and status. Inspect weight partition and affine precision; outside-hull invalidity is covered by the API's numerical tests.</p>
<h2 id="conditional-gaussian-simulation">Conditional Gaussian simulation</h2>
<p>{/* lens:GPUConditionalGaussianSimulation */}</p>
<p>::lens{scenario=gpu-spatial-analysis.conditional-gaussian-simulation rung=result caption="First of 16 jointly simulated four-site fields, conditioned on the synthetic observations using known mean 2 and Philox seed 347."}</p>
<p>::lens{scenario=gpu-spatial-analysis.conditional-gaussian-simulation rung=graph}</p>
<p>The result view shows the first realization. Captures also include all 64 realization values, four conditional mean/variance pairs and each replicate's factorization status. The seeded field is stochastic; its individual realization need not reproduce the affine data trend.</p>`};export{e as default};