# Executable interpolation lenses

These examples use six explicitly synthetic planar sites with response `2 + 0.5x - 0.25y` and four interior query sites. Every result is produced by the GPU contributor and captured with its graph, input views, settings and status outputs. The [interpolation contracts](./interpolation-expansion.md) describe model assumptions, bounds and numerical failure policies.

## Local universal kriging

{/* lens:GPULocalKriging */}

::lens{scenario=gpu-spatial-analysis.local-kriging rung=result caption="Affine universal kriging with six neighbors, a supplied exponential covariance and separate measurement-error variance."}

::lens{scenario=gpu-spatial-analysis.local-kriging rung=graph caption="The GPU neighborhood solve and scalar prediction capture; inspect results for variance, pivot ratio and retained count."}

The sample observation rows contain response, measurement-error variance `0.1`, variable index and validity. The result includes the latent point variance and numerical conditioning evidence.

## Local thin-plate interpolation

{/* lens:GPULocalRBF */}

::lens{scenario=gpu-spatial-analysis.local-rbf rung=result caption="The thin-plate kernel with an affine polynomial and smoothing 0.02 reproduces the synthetic affine field."}

::lens{scenario=gpu-spatial-analysis.local-rbf rung=graph}

The graph retains six neighbors per query and captures prediction, pivot ratio, retained count and status. These are point predictions with explicitly permitted extrapolation in the general API; this dataset uses interior queries.

## Sibson natural neighbors

{/* lens:GPUNaturalNeighbor */}

::lens{scenario=gpu-spatial-analysis.natural-neighbor rung=result caption="Sibson interpolation from actual stolen Voronoi areas, with captured query-major sample weights."}

::lens{scenario=gpu-spatial-analysis.natural-neighbor rung=graph}

The example captures all 24 query/sample weights together with cell area, contributing count and status. Inspect weight partition and affine precision; outside-hull invalidity is covered by the API's numerical tests.

## Conditional Gaussian simulation

{/* lens:GPUConditionalGaussianSimulation */}

::lens{scenario=gpu-spatial-analysis.conditional-gaussian-simulation rung=result caption="First of 16 jointly simulated four-site fields, conditioned on the synthetic observations using known mean 2 and Philox seed 347."}

::lens{scenario=gpu-spatial-analysis.conditional-gaussian-simulation rung=graph}

The result view shows the first realization. Captures also include all 64 realization values, four conditional mean/variance pairs and each replicate's factorization status. The seeded field is stochastic; its individual realization need not reproduce the affine data trend.
