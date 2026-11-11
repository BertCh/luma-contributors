# New contributors: directions, measurements and options (2026-11-11)

Question from the maintainer: which new contributors should be added, leaning on what the GPU is good at (neural networks included), at sizes that are reasonable in a browser, and still about geospatial statistics, analysis and presentation.

The plan that follows from this folder is [roadmap.md](../../roadmap.md) section 14. This file is the evidence and the reasoning. Written in this repository, not carried over.

## What was done

- A census of what exists: [census.md](./census.md) (every exported contributor and recipe, and the published `@luma.gl/graph` 10.0.0-alpha.12 building blocks).
- Five web research reports, one per direction, each ending in a ranked list with sources and a list of what could not be verified:
  - [research-fitted-models.md](./research-fitted-models.md): models fitted in the browser on the analyst's own data.
  - [research-pretrained-models.md](./research-pretrained-models.md): pretrained geospatial models and embedding products.
  - [research-statistical-simulation.md](./research-statistical-simulation.md): simulation, uncertainty and factorization methods without neural networks.
  - [research-presentation.md](./research-presentation.md): computations whose purpose is the map.
  - [research-browser-runtime.md](./research-browser-runtime.md): WebGPU limits, runtimes, and the choice between writing kernels and using ONNX Runtime Web.
- Three probes on the test GPU, kept as `.spec.ts.txt` with their numbers in [probe-results.json](./probe-results.json). To rerun one, copy it to `modules/analysis/test/utils/` without the `.txt` and run `yarn vitest run --project headless` on it. Each fails on purpose so that the report is printed; do not leave it in the test tree.

## What the library has today

About 600 contributors, all classical. The only learned-model code is eleven `RasterEmbedding*` classes (a per-pixel softmax linear probe, a k-nearest classifier, k-means, similarity and change over precomputed embeddings). There is no dense layer, no gradient, no tree ensemble and no pretrained weight anywhere.

Published luma.gl has `GPUMatMul`, `GPUMatVec`, `GPUConvolution` (one channel, direct or FFT), `GPUFFT2D` (powers of two up to 2048 per side, batched), `GPUJacobiPCG` (one right-hand side), and a vector-search package (`GPUKMeans`, `GPUIVFFlatIndex`, `GPUSimilaritySearch`). No contributor here uses `GPUMatMul` or `GPUJacobiPCG` yet. Neural layers would need what these lack: channels, bias, activation, normalisation, attention, any backward pass.

## Measurements (Apple M3 Pro, 18 GPU cores, headless Chromium, float32)

Plain WGSL through raw WebGPU, random data, every case checked against a CPU reference. Times are medians of five runs, confirmed by a second clock that ends on a mapped read. One multiply-accumulate (MAC) is one weight times one value added to a sum.

Inference ([neural-layer-throughput-probe.spec.ts.txt](./neural-layer-throughput-probe.spec.ts.txt)):

| Case | Work | Time | Rate |
| --- | --- | --- | --- |
| Dense layer 64 to 64 over 262,144 rows, one invocation per output | 1.07 billion MAC | 8.8 ms | 122 billion MAC/s |
| The same, one invocation per row | 1.07 billion | 2.2 ms | 488 billion/s |
| Whole coordinate network (32 to 64 to 64 to 1) in one kernel, 1,048,576 points | 6.5 billion | 10.6 ms | 614 billion/s |
| The same shape at 16 to 32 to 32 to 1, 4,194,304 points | 6.6 billion | 11.9 ms | 553 billion/s |
| 3 by 3 convolution, 32 to 32 channels, 512 by 512 | 2.4 billion | 24.8 ms | 97 billion/s |
| 3 by 3 convolution, 64 to 64 channels, 256 by 256 | 2.4 billion | 24.8 ms | 97 billion/s |

Shader compile time was 19 to 52 ms per kernel.

Training ([neural-field-fit-probe.spec.ts.txt](./neural-field-fit-probe.spec.ts.txt)): a coordinate network with Fourier features and two hidden layers, fitted to scattered samples of a synthetic surface (six hills and a ripple) by hand-written backpropagation and Adam, full batch. The gradient of each weight is summed in two fixed-order levels (256 rows per tile, then the tiles), so no kernel loops over the data. The error is the held-out root mean square error divided by the surface's standard deviation, from a float64 forward pass on the CPU with the weights read back.

| Samples | Parameters | Time per step | Held-out error after 250, 1,000, 2,000 steps | Time for those steps |
| --- | --- | --- | --- | --- |
| 65,536 | 5,825 | 4.0 ms | 0.26, 0.10, 0.06 | 1.0 s, 4.0 s, 8.0 s |
| 65,536 | 1,889 | 2.2 ms | 0.30, 0.11, 0.06 | 0.5 s, 2.1 s, 4.3 s |
| 262,144 | 5,825 | 14.6 ms | 0.24, 0.09, not run | 3.7 s, 14.6 s |
| 4,096 | 5,825 | 0.3 ms | 0.46, 0.29, 0.24 | 0.09 s, 0.33 s, 0.65 s |

The 262,144-sample case kept 281 MB of per-row state in one buffer. That fits this device, which was created with the adapter's limits, and it is more than twice the 128 MiB a default device allows for one binding. A contributor needs minibatches or a narrower per-row state.

Forest inference ([forest-inference-probe.spec.ts.txt](./forest-inference-probe.spec.ts.txt)): complete binary trees with random splits, one invocation per cell.

| Cells | Features | Forest | Time |
| --- | --- | --- | --- |
| 1,048,576 | 16 | 100 trees, depth 6 | 9.6 ms |
| 1,048,576 | 16 | 500 trees, depth 8 | 69 ms |
| 1,048,576 | 16 | 1,000 trees, depth 10 (11.7 MB) | 308 ms |
| 4,194,304 | 8 | 100 trees, depth 6 | 16.7 ms |

What the probes do not show: any other machine; real trained forests, which are unbalanced; a hash-grid encoder; attention; the cost of running through a luma.gl command graph instead of raw WebGPU; float16.

## Findings

1. **Small networks and forests are well inside the budget on this machine.** A per-point network of a few thousand weights evaluates a million points in about 10 ms and trains at 250 steps per second on 65,536 samples. The research agents had to assume a speed (0.1 to 0.5 trillion MAC per second); the measured figures are 0.09 to 0.1 for a naive convolution, 0.26 for training and 0.5 to 0.6 for a fused per-point network. So their time estimates are about right for convolution and pessimistic by 2.5 to 6 times for dense networks, on this GPU. Published figures for integrated GPUs span about 35 times (0.04 to 1.4 trillion), so a weaker laptop or a phone is still unmeasured.
2. **Float32 training converges and can be made repeatable.** The two-level fixed-order gradient sum is the pattern `regression-row-tiles.ts` already uses for regression sums. Published `GPUMatMul` cannot do this sum under the contract, because its inner dimension would be the row count.
3. **The cheapest valuable additions are not neural.** A tree-ensemble predictor is the step missing from the published AlphaEarth workflows (Google's regression tutorial uses a 50-tree random forest) and from what most analysts already train in Python. Gaussian random fields by FFT, a flow-based cartogram by FFT, and CLAHE are all built from published primitives.
4. **Pretrained image encoders are the expensive direction.** The smallest usable one with a clear licence (Galileo-Nano, 0.8 million parameters, MIT) needs a transformer layer set written in WGSL. Larger ones need tens of megabytes of weights and a wide set of layer types. Several attractive models have unclear or restrictive weight licences (OmniCloudMask, Dynamic World, OlmoEarth, MobileSAM); neural relief shading has no public weights.
5. **Neural fields are not a better kriging.** No published study shows them beating kriging on scattered points; the one direct comparison found tuned inverse distance weighting best. Their case is scale, a compact resolution-free surface and gap filling, with no uncertainty unless an ensemble and the existing conformal classes are added.
6. **Generic layers are luma.gl's business under the upstream-first rule.** A layer library here would be a re-implementation of something that belongs upstream. What fits here is a fused kernel named for its analysis purpose and a few shared kernels in `analysis-core`.
7. **ONNX Runtime Web can share a device.** Version 1.31.1 accepts an external `GPUDevice` and keeps tensors on the GPU, per its documentation; this was not tried here. WebNN ships in no stable browser. In-browser training in ONNX Runtime is CPU only.

## Five directions

Effort: S is days, M is one to two weeks, L is more. IDs are the roadmap's.

### 1. Models the analyst brings or fits (LRN)

No shipped weights, no licence questions, tabular or raster input.

| ID | Candidate | Effort | Note |
| --- | --- | --- | --- |
| LRN-01 | Forest inference: predictions of an imported random forest or boosted model over rows or a raster stack, per frame | M | Measured above. Oracle: scikit-learn (installed) |
| LRN-02 | Shared fitting kernels in `analysis-core`: row-tiled gradient sums, an Adam step, a seeded minibatch order, coordinate encoders | M | Needed by LRN-03, ENC-01, the autoencoder |
| LRN-03 | Neural field: a continuous surface through points or a raster, for interpolation at scale, gap filling and compression; Fourier features first, a hash grid second; two dimensions and space-time | L | Measured above for Fourier features. Oracle: PyTorch (installed) |
| LRN-04 | Gaussian mixture by expectation-maximization, with BIC | M | Oracle: scikit-learn |
| LRN-05 | Self-organizing map with a GeoSOM mode | S to M | An established geodemographics tool |
| LRN-06 | Permutation importance per covariate, optionally in spatial blocks | S | After LRN-01 |
| LRN-07 | Gradient-boosted trees trained in the browser (histogram method; squared, logistic and quantile loss) | L | Later |
| LRN-08 | TreeSHAP contribution maps | M | Later, after LRN-01 |
| LRN-09 | Ensemble option on the fitted networks, feeding the existing conformal classes | S | After LRN-03 |

Skipped: GNNWR (its input grows with the sample count; about 100 minutes for 10,000 points by estimate, and the reference is GPL), geographical random forest, quantile regression forests, UMAP (racy updates by design), sparse variational Gaussian processes (need a Cholesky across invocations).

### 2. Embeddings, deeper (EMB)

Small extensions of the eleven existing classes. AlphaEarth is CC BY 4.0 and now readable without an account; TESSERA is CC0.

| ID | Candidate | Effort |
| --- | --- | --- |
| EMB-01 | More decoders (TESSERA per-pixel scale, Presto uint16, float16) and the missing int8 encoder; verify each formula first | S |
| EMB-02 | More than 128 dimensions (192, 256, 384; 1,024 where the binding allows) | S |
| EMB-03 | Multi-year trajectory: cosine to a base year, year of largest step, cumulative angle | S |
| EMB-04 | Anomaly score: Mahalanobis distance from the covariance class, or distance to the nearest label | S |
| EMB-05 | Regression output on the linear probe; mean-of-references similarity (what Google's tutorial computes) | S |
| EMB-06 | Edge strength between neighbouring embeddings, as input to watershed and sieve | S, maybe |

LRN-01 over an embedding raster completes the published AlphaEarth regression recipe.

### 3. Simulation, uncertainty and factorization (SIM)

| ID | Candidate | Effort | Note |
| --- | --- | --- | --- |
| SIM-01 | Gaussian random field from a variogram model: FFT to 1024², random spectral modes beyond, conditioning by kriged residuals | S to M | Lifts the 4,096-row cap of conditional simulation; feeds SIM-02, SIM-03 |
| SIM-02 | Terrain error ensemble recipe: probable viewshed, depression probability, flow-path probability | M | Uncertainty a reader can see change |
| SIM-03 | Sensitivity indices (Sobol, Morris) for the weighted overlay and other per-frame contributors | M | Nothing like it exists |
| SIM-04 | Randomized SVD as a shared primitive | M | Moves the EOF eigen step off the CPU; unblocks SIM-06 and Moran eigenvector maps |
| SIM-05 | Geographically weighted PCA | M | Reuses the GWR neighbour grid |
| SIM-06 | Non-negative matrix factorization | S to M | Unblocks roadmap RS10 |
| SIM-07 | Zone raking (iterative proportional fitting across many zones) | S to M | Synthetic populations |
| SIM-08 | Movement states by hidden Markov model | M, maybe | Published `GPUSegmentedScan` is a uint32 prefix sum only, so the scan over matrix products would be local code |
| SIM-09 | Location obfuscation (donut, Gaussian, planar Laplace) with its effect on density and hot spots | S | |
| SIM-10 | Structured kernel interpolation: the kriging mean of 100,000 to 10 million points on a grid by FFT and conjugate gradients | M | First user of `GPUJacobiPCG`; variance left for later |

Skipped or kept excluded: many-chain MCMC and redistricting ensembles (command-node budget until PLAT-04), multiple-point and sequential indicator simulation (serial), wildfire and urban growth (excluded by the 2026-11-09 review), full INLA.

### 4. Presentation (CART)

Computations that produce data for a map. Several items in the brief already exist (`GPUTerrainHorizon` gives sky-view factor and openness; `GPUStreamlines` places evenly spaced lines).

| ID | Candidate | Effort | Picture |
| --- | --- | --- | --- |
| CART-01 | Flow-based contiguous cartogram (Gastner, Seguy and More 2018) on `GPUFFT2D` | M to L | Regions swell to their population |
| CART-02 | Point label placement | M | Stations labelled live; no public reference code, so a CPU twin is the oracle |
| CART-03 | CLAHE and histogram matching | M | Closes deferred R7.1. Oracle: scikit-image (installed) |
| CART-04 | Displacement interpolation between two densities, on the Sinkhorn kernel | M | Morning trips flow into evening trips |
| CART-05 | Point cluster hierarchy per zoom, with a blue-noise sample | M | Ten million points merging into bubbles; the algorithm is the research agent's design, not a paper's |
| CART-06 | Centred travel-time map | S | The street map bends around the pointer; also the agent's design |
| CART-07 | Small set: illuminated contours, value-suppressing uncertainty classes and exceedance probability, colour-vision simulation, guided filter, parallel-coordinate densities, pansharpening | S each | |
| CART-08 | Field-based t-SNE linked to the map | L | Later |
| CART-09 | Road strokes for network thinning | M | Later |

Skipped: neural relief shading (no weights), tile-grid assignment, necklace maps, spiral trees, typification.

### 5. Pretrained encoders (ENC)

| ID | Candidate | Effort | Note |
| --- | --- | --- | --- |
| ENC-01 | Location encoder (SatCLIP): coordinates in, a 256-value vector out, as covariates for regression | S to M | About 1.2 million weights (4.8 MB float32, an estimate), MIT. Dense layers and a sine only. The model card limits intended use to research and excludes defence and surveillance |
| ENC-02 | Transformer kernels and the Galileo-Nano encoder | L, maybe | The first image encoder worth it; 3.2 MB (estimate), MIT |
| ENC-03 | Presto on sample pixels (not rasters: about 89 million operations per pixel) | M after ENC-02, maybe | Weight licence not stated |
| ENC-04 | A separate optional package that wraps ONNX Runtime Web on the shared device, for larger encoders | Spike first | Not in `@luma-contributors/analysis` |

Skipped: MobileSAM and other promptable segmentation (40 MB, the widest layer set, already served by ONNX Runtime Web), OmniCloudMask, Dynamic World, OlmoEarth weights (use restrictions), Major TOM embeddings (share-alike).

## Size budget

From the runtime report, restated against the measurements. The planning speed is 0.1 trillion MAC per second: what a naive convolution reaches on the test GPU, and the report's estimate for the weakest supported laptop.

| Tier | What | Most parameters | Most weights | Most work per run |
| --- | --- | --- | --- | --- |
| 0 | Per-point or per-pixel network, trained or applied here | 100,000 | 0.4 MB | 5 billion MAC (about 10 ms on the test GPU, 50 ms at the planning speed) |
| 1 | Small pretrained network as WGSL | 2 million | 8 MB float32, 4 MB float16 | 10 billion MAC |
| 2 | Pretrained encoder in a separate package | 25 million | 50 MB float16 | 50 billion MAC per run, not per frame |

Also: one storage binding stays under 128 MiB unless the contributor checks the device limit at the start of `getCommandNodes`; a run stays in the tens of dispatches (Firefox costs about 1 ms per dispatch against 30 to 70 microseconds in Chrome and Safari, per one 2026 paper); weights arrive as a caller-supplied view and are never bundled in a package.

## Recommendation

My judgement, from the reports and the probes:

1. Start with the foundations that several candidates share: LRN-01's forest layout, LRN-02's fitting kernels, SIM-01's random field. Measure the probes on a second, weaker device before fixing the limits.
2. First wave, independent and each S or M: LRN-01, EMB-01 to EMB-05, SIM-01, CART-03, CART-06 and the CART-07 set.
3. Second wave, the visible ones: LRN-03 (neural field), ENC-01 (location encoder), SIM-02, SIM-03, CART-01, LRN-04, LRN-05.
4. By demand: the rest. ENC-02 to ENC-04 wait for a consumer and for decisions 1 and 4 below.

## Decisions for the maintainer

1. **Where generic layers live.** Recommended: fused kernels named for their analysis purpose here, the shared fitting kernels in `analysis-core`, and a luma.gl proposal for channel convolution, a dense layer with bias and activation, softmax and normalisation. The alternative, a layer set in this repository, conflicts with AGENTS.md.
2. **Where the new families live.** Source is allowed only in the analysis-owned directories. Recommended: no new directory. Forest, mixture and map models in `dataframe-analysis`; the neural field and random field beside interpolation in `gpu-spatial-analysis`; the cartogram beside the Dorling cartogram; CLAHE beside raster stretch; embeddings where they are.
3. **Pretrained weights.** Recommended: never in a package; the contributor takes a weights view; examples and the website fetch them and record them in `THIRD-PARTY-DATA.md`. ENC-01 needs a decision on SatCLIP's research-use wording.
4. **ONNX Runtime Web.** Recommended: not a dependency of the analysis package; a spike in an example decides whether ENC-04 is worth a package.
5. **Device limits.** The website and examples create devices with default limits or with the adapter's. The neural field's capacity depends on which.

## Not verified

- Every timing in the five research reports is an estimate; only the tables above were measured, on one machine.
- Weight file sizes except MobileSAM's are parameters times four bytes. No ONNX export of Presto, Galileo or SatCLIP was found.
- Licences of the weights of Presto, OmniCloudMask, CloudSEN12, Dynamic World, MobileSAM and the canopy-height models; the software licences of several oracles (GWmodel, WhiteboxTools, CartAGen, tiny-cuda-nn, jax-js); R-INLA's licence is listed two ways.
- The decode formulas for TESSERA and OlmoEarth int8 embeddings.
- Whether ONNX Runtime Web works on a device that luma.gl created.
- Several classic papers were cited from their usual addresses without being opened; each report lists which.
- Oracles installed on this machine (`/usr/bin/python3`): scikit-learn 1.6.1, PyTorch 2.7.0, scikit-image 0.24.0, OpenCV 4.12.0. Not installed: XGBoost, LightGBM, GSTools, POT, SALib.
