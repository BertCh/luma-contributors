# Fitted models in the browser: neural and gradient-based spatial models on WebGPU

Research for new contributors, 2026-11-11. Question: which neural, kernel and tree models can be *trained* on the analyst's own data, in the browser, in about one to ten seconds on an Apple M-class integrated GPU, as data-parallel WGSL kernels with hand-derived gradients, and which of them add something the ~600 existing contributors (see `census.md`) do not.

## 1. Planning assumptions

**Throughput.** Apple rates the M1 GPU at a theoretical 2.6 TFLOP/s float32 and the 10-core M2 at 3.6 ([Tom's Hardware](https://tomshardware.com/news/apple-m2-gpu-analysis)), but a measurement study found 1.36 TFLOP/s peak on the M1 and 2.9 on the M4 ([arXiv 2502.05317](https://arxiv.org/pdf/2502.05317v2)). jax-js reports matrix multiplication above 3 TFLOP/s on an M4 Pro through WebGPU ([jax-js blog](https://ss.ekzhang.com/p/jax-js-an-ml-library-for-the-web)). For planning I use **1×10¹¹ multiply-accumulates per second (0.2 TFLOP/s) sustained** for hand-written, unfused WGSL kernels that pass through global memory between layers. *This is an estimate, not a measurement*; all times below scale linearly from it, and the first task of any build is to measure it on the test GPU.

**Training cost rule.** For a dense layer, one training step costs about three times the forward multiply-accumulates (forward, the input gradient, and the weight gradient). This is the standard accounting, used below without further citation.

**The contract changes the kernel shape.** Every weight gradient `dW = Σ_batch aᵀ·δ` and every Gram matrix `ΦᵀΦ` is a sum over a data-sized range. `dev-docs/contributor-contract.md` forbids looping over rows inside one invocation and requires sums over data to use the fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`) or a fixed-order tree. A plain tiled `GPUMatMul` whose inner loop runs over the batch would violate this, so every training contributor needs a **split-K** weight-gradient kernel: partial products over fixed batch chunks, then a fixed-order tree reduction. A useful side effect: fixed-order or fixed-point gradients make training *bit-for-bit reproducible*, which no browser ML library I found claims.

**Workgroup memory.** The WebGPU default for `maxComputeWorkgroupStorageSize` is 16,384 bytes ([WebGPU spec, limits](https://www.w3.org/TR/webgpu/#limits)). tiny-cuda-nn's "fully fused" MLP keeps all activations of a 64-wide network in shared memory and its README warns it needs a very large amount of it ([Müller et al. 2021](https://arxiv.org/pdf/2106.12372)). A 64-wide float32 tile of 64 rows is exactly 16 KB, so fully fused training is at the limit; plan on layer-by-layer kernels first.

## 2. Neural fields: coordinate MLPs, Fourier features, SIREN, hash grids

**Question answered.** "Give me a continuous, resolution-free surface through these points (or this raster), fill the gaps, and let me query it anywhere"; also lossy compression of a DEM or a space-time cube into a few hundred kilobytes of weights.

**References.** Fourier features ([Tancik et al. 2020](https://arxiv.org/abs/2006.10739)); SIREN ([Sitzmann et al. 2020](https://arxiv.org/abs/2006.09661)); multiresolution hash encoding ([Müller et al. 2022](https://arxiv.org/abs/2201.05989)). Instant-NGP uses L = 16 levels, F = 2 features per entry, tables of 2¹⁴ to 2²⁴ entries, an MLP of two hidden layers of width 64, Adam with β₁ = 0.9, β₂ = 0.99, ε = 10⁻¹⁵, learning rate 10⁻², batches of 2¹⁸, and skips Adam steps for entries whose gradient is zero; colliding gradients are averaged; it fitted a gigapixel image to 38.59 dB PSNR in 2.5 minutes on an RTX 3090 ([paper text](https://ar5iv.labs.arxiv.org/html/2201.05989)). Oracle: [tiny-cuda-nn / instant-ngp](https://nvlabs.github.io/instant-ngp) (NVIDIA source-available licence, not open source; check before use) or a 60-line PyTorch script.

**Cost, worked.**

| Variant | Parameters | Training MAC per sample | Problem | Total MAC | Time at 10¹¹/s |
|---|---|---|---|---|---|
| Fourier features (2 → 128 features) + 3 hidden × 128 | ≈ 3 × 128² ≈ 49 k | ≈ 3 × 49 k ≈ 1.5×10⁵ | 1 M points, 20 epochs | 2×10⁷ × 1.5×10⁵ = 3×10¹² | ≈ 30 s (estimate) |
| same | | | 4096² raster, 1 epoch | 1.7×10⁷ × 1.5×10⁵ = 2.5×10¹² | ≈ 25 s (estimate) |
| Hash grid L = 16, F = 2 + MLP 32 → 64 → 64 → 1 | MLP ≈ 6.2 k; table 16 × 2 × T | ≈ 3 × 6.2 k ≈ 1.9×10⁴ plus 64 gathers and 64 scatter-adds (2-D bilinear, 4 corners × 16 levels) | 1 M points, 20 epochs | 2×10⁷ × 1.9×10⁴ ≈ 4×10¹¹ | ≈ 4 s plus atomics (estimate) |

The hash grid is the only neural-field variant inside the ten-second class at a million samples. Its encoding backward pass is a scatter-add into the table. WGSL has atomics only on 32-bit integers, so the scatter-add is a fixed-point `atomicAdd` on `i32`, which the contract favours anyway and which is order-independent. Compression only happens when the table is smaller than the data: at T = 2¹⁹ the table holds 16 × 2 × 2¹⁹ ≈ 16.8 M parameters, as many as a 4096² raster has cells. For DEMs, recent work uses cascaded SIRENs plus post-training compression ([arXiv 2605.22556](https://arxiv.org/pdf/2605.22556)) and amortized SIREN decoders on 1 m terrain ([arXiv 2606.00404](https://arxiv.org/abs/2606.00404)); I found no published hash-grid DEM codec.

**Against kriging.** The evidence is thin and mixed. On 100 sparse ECA&D climate datasets an optimized IDW beat both ordinary kriging and an implicit neural representation (MMGN) in accuracy and time ([Walczak 2025](https://arxiv.org/abs/2512.11832)). A 2026 ecology study reports "stable continuous representations with predictable computational cost" but, from its abstract, no kriging baseline ([Pregowska and Kalaji 2026](https://arxiv.org/abs/2604.18083)). Neural fields were proposed for ocean altimetry as a scalable alternative to optimal interpolation ([arXiv 2211.10444](https://arxiv.org/pdf/2211.10444)). I found no head-to-head study on scattered-point interpolation that a library could cite to claim neural fields are more accurate than kriging. Their honest selling points are scale (millions of samples, no covariance matrix), resolution-free querying and compactness, not accuracy, and they give no uncertainty.

**float32 risks.** Fourier features of geographic coordinates: `sin(2π·f·x)` with metre-scale x and high f loses all precision. Encode in a local, normalised frame (the library's global-coordinates conventions already do this). SIREN needs its specific initialisation and ω₀ = 30 scaling ([Sitzmann et al.](https://arxiv.org/abs/2006.09661)); WGSL `sin` of large arguments is implementation-defined in accuracy. Adam's ε = 10⁻¹⁵ is representable in float32 (normal numbers go down to about 1.2×10⁻³⁸), but GPUs may flush denormals, so the second-moment estimate of a rarely touched hash entry can reach zero; keep ε well above the denormal range.

**Verdict.** **Do** a hash-grid (2-D and 3-D space-time) neural field. **Maybe** Fourier/SIREN, only as encoder options in the same contributor. Present it as interpolation and compression, never as "better than kriging".

## 3. Neural extensions of spatial regression

| Method | What it adds | Reference | Code / licence | Cost class | Verdict |
|---|---|---|---|---|---|
| GNNWR / GTNNWR | GWR whose coefficient weights come from a neural network of distances instead of a kernel | [Du et al. 2020](https://gmd.copernicus.org/preprints/gmd-2024-62); [GMD 2024 package paper](https://gmd.copernicus.org/articles/17/8455/2024/) | [zjuwss/gnnwr](https://github.com/zjuwss/gnnwr), GPL-3.0 | minutes to hours | skip as published |
| DeepKriging | MLP on a multiresolution basis-function embedding of coordinates (plus covariates); no covariance matrix | [Chen, Li, Reich, Sun 2024](https://arxiv.org/abs/2007.11972), Statistica Sinica 34 | no repository found | seconds | do (as encoder option) |
| Space2Vec, Sphere2Vec, spherical harmonics + SIREN | Learned multi-scale or spherical location encoders | [Mai et al. 2020](https://arxiv.org/abs/2003.00824); [Mai et al. 2023](https://arxiv.org/abs/2306.17624); [Rußwurm et al. 2024](https://arxiv.org/abs/2310.06743) | [locationencoder](https://github.com/marccoru/locationencoder): no licence file | seconds | do (encoders only) |
| PE-GNN | GNN with learned positional encoder and an auxiliary Moran's I task | [Klemmer et al. 2023](https://proceedings.mlr.press/v206/klemmer23a.html) | [pe-gnn](https://github.com/konstantinklemmer/pe-gnn): no licence file | seconds | maybe |
| Kriging convolutional network | GNN whose messages carry neighbouring observed values | [Appleby et al. 2020](https://ojs.aaai.org/index.php/AAAI/article/view/5716) | PyTorch code linked from [arXiv 2306.09463](https://arxiv.org/abs/2306.09463) | seconds | maybe (after GNN) |
| Neural processes | Meta-learned predictive distribution over many functions | [Garnelo et al. 2018](https://arxiv.org/abs/1807.01622) | | needs many training tasks | skip |

**GNNWR arithmetic.** Its network input is the vector of distances from a location to all n training samples ([GMD 2024](https://gmd.copernicus.org/articles/17/8455/2024/)), and the PM2.5 example uses hidden layers of 1024, 512, 256 and 128 with batch normalisation, dropout and PReLU, trained with Adadelta for 2000 epochs (about 3 minutes on a desktop CPU for a sample count the paper does not state). For n = 10,000 the first layer alone has 10⁴ × 1024 ≈ 10⁷ weights; one epoch costs about 10⁴ samples × 10⁷ × 3 = 3×10¹¹ MAC, and 2000 epochs 6×10¹⁴ MAC, roughly 100 minutes at the planning rate (estimate). The input width grows with n, so it does not scale. A variant fed by a positional encoder instead of the distance vector would be fast, but it is no longer GNNWR and could not use the published package as an oracle. The library already has GWR, MGWR, GTWR and generalized GWR.

**DeepKriging arithmetic.** The embedding is compactly supported basis functions on nested grids, so each point touches only a few dozen basis functions per level (the paper's exact basis counts and network widths were not readable in the pages I fetched; *not verified*). Assume ≈ 40 non-zero basis values and three hidden layers of 100: forward ≈ 40 × 100 + 2 × 100² ≈ 2.4×10⁴ MAC, training ≈ 7×10⁴ per sample; 100 k points × 100 epochs = 10⁷ × 7×10⁴ = 7×10¹¹ MAC ≈ 7 s (estimate). DeepKriging is structurally a *fixed* multiresolution grid encoding followed by an MLP, the same pipeline as the hash grid with a different encoder. It also outputs density predictions for uncertainty ([abstract](https://arxiv.org/abs/2007.11972)). The authors claim advantages for non-Gaussian and nonstationary data, which is the case where the library's kriging is weakest.

**Positional encoders** are cheap feature maps (sinusoids at many scales; spherical harmonics on the globe) and should be a shared kernel, not contributors of their own.

## 4. Scalable Gaussian-process approximations

The library already fits GP hyperparameters well (Vecchia likelihood with Fisher scoring) and predicts with global, local and space-time kriging. The gap is prediction and posterior sampling at a million points or more.

| Method | Reference | Training cost (worked) | Uncertainty | Verdict |
|---|---|---|---|---|
| Random Fourier features | [Rahimi and Recht 2007](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines) | ΦᵀΦ for n = 10⁶, D = 1024: n·D² ≈ 10¹² MAC ≈ 10 s; n = 10⁵: ≈ 1 s (estimate); then a D × D solve | Variance starvation: overconfident away from data ([arXiv 2404.19157 §2.2.4](https://arxiv.org/pdf/2404.19157); [Wilson et al. 2020](https://arxiv.org/pdf/2002.09309)) | maybe |
| Structured kernel interpolation (KISS-GP) | [Wilson and Nickisch 2015](https://arxiv.org/abs/1503.01057) | per CG iteration: n × 16 interpolation weights + FFT on a 1024² grid (≈ 5·N·log₂N ≈ 10⁸); 100 iterations ≈ 10¹⁰ ≈ 0.1–1 s (estimate) | Mean is cheap; variance needs LOVE or sampling | do (posterior mean) |
| Sparse variational GP | [Hensman et al. 2013](https://arxiv.org/abs/1309.6835) | per minibatch step b·m² + m³; m = 256: m³ ≈ 1.7×10⁷ | Calibrated, the best of these | maybe (L) |
| Nyström | | like RFF with m landmarks | similar to RFF | skip (RFF covers it) |
| Deep kernel learning | [Wilson et al. 2016](https://arxiv.org/abs/1511.02222) | MLP + SKI + marginal-likelihood gradient | Good if the likelihood is right | skip (too much at once) |

Oracle for all of these: [GPyTorch](https://github.com/cornellius-gp/gpytorch), MIT licence, which implements SKI/KISS-GP and stochastic variational deep kernel learning.

**Why SKI fits this library.** With hyperparameters taken from the existing Vecchia fit, the posterior mean is the solution of `(W K_UU Wᵀ + σ²I) α = y`, where `W` is sparse interpolation onto a regular grid and `K_UU` on a stationary kernel is Toeplitz (block-Toeplitz in 2-D), so its product is an FFT convolution. Every piece exists in published luma.gl: `GPUFFT2D` for the product, `GPUJacobiPCG` for the solve, `GPUGridBinning` / `GPUGather` for `W`. This is kriging of a million points onto a 4096² raster, and it is the one GP candidate that needs no new linear-algebra machinery.

**float32 risks.** RFF normal equations square the condition number; solve with PCG on Φ rather than Cholesky of ΦᵀΦ, or add jitter. Per-pixel RFF variance costs D² per pixel (10⁶ × 1.7×10⁷ pixels ≈ 1.7×10¹³ MAC at 4096²), so it is not a per-frame quantity at that size. SKI's CG on a smooth kernel converges slowly in float32 when σ² is small relative to the signal; cap iterations and report the convergence status the contract requires. SVGP needs an m × m Cholesky each step; the library's `dense-solve-wgsl.ts` handles small systems per invocation, but m = 256 in one invocation is a serial m³/3 ≈ 5.6×10⁶-operation loop, which needs a blocked, multi-invocation Cholesky that does not exist yet.

## 5. Tree ensembles

**Inference of an imported forest (FIL-style).** RAPIDS FIL showed that layout matters: one tree per thread with nodes stored tree by tree does not coalesce, which motivated its reorganised layouts, and its current layouts are depth-first (default), layered and breadth-first ([RAPIDS 2019](https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35); [cuML FIL docs](https://docs.rapids.ai/api/cuml/legacy/fil/)). Dense storage of a depth-d tree needs 2^(d+1) − 1 nodes, so sparse storage is needed for deep trees ([NVIDIA](https://developer.nvidia.com/blog/sparse-forests-with-fil/)). FIL is being superseded by nvForest ([NVIDIA 2026](https://docs.nvidia.com/nvforest/latest/blog/2026-09-03-supercharge-tree-inference/)). Cost: 500 trees × depth 8 = 4,000 node visits per cell; at 1024² that is 4.2×10⁹ visits, at 4096² 6.7×10¹⁰. This is memory-bound and divergent, so the multiply-accumulate rate does not apply; I estimate sub-second at 1024² and seconds at 4096², *unmeasured*. Value: an analyst trains XGBoost, LightGBM or scikit-learn in Python and applies it to a raster stack in the browser, per frame. Oracles: [XGBoost](https://xgboost.readthedocs.io/en/stable/gpu/) (Apache-2.0), scikit-learn (BSD-3). **Do.**

**Histogram gradient boosting, trained in the browser.** XGBoost's GPU method sums gradients into histogram bins of pre-discretised features ([XGBoost 2018](https://xgboost.ai/2018/07/04/gpu-xgboost-update)); [Mitchell and Frank 2017](https://doi.org/10.7717/peerj-cs.127) is the reference. Each boosting round, per depth level: histogram build (n × p fixed-point atomic adds, `GPUHistogram` / `GPUSegmentedReduction`), split search (`GPUScan` over bins), row partition (`GPUCompaction`). Arithmetic for n = 10⁶, p = 20, depth 6: 1.2×10⁸ adds per round; 100 rounds = 1.2×10¹⁰. The only published timing I found: 10⁶ rows × 50 columns × 500 rounds in 13.87 s on a Pascal Titan X ([XGBoost 0.72 docs](https://xgboost.readthedocs.io/en/release_0.72/gpu/index.html)). Scaling that to 20 columns and 100 rounds gives 1/12.5 of the work, about 1.1 s on the Titan X; an M1 is several times slower, so **a few seconds to tens of seconds** (estimate). Quantile (pinball) loss gives quantile maps without the leaf-membership storage of a quantile regression forest. **Do (L).**

**Random forests, QRF, GRF, RFsp.** Random forest training is the same histogram machinery with bootstrap weights and feature subsampling, but at full depth and hundreds of trees it costs several times more than boosting; **maybe**, after boosting. Quantile regression forests ([Meinshausen 2006](https://jmlr.org/papers/v7/meinshausen06a.html); oracle [quantile-forest](https://github.com/zillow/quantile-forest), Apache-2.0) need the training rows per leaf at prediction time; **skip** in favour of quantile boosting plus the existing conformal intervals. Geographical random forest fits one local forest per location ([SpatialML vignette](https://cran.r-universe.dev/SpatialML/doc/SpatialML.html)): n forests, **skip**. RFsp ([Hengl et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6119462)) is a random forest with buffer distances to observations as covariates; its authors found it about as accurate as kriging but note kriging can win on small, near-linear data. It is a *feature recipe* (the library's `DistanceField` builds the buffers) over a forest, not a contributor.

**GPU TreeSHAP.** [GPUTreeShap](https://peerj.com/articles/cs-880.pdf) (Mitchell, Frank, Holmes 2022) splits each tree into root-to-leaf paths, bin-packs them and solves each path in a SIMT group: up to 19× faster than 40 CPU cores for SHAP values on one V100. Per row and tree the cost is about leaves × depth²: 100 trees × 64 leaves × 36 ≈ 2.3×10⁵ per cell, ≈ 2.3×10¹¹ for 10⁶ cells, a few seconds (estimate). Result: a map per covariate of its contribution. Oracle: XGBoost `pred_contribs` or [shap](https://arxiv.org/abs/1802.03888) (MIT). **Maybe (M)**, after inference.

## 6. Unsupervised and embedding methods

| Method | Adds over the library | Cost (worked) | Building blocks | Oracle | Verdict |
|---|---|---|---|---|---|
| Batch SOM and GeoSOM | Topology-preserving clusters (a 2-D map of prototypes); GeoSOM restricts the winner to units near the geographic winner ([Bação, Lobo, Painho 2005](https://www.novaims.unl.pt/labnt/geosom/GeoSOM.Publ.htm)) | n = 10⁶, 100 units, p = 20: 2×10⁹ MAC per epoch; 20 epochs 4×10¹⁰ ≈ 0.4 s | best-matching unit as in k-means assignment; neighbourhood-weighted fixed-point sums | [MiniSom](https://github.com/JustGlowing/minisom) (MIT; online training, so compare to a NumPy batch-SOM script) | do (S–M) |
| Gaussian mixture by EM | Soft clusters with covariance and membership probabilities, BIC for k | n = 10⁶, K = 10, p = 10 full covariance: ≈ 10⁹ per iteration; 100 iterations ≈ 1 s | responsibilities per row; fixed-point weighted sums; p × p Cholesky per component (`dense-solve-wgsl`) | scikit-learn `GaussianMixture` (BSD-3) | do (M) |
| Autoencoder anomaly map | Nonlinear reconstruction error over many bands; the library has `RasterEOFProjection` (linear) and `LocalOutlierFactor` | 200 → 64 → 8 → 64 → 200: ≈ 2.7×10⁴ forward, 8×10⁴ training per pixel; 10⁶ pixels × 10 epochs ≈ 8×10¹¹ ≈ 8 s | dense layer, Adam | PyTorch script | maybe |
| GCN / GraphSAGE on spatial weights | Nonlinear spatially lagged features ([Kipf and Welling 2017](https://arxiv.org/abs/1609.02907); [Hamilton et al. 2017](https://arxiv.org/abs/1706.02216)) | n = 10⁵, k = 8, hidden 64, 2 layers: ≈ 4.5×10⁸ forward per epoch, ≈ 1.4×10⁹ training; 200 full-batch epochs ≈ 2.8×10¹¹ ≈ 3 s | multi-column SpMV (SpMM) over the weights CSR, dense layer, Adam | PyTorch Geometric (MIT) | maybe |
| t-SNE | 2-D map of high-dimensional attributes; [Pezzotti et al.](https://arxiv.org/abs/1805.10817) computed repulsion as splatted fields in WebGL ([tfjs-tsne](https://research.google/blog/realtime-tsne-visualizations-with-tensorflowjs/), archived) | per iteration: kNN attraction n·k plus a grid field by FFT; 1000 iterations for 10⁵ points: seconds (estimate) | `GPUSimilaritySearch` kNN, `GPUGridBinning`, `GPUFFT2D` | openTSNE / scikit-learn | maybe |
| UMAP | Same purpose; no established WebGPU implementation (an unverified [umap-gpu](https://www.pkgstats.com/pkg:umap-gpu) package; [umap-js](https://github.com/tmcw/umap-js) is CPU) | SGD with negative sampling; GPU versions use racy updates | kNN, SGD | umap-learn (BSD-3) | skip (non-deterministic by design) |
| Spectral / Moran eigenvector embedding | Eigenvectors of the spatial-weights Laplacian, for eigenvector spatial filtering | Lanczos with `GPUGraphSpMV` | SpMV, reorthogonalisation | SciPy `eigsh` | out of scope here (not a fitted model), note for the linear-algebra plan |

## 7. Explainability and uncertainty for presentation

- **Permutation importance maps.** Model-agnostic: for each covariate, permute it (a random-key `GPUSort`), predict again, record the loss increase, optionally within spatial blocks. Cost is p × repeats × one prediction. Reference: [Fisher, Rudin, Dominici 2019](https://arxiv.org/abs/1801.01489). It is cheap once a model has a predict kernel. **Do (S)** as a shared contributor.
- **SHAP.** Exact and fast only for trees (TreeSHAP, above). Kernel SHAP for neural models costs thousands of predictions per point; skip.
- **Deep ensembles** ([Lakshminarayanan et al. 2017](https://arxiv.org/abs/1612.01474)) multiply training cost by M (typically 5); for the seconds-class models here, M = 5 parallel networks in one dispatch is affordable. **MC dropout** ([Gal and Ghahramani 2016](https://arxiv.org/abs/1506.02142)) costs T forward passes at prediction. Both give a spread, not a coverage guarantee. The existing conformal and geo-conformal contributors give marginal coverage; the natural combination is to use the ensemble spread as the normaliser in a normalised conformal score, and to score ensemble members with the existing `EnsembleScores`. The library's area of applicability already answers "is this location like the training data"; ensembles add model disagreement. Recommendation: an ensemble *option* on the MLP contributors (M seeds) plus normalised conformal, rather than separate contributors.

## 8. Precedent for training in a browser

| Library | Training on WebGPU? | Reported numbers | Licence | Use here |
|---|---|---|---|---|
| TensorFlow.js `tfjs-backend-webgpu` | Yes per its FAQ ([npm](https://www.npmjs.com/package/%40tensorflow%2Ftfjs-backend-webgpu)) | An early talk reported no significant out-of-the-box speedup ([W3C 2020](https://www.w3.org/2020/06/machine-learning-workshop/talks/fast_client_side_ml_with_tensorflow_js.html)); no current training benchmark found | Apache-2.0 | oracle for MLP and Adam |
| ONNX Runtime Web training | CPU only: the training component has only a CPU execution provider ([Microsoft 2024](https://opensource.microsoft.com/blog/2024/02/06/on-device-training-training-a-model-in-browser)) | none | MIT | not useful |
| jax-js | Yes, WebGPU with autodiff and JIT ([blog](https://ss.ekzhang.com/p/jax-js-an-ml-library-for-the-web)) | > 3 TFLOP/s matmul on M4 Pro; MNIST to > 99 % "in seconds" | MIT per a third-party listing; not checked in the repository | strongest oracle and throughput yardstick |
| webgpu-torch | Autograd yes; SGD and Adam listed as not done ([README](https://cdn.jsdelivr.net/npm/webgpu-torch@0.3.5/README.md)) | none | not checked | precedent only |
| tinygrad | WebGPU export; a browser PPO training demo exists ([Show HN](https://hn.svelte.dev/item/48136981)) | none | not checked | precedent only |
| Burn (`burn-wgpu`) | Native training on wgpu; browser examples I found are inference ([Burn blog](https://burn.dev/blog/cross-platform-gpu-backend)) | none | not checked | precedent only |

Summary: small-model training in the browser works, but nobody publishes reproducible throughput for MLP backpropagation on WebGPU; one commentator calls most browser-training projects demos ([vin.how](https://hn.jakechampion.name/reader/https://vin.how/blog/train-a-language-model-in-your-browser)). WebGPU per-dispatch validation overhead matters for many small layers ([arXiv 2604.02344](https://arxiv.org/pdf/2604.02344)), which argues for few, wide dispatches per step (a whole minibatch per kernel, and the luma.gl command graph to record the step once).

**How hand-written backprop is organised.** Per layer: forward `Z = A·W + b`, activation, keep `A` and `Z` (or recompute); backward `δ_l = (δ_{l+1}·Wᵀ) ⊙ σ′(Z_l)`, `dW = Aᵀ·δ` (split-K over the batch, fixed-order tree), `db = Σ δ` (same); Adam is an elementwise kernel over all parameters with bias correction ([Kingma and Ba 2015](https://arxiv.org/abs/1412.6980)). tiny-cuda-nn fuses all layers into one kernel with activations in shared memory so only inputs and outputs touch global memory ([Müller et al. 2021](https://arxiv.org/pdf/2106.12372)); that is the second optimisation, after correctness.

## 9. Ranked candidates

Fit-time classes: **A** under 1 s, **B** 1–10 s, **C** over 10 s, at the sizes stated; all are estimates from section 1's rate.

| Rank | Name | Computes | Building blocks | Fit class | Effort | Oracle | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | `GPUForestInference` | Predictions (and leaf indices) of an imported XGBoost / LightGBM / scikit-learn forest over points or a raster stack | gather, sparse tree layout, per-row traversal | A–B per frame (no fit) | M | XGBoost (Apache-2.0), scikit-learn (BSD-3) | **do**: most analysts already train trees in Python; per-frame prediction is new |
| 2 | `GPUStructuredKernelInterpolation` | Kriging posterior mean of 10⁵–10⁷ points onto a grid, hyperparameters from the Vecchia fit | `GPUFFT2D`, `GPUJacobiPCG`, grid binning, gather | A–B | M | GPyTorch KISS-GP (MIT) | **do**: scales existing kriging with published blocks only; variance by later work |
| 3 | `GPUNeuralField` (hash grid, optional Fourier / DeepKriging basis encoders) | Continuous surface through points or a raster, gap filling, compact weights; 2-D and space-time | positional encoder, dense layer forward/backward, fixed-point scatter-add, Adam | B at 10⁶ samples | L | PyTorch script; tiny-cuda-nn (licence to check) | **do**: new capability; present as interpolation/compression, not as more accurate than kriging |
| 4 | `GPUGaussianMixture` | EM soft clustering with full covariances, BIC | responsibilities kernel, fixed-point sums, `dense-solve-wgsl` | A–B | M | scikit-learn (BSD-3) | **do**: classic, deterministic, small |
| 5 | `GPUGradientBoostedTrees` | Histogram GBDT training (squared, logistic, quantile loss) | `GPUHistogram`, `GPUScan`, `GPUCompaction`, `GPUSegmentedReduction` | B–C | L | XGBoost / LightGBM | **do**: the strongest tabular learner; feeds rank 1 |
| 6 | `GPUSelfOrganizingMap` (with GeoSOM mode) | Batch SOM prototypes and assignments | k-means-style assignment, fixed-point weighted sums | A | S–M | NumPy batch-SOM; MiniSom (MIT) | **do**: cheap, an established geodemographics tool |
| 7 | `GPUPermutationImportance` | Loss increase per covariate, optionally in spatial blocks, for any contributor with a predict kernel | random-key `GPUSort`, reduction | depends on model | S | scikit-learn `permutation_importance` | **do** once ranks 1 or 5 exist |
| 8 | `GPUDeepKriging` | MLP on multiresolution Wendland basis + covariates, density output | shared encoder, dense layer, Adam | B | M | own PyTorch reproduction (no public code found) | **maybe**: fold into rank 3 as an encoder unless the density output is wanted |
| 9 | `GPUTreeShap` | Per-cell SHAP contribution maps for a forest | path extraction, per-path kernel | B | M | XGBoost `pred_contribs`, shap (MIT) | **maybe**: after ranks 1 and 5 |
| 10 | `GPUGraphConvolution` / PE-GNN | GCN or GraphSAGE regression on the spatial weights graph | SpMM, dense layer, Adam | B | M | PyTorch Geometric (MIT) | **maybe**: evidence of gain over spatial lag models is thin |
| 11 | `GPURandomFourierFeatureGP` | GP regression and posterior function samples via D random features | `GPUMatMul` (split-K), PCG | A–B | M | GPyTorch / NumPy | **maybe**: variance starvation; conditional simulation already exists |
| 12 | `GPUAutoencoderAnomaly` | Reconstruction error map | dense layer, Adam | B | S once shared pieces exist | PyTorch | **maybe** |
| 13 | t-SNE (FFT field) | 2-D embedding of attributes | kNN, grid binning, FFT | B | M | openTSNE | **maybe** |
| — | Sparse variational GP | calibrated GP at scale | blocked Cholesky (missing) | B | L | GPyTorch | **skip for now**: needs a multi-invocation Cholesky first |
| — | GNNWR / GTNNWR | neural-weighted GWR | | C (hours) | L | gnnwr (GPL-3.0) | **skip**: input width grows with n |
| — | Geographical RF, QRF, UMAP, neural processes, deep kernel learning | | | C or non-deterministic | L | | **skip**: reasons above |

## 10. Shared pieces

Four kernels would serve ranks 3, 8, 10, 11 and 12 and any later MLP:

1. **Dense layer forward and backward** (`Z = A·W + b`, activation, `δ·Wᵀ`, split-K `Aᵀ·δ` with fixed-order tree reduction). It is the contract-compliant counterpart of `GPUMatMul` for data-sized inner dimensions.
2. **Adam step** (elementwise, bias-corrected, per-frame learning rate, skip-on-zero-gradient option for sparse tables), with a loss-plateau convergence status and an epoch budget.
3. **Positional encoder** (Fourier multi-scale à la Space2Vec, spherical harmonics, Wendland multiresolution basis, hash grid with fixed-point scatter-add backward).
4. **Seeded minibatch sampler** (a permutation per epoch through `GPUSort` on hashed keys) so training is reproducible.

Ranks 1, 5, 7 and 9 share a **forest representation** (a sparse node array plus a JSON importer for XGBoost, LightGBM and scikit-learn).

## 11. What I could not verify

- No sustained throughput number for hand-written WGSL training on an M-class GPU; every time in this report derives from an assumed 10¹¹ MAC/s.
- No published head-to-head of neural fields against kriging on scattered-point interpolation; the one direct comparison found favours IDW.
- DeepKriging's basis counts and network widths, and whether code is public.
- The licences of `locationencoder`, `pe-gnn`, the KCN code, jax-js (third-party says MIT), webgpu-torch, tinygrad and Burn; tiny-cuda-nn's licence is NVIDIA's own and must be read before any use as an oracle.
- Whether TensorFlow.js WebGPU training is faster than its WebGL backend today.
- FIL and GPUTreeShap figures are for NVIDIA datacentre GPUs; the M-class figures above are scaled estimates.
- The GeoSOM DOI and the existence of a batch mode in R `kohonen` were not checked.
- Several classic references (Rahimi and Recht, Hensman et al., Wilson and Nickisch, Wilson et al. 2016, Tancik et al., Sitzmann et al., Sphere2Vec, Kipf and Welling, GraphSAGE, deep ensembles, MC dropout, Adam, TreeSHAP, Fisher et al., Meinshausen, the WebGPU limits section) are cited by their canonical URLs without being fetched in this session; their claims here are limited to what each paper is about.
