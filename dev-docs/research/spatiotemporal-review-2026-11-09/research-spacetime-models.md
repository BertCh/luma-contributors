# Model-based spatial and spatiotemporal statistics on WebGPU: research scouting report

Date: 2026-11-09. Scope: model-based spatial and space-time statistics and scalable inference (2022 to 2026), judged for
massively parallel float32 WGSL compute under the library's constraints. These are: no f64, dense solves of tens of
unknowns per site with thousands of sites at once, Jacobi-PCG up to about 1e6 unknowns, bounded and unrolled loops,
no data-dependent iteration without CPU readback, explicit capacities, and a pinned reference oracle for every
operation.

Citation rule: every URL listed was fetched in this session. "Search-only" means the claim came from a search-result
summary and the primary page was not opened, so treat it as UNVERIFIED. Versions and licences come from the CRAN or
PyPI pages fetched on 2026-11-09.

Existing library pieces this report assumes it can reuse (from `modules/analysis/src/gpu-spatial-analysis`):
`spatial-interpolation/gpu-local-kriging.ts`, `equilibrated-solve-wgsl.ts`, `conditional-simulation-blocked.ts`,
`dataframe-analysis/pair-statistics/gpu-cross-variogram.ts`, `spatial-regression/local-weighted-least-squares-wgsl.ts`,
`gwr-*`, `gpu-multiscale-geographically-weighted-regression.ts`, `spatial-exact-log-determinant.ts`,
`landscape-ecology/sparse-resistance-solver.ts` (Jacobi-PCG), `spatial-weights/*`, `space-time-tests/*` (Knox and
Mantel), `spatial-validation/*`, `point-process/*` and `neighbor-search/*`.

---

## 0. Ranked shortlist (analyst value divided by effort)

| Rank | Candidate | Value | Effort | Risk | Notes |
|---|---|---|---|---|---|
| 1 | Space-time variogram families and local space-time kriging (separable, product-sum, metric, sum-metric, simple sum-metric, Gneiting) | High | M | Low | Extends existing local kriging and variogram fitting. gstat 2.1-6 is the oracle. |
| 2 | Vecchia / NNGP log-likelihood, Fisher-scoring parameter estimation and prediction (spatial and space-time; scaled Vecchia) | Very high | M to L | Medium | Each conditional is a tiny dense solve, which is the platform's sweet spot. GpGp 1.0.0 (MIT) is the oracle. |
| 3 | Spatial conformal prediction intervals (GeoConformal, local spatial conformal) for any predictor | High | S | Low | Neighbour search plus a weighted quantile. Coverage can be checked by simulation. |
| 4 | Proper scoring rules for fields: CRPS (closed-form Gaussian and sample), energy score, variogram score | Medium-high | S | Low | scoringRules 1.1.3 is the oracle. Pairs with rank 3 and the existing CV. |
| 5 | Area of applicability / dissimilarity index (CAST `aoa`) | Medium-high | S to M | Low | A nearest-neighbour search in weighted predictor space. |
| 6 | GTWR and multiscale GTWR | High | S to M | Low to medium | A space-time kernel for the existing GWR / MGWR. The oracle is only partly available (see section 3). |
| 7 | Space-time Hawkes / ETAS by EM with bounded parent windows and binned (histogram) triggering | High (crime, seismic, epidemic) | M | Medium (float32 sums) | ETAS 0.7.2 is the oracle. A float32 GPU kernel has been shown to work at 1e6 events (see section 4). |
| 8 | Sinkhorn / unbalanced optimal-transport divergence between spatial fields and point patterns | Medium | M | Medium (log-domain f32) | POT 0.9.7.post1 (MIT) is the oracle. Useful for forecast verification. |
| 9 | Vecchia-Laplace for counts and binary data (latent GP GLMM) | High | L | Medium-high | GPBoost 1.8.0 (Apache-2.0) is the oracle. Uses CG on Vecchia precision. |
| 10 | EOF / DMD of space-time grids by randomized SVD | Medium | M | Medium | PyDMD 2025.8.1 (MIT) is the oracle. |
| 11 | CP / Tucker decomposition of origin-destination × time tensors (ALS) | Medium | M | Medium | TensorLy 0.10.0 (BSD-3) is the oracle. |
| 12 | BYM2 / Knorr-Held space-time disease mapping by Laplace approximation on a fixed hyperparameter grid | High | L | High (log-determinants in f32) | R-INLA is the oracle, but it is not on CRAN and its licence is UNVERIFIED. |
| 13 | RE-ESF / Moran eigenvector spatial filtering with spatially and temporally varying coefficients | Medium | M to L | Medium-high (eigensolver in f32) | spmoran 0.3.3 (GPL) is the oracle. |

---

## 1. Space-time geostatistics

### 1.1 Covariance and variogram families (rank 1)

**Question answered.** How does correlation decay jointly in space and time, and what is the best prediction at an
unobserved (location, time)?

**Primary sources.**
- Gräler, Pebesma and Heuvelink, "Spatio-temporal interpolation using gstat" (gstat vignette; the R Journal 2016).
  https://cran.r-project.org/web/packages/gstat/vignettes/spatio-temporal-kriging.pdf (PDF fetched and text
  extracted). It gives:
  - separable: `C(h,u) = Cs(h) Ct(u)`, with variogram `sill·(γs + γt − γs γt)` on standardised parts. Its Kronecker
    structure helps when every site is observed at every time.
  - product-sum: `C = k Cs Ct + Cs + Ct` with k > 0, and `sill_st = k·sill_s·sill_t + sill_s + sill_t`.
  - metric: `C = Cjoint(sqrt(h² + (κu)²))`, where κ (`stAni`) is spatial units per temporal unit.
  - sum-metric: `γ = γs(h) + γt(u) + γjoint(sqrt(h² + (κu)²))`, with separate nuggets.
  - simple sum-metric: nugget-free parts plus one joint nugget.
  - Fitting uses `fit.StVariogram` with weighted least squares, with several weightings, for example
    `Nj/(h² + stAni²·u²)` (fit.method 7), and L-BFGS-B.
  - Local kriging (`krigeST`, `nmax`): k-nearest neighbours in the metric space-time scaled by stAni (FNN). An option
    searches a larger metric neighbourhood and then keeps the neighbours with the largest covariance. The vignette
    says that "staged search ... or an octant search ... are not yet available in gstat". That is a clear place to
    add value.
  - In the PM10 example, sum-metric (weighted MSE 3.31) beats separable (6.82) and metric (10.05). Prediction uses
    `nmax = 50`.
- gstat 2.1-6, published 2026-03-30, GPL (≥ 2). https://cran.r-project.org/package=gstat
- Gneiting non-separable class, NRCSE technical report TRS-063 (2001), the precursor of JASA 97(458), 2002.
  https://sites.stat.washington.edu/NRCSE/pdf/trs63_nonsep.pdf (fetched, text extracted). The form is
  `C(h,u) = σ² / ψ(|u|²)^{d/2} · φ(|h|² / ψ(|u|²))`, where φ is completely monotone and ψ is positive with a
  completely monotone derivative. The worked family is `σ²/(a|u|^{2α}+1)^{βd/2} · exp(−c|h|^{2γ}/(a|u|^{2α}+1)^{βγ})`
  with interaction parameter β ∈ [0,1], and β = 0 gives the separable case. The JASA citation is search-only.
- GeoModels 2.3.0, 2026-10-30, GPL (≥ 3): Gaussian and non-Gaussian spatial and space-time models, likelihood and
  weighted pairwise composite likelihood, and (local) BLUP. https://cran.r-project.org/package=GeoModels. That its
  covariance list includes Gneiting, Gneiting-Matérn and Gneiting-Wendland comes from search-result summaries of the
  refman (UNVERIFIED in detail).
- GSTools: spatio-temporal support is the metric model only, via `temporal=True` plus a temporal anisotropy ratio.
  https://geostat-framework.readthedocs.io/projects/gstools/en/latest/examples/09_spatio_temporal/index.html.
  GSTools 1.7.0 is LGPL-3.0: https://pypi.org/pypi/gstools/json
- Prior art in WebGPU: the Rust crate kriging-rs 0.4.0 (MIT, 2026-04-26) offers ordinary, simple, universal and
  "binomial" kriging with separable or product-sum space-time variograms. It has a wgpu (WebGPU) path for
  "GPU-assisted RHS covariance for batch prediction" and defaults to f32. https://docs.rs/crate/kriging-rs/latest.
  This shows that browser space-time kriging already exists. To differentiate, add sum-metric and Gneiting models,
  space-time neighbour search, weighted space-time variogram fitting and full parity with the oracle.

**Parallel formulation.**
- Empirical space-time variogram: the existing pair-statistics kernel with a 2-D bin index (spatial lag bin × temporal
  lag bin). Use atomic or segmented reduction per bin, at about 15 × 10 bins.
- Fitting: the existing GPU variogram-fit path, with the model evaluated on the bin grid. There are at most 5 to 9
  parameters per model, so a CPU-driven L-BFGS-B or Gauss-Newton with GPU objective and gradient over the bins works,
  as does a batched multi-start grid of candidate parameter vectors evaluated at once, which suits WGSL better.
- Local space-time kriging: per prediction (s,t), find k neighbours in the metric space `(x, y, κt)`. This is a 3-D
  grid or kNN with κ scaling and reuses `grid-neighbor-scan.ts`. A "buffer" variant takes kNN of size B·k, evaluates
  the full covariance and keeps the top k by bounded partial selection. Then build the (k+1)×(k+1) system and run
  `equilibrated-solve-wgsl`, with k ≤ 50 as in the gstat example. An octant / staged search (spatial quadrant ×
  past/future) is a bounded variant that gstat lacks.
- Space-time conditional simulation: reuse the blocked Cholesky simulation with a space-time covariance.

**Float32 risk.** Large time-lag scaling: κ·u can dwarf h if units are seconds and metres, as gstat's base units are.
Mitigation: centre and rescale space and time to O(1) before the kernel and store κ in normalised units.
Near-singular systems from many co-located time slices at one station are another risk; mitigate with the existing
equilibration plus a nugget floor and a documented minimum nugget.

**Oracle.** gstat 2.1-6 (`variogramST`, `fit.StVariogram`, `krigeST`) for the five gstat families. GeoModels 2.3.0
for Gneiting. Both are GPL, so use them as oracles only.

**Capacity.** k ≤ 64 neighbours, about 1e6 prediction cells per dispatch batch, about 1e5 to 1e6 observations.
**Effort** M. **Risk** low.

### 1.2 Space-time Vecchia / DNNGP (folded into section 2)

Datta, Banerjee, Finley, Hamm and Schaap, "Nonseparable dynamic nearest neighbor Gaussian process models for large
spatio-temporal data", Annals of Applied Statistics 10(3), 2016. https://arxiv.org/abs/1510.07130 appeared in search
results and was not fetched, so treat it as UNVERIFIED. GpGp 1.0.0 (fetched; see section 2) ships space-time
covariance functions for Vecchia, which makes Vecchia the natural route to space-time maximum-likelihood estimation
rather than variogram fitting.

---

## 2. Scalable Gaussian processes

### 2.1 Vecchia / NNGP likelihood, estimation and prediction (rank 2)

**Question answered.** Maximum-likelihood (REML) covariance parameters and kriging at 1e5 to 1e6 points, without
variogram-binning arbitrariness. This also gives likelihood-based model comparison between covariance families (AIC).

**Primary sources.**
- Datta, Banerjee, Finley and Gelfand, NNGP, JASA 111 (2016). A valid process with sparse precision and flops linear
  in n per iteration. https://arxiv.org/abs/1406.7343
- Schäfer, Katzfuss and Owhadi, "Sparse Cholesky factorization by Kullback-Leibler minimization", SIAM J. Sci.
  Comput. (2021). The KL-optimal sparse inverse-Cholesky factor has a closed form that recovers Vecchia, is
  "embarrassingly parallel", and runs in O(N log(N/ε)^{2d}) time. https://arxiv.org/abs/2004.14455
- Katzfuss, Guinness and Lawrence, "Scaled Vecchia approximation for fast computer-model emulation". Vecchia in a
  learned scaled input space with Fisher scoring, near-linear in n. The fetched page shows it as an arXiv preprint
  (v4, 2021); the SIAM/ASA JUQ 2022 venue is search-only. Code: katzfuss-group/scaledVecchia.
  https://arxiv.org/abs/2005.00386
- Pan, Abdulah, Genton, Keyes, Ltaief and Sun, "GPU-accelerated Vecchia approximations ... using batched matrix
  computations" (2024). KBLAS batched linear algebra, up to 700×/833×/1380× over ExaGeoStat on GV100/A100/H100, and
  up to 1M locations on one 80 GB GPU. Precision and m are not stated on the abstract page.
  https://arxiv.org/abs/2403.07412v3
- GpGp 1.0.0 (2025-12-18, **MIT**): Vecchia likelihood, Fisher scoring (Guinness 2019), maxmin ordering and grouping
  (Guinness 2018), and spatial plus spatial-temporal covariances on Euclidean domains and spheres.
  https://cran.r-project.org/package=GpGp
- GPvecchia 0.1.8 (2026-01-31, GPL ≥ 2): MaxMin ordering and IC0. https://cran.r-project.org/package=GPvecchia
- spNNGP 1.0.2 (2026-05-28, GPL ≥ 2): Bayesian NNGP by MCMC. https://cran.r-project.org/package=spNNGP
- BRISC 1.0.6 (2024-10-02, GPL ≥ 2): NNGP estimation plus bootstrap inference. https://cran.r-project.org/package=BRISC

**Parallel formulation.** For ordered points i = 1..n with conditioning set c(i) of size m (m ≈ 10 to 30), each site
i solves an m×m system `K_cc b = K_ci` and computes `d_i = K_ii − K_ic b`. Then `log L = −½ Σ_i [log d_i + (y_i −
b·y_c)² / d_i] − (n/2) log 2π`. This is "thousands of small independent dense solves", the platform's sweet spot, and
the same solve as local kriging. Steps:
1. Ordering: maxmin ordering is sequential in its exact form. Options are a CPU-side approximate maxmin using a grid
   or k-d tree (allowed as setup), or a GPU "random order + coordinate jitter" ordering, which Guinness 2018 reports
   as competitive (UNVERIFIED, not fetched). Hilbert keys already exist in the library and are another cheap option,
   though weaker than maxmin.
2. Conditioning sets: kNN among previously ordered points. Use a bounded neighbour search filtered by order index.
3. Per-site kernel: build K_cc (m ≤ 32), Cholesky with jitter, then accumulate `log d_i` and the residual term.
   Optionally include closed-form partial derivatives for Fisher scoring. The per-site Fisher information
   contribution is a p×p matrix (p ≤ 6), reduced on the GPU.
4. CPU-driven outer Fisher scoring with a fixed iteration cap (for example 20), with one readback per iteration.
5. Prediction: append prediction points after the data in the ordering. Each prediction is the existing local
   kriging solve on its conditioning set, which is equivalent.
6. Space-time: swap in any of the section 1 covariances, with neighbours in κ-scaled 3-D. Scaled Vecchia is a
   per-dimension range scaling before the neighbour search.

**Float32 risk.** (a) The total log-likelihood is a sum of 1e6 terms, about 1e6 in magnitude, while the differences
between candidate parameters are O(1). Mitigation: compensated (Kahan or pairwise) tree reduction and reporting
`log L − log L(θ₀)` as a sum of per-site differences. Better, drive Fisher scoring from gradients, where cancellation
is local, rather than from likelihood differences. (b) Ill-conditioned K_cc for smooth (Gaussian, high-ν Matérn)
kernels. Mitigation: a nugget floor, equilibration and m ≤ 30. (c) `log d_i` as d_i → 0: clamp with a recorded flag.
The 2024 GPU paper does not state its precision; whether float32 Vecchia MLE matches float64 within tolerance is
UNVERIFIED and should be the first experiment (a GpGp oracle comparison on n = 1e4).

**Reuses.** Neighbour search, the local kriging kernel, equilibrated solve, the variogram covariance library and the
GPU variogram fit (initial values). **Capacity.** n ≤ 1e6, m ≤ 32, p ≤ 8. **Effort** M for likelihood and
prediction, L with Fisher scoring and space-time. **Risk** medium. This is the strategic piece: it unlocks rank 9 and
gives likelihood-based space-time model selection.

### 2.2 Vecchia-Laplace for non-Gaussian data (rank 9)

- Zilber and Katzfuss, "Vecchia-Laplace approximations of generalized Gaussian processes for big non-Gaussian
  spatial data", CSDA 153 (2021). https://arxiv.org/abs/1906.07828v4
- Kündig and Sigrist, "Iterative methods for Vecchia-Laplace approximations for latent Gaussian process models", JASA
  (2024). Iterative solvers with preconditioners replace Cholesky, giving "an order of magnitude" speed-up and a
  threefold CRPS improvement on a large satellite dataset. https://arxiv.org/abs/2310.12000. That it uses conjugate
  gradients and stochastic Lanczos quadrature comes from search snippets only (UNVERIFIED).
- GPBoost 1.8.0 (Python), **Apache-2.0**: https://pypi.org/pypi/gpboost/json. That it implements the VIF / iterative
  Vecchia-Laplace methods (Gyger, Furrer and Sigrist 2025, arXiv 2507.05064) is search-only (UNVERIFIED).

**Formulation.** The latent precision is `Q = Bᵀ D⁻¹ B`, where B is the sparse Vecchia factor with m+1 entries per
row, built by the section 2.1 kernel. Newton steps on the latent mode solve `(Q + W) x = r`, with W diagonal from the
Poisson or Bernoulli likelihood. Q's matvec is two sparse triangular-pattern matvecs (B, then Bᵀ, through a scatter /
transpose kernel, which `spatial-weights-transpose` already provides). Solve with the existing Jacobi-PCG for a fixed
iteration count and a fixed number of Newton steps (for example 10). The Laplace log-marginal needs
`log det(Q + W) − log det Q`. Here `log det Q = −Σ log d_i` is exact and cheap, but `log det(Q+W)` needs stochastic
Lanczos quadrature (SLQ) or a Cholesky.

**Float32 risk.** The SLQ log-determinant in f32 is noisy, and Lanczos loses orthogonality. Mitigation: fixed probe
count (for example 16 to 32), a fixed Lanczos depth with full reorthogonalisation in shared memory (depth ≤ 64), and
reporting the Monte Carlo standard error. For parameter estimation, use gradient-based updates with stochastic trace
estimators instead of likelihood differences. **Effort** L. **Risk** medium-high.

### 2.3 Other scalable GP approaches, judged

| Method | Source (fetched) | Fit to constraints | Verdict |
|---|---|---|---|
| Exact GP by CG / BBMM (GPyTorch, KeOps) | Wang et al., NeurIPS 2019, "exact GP on over a million points" via partitioned kernel matmuls and CG. https://arxiv.org/abs/1903.08114 | Dense O(n²) matvec per CG iteration; f32 CG on dense smooth kernels converges poorly without a preconditioner (partial pivoted Cholesky). | Not first. Vecchia gives similar accuracy at O(n m³). Keep tiled kernel matvec as an optional building block. |
| GPJax 1.0.0 (MIT) | https://pypi.org/pypi/gpjax/json | Research library. | A possible oracle for RFF and sparse variational GP only. |
| Fixed-rank kriging (FRK 2.3.2, GPL ≥ 2; spatial and space-time; non-Gaussian since v2) | https://cran.r-project.org/package=FRK | Low rank r ≤ a few thousand gives an r×r dense solve, too big for one per-site system and too small to need CG. Possible as a single blocked Cholesky on the CPU or a batched GPU kernel. | Medium. Oversmooths small-scale structure. Low priority. |
| Multi-resolution approximation (M-RA) | Katzfuss, JASA 2017; parallelisable over distributed memory. https://arxiv.org/abs/1507.04789 | Recursive tree of small dense solves per level, which is GPU-able level by level, but the bookkeeping is heavy and there is no maintained CRAN oracle (not checked). | Skip for now. |
| SPDE / GMRF (INLA, inlabru, fmesher) | fmesher 0.8.0 (MPL-2.0, mesh and FEM matrices): https://cran.r-project.org/package=fmesher. inlabru 2.15.0 (GPL ≥ 2, LGCP likelihood): https://cran.r-project.org/package=inlabru | The SPDE precision Q = κ⁴C + 2κ²G + G C⁻¹ G is sparse, so it fits Jacobi-PCG. Meshing is CPU work. Log-determinants are again the obstacle. | The fmesher FEM matrices (MPL-2.0, a weak-copyleft per-file licence, so it may be readable but check before porting) could feed a GPU PCG for SPDE-kriging (posterior mean only). Full INLA: no. |
| Random Fourier features | No primary source fetched (UNVERIFIED). | A pure matmul plus an R×R solve. Trivial on GPU. | A cheap smooth-surface baseline. Poor for short ranges. Low priority. |
| DeepKriging / neural kriging | Chen, Li, Reich and Sun, arXiv 2020 / 2022, a basis-function embedding plus a DNN. https://arxiv.org/abs/2007.11972 | Training is data-dependent and iterative. Inference is a matmul. | Not worth it: no stable oracle, training loop. |
| Sparse inverse Cholesky (KL) as a preconditioner | https://arxiv.org/abs/2004.14455 | Same per-column kernel as Vecchia. | A by-product of section 2.1: it gives an excellent PCG preconditioner for any dense covariance or SPDE system. |

---

## 3. Space-time regression

### 3.1 GTWR and multiscale GTWR (rank 6)

- Huang, Wu and Barry, "Geographically and temporally weighted regression ...", IJGIS 24(3) 2010 (search-only,
  UNVERIFIED): GTWR uses a space-time distance `d² = λ·ds² + μ·dt²` in the GWR kernel.
- Geniaux, Martinez and Soubeyrand, "A top-down scale approach for multiscale GTWR" (arXiv 2603.22990, March 2026,
  submitted to Spatial Statistics): separable space-time kernel, per-covariate spatial and temporal bandwidths by
  coordinate-wise search over ordered grids with AICc, and importance-ordered backfitting.
  https://arxiv.org/abs/2603.22990. That grid search over ordered candidates is exactly a bounded, unrolled loop.
- Yu and Fotheringham, IJGIS 2024 (calibration of MGTWR, effective number of parameters), and a non-iterative
  MGTWR (MDPI Mathematics 2025): search-only, UNVERIFIED.
- Oracle gap: mgwr 2.2.1 (BSD-3) does **not** include GTWR. https://pypi.org/pypi/mgwr/json. GWmodel's `gtwr()` is
  the likely R oracle (GPL; version not fetched, UNVERIFIED). Geniaux's mgwrsar is another possibility (not
  verified).
- FastGWR (Li, Fotheringham, Li and Oshan, IJGIS 2019; MPI-parallel, 1.3M observations, MIT): search-only.

**Formulation.** Identical to the existing GWR / MGWR kernels. Only the weight function changes, to separable
`K_s(ds/bs)·K_t(dt/bt)` or metric. Restrict the temporal kernel to past-only (asymmetric) as an option for
forecasting use. Bandwidth search uses the existing golden-section nodes, extended to a 2-D grid.

**Float32 risk.** Same as GWR. Watch time scaling (use normalised units). **Effort** S to M. **Risk** low once an
oracle is pinned.

### 3.2 Eigenvector spatial filtering / RE-ESF with spatially and temporally varying coefficients (rank 13)

- spmoran 0.3.3 (2024-12-05, GPL ≥ 2): ESF and RE-ESF, and "spatially / spatio-temporally varying coefficient
  models", using Moran eigenvectors as basis functions to approximate GPs.
  https://cran.r-project.org/package=spmoran
- Murakami and Griffith, Geographical Analysis 51(1) 2019 (fast ESF and RE-ESF for large samples): search-only.

**Formulation.** The top L (≈ 200) eigenvectors of `M C M` (doubly centred connectivity), then a mixed model with L
random coefficients per varying covariate. The GPU eigensolver needs blocked Lanczos or subspace iteration with
matvecs against C, which can reuse the chunked spatial-weights algebra. The final REML is a small dense problem
(L × covariates), which is fine. **Float32 risk.** Eigenvector accuracy for clustered spectra. Mitigate with subspace
iteration and Rayleigh-Ritz in a small dense block, and compare spans rather than vectors against the oracle.
**Effort** M to L.

### 3.3 Geographically weighted machine learning

- SpatialML 1.8.2 (GPL ≥ 2): geographically weighted random forest with bandwidth search.
  https://cran.r-project.org/package=SpatialML
- GNNWR and GTNNWR (Du et al. 2020; Wu et al., IJGIS 2020): search-only.
- Neural spatio-temporal point processes (Chen, Amos and Nickel, ICLR 2021; code CC BY-NC 4.0): search-only.

Verdict: **not worth doing** as GPU training on this platform. Training loops are data-dependent, random-forest
splits are sequential, and the oracles are stochastic. Better value: supply their **validation scaffolding** (sections
5.1 to 5.4), which works for any externally trained model.

### 3.4 Spatial and spatio-temporal panels

- splm 1.6-5 (2023-12-20, GPL-2): ML and GM spatial panel estimation and diagnostics.
  https://cran.r-project.org/package=splm. The library already has balanced FE / RE panels.
- Missing pieces: SDM and SDEM panels (just add WX columns, so S), and dynamic spatial panels (time lag + space-time
  lag `W y_{t−1}`). The latter need bias-corrected QML (Yu, de Jong and Lee 2008, UNVERIFIED) or GMM. Bias-corrected
  QML reuses the existing log-determinant and the spatial-lag ML path. Oracle: splm does not do dynamic panels (as
  far as the description shows). Stata `xsmle` and the R `SDPDmod` package are UNVERIFIED. Effort M. Value medium
  for econometric users.

### 3.5 Bayesian hierarchical space-time disease mapping (BYM2, Knorr-Held) (rank 12)

- Riebler, Sørbye, Simpson and Rue (2016), BYM2: the scaled ICAR plus i.i.d. mixing, with separable PC priors.
  https://arxiv.org/abs/1601.01180. The Statistical Methods in Medical Research venue is search-only.
- Knorr-Held, Statistics in Medicine 19 (2000), four interaction types (I to IV) as Kronecker products of
  structure matrices: search-only, UNVERIFIED.
- The library already has empirical Bayes rate smoothing (`rate-smoothing/*`).

**Feasibility of a Laplace approach.** For fixed hyperparameters θ, the latent field `x | y, θ` (areas × times) has
sparse precision `Q(θ) = τ_s R_s ⊗ I + τ_t I ⊗ R_t + τ_st R_s ⊗ R_t (+ i.i.d.)`. Newton with diagonal Poisson W and
Jacobi-PCG matvecs is GPU-friendly: Kronecker matvecs are two sparse matmuls. The integrated nested Laplace step needs
`log det(Q + W) − log det Q` on a θ-grid (about 30 to 100 points for 3 to 4 hyperparameters). Intrinsic Q is
rank-deficient (sum-to-zero constraints). In f32 those determinants are the failure point. For Kronecker structure,
however, `log det(R_s ⊗ R_t + ...)` can be computed from the **eigenvalues of R_s and R_t**: a one-off dense
eigendecomposition of a few thousand areas on the CPU, plus a per-time small eigenproblem. That makes the
Gaussian-likelihood case exact and the Poisson case approximate (SLQ for the W perturbation).
Verdict: a **posterior mode plus Gaussian marginal approximation on a fixed θ-grid for BYM2 (space only)** is feasible
(M to L). The full Knorr-Held type IV with INLA-grade accuracy is high-risk. Oracle: R-INLA (not on CRAN; licence not
verified in this session). CARBayesST would be an MCMC oracle (not fetched, UNVERIFIED).

---

## 4. Point process models

### 4.1 Space-time Hawkes / ETAS by EM with bounded parents (rank 7)

**Question answered.** How much of the activity is background and how much is triggered, what are the triggering
range and decay, and what is the short-term conditional intensity forecast (seismic aftershocks, near-repeat crime,
epidemic clusters)?

**Primary sources.**
- Veen and Schoenberg, "Estimation of space-time branching process models in seismology using an EM-type algorithm"
  (UCLA preprint 2006; later JASA 2008). It treats the branching structure as missing data. The E-step computes
  triggering probabilities and the M-step maximises the expected complete-data log-likelihood, which is more stable
  than direct ML. https://escholarship.org/content/qt7zm0v7tg/qt7zm0v7tg.pdf (PDF fetched, text extracted)
- ETAS (R) 0.7.2 (2025-10-24, GPL ≥ 2): space-time ETAS by stochastic declustering (Zhuang, Ogata and Vere-Jones
  2002), C++ with optional OpenMP. https://cran.r-project.org/package=ETAS. The function docs say it is for "small and
  medium-size" catalogues, and for large ones recommend a parallel Fortran code. The θ vector is (μ, A, c, α, p, D, q,
  γ). The algorithm alternates ML for θ, background probabilities φ_i = μ/λ, and kernel-smoothed background updates.
  https://rdrr.io/cran/ETAS/man/etas.html
- Holbrook, Loeffler, Flaxman and Suchard (2020), GPU and multi-core Hawkes likelihoods, ">100-fold" speed-ups,
  85,000 gunshots. https://arxiv.org/abs/2005.10123
- Ko, Suchard and Holbrook (2024), "Scaling Hawkes processes to one million COVID-19 cases". Multi-GPU OpenCL, O(N²)
  likelihood, log-sum-exp, flooring of sums below 1e-40. **The single-precision kernel "runs reliably" and is 7.2×
  faster than the earlier double-precision kernel at 1M observations**, whereas the earlier work had said single
  precision was unreliable above 100,000 observations. https://arxiv.org/html/2407.11349v1
- Mohler et al., "Self-exciting point process modeling of crime", JASA 2011 (nonparametric stochastic declustering
  with KDE): search-only. Marsan and Lengliné 2008 (model-independent stochastic declustering, histogram triggering):
  search-only.

**Parallel formulation (EM, fixed iterations).**
1. Sort events by time. For each event j, the candidate parents are i < j with t_j − t_i ≤ T_max and
   |s_j − s_i| ≤ R_max. That is a bounded space-time neighbour list (capacity K ≤ 256), reusing the
   `trajectory-encounters` and `space-time-tests` pair kernels. Overflow is counted and reported.
2. E-step (per event j, independent): `p_ij = g(t_j − t_i, s_j − s_i; θ) · κ(m_i) / λ_j` and
   `p_jj = μ(s_j)/λ_j`, computed in log space with max-subtraction.
3. M-step: for a **binned (histogram) triggering kernel** (Marsan-Lengliné style), each bin's height is
   `Σ p_ij` over pairs falling in the bin divided by the exposure. This is an atomic or segmented reduction with a
   closed form and no inner optimiser. For parametric ETAS (Omori-Utsu, power-law space), use closed-form or
   few-step Newton updates with p ≤ 8 parameters (CPU, from GPU-reduced sufficient statistics). Background
   μ(s) is a weighted KDE using p_jj as weights, which reuses `point-density` / kernel intensity.
4. Fixed EM iteration count (for example 50 to 200) with the log-likelihood read back every N iterations for a
   convergence report, not for control flow.
5. Integral (compensator) term: closed form for exponential / Omori kernels in time, and for a Gaussian or
   power-law kernel in space, either integrate over the study window by quadrature or approximate by ℝ² as
   Ko et al. do (and document it).

**Float32 risk.** λ_j is a sum of up to K terms spanning many orders of magnitude, so use log-sum-exp. The total
log-likelihood over 1e6 events needs Kahan or pairwise reduction. Times should be stored relative to the window start
in days, because absolute Unix seconds lose precision in f32. Power-law tails with (t + c)^{-p} for small c are fine
in log space. Ko et al. above is direct evidence that f32 is viable with these mitigations.

**Oracle.** ETAS 0.7.2 for parametric space-time ETAS. For histogram triggering, compare with a small hand-written
reference EM in the test suite against a simulated catalogue with known parameters. **Capacity.** N ≤ 1e6 events,
K ≤ 256 parents. **Effort** M. **Risk** medium. **Value** high, and it is a natural partner to the existing Knox and
Mantel tests and emerging hot spots.

### 4.2 Log-Gaussian Cox processes

- lgcp (Taylor, Davies, Rowlingson and Diggle): spatio-temporal LGCP, "main computational tool ... MCMC".
  https://arxiv.org/abs/1110.6054. The details on FFT circulant embedding and MALA come from search snippets only.
- inlabru 2.15.0: LGCP via INLA and SPDE (above).

**GPU route.** On a regular grid, `log λ = β + Z`. A Laplace mode-find (Newton with Poisson W) where the prior
precision is either (a) GMRF / SPDE sparse, which gives PCG, or (b) stationary on a torus, which gives FFT-based
solves. Option (b) needs a WGSL FFT; check whether the library already has one before costing it. The posterior mode
and a Laplace variance map give a useful "smoothed intensity with uncertainty". Effort M to L. MCMC (MALA) is **not
worth doing**.

### 4.3 Neural spatio-temporal point processes

Chen, Amos and Nickel, ICLR 2021 (neural ODE / CNF; repository CC BY-NC 4.0): search-only. **Not worth doing**: it
needs adjoint ODE training and has a non-commercial licence.

---

## 5. Uncertainty and validation

### 5.1 Spatial conformal prediction (rank 3)

- Lou, Luo and Meng, "GeoConformal prediction: a model-agnostic framework of measuring the uncertainty of spatial
  prediction" (arXiv 2412.08661, v3 April 2025). It adds geographical weighting to conformal prediction: a
  geographically weighted quantile of calibration nonconformity scores per test location. They report 93.67% coverage
  against ≤ 81% for bootstrap (XGBoost house prices), and that the uncertainty tracks kriging variance.
  https://arxiv.org/abs/2412.08661v3. Code availability: not confirmed.
- Mao, Martin and Reich, "Valid model-free spatial prediction", JASA 119 (2024), pp. 904-914 (global spatial
  conformal plus local spatial conformal prediction under approximate local exchangeability).
  https://arxiv.org/abs/2006.15640v2
- Jiang and Xie, "Spatial conformal inference through localized quantile regression" (arXiv 2412.01098; listed at
  ICML 2026 per search). The fetch failed (HTTP 500), so this is UNVERIFIED.

**Formulation.** Split conformal: the calibration residuals |y − ŷ| come from the existing spatial CV folds. For each
prediction location, take the k nearest calibration points (or all within the bandwidth), with kernel weights
`w = K(d/b)`. The weighted (1−α)(1 + 1/n_eff) quantile is a bounded partial sort or a histogram quantile in shared
memory, and k ≤ 256 is fine. Adaptive scores come from dividing by the kriging standard deviation (normalised
conformal), which makes it conformalised kriging. **Float32 risk.** Negligible. **Oracle.** No pinned package was
found. Validate by (a) a CPU reference of the weighted-quantile formula and (b) empirical coverage in simulation
(Monte Carlo coverage within binomial tolerance). **Effort** S. **Value** high. It works for every predictor the
library has, plus external ML predictions.

### 5.2 Proper scoring rules (rank 4)

- scoringRules 1.1.3 (2024-09-18, GPL ≥ 2): https://cran.r-project.org/package=scoringRules. The vignette (Jordan,
  Krüger and Lerch; Journal of Statistical Software) gives `crps_sample`, closed-form CRPS for parametric families,
  energy score `es_sample` (Gneiting et al. 2008) and variogram score `vs_sample(p = 0.5)`.
  https://cran.r-project.org/web/packages/scoringRules/vignettes/article.pdf (PDF fetched, text extracted)

**Formulation.** Gaussian CRPS is closed-form per site and needs only an erf implementation in f32. For sample CRPS
with m ensemble members, `E|X − y| − ½ E|X − X'|` is O(m log m) after sorting, or O(m²) pairs with m ≤ 256. The
energy score over a field with ensembles from conditional simulation is O(m² d), so tile it. The variogram score is
O(d²) site pairs, so restrict it to neighbour pairs (a documented choice, and the weights argument matches). This is
the natural evaluation of the existing conditional Gaussian simulation. **Float32 risk.** Low; use pairwise sums.
**Effort** S.

### 5.3 Spatial cross-validation and area of applicability (rank 5)

- kNNDM (Linnenbrink, Milà, Ludwig and Meyer, GMD 17, 2024). It runs a KS test for clustering, builds candidate
  clusterings with q from a log sequence, merges into k folds along PC1, and picks the candidate minimising the
  Wasserstein W between the CV and prediction NND ECDFs. 4.8 days became 1.2 minutes against NNDM LOO.
  https://gmd.copernicus.org/articles/17/5897/2024/gmd-17-5897-2024.html. **Already in the library.**
- Area of applicability: Meyer and Pebesma (Methods in Ecology and Evolution 2021). Dissimilarity index = distance to
  nearest training point in importance-weighted, standardised predictor space, divided by the mean training distance,
  with a threshold from the CV-fold DI of training data. https://arxiv.org/abs/2005.07939
- CAST 1.1.2 (2026-09-09, GPL ≥ 2): NNDM / kNNDM CV and AOA. https://cran.r-project.org/package=CAST

**Formulation.** A brute-force or tiled kNN in p-dimensional predictor space (p ≤ 32): tiled pairwise distances with
a min-reduce, which is an existing GPU pattern. The training-DI threshold respects folds (fold-masked nearest
neighbour). The local point density (LPD) extension (EGU24, search-only) is a count within the threshold. **Effort**
S to M. **Oracle** CAST `aoa()`.

---

## 6. Other GPU-friendly methods

### 6.1 Optimal transport / Sinkhorn divergence (rank 8)

- Francis, Cotter and Mittermaier, entropic unbalanced OT and the Sinkhorn divergence for spatial precipitation
  forecast verification (Meteorological Applications, accepted; arXiv 2412.16063 v3, July 2025). Robust to the
  "double penalty" problem, and it gives morphing and feature assignment. https://arxiv.org/abs/2412.16063
- Struleva, Hundrieser, Schuhmacher and Munk (arXiv 2509.04225, 2025): convergence rates for empirical unbalanced OT
  (Kantorovich-Rubinstein) between intensity measures of spatio-temporal point processes (Poisson cluster, Hawkes,
  Neyman-Scott, LGCP). This is theoretical support for OT as a two-sample comparison of point patterns.
  https://arxiv.org/abs/2509.04225
- POT 0.9.7.post1 (MIT): https://pypi.org/pypi/POT/json. It is the oracle and may even be readable for ideas.

**Formulation.** On grids, use convolutional Sinkhorn: the Gibbs kernel exp(−|x−y|²/ε) is separable, so each
iteration is two 1-D Gaussian blurs. That is the existing focal-statistics pattern, at O(n·r) per iteration. On point
sets, a dense n×m kernel (n, m ≤ about 2e4) tiled with log-sum-exp reductions. Unbalanced KL marginals add a
pointwise exponent (ρ/(ρ+ε)). Use a fixed iteration count with ε-scaling over a fixed schedule, and read back the
marginal error for reporting. The debiased divergence S = OT(a,b) − ½OT(a,a) − ½OT(b,b) takes three runs.
**Float32 risk.** Small ε underflows exp(−C/ε), so use log-domain Sinkhorn and keep ε ≥ about 1e-3 of the diameter²
for f32. Cancellation in the debiased divergence: report the components too. **Effort** M. **Value** medium
(forecast verification, comparing distribution maps, flow re-assignment).

### 6.2 EOF / DMD of space-time fields (rank 10)

PyDMD 2025.8.1 (MIT): https://pypi.org/pypi/pydmd/json. Formulation: randomized SVD of the snapshot matrix (n sites ×
T times), meaning a GEMM with a Gaussian sketch (rank r + oversampling ≤ 64), a QR of the n×(r+p) sketch
(TSQR / CholeskyQR2 in blocks), then a small dense SVD on the CPU or a single workgroup. DMD then uses the r×r reduced
operator eigen-decomposition (small, CPU). **Float32 risk.** CholeskyQR loses orthogonality, so use CholeskyQR2 or
shifted CholeskyQR3. Effort M.

### 6.3 Tensor decomposition of origin-destination × time (rank 11)

TensorLy 0.10.0 (BSD-3): https://pypi.org/pypi/tensorly/json. CP-ALS: each mode update is an MTTKRP (sparse,
parallel over nonzeros) plus an R×R solve (R ≤ 32, the sweet spot). Non-negative CP via multiplicative updates or HALS
is equally parallel. Fixed iteration count, random restarts as a batch. Float32: normalise factors each sweep. Effort
M. Value medium for mobility analysis, and it reuses `gpu-network` flow aggregation.

### 6.4 Spectral graph methods

These overlap with section 3.2: Lanczos or subspace iteration on Laplacians / Moran operators built from spatial
weights. A shared "top-L eigenvectors of a sparse symmetric operator" building block would serve RE-ESF, spectral
clustering and graph-signal smoothing. Float32 subspace iteration with Rayleigh-Ritz is robust for moderate L.

---

## 7. Reference software status (fetched 2026-11-09)

| Package | Version / date | Licence | Use |
|---|---|---|---|
| gstat (R) | 2.1-6, 2026-03-30 | GPL (≥ 2) | Space-time variograms and kriging oracle |
| GeoModels (R) | 2.3.0, 2026-10-30 | GPL (≥ 3) | Gneiting and non-separable oracle |
| GSTools (Py) | 1.7.0 | LGPL-3.0 | Metric space-time oracle |
| GpGp (R) | 1.0.0, 2025-12-18 | **MIT** | Vecchia likelihood, Fisher scoring, space-time; readable |
| GPvecchia (R) | 0.1.8, 2026-01-31 | GPL (≥ 2) | Vecchia, maxmin |
| spNNGP (R) | 1.0.2, 2026-05-28 | GPL (≥ 2) | NNGP (MCMC) |
| BRISC (R) | 1.0.6, 2024-10-02 | GPL (≥ 2) | NNGP estimation |
| GPBoost (Py) | 1.8.0 | **Apache-2.0** | Vecchia-Laplace, GLMM-GP; readable |
| FRK (R) | 2.3.2, 2026-01-16 | GPL (≥ 2) | Fixed-rank kriging |
| fmesher (R) | 0.8.0, 2026-06-23 | MPL-2.0 | Meshes, FEM matrices |
| inlabru (R) | 2.15.0, 2026-07-28 | GPL (≥ 2) | LGCP, SPDE |
| R-INLA | not on CRAN (search-only) | UNVERIFIED | Laplace oracle |
| spmoran (R) | 0.3.3, 2024-12-05 | GPL (≥ 2) | ESF, RE-ESF, STVC |
| splm (R) | 1.6-5, 2023-12-20 | GPL-2 | Spatial panels |
| mgwr (Py) | 2.2.1 | BSD-3 | GWR / MGWR (no GTWR) |
| SpatialML (R) | 1.8.2, 2026-07-06 | GPL (≥ 2) | GW random forest |
| ETAS (R) | 0.7.2, 2025-10-24 | GPL (≥ 2) | Space-time ETAS oracle |
| CAST (R) | 1.1.2, 2026-09-09 | GPL (≥ 2) | kNNDM, AOA |
| scoringRules (R) | 1.1.3, 2024-09-18 | GPL (≥ 2) | CRPS, ES, VS |
| POT (Py) | 0.9.7.post1 | MIT | Sinkhorn / unbalanced OT |
| PyDMD (Py) | 2025.8.1 | MIT | DMD |
| TensorLy (Py) | 0.10.0 | BSD-3 | CP / Tucker |
| GPJax (Py) | 1.0.0 | MIT | GP research library |
| kriging-rs (Rust / WASM / wgpu) | 0.4.0, 2026-04-26 | MIT | Prior art: WebGPU space-time kriging (separable, product-sum) |

Reported GPU results: Vecchia with batched linear algebra, 700 to 1380× over ExaGeoStat and 1M locations on one GPU
(https://arxiv.org/abs/2403.07412v3). Hawkes, >100× over single-core (https://arxiv.org/abs/2005.10123) and 1M
events on 32 GPUs, with f32 7.2× faster than f64 (https://arxiv.org/html/2407.11349v1). Exact GP by CG on 1M points
(https://arxiv.org/abs/1903.08114). ExaGeoStat mixed-precision (FP64/FP32/FP16) MLE with accuracy "close to" double
is search-only (UNVERIFIED). No WebGPU implementation of Vecchia, Hawkes EM, spatial conformal or sum-metric
space-time kriging was found.

---

## 8. Not worth doing on this platform

| Method | Reason |
|---|---|
| Full MCMC (spNNGP, lgcp MALA, Bayesian Hawkes, CARBayesST, Stan models) | Long sequential chains, data-dependent accept / reject, and stochastic oracles that are hard to pin. The Laplace / EM / Fisher-scoring alternatives above capture most of the analyst value. |
| Full INLA (nested Laplace with Gaussian-copula marginals, arbitrary hyperparameter dimension) | Many sparse Cholesky factorisations and log-determinants, where f32 is the weak point. Offer only the restricted BYM2 Laplace on a grid. |
| Exact dense-GP CG at 1e5 to 1e6 | O(n²) per iteration with poor f32 convergence on smooth kernels. Vecchia dominates. |
| Neural STPPs, GNNWR / GTNNWR, DeepKriging training, GW random forest training | Iterative stochastic training, non-commercial or absent oracles, and non-reproducible results. Provide validation (conformal, AOA, CRPS) for externally trained models instead. |
| Multi-resolution approximation (M-RA) | Complex recursive bookkeeping for a gain that Vecchia already gives. No oracle verified. |
| Dense Cholesky conditional simulation beyond a few hundred sites | Already capped. Vecchia-based simulation (sequential conditional draws in maxmin order) is the scalable path but is inherently sequential. A **blocked / coloured** Vecchia simulation is research-grade, so skip it. |

---

## 9. Suggested sequencing

1. Space-time variograms and local space-time kriging (gstat parity, plus octant search as the differentiator).
2. Scoring rules and spatial conformal intervals (small, and they immediately raise the value of 1 and of all
   existing predictors).
3. Vecchia likelihood kernel, with an f32-versus-GpGp accuracy study on n = 1e4 first. Then Fisher scoring, then
   space-time covariances and scaled Vecchia.
4. GTWR (once an oracle is pinned: GWmodel `gtwr` or a CPU reference).
5. Hawkes / ETAS EM with binned triggering (oracle ETAS 0.7.2 plus simulation recovery).
6. AOA, then Sinkhorn divergence, then Vecchia-Laplace (GPBoost oracle), and then optionally BYM2 Laplace.

## 10. Unverified items to check before committing to work

- The float32 accuracy of the Vecchia log-likelihood and Fisher scoring against GpGp (not reported in any source
  opened).
- That GPBoost implements the Kündig and Sigrist iterative Vecchia-Laplace (search-only).
- A GTWR oracle (GWmodel version and licence not fetched).
- R-INLA licence and version.
- The Gneiting variants in GeoModels (refman not fetched directly).
- LSCP (arXiv 2412.01098) content (fetch failed).
- Mohler 2011, Marsan-Lengliné 2008, Huang 2010, Knorr-Held 2000, FastGWR, GNNWR, neural STPP and DNNGP details
  (search-only).
