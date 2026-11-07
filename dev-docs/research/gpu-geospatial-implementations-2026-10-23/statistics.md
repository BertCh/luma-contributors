# GPU prior art: spatial statistics, geostatistics and sampling

Research date: 2026-10-23. Read the supplied AGENTS.md, the coverage backlog, and relevant source in the current dirty working tree. No repository files were changed, no dependencies installed and no tests or formatting run. External code was inspected, not built or benchmarked. Engineering recommendations below are inferences; upstream CUDA results do not predict WebGPU speedups. `IN`, not `IP`, is this backlog's interpolation prefix.

## Reconciled implementation baseline

Paths below are relative to `modules/experimental/src/`.

* [gpu-spatial-analysis/spatial-interpolation/gpu-kriging.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging.ts) already implements local ordinary kriging, deterministic neighbor ordering, duplicate averaging, partial-pivot elimination, variance, leave-one-out predictions and conditioning diagnostics. Its compile-time neighbor cap is **16**. Exact hits have zero variance; its nugget represents microscale variation, explicitly not measurement error. IN02–IN03 must extend these semantics rather than replace an assumed missing predictor.
* [gpu-dataframe/pair-statistics/variogram-model.ts](../../../modules/graph/src/gpu-dataframe/pair-statistics/variogram-model.ts) fits spherical/exponential/Gaussian models **on the CPU**, using a 200-candidate logarithmic range ladder, golden-section refinement and nonnegative nugget/sill fits. It supports pair, unweighted and iterative Cressie weights. IN01 is a resident fitting extension of this actual algorithm.
* Gaussian [spatial-regression/gpu-geographically-weighted-regression.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression.ts) already exposes AICc bandwidth ladders, refinement, GPU golden-section search and local condition numbers. [gpu-generalized-geographically-weighted-regression.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts) already supports adaptive/fixed Poisson and logistic fits, but explicitly excludes bandwidth search and inference. ST06 therefore concerns remaining VIF/variance-decomposition surfaces, model-specific search/inference and MGWR uncertainty, not all GWR bandwidth selection.
* [spatial-regression/gpu-spatial-two-stage-least-squares.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares.ts) already has instrument and conditioning diagnostics; its documented covariance is homoskedastic and explicitly excludes HAC/heteroskedastic corrections. ST01 is a real gap.
* [spatial-clustering/gpu-hierarchical-density-clustering.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering.ts) already constructs an exact mutual-reachability MST with BVH-pruned Boruvka rounds, a dendrogram and condensed-tree extraction. PP04 adds OPTICS ordering/extraction, not HDBSCAN.
* [permutation-inference/permutation-random.ts](../../../modules/graph/src/gpu-spatial-analysis/permutation-inference/permutation-random.ts) already provides Philox4x32-10, unbiased bounded integers, purpose-separated streams and Feistel permutations in CPU/WGSL forms. [gpu-local-permutation-replay.ts](../../../modules/graph/src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay.ts) already replays draws for inspection. ST07/PP02 should build on these.
* `analysis-core/contracts/{status,inference}.ts` already supplies bounded iteration, approximation, invalidity and solver contracts. Current dense Cholesky/eigenvalue helpers actually live in [gpu-spatial-analysis/spatial-regression/spatial-regression-solve.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/spatial-regression-solve.ts); “small solves in analysis-core” is a prospective ownership direction, not today's implementation location.

## Seven useful primary-source families

### 1. GWmodel CUDA: ST06, with solver lessons for ST02–ST05

**Evidence: actual CUDA implementation.** The upstream [GWmodel CUDA kernel](https://github.com/GWmodel-Lab/GWmodel/blob/60dc1cee77957aa4f899ca1bd4898d151dc579af/src/GWmodelCUDA/GWmodelKernel.cu) implements distance, adaptive/fixed weighting and weighted-design operations. Adaptive weighting sorts distances with Thrust. Its [package metadata](https://github.com/GWmodel-Lab/GWmodel/blob/60dc1cee77957aa4f899ca1bd4898d151dc579af/DESCRIPTION) declares GPL >=2; treat it as algorithm and differential-reference material rather than assuming its code can enter luma's MIT files.

**Transfer:** batch many local weighted designs; retain preparation across candidate evaluations. In luma, extend [gpu-generalized-geographically-weighted-regression.ts](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression.ts) using the existing Gaussian candidate-search graph and shared solve/status conventions. Local VIF requires more covariance information than a scalar condition number.

**Port risks:** upstream uses double precision, CUDA allocation and host loops; a WGSL implementation needs centered/scaled f32 designs, bounded resident search and explicit failed-fit scoring. Sorting every observation for every focus is a comparison baseline, not automatically the best architecture for luma's indexed neighborhoods. Check duplicated distances, singular neighborhoods, separation and candidate ties; compare coefficients, information matrices, AICc/deviance and search traces under the same model.

### 2. CuPy RBFInterpolator: IN05, supporting IN02–IN03 dense-system design

**Evidence: GPU arithmetic with host orchestration.** [CuPy's source](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/cupyx/scipy/interpolate/_rbfinterp.py) forms the augmented kernel/polynomial system, scales the polynomial domain, queries neighbors and deduplicates sorted neighbor sets before solving. Crucially, local mode calls `inv.tolist()` and loops over neighborhoods in Python: this is useful GPU prior art, not an entirely GPU-resident implementation. The [project license](https://github.com/cupy/cupy/blob/9aac5cf188c24adf2e6c5051278a1491d0101a1f/LICENSE) is MIT; preserve any file-specific upstream provenance when adapting.

**Transfer:** share solves among identical neighborhoods, preserve a query-to-neighborhood map and evaluate multiple output fields against one factorization. Reuse luma's interpolation indexing; add bounded augmented solves rather than global quadratic storage.

**Port risks:** the polynomial-augmented matrix is generally indefinite; existing SPD Cholesky is not a drop-in solver. Start with partial-pivot LU or an explicitly valid block formulation. Establish polynomial rank, smoothing and extrapolation contracts. Test thin-plate zero-distance handling, collinear samples, exact interpolation at smoothing zero and affine reproduction with degree one.

### 3. ParallelVecchiaGP / batched Vecchia paper: IN02, IN04, IN06

**Evidence: GPU implementation plus primary research paper.** [Pan et al., 2024](https://arxiv.org/html/2403.07412v1) evaluate GPU batched conditional covariance calculations with KBLAS, replacing a global dense Gaussian-process likelihood with bounded neighbor conditionals. The linked [ParallelVecchiaGP repository](https://github.com/kaust-es/ParallelVecchiaGP/tree/d7e8260deb439520b2b00ca24b53f873c785d150) contains CUDA kernels and estimation/prediction code; [its license](https://github.com/kaust-es/ParallelVecchiaGP/blob/d7e8260deb439520b2b00ca24b53f873c785d150/LICENSE) is BSD-3-Clause. The paper's CC-BY license is separate.

**Transfer:** resident batches of similarly sized covariance systems, reusable neighbor preparation and explicit approximation settings. This supports an experimental architecture for local prediction/simulation, not a claim that Vecchia equals ordinary kriging or fulfills IN04's valid coregionalization model.

**Port risks:** conditioning-set ordering is part of the approximation. Browser scratch memory makes simultaneous storage of every local matrix unattractive; process explicit bounded batches. Kriging-with-drift systems and PSD covariance factors need distinct solver contracts. Validate likelihood/prediction error versus dense small cases, then ensemble mean/covariance and conditioning behavior for IN06. Do not reuse the paper's large speedup ratios as evidence for luma.

### 4. RAPIDS cuML / RAFT / cuVS: PP04 and shared neighborhoods for PC01–PC02

**Evidence: production CUDA pipeline.** The current [cuML HDBSCAN runner](https://github.com/rapidsai/cuml/blob/93418baf59a121058189d654447447803859aab7/cpp/src/hdbscan/runner.cuh) separates mutual-reachability linkage, hierarchy condensation and cluster extraction, using RAFT resources and cuVS graph construction. This is valuable modular GPU prior art; it is **not evidence of a RAPIDS OPTICS implementation**. The [cuML license](https://github.com/rapidsai/cuml/blob/93418baf59a121058189d654447447803859aab7/LICENSE) and [RAFT license](https://github.com/rapidsai/raft/blob/main/LICENSE) are Apache-2.0; inherited third-party notices remain relevant.

**Transfer:** reusable neighbor/core-distance products and explicit exact/approximate graph choices. Keep PP04's predecessor, reachability and extraction outputs separate from current HDBSCAN labels. PC01 needs a genuinely 3D neighborhood adapter rather than feeding a planar API z coordinates indirectly.

**Port risks:** dynamic frontiers and repeated minimum selection can serialize OPTICS; prototype the frontier cost before committing. Pin self-neighbor conventions: the inspected runner adjusts `min_samples` for scikit-learn-contrib compatibility, while luma documents self-inclusive semantics. Compare partition invariants rather than raw label IDs, and separately inspect reachability/core distances, equal-distance tie policies, neighborhood recall, truncation and disconnected components.

### 5. Random123: ST07, PP02, PP05 and IN06 reproducibility

**Evidence: existing CPU/CUDA/OpenCL implementation.** [Random123](https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150) is a counter-based RNG family; the [license](https://github.com/DEShawResearch/random123/blob/9545ff6413f258be2f04c1d319d99aaef7521150/LICENSE) records BSD-style redistribution and non-endorsement conditions. Luma already implements Philox; no replacement is proposed.

**Transfer:** extend purpose tags so simulation, fold assignment and sampling cannot accidentally reuse streams. Associate counters with stable frame/row, replicate and operation identities rather than dispatch ordering. Preserve current replay instrumentation for multivariate adapters.

**Port risks:** WGSL lacks a native u64 multiply-high path matching CUDA, but luma already supplies exact 16-bit partial products. Add known-answer vectors and invariance across batch partitioning. Counter generators alone do not establish uniform permutation laws, a valid spatial sampling design or a global envelope test; validate these constructions independently. ST07 must jointly permute the intended multivariate observation, with common masks and explicit multiple-testing families.

### 6. PySAL spreg TSLS: ST01–ST05 correctness oracle, not GPU prior art

The official [TSLS contract](https://pysal.org/spreg/generated/spreg.TSLS.html) exposes White and HAC covariance, kernel weights with unit diagonal, and the `sig2n_k` convention. Pin [spreg source](https://github.com/pysal/spreg/tree/4dbc809242d9fe8f5fca0257c7dc9c957fc1ce36) for fixtures rather than relying on moving development documentation. Its [license](https://github.com/pysal/spreg/blob/4dbc809242d9fe8f5fca0257c7dc9c957fc1ce36/LICENSE.txt) is BSD-style with three conditions. This reference defines estimator behavior; it provides no WebGPU implementation or timing evidence.

**Transfer:** add ST01 as a covariance stage over existing 2SLS residuals/instruments. Form the correct projected-design sandwich and apply sparse kernel multiplication without materializing a dense n-by-n matrix. Keep estimator, HC convention, finite-sample correction and HAC kernel independently identifiable.

**Port risks:** wrong matrix ordering can produce plausible but incorrect standard errors. Verify the full covariance, symmetry, coefficient-order mapping and diagonal standard errors; compare white-noise, heteroskedastic, spatially correlated and singular-instrument fixtures. Start with one explicit convention. ST02 restrictions and ST03–ST05 likelihood variants remain separate estimator projects.

### 7. EPA spsurvey GRTS: PP05 design oracle, linked to ST10 evaluation

**Evidence: CPU design reference, not established GPU port.** EPA's official [grts documentation](https://usepa.github.io/spsurvey/reference/grts.html) describes finite-point/line/area frames, strata and equal/unequal inclusion schemes, and returns design information. Pin [spsurvey source](https://github.com/USEPA/spsurvey/tree/68c429bc4f1682a27dd029e3b13b8292d6bf7714) for exact fixtures; its [metadata](https://github.com/USEPA/spsurvey/blob/68c429bc4f1682a27dd029e3b13b8292d6bf7714/DESCRIPTION) declares GPL >=3. GPU sorting/random traversal is an implementation proposal; random Hilbert ordering alone is not evidence of GRTS equivalence.

**Transfer:** begin with finite-point equal-probability selection using existing spatial keys, scans and Philox. Return original IDs, inclusion probabilities, reciprocal weights and seed/provenance, with the frame interpretation explicit.

**Port risks:** clipping, omitted cells and capacity truncation can alter inclusion probabilities. Validate many-replicate empirical inclusion frequencies, sample-size invariants, design-weighted totals and spatial balance against random sampling. Keep ST10's train/test folds distinct from probability samples: fold generation must also enforce buffer separation and fit preprocessing, bandwidth and variogram choices inside training folds.

## Recommended contributor slices and acceptance evidence

1. **IN01 resident fitting:** translate the current constrained range search faithfully; batch bins/candidate models, expose residual and exhausted-search status, and preserve all current weight conventions. Compare parameter/objective results against the existing CPU implementation and an independent pinned reference. Benchmark a few hundred bins and repeated refits: avoiding readback may matter more than arithmetic acceleration.
2. **ST01 robust covariance:** extend existing 2SLS without changing point estimates. Full covariance fixtures and kernel provenance are the deliverable, alongside measured sparse-work scaling.
3. **IN05 bounded local RBF:** use existing neighbor preparation and an augmented dense solver with rank diagnostics. Establish a supported neighbor/polynomial envelope before broad kernel coverage.
4. **ST06 remaining inference; ST07/PP02 simulation composition:** reuse search and Philox infrastructure, preserving model-specific uncertainty and the distinction between pointwise and global tests.
5. **PP05 and ST10:** deliver narrow designs with statistical validation; PP04 waits for a convincing WebGPU frontier prototype.

For each slice report compile, preparation/indexing, upload, execution, optional readback, peak scratch storage and repeated-update cost separately. Exercise small CPU-favorable cases, clustered/skewed data, large coordinate offsets, near-singular neighborhoods, empty masks and forced capacity/iteration exhaustion. Use f64 CPU references plus independent invariants; publish tolerances as a function of conditioning rather than promising blanket bitwise parity. Preserve chunk boundaries and caller ownership; default binding budgets and explicit prepared-input invalidation apply to all reused neighborhoods. No new Arrow dependency belongs in experimental kernels. This domain review has not certified current numerical execution; implementation work must follow the AGENTS.md build and test gates.
