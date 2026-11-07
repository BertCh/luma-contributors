# Primary sources for geospatial performance practices

The eligible-strategy timing follow-up rechecked Merrill and Garland's original author abstract,
Netlib Templates section 4.2.5, Google Benchmark's random-interleaving guidance and Chrome's
developer timing guidance on October 24. The [new report](shared-spmv-strategy-performance-2026-10-24.md)
and [decision brief](performance-decision-brief-2026-10-24.md) separate those design principles from
local timings and numerical findings. The active-iteration and metadata-cost proposals are source
review and experiment specifications, not executed optimization results.

The [portable resistance follow-up](portable-resistance-research-2026-10-24.md) rechecks the original
merge-based SpMV abstract, current Circuitscape convergence guidance, the October 21 WGSL
reassociation rules and Chrome timestamp controls. The measured core SpMV attribution is local
evidence; these references support the proposed decomposition and validation practices.

Reviewed October 23, 2026, with the simulation and watershed references below added October 24.
The October 24 planning follow-up rechecked S2RegionCoverer, scikit-image watershed, GPML chapter 2,
Circuitscape solver guidance, merge-based SpMV, Chrome timestamp guidance and the
[October 21 WGSL specification](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#sync-builtin-functions).
Their application to the next experiments is recorded in [next research decisions](next-research-decisions.md).
The latest planning revision rechecked WGSL uniform workgroup loads, Chrome timestamp flags,
GPML factor/mean dependencies, S2 covering, Priority-Flood, Circuitscape residual checks and the
original Leiden paper. The variogram stage attribution and exploratory timing are local evidence,
not conclusions drawn from those external sources.
The latest source audit rechecked the merge-based SpMV author abstract, Circuitscape solver
guidance, the original Leiden paper, the cuGraph algorithm guide, GPML chapter 2 and WGSL.
The [remaining experiments](remaining-experiment-specifications.md) distinguish those references
from local source deductions and exact-arithmetic counterexamples. In particular, Circuitscape's
relative-residual policy is not luma's existing resistance acceptance rule, and the cuGraph guide's
summary complexity does not bound luma's complete multi-level workflow.
This register separates normative platform requirements, published
algorithms, implementation examples and semantic references. The recommendations in this pack
are our engineering inferences from these sources and the local evidence. No external performance
number is used to predict luma throughput. Documentation and moving source branches should be
pinned to a revision before implementation or code reuse.

The [irregular-workload planning follow-up](irregular-optimization-plan-2026-10-24.md) adds the
original WarpSelect, ECL-CC and FastSV references and rechecks WGSL synchronization, merge-based
SpMV and Circuitscape's true-residual guidance. Local staged CPU/static checks are retained
separately from GPU execution evidence.

## WebGPU and measurement

- **Transient resource lifetime and allocation**, AMD GPUOpen:
  [D3D12 Memory Allocator aliasing guidance](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/resource_aliasing.html)
  and [RPS transient resources](https://gpuopen.com/learn/rps-tutorial/rps-tutorial-part2/),
  rechecked October 24. Nonoverlapping execution lifetimes enable memory reuse; allocation
  capacity and alignment must accommodate each occupant. D3D12 aliasing also has explicit
  synchronization and initialization requirements. These are native-API mechanisms, not WebGPU
  APIs or evidence that a particular luma policy wins. Our engineering inference is to audit
  scheduled lifetimes and initialization, measure allocation count alongside requested capacity,
  and distinguish graph buffer bytes from driver heap residency. The local PCG study demonstrates
  why removing commands can still increase transient capacity when small, long-lived predicates
  occupy large reusable buffers. External guidance supplies no luma speedup estimate.

- **Google Benchmark user guide**, official project documentation:
  [random interleaving](https://google.github.io/benchmark/user_guide.html#random-interleaving).
  Rechecked October 24. Motivates interleaving repetitions to reduce exposure to changing system
  state and keeping warmup separate. It does not establish stability for our WebGPU measurements;
  the [local A/A study](pcg-measurement-and-numerics-2026-10-24.md) supplies that evidence and retains
  its exceedances. The same follow-up rechecks Netlib Templates section 4.2 for residual versus
  forward error; measured numerical differences are local results, not external performance claims.

- **WebGPU specification**, W3C/GPUWeb: [published specification](https://www.w3.org/TR/webgpu/)
  and [official source text](https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs).
  Normative basis for device limits, asynchronous pipeline creation, queue completion, buffer
  mapping and timestamp feature negotiation. The default `maxStorageBufferBindingSize` is
  134,217,728 bytes; a device may expose and an application may request more. Our Apple captures
  with raised limits do not demonstrate baseline portability. The source text was used where the
  rendered specification was too large to retrieve.
- **WGSL specification**, W3C: [memory model](https://www.w3.org/TR/WGSL/#memory-model),
  [synchronization built-ins](https://www.w3.org/TR/WGSL/#sync-builtin-functions) and
  [limits](https://www.w3.org/TR/WGSL/#limits). Normative basis for relaxed atomic semantics,
  workgroup-scoped barriers and minimum-supported complexity limits. An atomic flag does not
  establish publication of unrelated storage writes between workgroups. Optional subgroup support
  does not justify assuming a fixed CUDA warp size. Private-array size alone cannot establish
  register spilling or occupancy.
- **WebGPU developer features**, Chrome, June 2025:
  [timestamp and development flags](https://developer.chrome.com/docs/web-platform/webgpu/developer-features).
  Documents 100-microsecond timestamp quantization and a development flag that disables it.
  Record flags and separate diagnostic profiling from production-path latency measurements.
- **CUDA C++ Best Practices Guide**, NVIDIA:
  [official guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).
  Supports minimizing transfers, reusing resident data, coalescing access and exposing enough
  parallel work. Its memory, stream and occupancy mechanisms are CUDA-specific; the general
  principles require separate WebGPU implementations and measurements.

## Sparse work and shared primitives

- **Merge-Based Sparse Matrix-Vector Multiplication**, Merrill and Garland, 2016:
  [author publication page](https://mgarland.org/papers/2016/spmv/).
  Balances CSR traversal across nonzeros and row boundaries rather than assigning every row the
  same work unit. Relevant to hubs, long segments and luma's existing merge-path infrastructure.
  It supports investigating skew-aware dispatch; it does not prove every short-row case benefits.
- **ModernGPU segmented reduction**, implementation documentation:
  [segmented reduction](https://moderngpu.github.io/segreduce.html).
  Provides a concrete decomposition of irregular segments, partial reductions and carry handling.
  Useful for sparse sufficient statistics and community aggregation. Study empty segments, order
  and numerical behavior before adapting a CUDA implementation.
- **FidelityFX Parallel Sort**, AMD:
  [technique documentation](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/parallel-sort/).
  Documents a staged count, reduction, prefix and scatter design. This supports reusing common
  histogram/scan machinery and measuring full pipelines. It is not a reason to replace luma's
  existing sort without a consumer-level advantage.
- **GPUPrefixSums**, implementation research:
  [repository](https://github.com/b0nes164/GPUPrefixSums).
  Explores decoupled look-back/fallback and bounded spinning with recomputation. Relevant as an
  experimental scan alternative, especially on varied scheduling architectures. Published device
  tests are not a WebGPU-wide forward-progress proof; validate memory visibility, full-width
  counters and all target backends before promotion.

## MAGMA factorizations

- **Gaussian Processes for Machine Learning**, Rasmussen and Williams, 2006:
  [chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
  Equations 2.23–2.24 give conditional mean and joint covariance; algorithm 2.1 uses Cholesky
  solves. Supports the prepared-factor dependency analysis and distinguishes joint predictions
  from independent marginal draws. It does not certify luma's f32 pivot tolerance.
- **LAPACK DPOTRF**, official implementation:
  [source](https://netlib.org/lapack/explore-html/d0/d8a/dpotrf_8f_source.html).
  A blocked Cholesky reference using diagonal factorizations, triangular solves and matrix updates.
  Its positive-definite requirement excludes the zero-pivot query behavior that luma supports;
  adapting its structure still requires a semidefinite policy and WebGPU synchronization design.
- **Batched One-Sided Factorizations of Tiny Matrices Using GPUs: Challenges and Countermeasures**,
  Abdelfattah, Haidar, Tomov and Dongarra, 2018:
  [institutional publication record and abstract](https://research.manchester.ac.uk/en/publications/batched-one-sided-factorizations-of-tiny-matrices-using-gpus-chal/).
  Describes specialized LU, QR and Cholesky for batches of tiny matrices, including register
  blocking and parallelism across systems. The abstract supports size-specific mappings; the
  full manuscript was not retrieved for this review. A single 256 by 256 joint factor is outside
  that tiny-matrix framing. Choose Cholesky only for an appropriate positive-definite system;
  augmented kriging/RBF systems need a different numerical argument.
- **GPU-Accelerated Vecchia Approximations of Gaussian Processes for Geospatial Statistics**, 2024:
  [paper](https://arxiv.org/html/2403.07412v1).
  Uses bounded conditional neighborhoods and batched local dense operations to avoid global
  cubic factorization. Its approximation, ordering and neighbor selection change the statistical
  contract. Relevant as a separately named large-data mode, not an exact replacement for full
  joint conditional simulation. CUDA results and precision choices need new WebGPU evaluation.
- **SciPy RBFInterpolator**, official API reference:
  [reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html).
  Explains global interpolation's quadratic memory growth and the explicit neighbor option.
  Useful for documenting the distinction between global and local RBF. It is a CPU API/semantic
  reference, not a GPU kernel blueprint or a measured luma crossover.

## Spatial indexing and geometry

- **S2 cell hierarchy**, S2 Geometry:
  [developer guide](https://s2geometry.io/devguide/s2cell_hierarchy.html).
  Establishes the six-face hierarchy, four children per cell and level structure. Supports
  hierarchical rejection of disjoint subtrees and explicit distinction between fixed-level output
  and mixed-level coverings. A minimum-level constraint can prevent a small cell budget from
  being achievable.
- **S2RegionCoverer**, Google S2 reference implementation:
  [source](https://raw.githubusercontent.com/google/s2geometry/master/src/s2/s2region_coverer.cc).
  Demonstrates bounding candidates, containment/rejection and prioritized refinement. It is a
  useful algorithm/semantic reference for replacing flat enumeration. Parallel frontier ordering
  can change a budget-limited prefix, so containment, ordering and exhaustion need explicit tests.
- **cuSpatial spatial functions**, RAPIDS:
  [API documentation](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/).
  Provides quadtree and candidate-filtered spatial joins. Supports broad-phase pruning followed
  by exact geometry checks, including preparation and intermediate candidate costs. The right
  index still depends on distribution and update rate; it does not establish BVH or quadtree
  superiority over luma's measured grids.
- **Adaptive Precision Floating-Point Arithmetic and Fast Robust Predicates for Computational
  Geometry**, Shewchuk:
  [author's predicates page](https://www.cs.cmu.edu/~quake/robust.html).
  Explains orientation/incircle failures near degeneracy and adaptive precision predicates.
  Supports retaining robust topology checks. Its arithmetic assumptions and CPU implementation
  require attention before translation to WGSL; choosing a looser epsilon is not equivalent.
- **gCDT**, GPU constrained triangulation research implementation:
  [project page](https://min-tang.github.io/gCDT/).
  Demonstrates parallel insertion, flipping and constraint processing as an alternative design
  space for triangulation. CUDA/device-specific performance does not imply a WebGPU target.
  Constraints, duplicate/degenerate sites, topology correctness and preparation costs remain
  acceptance criteria.

## Raster and terrain

- **scikit-image watershed**, official API documentation:
  [watershed reference](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed).
  Documents priority and queue-entry-time tie handling, connectivity, masking and watershed-line
  options. Supports treating queue order as an output-defining choice. Full parity with luma's
  path-cost, update-age and bounded-execution rules is not established by the shared method name.
- **Parallel Watershed Partitioning: GPU-Based Hierarchical Image Segmentation**, Yeghiazaryan,
  Gabrielyan and Voiculescu, 2024:
  [paper](https://arxiv.org/html/2410.08946v1).
  Studies parallel watershed stages including plateau handling, path reduction and union-find.
  Useful for a scalable segmentation method. Different plateau/partition rules must not be
  assumed equivalent to luma's current marker-controlled FIFO tie behavior.
- **Parallel Priority-Flood Depression Filling For Trillion Cell Digital Elevation Models On
  Desktop Workstations Or Distributed Systems**, Barnes, 2016:
  [paper abstract and manuscript link](https://arxiv.org/abs/1606.06204).
  Uses tiled processing and boundary information for large DEM depression filling. This is a
  terrain decomposition reference, not proof of a GPU algorithm or marker-watershed equivalence.
  Global flow/outlet consistency must survive tile boundaries.
- **CuPy fftconvolve**, official reference:
  [API documentation](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.signal.fftconvolve.html).
  Documents FFT convolution, output modes and cases where it can lose when few outputs are
  needed. Supports a measured crossover between direct, separable and FFT approaches. Include
  transforms, padding, float conversion and exact boundary semantics in comparisons.
- **exactextract operations**, official reference:
  [operation semantics](https://isciences.github.io/exactextract/operations.html).
  Defines coverage fractions and independent weights for raster/vector statistics. Useful as
  an independent semantic reference for weighted reductions, denominators and conservation.
  It does not provide evidence for luma GPU performance.

## Networks and sparse solves

- **A High-Performance Connected Components Implementation for GPUs**, Jaiganesh and Burtscher,
  2018: [ECL-CC author page](https://userweb.cs.txstate.edu/~burtscher/research/ECL-CC/).
  Describes CSR-based union-find and degree-dependent thread/warp/block work assignment.
  Supports investigating edge-load balance in resistance connectivity. CUDA asynchronous execution
  and its performance are not a portable WebGPU implementation or completion bound.
- **FastSV: A Distributed-Memory Connected Component Algorithm with Fast Convergence**, Zhang,
  Azad and Hu, 2020: [version 2](https://arxiv.org/abs/1910.05971v2).
  Provides a distinct hooking and sparse-linear-algebra approach to connected components.
  The abstract supports this algorithm-family comparison; distributed-memory performance does
  not predict luma latency or establish a bound for a different local hooking implementation.

- **cuGraph Louvain**, RAPIDS:
  [implementation](https://raw.githubusercontent.com/rapidsai/cugraph/main/cpp/src/community/louvain_impl.cuh)
  and [algorithm guide](https://docs.rapids.ai/api/cugraph/stable/graph_support/algorithms/louvain_community/).
  Source shows cluster-weight aggregation, moves by modularity delta and graph coarsening.
  Supports a sparse replacement for repeated dense objective evaluation. Parallel move schedules
  can change heuristic local optima; validate the objective, self-loop/directed conventions and
  reproducibility policy instead of requiring identical labels in every tie case.
- **Circuitscape computation**, official documentation:
  [solver guidance](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/).
  Discusses sparse direct versus iterative methods, AMG preconditioning, multiple right-hand
  sides and true residual checking. Relevant to resistance recipe profiling and amortization.
  Preconditioning has construction and memory costs; CPU package experience is not evidence that
  AMG will improve the current eight-iteration hub fixture on WebGPU.

## Point clouds clustering and movement

- **Billion-scale similarity search with GPUs**, Johnson, Douze and Jégou, 2017:
  [version 1, sections 4 and 5](https://arxiv.org/html/1702.08734v1#S4).
  Separates streaming k-selection and distance production, using local queues and cooperative
  merging. This motivates exact local-list experiments and avoiding a full distance matrix.
  WarpSelect's CUDA register/warp mechanisms and reported gains do not transfer directly to WGSL.
  Luma additionally needs full radius-support counts and stable original-ID ties.

- **Estimating Surface Normals in a PointCloud**, PCL:
  [tutorial](https://pointclouds.org/documentation/tutorials/normal_estimation.html).
  Defines neighborhood covariance, scale selection and normal orientation. Supports centered
  reductions, explicit support and degeneracy tests. Bounded output capacity must not silently
  replace the specified neighborhood with an arbitrary subset.
- **ICP registration**, Open3D:
  [tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
  Separates correspondence quality, inlier RMSE and convergence behavior, and distinguishes
  point-to-point from point-to-plane objectives. Supports measuring time to an agreed alignment
  quality, in addition to fixed-iteration throughput.
- **cuML DBSCAN**, NVIDIA/RAPIDS:
  [API reference](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.DBSCAN/).
  Documents a memory bound for pairwise-distance batches, explicitly not total algorithm memory.
  Useful for exposing temporary-memory controls. Batching alone does not eliminate quadratic
  dense-neighborhood work.
- **scikit-learn OPTICS**, official API reference:
  [reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html).
  Describes its nearest-neighbor preprocessing and expansion without a heap, yielding quadratic
  time. Useful for selecting and labeling an independent CPU baseline; this reference must not
  be advertised as the fastest available CPU algorithm.
- **Parallel Kalman filtering**, TensorFlow Probability:
  [experimental API reference](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/parallel_filter/kalman_filter).
  Describes associative-prefix filtering with logarithmic temporal depth and linear work, with
  larger constants. A candidate for few long tracks; many short tracks already expose parallelism
  across tracks. Missing-data and covariance semantics must match the existing filter.
- **cuDTW**, research implementation:
  [project page](https://gravitino.github.io/cudadtw/).
  Provides CUDA time-series alignment and distance-search implementation context. Useful for
  investigating tiling and dependency scheduling. This review does not establish that its exact
  pruning rules match any luma movement method; prove bounds before transferring them.

## Numerics and reproducibility

- **Parallel central moments**, Pébay, Terriberry, Kolla and Bennett, 2015:
  [Sandia publication record](https://www.sandia.gov/research/publications/details/formulas-for-robust-parallel-computation-of-arbitrary-order-arbitrary-varia-2015-02-01/).
  Reviewed October 24. Supports mergeable centered statistics for SUR row reductions. The
  publication abstract was reviewed; no new performance result or WGSL accuracy guarantee follows.
- **LAPACK least-squares drivers**, official users' guide:
  [QR/LQ and rank-revealing alternatives](https://netlib.org/lapack/lug/node27.html).
  Reviewed October 24. Supports an independent QR/SVD diagnostic rather than relying only on a
  second normal-equations implementation for difficult SUR inputs. Rank-deficient minimum-norm
  solutions are not permission to change the contributor's failure contract.
- **From Louvain to Leiden: guaranteeing well-connected communities**, Traag, Waltman and
  van Eck, 2019: [author manuscript record](https://arxiv.org/abs/1810.08473).
  Reviewed October 24. Establishes that Louvain can produce disconnected communities and motivates
  a separately evaluated refinement method. It does not establish luma performance.
- **cuGraph Louvain API, version 26.08**:
  [versioned API documentation](https://docs.nvidia.com/cugraph/26.08/api_docs/api/cugraph/cugraph.louvain/).
  Reviewed October 24. The documented implementation supports undirected graphs only; use it as
  an undirected comparison, not an oracle for luma's directed objective and self-loop conventions.
- **CCCL determinism**, NVIDIA:
  [determinism documentation](https://nvidia.github.io/cccl/unstable/cccl/determinism.html).
  Distinguishes unconstrained, run-to-run and cross-GPU reproducibility policies. Supports stating
  reduction guarantees explicitly and testing floating-point order sensitivity. CUDA library
  guarantees do not automatically apply to WGSL or luma's current primitives.
- **Random123**, D. E. Shaw Research:
  [official repository](https://github.com/DEShawResearch/random123).
  Defines counter-based generators, including Philox. Supports tying random draws to logical
  sample/replicate identity instead of dispatch shape. Preserve the existing counter/key mapping
  when changing batching or factorization; a generator name alone does not establish identical
  normal transforms or cross-device floating-point samples.

## Local evidence and reuse policy

The [findings](README.md) link each timing to its retained capture. The
[evidence manifest](evidence.json) records hashes and extracted observations, including the full
matched variogram comparison. These hashes describe files at research time, not necessarily the
source state that generated an earlier benchmark. The [algorithm review](algorithm-practices.md)
links the specific implementation paths supporting each structural finding.

No third-party implementation code was copied. Before any adaptation, pin the upstream revision,
inspect its license and transitive dependencies, preserve attribution and verify WebGPU memory and
progress assumptions. A paper, an open repository and a reusable implementation are distinct forms
of evidence. The earlier [implementation survey](../gpu-geospatial-implementations-2026-10-23/README.md)
contains further implementation and licensing leads.

## Capture audit and repeated resistance queries

Reviewed October 24 for the [evidence audit](research-decisions-and-evidence-audit-2026-10-24.md).

- **WebGPU query creation**, [specification](https://www.w3.org/TR/webgpu/#dom-gpudevice-createqueryset)
  and [reviewed source](https://raw.githubusercontent.com/gpuweb/gpuweb/main/spec/index.bs):
  query-set count must not exceed 4,096. This explains the retained chain/grid diagnostic errors;
  it is not a requestable device-limit increase. The published HTML exceeded the fetch limit,
  so the official source was used for the validation rule.
- **Chrome developer features**, [official guidance](https://developer.chrome.com/docs/web-platform/webgpu/developer-features):
  default timestamp quantization and developer flags remain material to short-stage attribution.
- **Circuitscape computation guidance**, [official documentation](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/):
  scalar pairwise shortcuts, requested maps, true-residual validation and solver reuse motivate
  distinct consumer experiments. No Circuitscape timing is used to predict luma performance.
- **Effective Resistance and Schur Complements**, Daniel Spielman, October 10, 2018:
  [Yale lecture 13](https://www.cs.yale.edu/homes/spielman/561/lect13-18.pdf). The unit-flow
  quadratic form supports the proposed fixed-anchor response derivation. The pair-error budget,
  storage plan and API constraints in the audit are our engineering deductions.

## Resumed research review

Rechecked October 24, 2026: WGSL static access and synchronization, the WebGPU source
specification, Chrome timestamp behavior, merge-based SpMV, Circuitscape convergence and
preparation reuse, S2RegionCoverer, and the Leiden author manuscript. The
[current assessment](current-assessment-and-research-plan.md) applies these sources to the retained
captures and records the distinction between observed failures and source-based inferences.

- **Templates for the Solution of Linear Systems**, Barrett and colleagues, second edition:
  [Netlib text](https://www.netlib.org/templates/templates.pdf), chapter 4.2. Reviewed the stopping
  criteria and preconditioned-residual discussion. Supports keeping the original-system residual
  distinct from a preconditioned stopping statistic; interpreting solution error additionally
  depends on conditioning. This source does not define luma's physical acceptance rule.
- **WGSL shader interface**, W3C, October 21, 2026:
  [versioned resource interface](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#shader-interface).
  Resource membership depends on static access. This supports diagnosing the empty-entry Jacobi
  shader's unused row-offset bindings; the retained WebGPU error is the direct execution evidence.

## Certificate arithmetic follow-up

Reviewed October 24, 2026. The [numerical decision memo](residual-certificate-research-2026-10-24.md)
uses these primary sources alongside the local eight-case arithmetic artifact:

- Higham, *The Accuracy of Floating Point Summation* (1993),
  [author-hosted paper](https://nhigham.com/wp-content/uploads/2023/10/high93s.pdf): ordering,
  cancellation and comparisons among summation methods; no universal superiority claim.
- [WGSL floating-point evaluation](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#floating-point-evaluation):
  implementation freedoms qualify the CPU model and any proposed portable error proof.
- [CCCL determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html): distinct scopes
  of reproducibility; an analogy for documenting requirements, not a WebGPU guarantee.
- [Circuitscape solver guidance](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/):
  independent true-residual acceptance and single-precision limitations.

The Berkeley ReproBLAS project page was discoverable in search but returned HTTP 403 when opened
in this review. No implementation or WebGPU compatibility claim relies on that inaccessible page.

## Interleaved hybrid comparison and domain validation

Rechecked October 24, 2026 for the explicitly selected baseline/hybrid comparison:

- [Google Benchmark random interleaving](https://google.github.io/benchmark/user_guide.html#random-interleaving)
  supports interleaving to reduce exposure to system-state changes. The local 15-pair A/A result
  qualifies this harness on its tested controls; the guide does not establish luma's error rate.
- [Merrill and Garland's merge-based SpMV](https://mgarland.org/papers/2016/spmv/)
  motivates balancing CSR rows and entries. The hybrid's 256-entry cutoff and preserving scalar
  short-row arithmetic are local design decisions, not recommendations made by that paper.
- [Templates, section 4.2](https://www.netlib.org/templates/templates.pdf)
  distinguishes residual-based stopping from solution-error guarantees. The
  [domain validation plan](pcg-domain-validation-plan-2026-10-24.md) therefore keeps independent
  currents, energy, resistance and stored-system residual checks alongside PCG telemetry.


## Weighted network precision and acceptance

Rechecked October 24 for the [domain results](pcg-domain-numerics-2026-10-24.md):
[WGSL floating-point types](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#floating-point-types)
defines f32 as IEEE binary32. The local exact rational artifact derives its current-grid bound from
that representation; WGSL evaluation freedoms remain separate from the CPU rounding diagnostic.
[Templates section 4.2](https://www.netlib.org/templates/templates.pdf) supports keeping recursive
stopping, independent true residual and solution accuracy distinct. The residual acceptance
regression is established by the local fixed-potential GPU comparison, not inferred from either
reference.

## Real-network validation candidates

The [real-network validation plan](real-network-validation-plan-2026-10-24.md) records the October 24
candidate-source review and its acquisition limits. Official DIMACS and SNAP metadata identify
small and large road graphs; their files and checksums have not yet been acquired. SciPy's official
`cg` and `spsolve` API pages support the proposed independent-reference procedure. The plan separates
source facts from our proposed graph transformations, convergence checks and acceptance rules.


## Generic allocator harness follow-up

Rechecked October 24, 2026. The versioned [WGSL reserved-word list](https://www.w3.org/TR/2026/CRD-WGSL-20261021/#reserved-words)
includes `target`; the repaired harness's browser log directly reports that shader-creation error.
TypeScript checks do not compile embedded WGSL. Shader construction and small GPU readback
fixtures therefore need an execution check before a larger allocator comparison. This is a local
validation-process lesson, not evidence of an allocator defect.

The [AMD aliasing guidance](https://gpuopen-librariesandsdks.github.io/D3D12MemoryAllocator/html/resource_aliasing.html)
was also rechecked: reuse depends on disjoint use ranges and storage large enough for every
occupant. Its explicit D3D12 aliasing barriers and initialization rules are backend-specific;
these studies reuse WebGPU buffer objects through the existing graph allocator. The local
allocation maps and readbacks establish the measured behavior.

## Isolated follow-up and latency preparation

Rechecked October 24, 2026. [Google Benchmark's guide](https://google.github.io/benchmark/user_guide.html#random-interleaving)
supports interleaving to reduce system-state effects and distinguishes warmup from measured work.
The planned balanced schedule and its local drift limits remain our experimental choices.
[Chrome's developer-feature guidance](https://developer.chrome.com/docs/web-platform/webgpu/developer-features)
documents timestamp quantization and the developer flag that changes it; record that setting
when comparing GPU timestamps. This does not turn timestamps into complete application latency.
[Merrill and Garland's paper](https://mgarland.org/papers/2016/spmv/) addresses uneven CSR work;
it supports testing skewed rows, while local crossover and complete-consumer benefit still require
measurement. None of these sources establishes a speedup for the new allocator policies.
