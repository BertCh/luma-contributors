# Algorithm practices for GPU geospatial contributors

The proposals below distinguish algorithmic work, available parallelism, memory behavior and
application overhead. A contributor with one invocation is not automatically defective: a cursor
update or tiny final solve can be efficient. The problem is allowing that invocation to own a loop
whose length grows with the dataset. Numerical contracts and existing GPU primitives constrain
each proposed replacement.

## WebGPU execution and memory

WGSL atomics use relaxed ordering. Workgroup barriers do not synchronize separate workgroups, and
an atomic flag does not publish unrelated non-atomic writes across them. Use separate dispatches
for global producer/consumer phases unless the complete protocol has a valid memory-model proof.
Collective operations require the appropriate uniform participation; tail lanes must not exit
before a barrier used by active lanes. Subgroup assumptions must follow supported features and
actual subgroup size, with a portable fallback. [WGSL memory model](https://www.w3.org/TR/WGSL/#memory-model)

For luma, this rules out blindly copying CUDA persistent-queue or spin-wait kernels. The existing
multi-dispatch scan and graph resource dependencies are a sound starting point. GPU-written
indirect arguments can suppress empty work without control readback, but declaring hundreds of
gated nodes still costs host graph construction and encoding. Inspect both costs when considering
the existing [GPU program lowering](../../../modules/graph/src/gpu-core/gpu-program-lowering.ts).

Use explicit input capacities, live counts and output counts. Check each bound storage view against
the device's binding limit as well as allocation limits; a legal allocation can still be too large
for a binding. Record default-limit and raised-limit runs separately. The WebGPU core default
storage binding limit is 128 MiB; test device limits rather than treating the local approximately
4 GiB allowance as universal. [WebGPU limits](https://www.w3.org/TR/webgpu/#limits)

Coalesced access, fewer global-memory passes and device residency are established GPU practices.
Their useful translation here is to keep positions/attributes in reusable contiguous views, order
candidate traversal spatially, and fuse compatible reductions where resource pressure allows it.
CUDA warp widths, stream APIs, occupancy calculators and cache assumptions are not WebGPU contracts.
[CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

Measure fusion rather than maximizing it. A fused shader can increase private arrays, workgroup
memory, binding count or compile cost enough to lose. Split complex paths at numerical or resource
boundaries, retaining a short common path for tiny inputs. Use source inspection to propose variants;
do not infer register spills solely from private-array size.

## Variogram fitting and regression

### Parallelize independent fit candidates

The original [variogram fitter](../../../modules/graph/src/gpu-dataframe/pair-statistics/gpu-variogram-model-fit.ts)
launched one invocation per model, scanning bins for every candidate range and refinement step.
Candidate-parallel search and cooperative refinement are now integrated and measured. The
[refinement report](variogram-refinement-integration-2026-10-24.md) separates their measurements
and remaining acceptance gates. The decomposition below explains the
design and further batching opportunities; it is not a description of an unchanged serial fitter.

Decomposition and evaluation choices:

1. Reduce minimum/maximum usable lag and valid-bin count once.
2. Evaluate independent `(model, range candidate)` combinations in parallel. Start with one lane
   per candidate for very short bin arrays and a cooperative workgroup per candidate for larger
   arrays; compare both rather than fixing a crossover in advance.
3. Reduce the weighted linear-fit sufficient statistics, preserve nonnegative nugget/sill candidate
   checks, and compute the same objective. Reuse intermediate statistics where algebra permits it.
4. Reduce candidate objectives with stable tie ordering and preserve the boundary optimum status.
5. Refine the selected brackets, parallel across models/fits and cooperative across bins. Cressie
   reweighting depends on the preceding fit and must remain a sequence of complete rounds.

This changes the parallel decomposition, not the model search domain, weights or stopping rule.
Memory should scale with candidate partials rather than materializing every candidate-by-bin value.
For a single tiny fit, extra dispatches may cost more than the arithmetic saved; retain that as an
A/B case. Batched fitting across many regions offers additional independent work.

The primary GPU reduction references are [ModernGPU segmented reduction](https://moderngpu.github.io/segreduce.html)
and [merge-based SpMV](https://mgarland.org/papers/2016/spmv/). The fitting decomposition is now a
retained implementation derived from luma's objective. These external references support its
reduction patterns; the linked local report supplies the variogram performance evidence.

### Reduce data before solving small systems

The ordinary branch of the [SUR implementation](../../../modules/graph/src/gpu-spatial-analysis/spatial-regression/gpu-seemingly-unrelated-regression.ts)
now separates centered/scaled means, initial system statistics, residual covariance and final
generalized least-squares statistics into staged nodes. See the [integration report](sur-integration-2026-10-24.md).
The general practice is to use tiled partials and a deterministic combine for large row passes,
while keeping a genuinely small matrix solve serial when appropriate. Spatial SUR remains a
separate path. This distinguishes a tiny solve from dataset-sized work needed to construct it.

Preserve equation-specific design layout, complete-case rules, intercept reconstruction, covariance
and degrees of freedom. Compare coefficients, covariance, residuals and statuses, not only fitted
values. Near-collinear designs need a conditioning failure or a separately specified robust solver;
silently adding a ridge changes the estimator.

For GWR, local statistics and permutation tests, reuse resident neighborhood/CSR structures when
their inputs are unchanged. Compact-support kernels can prune spatially; an exact Gaussian kernel
has unbounded support. A cutoff needs an explicit approximation contract. Adaptive bandwidths must
preserve distance ties and effective-neighbor semantics. Sequential permutation stopping changes
the inference procedure and must remain opt-in, with the relevant error guarantee.

## Interpolation and joint simulation

### Select solvers by matrix structure and size

MAGMA research explicitly distinguishes mappings for batches of tiny matrices, including LU, QR and
Cholesky. Register blocking and cooperation depend on system size and batch count. That is useful
evidence for size buckets, not justification to translate CUDA warp code directly.
[Tiny factorizations](https://research.manchester.ac.uk/en/publications/batched-one-sided-factorizations-of-tiny-matrices-using-gpus-chal/)

The proposed shared contract should include matrix size/stride, right-hand-side layout, valid rows,
rank/pivot status and residual evidence. Cholesky is suitable for positive-definite systems;
augmented kriging and polynomial RBF systems require a solver appropriate to their generally
indefinite structure. Do not substitute explicit inverses for solves. Reuse luma's existing
centering, scaling, pivot and failure conventions.

Local neighborhoods can bound both solve size and memory. SciPy's RBF interface makes neighbor
restriction explicit because global coefficient storage grows quadratically. This supports the
API distinction between local and global interpolation, not a claim that they produce identical
surfaces. [RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)

### Separate factorization from realizations

[Conditional simulation](../../../modules/graph/src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts)
shares one factor across replicates. Its limit is 128 observations plus 128 jointly simulated
queries. The [retained implementation](simulation-integration-2026-10-24.md) separates
validation/compaction, cooperative factorization, conditional moments and realizations. One
64-lane workgroup factors the matrix column by column. MAGMA's tiny-matrix results do not directly
establish the best mapping for this single matrix of order 256.

Preserve semidefinite query support, including exact observed locations, and the explicit failure
for singular observation blocks. A conventional strictly-positive Cholesky routine is not an
equivalent replacement. Test jitter-free exact-hit behavior and joint covariance; pointwise mean
and variance alone cannot detect broken cross-query dependence.

The [preparation-reuse implementation](simulation-reuse-integration-2026-10-24.md) now offers
opt-in reuse on `GPUConditionalGaussianSimulation`, with separate geometry and observation
generations and explicit invalidation after input writes. Its validity depends on
positions, query locations/order, covariance parameters, observation validity and measurement-error
variance. Changing observation values generally changes the conditional mean without necessarily
changing covariance; changing the seed changes realizations. Caller-declared versioning and
ownership are required for mutable GPU inputs. Submit earlier encodings before changing controls,
and destroy compiled graphs before the producer. There is no hidden cache or inferred buffer-content
identity; the default path still prepares on every encoding.

For substantially larger joint problems, Vecchia methods replace one global factorization with
bounded conditioning sets and batched local systems. They alter the statistical model and depend
on ordering/neighborhood choices. Offer such a method separately, with accuracy and covariance
evaluation, rather than describing it as an exact dense-simulation optimization.
[GPU Vecchia research](https://arxiv.org/html/2403.07412v1)

### Account for interpolation preparation

The very fast indexed natural-neighbor fixture receives both triangulation and a containing
triangle for each query. Report three costs: building topology, locating query seeds, and evaluating
interpolation. Reuse the first when sites are static; moving queries still need valid seed updates.
Compare resident queries with resident queries, and complete build/query workflows with complete
workflows. The current [scaling fixture](../../../modules/graph/test/gpu-spatial-analysis/spatial-interpolation/interpolation-scaling-fixtures.ts)
documents the excluded CPU fixture preparation.

## S2 covering and spherical geometry

S2's reference region coverer prunes disjoint cells, accepts contained cells and selectively refines
boundary cells. Its priority queue favors larger cells and uses an initial covering of a bounding
cap. `max_cells` trades approximation tightness against work and output size; a minimum level may
force more cells. This is a CPU reference for semantics and pruning, not an existing GPU kernel.
[S2 hierarchy](https://s2geometry.io/devguide/s2cell_hierarchy.html) and
[coverer source](https://raw.githubusercontent.com/google/s2geometry/master/src/s2/s2region_coverer.cc)

Two distinct luma contracts need distinct changes:

- [Uniform cap covering](../../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering.ts)
  currently examines `6 * 4^level` cells per cap in one invocation. Parallelize cap/cell candidates,
  reduce counts and compact selected IDs in canonical order. Hierarchical rejection can skip
  disjoint subtrees, but accepted output still has to reach the requested uniform level. Returning
  fewer coarse cells would change this API's result.
- [Adaptive polygon covering](../../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering.ts)
  already prunes recursively but does so in a serial stack. Test a bounded breadth-first frontier:
  classify cells in parallel, emit accepted cells, scan child counts, expand only unresolved cells,
  and continue by indirect dispatch. Reuse conservative ring/cap bounds to avoid repeated edge work.

The [October 24 classifier integration](s2-integration-2026-10-24.md) first removes repeated
cell-geometry construction while preserving the traversal body. Three isolated level-6 octant
sessions improve from 68.8–69.0 ms to 17.4–17.7 ms, with exact ordered-output checks. Frontier
parallelism and complex-ring validation remain separate profiling and implementation questions.

A parallel schedule changes visitation order. Stable final sorting can restore order after complete
execution, but under a visited-cell budget the explored prefix may differ. Either reproduce the
existing truncated-prefix contract or specify the incomplete-output policy explicitly. Preserve
interior versus exterior guarantees, tolerance, hole/complement handling and distinct output
capacity versus exploration-budget status. Include dateline, poles, face edges and tiny caps.

Do not label arbitrary ring clipping as a general spherical overlay optimization. Intrinsic
spherical overlay needs its own topology and construction policy; the current small bounded
measurements do not establish continent-scale performance.

## Watershed and terrain

### Improve queue work before changing labels

The [marker-controlled watershed](../../../modules/graph/src/gpu-raster/gpu-raster-watershed-segmentation.ts)
uses maximum path elevation and FIFO discovery age. Its former full pending-cell scan yielded
quadratic work. The [October 24 integration](watershed-integration-2026-10-24.md) replaces it with
an indexed binary heap ordered by the same `(cost, age)` key: still serial, but less total selection
work. It preserves neighbor enumeration and update-age semantics with one live entry per cell.
The retained three-session comparison supports a large improvement at 24 by 24; the asymptotic
argument alone does not establish another device's latency or performance at larger region sizes.

Parallel Priority-Flood demonstrates tile decomposition and boundary/spill summaries for depression
filling. Image watershed research demonstrates plateau processing, path reduction and union-find.
These references concern related but different outputs. Neither establishes equivalence to luma's
marker FIFO labels. A parallel method that changes plateau allocation needs a separate method
contract. [Parallel Priority-Flood](https://arxiv.org/abs/1606.06204) and
[parallel watershed partitioning](https://arxiv.org/html/2410.08946v1)

For an exact replacement, test equal-height competing markers, disconnected valid regions,
masked barriers, four/eight connectivity and forced exhaustion. For a separately named scalable
method, define the minimum-spanning-forest or plateau tie rule first, then test whole-domain and
tiled equivalence under that rule.

### Reuse terrain structure and choose the appropriate raster algorithm

luma already has [spill-graph filling](../../../modules/graph/src/gpu-terrain/hydrology/terrain-flow-spill-fill.ts),
tree contraction, distance transforms, summed-area tables and horizon sweeps. The task is tuning
and extending difficult cases rather than introducing those algorithms again. Test long narrow
drainage chains, large flats, nested pits and branching MFD flow separately. A finite halo alone
cannot summarize arbitrary upstream drainage; tiled global results need explicit boundary flow
dependencies and conservation checks.

Use direct stencil work for small neighborhoods, separable passes for separable kernels, summed-area
tables for rectangular sums, and FFT convolution only when the kernel/support/output regime
justifies its setup and padding. CuPy notes that FFT convolution can lose when only a small subset
of outputs is needed. [CuPy FFT convolution](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.signal.fftconvolve.html)

Nodata and boundary rules are part of the operator. Normalized smoothing may require separate
value and valid-weight passes; changing that denominator changes the result. Exact terrain sums
need their existing integer/local-origin treatment. Do not replace a documented precise SAT with
a large global f32 sum merely to reduce dispatches.

For fractional zonal processing, keep cell coverage and statistical weight distinct. Weighted
means can merge numerator/denominator states; quantiles cannot be merged by averaging tile
quantiles. Avoid allocating a dense zone-by-cell table when sparse coverage tuples suffice.
[exactextract operation definitions](https://isciences.github.io/exactextract/operations.html)

## Networks and resistance

### Replace dense Louvain reevaluation with sparse gain computation

cuGraph's implementation computes cluster weights, updates clustering by delta modularity and
coarsens the graph. It also constrains move direction to prevent simultaneous vertex swaps. Those
are useful algorithmic patterns; its host control loops and CUDA synchronization are not a portable
WebGPU implementation. [cuGraph source](https://raw.githubusercontent.com/rapidsai/cugraph/main/cpp/src/community/louvain_impl.cuh)

Proposed luma prototype:

1. Reuse CSR topology and calculate weighted degree totals.
2. Aggregate each vertex's incident weight by neighboring community using bounded local aggregation
   or segmented sort/reduce, with a separate path for high-degree vertices.
3. Compute local modularity gains using community totals rather than rescanning the dense matrix
   for every possible move.
4. Apply a deterministic conflict/move policy, recompute the actual objective and gate continuation.
   Simultaneous positive local gains do not guarantee a positive combined gain.
5. Coarsen and repeat with explicit level, iteration and convergence statuses.

The existing [32-vertex implementation](../../../modules/graph/src/gpu-network/network-optimization/gpu-network-louvain.ts)
supports directed/undirected and self-loop conventions that must survive. Parallel move schedules
can produce different locally optimal partitions. Define whether compatibility means exact labels,
stable labels for a seed, or objective/partition invariants. Never promise the same heuristic optimum
without evidence. The tutorial-level claim that Louvain is simply linear in edges is insufficient
for planning: rounds, coarsening and degree skew also matter.

### Separate sparse setup from iterative solve

The [resistance recipe](../../../modules/graph/src/gpu-spatial-analysis/landscape-ecology/sparse-resistance.ts)
already constructs a grounded Laplacian and invokes [Jacobi-PCG](../../../modules/graph/src/gpu-core/gpu-pcg.ts).
Profile component discovery, edge sorting/CSR, reductions, SpMV, encoding and convergence separately.
Reuse topology and a preconditioner across right-hand sides when conductances and grounding permit
it; do not reuse an invalid factor merely because the buffer object is unchanged.

Circuitscape documents the memory/convergence tradeoff between preconditioned CG and direct
factorization, and the value of batching right-hand sides. This motivates comparing stronger
preconditioning only after setup cost and true residual are measured. An AMG hierarchy can reduce
iterations while losing on a single small solve; it is not an automatic next implementation.
[Circuitscape solver guidance](https://docs.circuitscape.org/Circuitscape.jl/stable/compute/)

Validate grounded components, disconnected source/sink, zero/large conductances, current conservation
and true residual `norm(Ax-b)/norm(b)`, with an absolute criterion for zero right-hand sides.
Iteration count and preconditioned residual alone cannot certify the requested solution accuracy.

### Keep exhaustive optimization visibly bounded

The [facility optimizer](../../../modules/graph/src/gpu-network/network-optimization/gpu-network-facility-optimization.ts)
enumerates bounded cases. Parallelizing independent candidates can improve a fixed bound but does
not cure combinatorial growth. Larger instances need a separately specified heuristic, branch-and-bound
or mathematical-programming method with incumbent, feasibility and completeness status. Do not
report a capped search as a full optimum. The same principle applies to bounded alternative routes.

For reachability and centrality, reuse existing frontier and graph-program machinery. Frontier gating
reduces GPU work after convergence; inspect declared node count and CPU encode cost separately.
Benchmark grids, chains, disconnected components and power-law hubs before choosing delta-stepping
or another scheduler from external results.

## Spatial queries and constructive geometry

cuSpatial's indexed join uses polygon/quadrant candidates before exact point-in-polygon tests. This
illustrates the broad-phase/narrow-phase separation already present in luma. Measure candidate
inflation and index build/update costs, not only accepted-pair throughput.
[cuSpatial indexed joins](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/)

Retain the current spatially ordered feature/index path. Sweep uniform, clustered, overlapping,
elongated and mixed-size features; grid occupancy or box overlap can defeat apparently favorable
asymptotic complexity. Large query polygons require edge-level pruning as well as feature-level
pruning. Prepared indexes require explicit versioning when coordinates or masks change.

Parallel insertion/flipping and conflict resolution are established GPU triangulation techniques;
gCDT provides a current research implementation. Compare its algorithmic phases with luma's already
implemented parallel triangulation, especially conflicts and repeated cleanup. Do not interpret
the project's RTX 4090 throughput as a target measured on browser WebGPU.
[gCDT](https://min-tang.github.io/gCDT/)

Robust predicates should use fast certified filters followed by an exact path for uncertain signs.
Shewchuk explains why ordinary floating determinant signs can fail near degeneracy. luma's existing
integer-significand predicates have their own supported domains; exact predicate signs do not make
intersection coordinates or arbitrary spherical constructions exact.
[Robust predicates](https://www.cs.cmu.edu/~quake/robust.html)

Keep robust overlay semantics and test degeneracies before selecting a faster path. Where filtering
leaves a small active set, live-count sort/scan and compacted conflict processing are promising shared
optimizations. Track useful items versus allocated capacity to establish that benefit.

## Point clouds and clustering

The million-point lattice results are strong, but the coincident-point case exposes a different
problem. A grid or BVH cannot prune true neighbors that all occupy the same location. The
[retained cooperative path](neighborhood-cooperative-integration-2026-10-24.md) assigns one
workgroup per query, merges exact top-k lists and counts complete support. Serial remains the
default because sparse inputs can regress; multiple tiles per query remain further work. Reduce
support and moments separately from storing selected neighbor IDs. Exact full support cannot be
replaced by a capacity-limited prefix. Preserve stable top-k IDs, distances and tie semantics.

Normal estimation depends on neighborhood scale and covariance, with orientation ambiguity unless
a viewpoint or another rule resolves it. Keep neighborhood selection, centered moments, eigensolve
and orientation individually testable. [PCL normal estimation](https://pointclouds.org/documentation/tutorials/normal_estimation.html)
Record support, truncation and degeneracy with runtime, especially for lines, planes and duplicates.

For ICP, retain a target index across iterations and consider an explicitly selected multiscale
workflow. Measure time to a specified fitness/RMSE/transform target, not just time for four
iterations. Point-to-plane versus point-to-point and robust losses change the objective and must
be identified. [Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)

DBSCAN's neighborhood work can grow dramatically with radius and density; batching a distance
calculation does not bound all algorithm memory. cuML explicitly separates its distance-batch memory
setting from total DBSCAN usage. [cuML DBSCAN](https://docs.nvidia.com/cuml/latest/api/generated/cuml.cluster.DBSCAN/)
Benchmark candidate counts, cluster skew and output cardinality. For HDBSCAN include MST and tree
construction/extraction, not just neighbor queries, and distinguish exact from approximate neighbors.

For OPTICS, compare complete orderings with complete orderings at the same minimum-samples and
distance settings. scikit-learn documents a quadratic implementation, so it is a useful named
baseline rather than proof of the fastest possible CPU algorithm.
[OPTICS reference](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html)
Do not extrapolate the local 4,096-point bound or its JavaScript-oracle ratio to large datasets.

## Movement and sequential algorithms

Parallelism across many short tracks can be sufficient. A few very long tracks expose the sequential
time dimension. TensorFlow Probability describes an associative-scan Kalman formulation with
logarithmic sequential depth and larger constant costs; it explicitly notes that a sequential
filter may still win for throughput. [Parallel Kalman filter](https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/parallel_filter/kalman_filter)

Apply that distinction to a separate long-track experiment, preserving missing-observation,
covariance and timestamp semantics. For DTW/Fréchet, use admissible bounds and threshold decisions
when requested; compute full distances for survivors. Banded DTW is exact only for its declared
band-constrained problem unless the unconstrained optimum is otherwise certified. An arbitrary
band is not an exact acceleration of unrestricted DTW. The
[CUDA-DTW research implementation](https://gravitino.github.io/cudadtw/) is relevant to alignment
decomposition, not proof of matching luma's geometry metrics.

Report survivor fraction, length distribution and dynamic-programming storage. The existing
95-percent-pruned timing changes workload selectivity; it is not a matched CPU/GPU speedup.

## Shared reductions and reproducibility

[Merge-based SpMV](https://mgarland.org/papers/2016/spmv/) partitions CSR work to bound load even
under uneven nonzeros. luma already exposes [adaptive strategies](../../../modules/graph/src/gpu-core/gpu-spmv-strategy.ts)
and [load-balanced traversal](../../../modules/graph/src/utils/load-balanced-for-each.ts). Extend
those before adding another CSR engine. Average degree alone is insufficient for a million-edge hub.

[FidelityFX Parallel Sort](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/parallel-sort/)
documents the count, reduce, scan and scatter stages. luma has this substrate. Its performance
opportunity is live-prefix propagation and workload-dependent digit/tiling selection, not a new
radix-sort implementation. The [WebGPU prefix-sum experiments](https://github.com/b0nes164/GPUPrefixSums)
also identify forward-progress risks. Do not replace the portable hierarchical scan with a
spinning protocol without memory, progress and full-width-uint32 proofs.

CCCL distinguishes reproducibility across repeated runs from reproducibility across GPU
architectures, qualified by build and tuning. Use equally precise language for luma's floating
reductions. Fixed-order f32 summation, integer fixed-point accumulation, and reproducible random
counters offer different guarantees. [CCCL determinism](https://nvidia.github.io/cccl/unstable/cccl/determinism.html)

Random123 supports counter/key-based generation, which fits luma's existing Philox mapping.
Assign identity by observation, replicate and purpose, independently of scheduling and chunking.
Integer stream replay does not imply identical transcendental normal transforms on every driver.
[Random123](https://github.com/DEShawResearch/random123)

When a faster reduction changes accumulation order, validate the promised numerical tolerance and
failure classification. Do not turn an oracle into a tolerance-relaxation mechanism. Keep exact
integer counts, stable sorting and overflow status exact while treating floating error separately.

## Solver diagnostics need independent stored-solution checks

The [strategy numerical survey](shared-spmv-strategy-numerics-2026-10-24.md) adds a local example
with exact matrix condition number 129/127: forced scalar PCG reports recursive squared residual
near 1e-26 after tighter tolerances while the stored solution retains a true maximum residual of
3.662109375e-4. Default and cooperative alternatives pass. This is observed evidence for retaining
independent physical checks and qualifying accuracy before ranking performance; the recursive
criterion alone does not certify the downstream output requirement. The example does not by
itself select a residual-refresh or accumulation fix.
