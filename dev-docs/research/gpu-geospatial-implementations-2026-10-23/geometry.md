# GPU prior art: geometry, point clouds, coordinates and cells

Research snapshot: October 23, 2026. Read-only review of this working tree; no dependency installation, builds, formatting or repository edits. External code was inspected, not executed. Moving URLs below are discovery evidence; implementation work must pin commits. Recommendations and benchmark thresholds are engineering proposals, not measured luma.gl results.

## Reconcile the actual baseline first

Several tempting “new contributors” already exist. [gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts](../../../modules/graph/src/gpu-spatial-analysis/geometry-topology/gpu-snap-rounded-noding.ts) implements GPU hot-pixel noding, exact lattice intersections, deterministic source ordering and bounded capacity reporting. Its lattice step rounds up to a power of two and can grow with extent: this is not arbitrary-grid GEOS precision parity. [utils/exact-orientation-wgsl.ts](../../../modules/graph/src/utils/exact-orientation-wgsl.ts) and [delaunay-tessellation/gpu-delaunay-tessellation.ts](../../../modules/graph/src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.ts) already implement filtered integer-limb predicates. The latter's exact incircle path has a 55-bit exponent-span bound and falls back to float sign outside it. [geometry-edit/gpu-geometry-cleanup.ts](../../../modules/graph/src/gpu-spatial-analysis/geometry-edit/gpu-geometry-cleanup.ts) already rounds vertices and removes repeats/collapses; GE03 concerns geometry-level topology and collapse policies.

[neighbor-search/gpu-neighbor-search.ts](../../../modules/graph/src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search.ts) provides planar grid/BVH and spherical grid searches with `float32x2` inputs and maximum k=32. Its comments record that measured 200k-point workloads favored grids, so importing another BVH without workload evidence is unjustified. PC01 needs a genuine 3D contract. S2 point indexing and boundary geometry already exist in `cell-indexing`; [cell-topology/gpu-cell-topology.ts](../../../modules/graph/src/gpu-spatial-analysis/cell-topology/gpu-cell-topology.ts) accepts only H3/quadbin and [gpu-cell-measures.ts](../../../modules/graph/src/gpu-spatial-analysis/cell-indexing/gpu-cell-measures.ts) only H3. CR05 therefore means topology, covering and measures, not adding S2 IDs.

All paths above and below are relative to `modules/experimental/src/` unless stated otherwise.

## 1. cuSpatial: candidate expansion, scan and compaction

**Evidence: historical CUDA implementation, Apache-2.0 verified in the source header.** The [repository](https://github.com/rapidsai/cuspatial) was archived July 28, 2025; use pinned algorithm evidence, not a maintained runtime dependency. The [indexed point-in-polygon source](https://raw.githubusercontent.com/rapidsai/cuspatial/branch-25.04/cpp/include/cuspatial/detail/join/quadtree_point_in_polygon.cuh) expands polygon/quadtree pairs through prefix sums, permutation iterators and predicate compaction. It explicitly accumulates candidate counts in uint64 because a valid input can generate more candidate pairs than uint32 can hold. Its implementation also restricts multipolygons to one polygon each. The [official indexing API](https://docs.rapids.ai/api/cuspatial/stable/api_docs/spatial/) documents Morton ordering and a separate mapping to original point IDs.

**Use: GE02 subdivision; GE04 segment snapping; UR05 visibility candidates.** Preserve luma's existing join kernels; borrow the separation of candidate-count calculation, enumeration and narrow phase. GE02 can feed existing rectangle clipping/polygonization and retain original feature IDs through piece tables. Candidate counts and stable provenance are more valuable here than another point-in-polygon implementation.

**Portability/acceptance:** use bounded scans or explicit multiword counts, never silently wrap u32; CUDA allocation and iterator machinery do not port directly. Compare unsplit versus split joins on long coastlines, dense holes and skewed urban polygons at 10k/100k/1m points. Report piece-build cost, candidate reduction, output memory and repeated-query break-even. Require union reconstruction and source-ID equivalence; a smaller candidate table alone is insufficient.

## 2. PCL GPU normals: centered moments and small eigensolves

**Evidence: actual CUDA, BSD three-clause terms verified.** [PCL normal_3d.cu](https://raw.githubusercontent.com/PointCloudLibrary/pcl/master/gpu/features/src/normal_3d.cu) assigns a warp to each neighborhood, computes its centroid, then six centered covariance terms, and invokes a symmetric eigen routine. Fewer than three neighbors produce NaNs. The implementation depends on CUDA warp reductions and a block memory fence.

**Use: PC01**, then PC02, PC05 and PC06. Add a focused 3D neighbor output carrying CSR offsets, original IDs, squared distances, actual support and truncation status. Consume that in covariance/normals/shape-feature contributors. Do not retrofit planar distance semantics by dropping z. Reuse existing sort/scan and status contracts; choose 3D grid or hierarchy through measurements.

**Portability/acceptance:** replace fixed warp assumptions with workgroup reductions or feature-gated subgroups and a correct fallback. Uniform barrier participation matters. Local origins and centered accumulation reduce f32 cancellation on projected coordinates. Test translated planes, spheres, edges, duplicate/collinear points, extreme density and neighborhoods crossing chunk seams. Check eigen residuals, covariance symmetry, unit normals and sign-independent angular error against double-precision PDAL fixtures. Sweep k=8/16/32/64 and radii; benchmark full indexing plus features at 100k–1m points, not eigensolves alone.

## 3. Cupoch: composable point-cloud kernels, with cautionary contracts

**Evidence: actual CUDA; MIT terms verified in individual files.** [Estimate normals](https://raw.githubusercontent.com/neka-nat/cupoch/main/src/cupoch/geometry/estimate_normals.cu) combines neighbor lookup, reduce-by-key cumulants and a 3×3 eigensolve. Its raw-moment covariance subtracts products of means, and insufficient support can yield an artificial +z normal. This is useful comparative evidence, not a policy luma should inherit. [Registration](https://raw.githubusercontent.com/neka-nat/cupoch/main/src/cupoch/registration/registration.cu) separates nearest correspondences, compacted inliers, fitness/RMSE, transform estimation and iterative updates; its outer ICP loop is host controlled.

**Use: PC01 and PC05**, with PC06 as a later consumer. Build explicit graph stages for correspondence generation, robust masks, reductions, transform solve and update. Expose each iteration's support/residual and bounded convergence status. CPU/host-controlled CUDA ICP is not evidence of an entirely GPU-resident scheduler.

**Portability/acceptance:** map Thrust collectives onto existing scan/reduction contributors, reuse analysis-core small solves, and avoid float atomics through segmented reductions. Compare point-to-point ICP first against pinned PDAL fixtures: known rigid transforms, partial overlap, outliers, no correspondences and poor initialization. Report registration failure separately from zero error. Measure 10/30/100 bounded iterations over multiple independent tiles, including scratch and submission costs; require transform orthogonality and monotone objective only for the selected ICP/rejection variant.

## 4. gCDT: parallel triangulation maintenance, not a missing triangulator

**Evidence: 2026 algorithm paper plus public CUDA.** The [authors' project](https://min-tang.github.io/gCDT/) links their paper and [implementation](https://github.com/yingtix/gCDT), describing parallel edge flips and constraint insertion. Inspected `addConstraints.cu` uses atomic allocation and an unsigned-long-long atomic sum; the README explicitly calls out CUDA int128 support. No project license was found in the inspected tree, so treat it as algorithm-study material until reuse rights are established. Published RTX 4090 timings do not transfer to WebGPU.

**Use: GE06 ratio hulls and IN07 natural neighbors**, plus scalability review of existing constrained triangulation. Reuse `delaunay-tessellation` and [polygon-triangulation/gpu-constrained-delaunay-triangulation.ts](../../../modules/graph/src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation.ts). Ratio hulls need border-triangle removal and hole policy; natural neighbors need bounded query cavities and stolen-area weights. Neither is delivered by faster triangulation alone.

**Portability/acceptance:** redesign wide atomics, int128 arithmetic and inter-kernel state transitions into bounded worklists and explicit graph passes. Benchmark many small independent triangulations as well as one large set, including cocircular, duplicate, collinear and highly constrained cases. Validate adjacency reciprocity, constraint preservation and local Delaunay tests. IN07 additionally requires partition of unity, nonnegative weights, linear precision and explicit outside-hull invalidity.

## 5. Shewchuk: filtered exactness as an audit oracle

**Evidence: algorithm paper and CPU C reference, not GPU code.** [The author's implementation page](https://www.cs.cmu.edu/~quake/robust.html) supplies adaptive orientation/incircle predicates and explicitly places that predicate code in the public domain; Triangle itself has different terms.

**Use: GE03–GE06 robustness and CR04 prerequisites.** Existing integer-limb WGSL is a sensible adaptation because shader reassociation/contraction can invalidate error-free floating transforms. Do not replace it reflexively with C expansion arithmetic. Add a predicate audit corpus comparing identical f32 input bit patterns against CPU exact results, especially subnormals and exponent-span boundaries. Turn uncertified outcomes into observable status where topology depends on them. GE03 should assemble existing cleanup, snap-rounded noding and make-valid paths with explicit valid-output/keep-collapsed policies and stated grid restrictions. Acceptance includes idempotence, topology invariants and collapse provenance; arithmetic exactness alone does not certify reconstruction correctness.

## 6. S2: integer topology first, spherical overlay later

**Evidence: CPU implementation/oracle, Apache-2.0 headers verified.** [S2 cell IDs](https://raw.githubusercontent.com/google/s2geometry/master/src/s2/s2cell_id.cc) provide Hilbert lookup/state transforms and face-crossing neighbors; [region covering](https://raw.githubusercontent.com/google/s2geometry/master/src/s2/s2region_coverer.cc) separates intersection/containment and priority refinement. [Hierarchy documentation](https://s2geometry.io/devguide/s2cell_hierarchy.html) defines six faces and levels 0–30.

**Use: CR05 before CR04.** Extend existing two-u32 cell keys with parent/child and neighbor operations; preserve integer keys through level 30. Then add cell measures and bounded covering. Translating a CPU priority queue into breadth-first GPU frontiers changes ordering and potentially cover quality, so promise coverage invariants rather than identical cell lists unless specifically matched.

**Acceptance:** exhaustive low-level face transitions, sampled high levels, cube corners, antimeridian, poles, mixed levels and exhausted capacities. Require neighbor reciprocity and containment guarantees; test interior versus exterior covers separately. CR04 spherical ring assembly and robust arc intersection remain a distinct, substantially harder project. No convincing portable GPU S2 overlay implementation was established in this review.

## 7. PROJ: explicit prepared grid operations

**Evidence: CPU code and semantic oracle.** [Horizontal-grid docs](https://proj.org/en/stable/operations/transformations/hgridshift.html) specify ordered grids, optional resources and temporal gating; [implementation](https://raw.githubusercontent.com/OSGeo/PROJ/master/src/transformations/hgridshift.cpp) shows forward/inverse dispatch and units. No license claim is made here.

**Use: CR02–CR03.** [gpu-project/projection-program.ts](../../../modules/graph/src/gpu-project/projection-program.ts) currently describes two-dimensional axis/unit/affine/analytic/adaptive stages, not a native datum-grid operation. Add caller-prepared grid metadata, ordered selection, masks and inverse-iteration status at this explicit boundary; vertical correction needs a separately designed height port. Do not describe adaptive projection sampling as general datum-grid parity.

**Portability/acceptance:** use explicit loads/interpolation to control edges and nodata; hardware filtering alone cannot express overlapping-grid priority. Compare pinned PROJ plus supplied grids at boundaries, overlap, missing data, inverse failures and known vertical sign changes. Measure repeated transforms with grids resident, then include upload cost. Separate f32 coordinate error from interpolation error with local origins or existing split-precision infrastructure.

## 8. GeographicLib: difficult geodesics and truthful errors

**Evidence: CPU implementation, MIT/X11 verified.** [Geodesic.cpp](https://raw.githubusercontent.com/geographiclib/geographiclib/main/src/Geodesic.cpp) contains specialized inverse initialization and safeguarded iteration rather than relying only on Vincenty convergence.

**Use: CR01.** Extend existing geometry-measures kernels and ellipsoidal results; first freeze differential fixtures and error envelopes, then prototype hard-case fallback batches. The repo's well-conditioned spherical central-angle code should remain separate from ellipsoidal inverse accuracy. Porting GeographicLib's double-precision numerical contract directly to f32 is unjustified.

**Acceptance:** dense near-antipodal sweeps, nearly coincident points, poles, long edges, ring holes and orientation reversal; report convergence and error distributions against the CPU oracle. Measure easy/hard workload mixtures and high-precision scratch cost before promising a universal GPU path.

## Implementation ordering and common performance gate

Prioritize PC01's 3D neighborhood contract and GE03 policy closure; they unlock several useful workflows using existing infrastructure. Prototype CR05 integer topology and GE02 repeated joins next. Keep GE08 straight skeletons as P3: none of these sources establishes robust GPU event scheduling for simultaneous events and holes.

Use the [WGSL specification](https://www.w3.org/TR/WGSL/) as the portability boundary: no baseline native f64/int64/int128 arithmetic, integer atomics only, and workgroup-scoped synchronization. Global phases require explicit dispatch boundaries; CUDA warp assumptions need redesign. Across all prototypes report compilation, upload, resident execution, optional readback, submissions and peak scratch. Include small CPU-favored workloads, skewed workloads, whole-domain/tiled equality, forced overflow and iteration exhaustion. Preserve source chunks/ownership and keep Arrow conversion in `@luma.gl/arrow`. A CUDA speedup, constructor test or throughput chart without numerical status is not acceptance evidence.
