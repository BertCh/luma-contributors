# GPUDorlingCartogram

{/* lens:GPUDorlingCartogram */}

::lens{scenario=gpu-spatial-analysis.dorling-cartogram rung=graph}

Constructs a bounded planar circle cartogram entirely in a caller-encoded GPU graph. Input
`positions` are initial centers in equal units on both axes; project geographic coordinates first.
`values` determine circle area: `pi * radius² = value * areaScale` (`areaScale` defaults to 1).
Rows and source IDs are preserved. Optional `sourceIds` default to zero-based input rows.

Each synchronous iteration sums half-penetration forces over exact spatial neighbors (all other
circles), averages by overlapping neighbor count, and attracts the circle toward its original
center by `attraction * (origin - center)`. `attraction` defaults to 0.01 and accepts 0..1.
Coincident circles separate along the x axis using row order. This is a deterministic, geography
anchored Dorling variant; it does not reproduce GeoDa's neighborhood forces or coordinates.
Attraction and competing forces may leave unresolved overlap. No convergence guarantee is made.

`maximumIterations` is a static budget from 0 to 256 (default 100), with at most 1024 input rows. GPU
checks after initialization and every pass freeze the centers when maximum penetration is at
most `overlapTolerance` (default 0.001 position units). A stationary overlapping layout remains
unconverged. There is no CPU iteration decision, submission, readback, or fallback.

Caller-owned `output` views are:

| View | Format and length | Meaning |
| --- | --- | --- |
| `positions` | float32x2, input rows | Final centers |
| `radii` | float32, input rows | Area-derived radii |
| `sourceIds` | uint32, input rows | Input IDs, including excluded rows |
| `metrics` | float32x2, one row | Maximum residual penetration; maximum distance from original center |
| `status` | uint32, four words | Iterations used; converged flag; budget exhausted flag; excluded rows |

Zero values produce stationary zero-radius points that exert no force. Negative or nonfinite
values, nonfinite coordinates, coordinates outside ±1e15, and derived radii above 1e15 are
excluded and produce NaN centers/radii. A zero area scale produces zero radii. Empty/all-excluded
inputs converge vacuously with zero metrics and zero iterations. Nonempty inputs require packed
views, matching row counts, and separate output buffers. Borrowed input ownership is preserved;
chunked vectors require explicit caller packing. Float32 precision can prevent small corrections
at large coordinate magnitudes; inspect residuals and subtract a common coordinate origin.

Each pass has O(n²) exact pair queries distributed over rows; the scalar diagnostic reduction is
O(n). Temporary storage is O(n). Kernels use at most seven storage buffers, within default WebGPU
limits. Larger interacting layouts require a future indexed implementation. Independent partitions
do not resolve overlaps across partition boundaries and are not equivalent to one global layout.

The pinned behavioral reference is Luc Anselin's *Exploring Spatial Data with GeoDa: A Workbook*,
2005, section 11.3.2, printed pages 83–85,
[GeoDa workbook](https://geodacenter.github.io/docs/geodaworkbook.pdf), SHA256
`dc3f15071e278cc824c4f3de74ad6dbb7463ff11c79cbef76bb9e92f1598f635` (retrieved 2026-10-25).
It describes iterative nonlinear placement and optional additional improvement iterations. This
implementation compares those behaviors and circle invariants; the analytic pair fixtures are
independently derived and are not GeoDa output coordinates. GeoDa outlier styling, adjacency
attraction, polygon centroid extraction, automatic area scaling and geographic projection are
outside this contract.
