# Things noticed in existing GPU contributors while writing their CPU twins (2026-11-11)

Reported by the implementors and reviewers of roadmap section 13 while they read the GPU contributors. None was fixed in that pass, which created files and edited none. Each row is a lead: the line numbers are from the tree as it was on the morning of 2026-11-11, and only the rows marked "measured" were run on a GPU. Verify before acting.

Paths are relative to `modules/analysis/src`.

## Results that can be wrong or undefined

| Contributor | Where | What |
| --- | --- | --- |
| `GPUTransitionDiagnostics` | `gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics.ts:463-479` | Stationary weights are not rescaled during back-substitution. For a 36-class irreducible chain with `counts[i, i+1] = 1e9`, `counts[i, i-1] = 1` and `counts[35, 35] = 1e9`, the float32 weights overflow and the distribution is NaN while `converged` stays 1 (reasoned from the kernel, not run). The CPU twin rescales |
| `GPUTransitionDiagnostics` | same file, `:449` and `:132` | The elimination tests escape to lower-numbered states, which does not enforce the irreducibility the documentation requires. `[[1,0],[1,0]]` gives `[1,0]`; relabelling the states can make the elimination fail. The result depends on state numbering |
| `GPUTransitionDiagnostics` | same file, `:277`, `:301`, `:398` | Counts are summed in uint32 and can wrap without detection, so a non-empty row can look empty |
| Terrain flow tracing (`GPUTerrainWatersheds`, `GPUTerrainFlowDistance`) | `gpu-terrain/hydrology/terrain-flow-tracing.ts:259` | In-place pointer doubling can collapse a two-cell cycle into a self-pointer, where the documentation says a cycle stays unresolved |
| `GPUTerrainStreamOrder` | `gpu-terrain/hydrology/gpu-terrain-stream-order.ts:514` and `:599` | The Euler-tour path and the pull fallback give different orders for acyclic tributaries that enter a cycle |
| `GPUProfilePeaks` | `gpu-terrain/terrain-features/gpu-profile-peaks.ts:65`, `:563` | Finite float32 samples can make the prominence subtraction overflow; `+Infinity` then collides with the unresolved-value sentinel |
| `GPUSolarPosition` | `gpu-terrain/terrain-illumination/gpu-solar-position.ts:248` | A plain `atan2` whose second argument can be zero, where contract section 11 requires `guardedAtan2`. Effect on the GPU not measured |
| `GPUHealpixPointToCell` | `gpu-spatial-analysis/healpix-cells/healpix-index-wgsl.ts:38` | Longitude wrapping subtracts nearly equal float32 values, so a large finite longitude loses its remainder and still gets validity 1 |
| `GPUNetworkTravelTimePercentiles` | `gpu-network/network-time-geography/gpu-network-travel-time-percentiles.ts:251` | `1e-9` is subtracted before `ceil`, which can select a lower rank than the documented nearest-rank rule for a percentile just above a rank boundary. The twin keeps the same convention; decide whether it is intended |
| `GPUNetworkPathExtraction` | `gpu-network/network-analysis/network-path-extraction-passes.ts:205` | Path-row totals are uint32 scan results with no construction bound, so `requiredCount` and offsets can wrap for very long repeated paths |
| `GPUNetworkPathExtraction` | `gpu-network/network-analysis/gpu-network-path-extraction.ts:282` | With no targets the early return leaves the ID buffers unwritten, so edge padding differs from the non-empty case and a reused output keeps old contents |
| `GPURasterFractionalZonalDistribution` | `raster-analysis/gpu-raster-fractional-zonal-distribution.ts:312`, `:347` | Area validation is scheduled only when `observationWeights` or `cellAreas` is given, so a zero determinant in `parameters` alone gives zero areas without `INVALID_TUPLE` |
| `GPUBrownianBridgeUtilization` | `gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization.ts:122`, `:295`, `:315` | The normalisation claim needs a qualifier: centre-sampled mass can exceed one on a coarse grid, and zero-variance segments count in the duration denominator but add no density, so the mass can be below one |
| `GPUBrownianBridgeUtilization` | same contributor | Measured on an Apple GPU: a NaN motion variance gives a density of 0.318 for the stationary two-fix case; the input is not defined by the documentation |
| `GPUTrajectoryRecursions` | `gpu-spatial-analysis/trajectory-recursion/gpu-trajectory-recursions.ts` | Measured on an Apple GPU: a NaN position in a three-fix track gives revisits `[2, 2, 2]`; the input is not defined by the documentation |
| `GPUTerrainPeakProminence` | `gpu-terrain/terrain-features/gpu-terrain-peak-prominence.ts:169-171` | Measured on an Apple GPU: the source normalises a zero edge saddle to `+0`, but for elevations `[-0, -0]` the published key saddle is `-0`. The normalisation does not survive the shader compiler (contract section 11). The values are equal; only the sign bit differs |
| `GPUWardRegions` | `gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions.ts:737` | `iterationLimitReached` can be 1 for invalid input with a zero iteration budget; the mutual-nearest finalizer at `:1212` guards against it, the sequential one does not |

## Construction-time validation and contract departures

| Contributor | Where | What |
| --- | --- | --- |
| `GPUTrajectoryStaypoints` | `gpu-spatial-analysis/trajectory-staypoints/gpu-trajectory-staypoints.ts:240`, `:354` | `maximumSearchRows` is checked against its upper bound only; a negative or fractional value passes and becomes an invalid unsigned WGSL literal |
| `GPUHealpixPointToCell` | `gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell.ts:147` | An unknown `scheme` silently selects nested indexing |
| `GPUMaximumInscribedCircle` | `gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle.ts:80`, `:94` | Tolerance and output-length errors omit the contributor ID and bypass the named validators |
| `GPUMaximumInscribedCircle` | same file, `:116` | Feature-proportional cell storage is not checked against the storage binding size |
| `GPUMaximumInscribedCircle` | same file, `:150` | `tolerance` is a WGSL literal although it changes neither a shape nor a loop bound (contract section 5) |
| `GPUProfilePeaks` | `gpu-terrain/terrain-features/gpu-profile-peaks.ts:342`, `:504` | `minimumSide` accepts values up to `Number.MAX_SAFE_INTEGER` and is emitted as a `u32` literal |
| `GPUNetworkTravelTimePercentiles` | `gpu-network/network-time-geography/gpu-network-travel-time-percentiles.ts:224` | A negative window step is accepted when each departure is valid, against the documented non-negative step |
| `GPUNetworkTravelTimePercentiles` | same file, `:230` | The nested routing contributor is built, and its shape and budget validated, in `getCommandNodes` instead of the constructor |
| `GPUContiguityWeights` | `gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights.ts:742` | Intermediate pair overflow is folded into `overflow`; contract section 9 keeps it apart as `candidateOverflow` |
| `GPUInequality` | `dataframe-analysis/composite-indicators/inequality-parameters.ts:23` | The zone limit assumes 44 bytes per zone; the statistics stride in `inequality-wgsl.ts:12` is ten floats (40 bytes), with counts stored apart. The limit looks stale or more conservative than needed |
| `GPULocalFrameProjection` | `gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection.ts:88` | The compiled destination origin is changed without recomputing the error diagnostics (its documentation says so) |

## Documentation and test references

| Where | What |
| --- | --- |
| `raster-analysis/raster-time-series/gpu-raster-geometric-median-composite.ts:38`, `:209` | The documentation says an estimate that lands on an observation "stays there"; the update clamps its denominator but still includes the other observations, so the estimate can move |
| `modules/analysis/test/gpu-spatial-analysis/change-points/stx06-change-point-reference.ts:142` | The Pettitt test reference returns `0xffffffff` for a constant finite series; the kernel (`gpu-pettitt-test.ts:142`, `:162`) returns the second finite slice. One of the two should change |
