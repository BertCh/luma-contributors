# Dasymetric redistribution

`addDasymetricRedistributionRecipe` redistributes nonnegative source totals using fractional
source-polygon coverage and an ancillary raster. Targets own whole raster cells through a
`uint32` target-zone raster. The recipe runs coverage, calibration, eligibility, weighted areal
interpolation, transfer and mass accounting on the GPU. It performs no readback or CPU fallback.

```typescript
import {addDasymetricRedistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';

const result = addDasymetricRedistributionRecipe(graph, {
  grid,
  source: {kind: 'coverage', coverage},
  sourceTotals,
  targetZones,
  targetZoneCount: 4,
  ancillary: {id: 'density', format: 'float32', storage: {kind: 'buffer', values: density}},
  mask: eligibleCells,
  denominator: 'overlap',
  pairCapacity: 32
});
```

`source: {kind: 'polygons', zones, tupleCapacity, parameters}` also constructs fractional coverage from
packed planar polygons; `parameters` is the `getGPURasterFractionalCoverageParameterValues(grid)` view. `sourceTotals.length` is the source-feature count. Feature rows and
target IDs remain stable, including empty sources and targets. Prepared coverage uses the
`GPURasterFractionalCoverage` output layout and retains its original status views.

## Measure and conservation

A valid source/cell tuple contributes
`fraction * calibratedAncillaryValue * abs(affineDeterminant)` to its source's eligible measure.
The target-zone raster assigns that measure to at most one target. An ID at or above
`targetZoneCount` means uncovered support. The recipe does **not** multiply independent source
and target fractions or claim arbitrary fractional target-polygon intersection.

The optional mask is an eligibility decision: zero excludes the cell. Ancillary zero also
excludes land. Native nodata and band validity are resolved before calibration. Missing,
nonfinite or negative calibrated ancillary values contribute no eligible measure; their
`intermediates.cellWeights` are NaN, while explicitly excluded cells have zero weight.
Ancillary weights describe relative density per square affine-coordinate unit; a binary
eligibility raster is a special case. Geographic/spherical cell-area weighting is not provided.

For a source with total `M`, eligible measure `A` and target-covered measure `C`:

| Denominator | Allocated source total | Unallocated source total |
| --- | --- | --- |
| `overlap` (default), `C > 0` | `M` | `0` |
| `zone`, `A > 0` | `M * C / A` | `M * (1 - C / A)` |
| No eligible target support | `0` | `M` |

Zero eligible measure is valid accounting, with `sourceValidity = 0` and the whole original
mass in `sourceUnallocatedTotals`. Its transferred fraction is NaN because the denominator
is undefined. Targets with no allocated source support have zero totals and support weights.
Source features can overlap: their masses remain independent. `targetSupportWeights` adds
source measures; it is not the geometric union area or a fraction bounded by one.

The ledger preserves the existing areal-interpolation float32 results. It does not clamp
fractions or negative residuals caused by roundoff. Verification uses absolute `1e-4` plus
relative `1e-5` tolerance against independently computed values, and the same tolerance scaled
by total source mass for conservation. A small negative residual within this tolerance is
numerical roundoff, not physical negative mass. This tested tolerance is not a universal error
bound for arbitrarily ill-conditioned workloads; callers should inspect their mass residuals.

## Outputs and evidence

- `targetTotals`, `targetSupportWeights`: one row per target.
- `sourceEligibleWeights`, `sourceTransferredFractions`, `sourceAllocatedTotals`,
  `sourceUnallocatedTotals`, `sourceValidity`: one row per original source.
- `status.complete`: execution evidence is complete and published arithmetic is finite.
- `status.zeroEligibleCount`: sources with zero eligible measure; these retain their totals.
- `status.invalidTupleCount`, `status.invalidSourceCount`, `status.arithmeticInvalidCount`:
  malformed row values, invalid extensive totals and arithmetic failures, respectively.
- `status.stages`: original coverage count/required-count/overflow, coverage validation,
  areal pair overflow/required count, source-value, arithmetic and conservation evidence.

Coverage failure, invalid source totals or tuple values, truncated pair output, and detected
float32 arithmetic overflow set `complete = 0` and invalidate published numeric outputs with
NaN. This includes a finite ancillary value times finite cell area overflowing float32, source
support sums overflowing, and target support or mass sums overflowing. `sourceValidity` becomes
zero. Coverage status and the caller's input buffers remain available for diagnosis.

`intermediates` retains the original grid, coverage, ancillary band, mask, source-total and
target-zone views, plus the derived cell weights, expanded tuple identities and weights, and
interpolation CSR. All output overrides must be distinct from one another and from inputs.

## Prepared partitions and bounds

Prepared coverage must contain **unique `(source, cell)` tuples**, with fractions in `[0, 1]`
and valid IDs. Uniqueness and source geometry topology are caller preconditions;
`status.complete` does not certify them and the recipe does not deduplicate malformed input.
Generated coverage supplies one row per source/cell. Halo cells must not be repeated.

Concatenate disjoint partition cores, remapping cell IDs to the common grid, **before**
normalization. A source can span several cores; independently normalized tile allocations
cannot be added. The tests reorder two disjoint core partitions and compare the result with
the complete-domain oracle. This is prepared-partition composition, not automatic tile IO or
cross-tile geometry stitching.

`targetZoneCount * tupleCapacity` is bounded by 4,194,304 for the explicit support summary.
Polygon coverage also retains its own 4,194,304 source/cell candidate limit. Weighted areal
interpolation's device storage and pair-count limits still apply. The grid determinant must
be finite, positive in magnitude and representable as nonzero float32. These are shape limits,
not performance claims.

## Reference evidence

The generator `modules/analysis/test/oracles/dasymetric-redistribution.py` records package
versions, units, conventions and tolerances in its fixture. Fractional-boundary and continuous
weight cases use a **composed oracle**: Shapely/GEOS source/cell intersections followed by an
independent weighted extensive-allocation calculation. Binary, aligned-cell cases additionally
call [Tobler's `masked_area_interpolate`](https://pysal.org/tobler/stable/generated/tobler.dasymetric.masked_area_interpolate.html)
on a temporary GeoTIFF, with `allocate_total` selected explicitly. They do not imply full Tobler
option parity. Tobler's raster extraction has different partial-boundary behavior, so partial
source-cell cases are labeled composed rather than direct Tobler parity.

Pinned direct dependencies are Tobler 0.14.0, Shapely 2.1.2, GeoPandas 1.1.1, Rasterio 1.4.3 and
NumPy 2.2.6; additional runtime versions are recorded in the generated JSON. GPU tests cover
both denominator modes, empty support, excluded and missing land, source polygon composition,
prepared partitions, changes to masks/values/target ownership, failure recovery, truncation,
skewed measures and arithmetic overflow. Node construction tests cover ownership and default
storage-binding limits.

## Executable example

{/* lens:addDasymetricRedistributionRecipe */}

The example retains the source mass ledger and the actual eligibility and completion captures.

::lens{scenario=gpu-spatial-analysis.dasymetric-redistribution-recipe rung=graph}
