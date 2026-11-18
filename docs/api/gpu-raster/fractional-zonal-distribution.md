# GPURasterFractionalZonalDistribution

Computes coverage-weighted quantiles, population and reliability sample variance, and effective
sample support from the `(zoneId, cellId, fraction)` output of `GPURasterFractionalCoverage`.
The value band contains intensive cell observations. Statistical weights are coverage fractions times optional independent per-cell observation weights. Optional per-cell areas remain separate from those weights; there is no implicit conversion of cell totals into densities.

This is RS01's bounded planar distribution contract. All computation, sorting and grouping stays
inside the command graph. Construction does not submit GPU work or read data back to the CPU.
The input buffers and all outputs belong to the caller.

## Inputs and shapes

| Property | Contract |
| --- | --- |
| `grid` | Raster dimensions and an invertible planar affine transform; area is the absolute affine determinant. The determinant and the maximum covered area must be representable in float32. |
| `input` | A buffer raster band. Existing band canonicalization applies calibration, masks, nodata and finite-value filtering. |
| `zoneCount` | Dense source-zone rows, from 0 through 65,536. Empty zones retain their row. |
| `coverage` | The complete existing fractional-coverage output object: `zoneIds`, `cellIds`, `fractions`, `zoneValidity`, and `status.count`, `status.requiredCount`, `status.overflow`. Tuple capacity is 1 through 65,536; the active count may be zero. |
| `probabilities` | A float32 view of 1 through 32 probabilities. Values may change between encodings. Each must be finite and in `[0,1]`; invalid probabilities yield NaN and zero quantile validity. |
| `observationWeights` | Optional packed float32 cell column. Finite nonnegative reliability weights; zero excludes a cell from distribution statistics but retains its valid area. Negative/nonfinite weights invalidate the zone. Default one. |
| `cellAreas` | Optional packed float32 cell column. Positive finite areas supplied by the caller, including geographic/geodesic areas. Default absolute planar affine determinant, or ground area with `cellSizeMode`. Areas never silently become statistical weights. |
| `cellSizeMode` | `'uniform'` (default, planar coordinate units), `'geographic'` (degrees) or `'web-mercator'` (EPSG:3857 meters). The latter two multiply the default cell area by the WGS84 ground area per world area of the cell's row (row edges evaluated at the row center), so `coveredArea` and `validArea` are ground square meters. The row edges are read from `parameters` (slots 1 and 5) on every encoding when it is given, else baked from `grid.affine`. Needs an axis-aligned grid; rejected together with `cellAreas`. The mode is compile-time. |
| `parameters` | Optional. The per-frame georeference view given to `GPURasterFractionalCoverage` (`getGPURasterFractionalCoverageParameterValues`). Without `cellAreas`, the cell area is then read from it on every encoding as the absolute determinant, so the area outputs follow a georeference repacked for another cell size. With neither `cellAreas` nor `parameters` the cell area is the determinant of `grid.affine` fixed at compile time. |
| `quantileMethod` | `'exactextract'` (default) distinct-value interpolation, or `'inverse-cdf'` without interpolation. Both sort the exact observed float32 values; neither bins a continuous range. |
| `output` | Caller-owned, nonaliasing buffers described below. |

Tuple buffers have matching capacities and only their active prefix is read. Fractions must be
finite and in `[0,1]`; zero contributes no observation. A `(zoneId, cellId)` pair must occur once.
Duplicates are rejected, including a duplicate carrying zero weight. Negative/nonfinite observation weights, nonpositive/nonfinite cell areas, or overflowing combined weights also invalidate the zone. Different zones may refer to
the same cell. Invalid source-cell IDs invalidate their zone; an out-of-range zone ID invalidates
all zones because its intended destination cannot be recovered.

The input band, fractions, IDs, probabilities, counts, validity and overflow words are read again on
every encoding. There is no prepared cache to invalidate. Sizes, calibration and the grid affine
are construction-time settings, except the cell area when `parameters` is supplied (see above):
pass it whenever the coverage georeference is rewritten between encodings.
Prepared coverage may also supply `status.candidateOverflow`; both operation forms preserve this
optional input and suppress dependent results when it is nonzero. `requiredCount` must equal
`count`, which must fit the tuple capacity. Coverage status inputs (`count`, `requiredCount`,
`overflow` and optional `candidateOverflow`) must each contain at least one word. Only the first
logical word is read, including for offset views; spare words do not affect completeness.
Ground modes check both packed cross-axis words on every run, suppressing all affected results
when nonzero; a later aligned frame restores validity.

## Outputs and definitions

For valid cells of one zone, let `w` be coverage fraction times observation weight, `W = sum(w)`, `W2 = sum(w*w)`,
`mean = sum(w*x)/W`, and `M2 = sum(w*(x-mean)^2)`.

| Output | Shape | Meaning |
| --- | --- | --- |
| `means` | `zoneCount` float32 | Coverage-weighted mean. |
| `populationVariances` | `zoneCount` float32 | `M2/W`; zero for singleton or constant observed values. |
| `sampleVariances` | `zoneCount` float32 | Reliability-weighted correction `M2/(W-W2/W)`. NaN with fewer than two observed source cells or a nonpositive denominator. Fractions are not frequency counts. |
| `effectiveSampleSizes` | `zoneCount` float32 | Kish effective support `W*W/W2`; zero without observations. This measures weight concentration, not independent spatial information or inferential degrees of freedom. |
| `weightSums`, `squaredWeightSums` | `zoneCount` float32 each | `W` and `W2`, computed from individual source cells before equal values are grouped. |
| `coveredArea`, `validArea` | `zoneCount` float32 each | Sum of fraction times the corresponding cell area, over all accepted cells or only finite unmasked values. Zero observation weight does not remove valid area. Nodata remains visible as missing support. |
| `quantiles` | `zoneCount * probabilities.length` float32 | Zone-major quantiles: `zone * probabilities.length + probabilityIndex`. |
| `quantileValidity` | Same shape, uint32 | One for a valid finite quantile, otherwise zero. |
| `validity` | `zoneCount` uint32 | One for a complete zone with supported finite mean/population results. Sample variance additionally needs sufficient sample support. |
| `status` | `zoneCount` uint32 | Bit mask below; overwritten on every encoding. |

A complete empty/all-nodata zone has zero support and area sums as appropriate, NaN distribution
statistics and zero validity. Incomplete or malformed coverage yields NaN numeric outputs and zero
validity, so a truncated set of tuples cannot appear to be a complete distribution.

### Quantile convention

By default quantiles match the **exactextract 0.3.0 distinct-value histogram convention**. Equal finite values
are grouped and their combined coverage/observation weights added. For `m` sorted distinct values, group weight `g[j]`,
and cumulative weight strictly before group `j`, plotting positions are
`p[j] = j*g[j] + (m-1)*cumulativeWeightBefore[j]`. The requested position is `q*(m-1)*W`.
Linear interpolation occurs between adjacent plotting positions. A singleton distinct value is
returned for every valid probability; endpoints return the observed minimum and maximum.

This convention differs from an inverse empirical CDF and from applying unweighted type-7
quantiles to replicated cells. Grouping equal values affects quantiles, but does **not** collapse
source cells when calculating `W2`, sample variance or effective support. The reference generator
calls the external library directly rather than reimplementing its quantile algorithm.

`quantileMethod: 'inverse-cdf'` selects the first distinct value whose cumulative positive weight is at least `q*W`; `q=0` selects the minimum. This matches NumPy 2.2.6 `quantile(method='inverted_cdf', weights=...)`. Zero-weight observations are absent from either distribution. Direct exactextract `weighted_mean` / `weighted_variance`, NumPy weighted quantiles and `cov(aweights=...)` are executed by `raster-round-two.py`.

### Status bits

`GPU_RASTER_FRACTIONAL_ZONAL_DISTRIBUTION_STATUS` exports these flags:

| Bit | Name | Meaning |
| --- | --- | --- |
| 1 | `INCOMPLETE_COVERAGE` | Upstream tuple or candidate overflow, inconsistent active/required count, failed source-zone validity, or a non-axis-aligned effective ground-area frame. |
| 2 | `INVALID_TUPLE` | Invalid fraction, cell or zone ID, or duplicate cell within a zone. |
| 4 | `NO_VALID_SUPPORT` | No positive-weight finite observations. |
| 8 | `INSUFFICIENT_SAMPLE_SUPPORT` | Fewer than two observed source cells or a nonpositive sample-variance denominator. A valid singleton still has valid population results. |
| 16 | `NON_FINITE_RESULT` | Float32 arithmetic exceeded finite moment/support range. |

Invalid probabilities are represented by `quantileValidity`, independently of valid moment outputs.

## Example

The output buffers are allocated by the application. `coverage` is the output of an earlier
`GPURasterFractionalCoverage` producer in the same graph.

```typescript
import {GPURasterFractionalZonalDistribution} from '@luma-contributors/analysis/raster-analysis';

// probabilities contains [0.25, 0.5, 0.75]. Quantile outputs need zoneCount * 3 rows.
graph.add(new GPURasterFractionalZonalDistribution({
  grid,
  input: observationBand,
  zoneCount,
  coverage,
  probabilities,
  output: {
    means,
    populationVariances,
    sampleVariances,
    effectiveSampleSizes,
    weightSums,
    squaredWeightSums,
    coveredArea,
    validArea,
    quantiles,
    quantileValidity,
    validity,
    status
  }
}));
```

## Partition and numerical limits

For tiled inputs, prepare an explicit tuple union with disjoint cell cores and cell IDs rebased to
the explicitly assembled value band. Exclude halos. This operation does not merge source chunks
or tile summaries. Never average tile quantiles: the combined weighted histogram is required.
Repeated source cells are rejected to protect support and area accounting.

GPU radix sorting produces deterministic `(zone,value,cell)` order; moments use a centered,
two-pass calculation to avoid subtracting two large raw second moments. A separate GPU pass
materializes centered observations before reduction, preventing backend arithmetic reassociation
from restoring a large baseline inside the residual calculation. Reductions are float32,
so extreme dynamic ranges can lose small contributions. A nonpositive rounded sample denominator
is reported as insufficient support, and non-finite moment results set bit 16.

The tuple/zone/probability bounds are implementation limits, not a measured throughput promise.
Per-zone moment scans stripe sums across 256 local partials and reduce a fixed tree to limit float32 accumulation drift; the scan itself is serial. Each quantile scans its zone's exact distinct-value histogram. A large skewed
zone with 32 quantiles can dominate execution. Every pass respects the default eight-storage-buffer
limit; sort scratch is graph-owned and proportional to tuple capacity. There is no claim of a
browser speed advantage or production-scale envelope until dedicated GPU measurements are recorded.

## Reference and verification

The pinned generator is
`modules/analysis/test/oracles/fractional-zonal-distribution.py`, using Python 3.12,
exactextract 0.3.0, NumPy 2.3.4 and Shapely 2.1.2. Direct `exact_extract` calls provide quantiles,
mean, population variance and coverage count. `numpy.cov(aweights=coverage, ddof=1)` independently
provides sample variance; sum/dot composition provides effective support. Geometry fixtures use
exactly representable planar strips and a partial polygon with a hole.

Tests cover repeated values, unequal weights, singleton/empty/all-nodata zones, tiny positive
support, large offsets with small variance, malformed/duplicate tuples, capacity exhaustion,
source triangulation failure, per-frame updates, overlapping zones and explicit disjoint tile
composition. Node kernel audits and numerical GPU parity are separate checks.

References: [exactextract operations](https://isciences.github.io/exactextract/operations.html),
[version-pinned histogram integration](https://github.com/isciences/exactextract/blob/v0.3.0/src/raster_stats.h),
[version-pinned quantile implementation](https://github.com/isciences/exactextract/blob/v0.3.0/src/weighted_quantiles.cpp),
[exactextract 0.3.0](https://pypi.org/project/exactextract/0.3.0/).

## Executable examples

These examples use explicit synthetic inputs and expose real GPU graph captures.

### GPURasterFractionalZonalDistribution

{/* lens:GPURasterFractionalZonalDistribution */}

::lens{scenario=gpu-raster.raster-fractional-zonal-distribution rung=graph}
