# GPURasterFractionalCategoricalStatistics

`GPURasterFractionalCategoricalStatistics` summarizes categorical raster values over fractional
zone coverage. It consumes `GPURasterFractionalCoverage` tuples, groups them by zone and category,
and returns class coverage, planar class area, composition and majority/minority categories.
It does not count intersected cells or apply observation weights.

```typescript
import {GPURasterFractionalCategoricalStatistics} from '@luma-contributors/analysis/raster-analysis';

graph.add(new GPURasterFractionalCategoricalStatistics({
  grid,
  input: {
    id: 'land-cover',
    format: 'uint32',
    storage: {kind: 'buffer', values: categories},
    validity: categoryValidity,
    noDataValue: 0xffffffff
  },
  zoneCount,
  categoryCapacity: 6,
  coverage,
  output: {
    coverageSums,
    classAreas,
    proportions,
    coveredArea,
    validArea,
    missingArea,
    majority,
    minority,
    validity
  }
}));
```

All arguments representing data are graph views over caller-owned buffers. `coverage` contains
`zoneIds`, `cellIds`, `fractions`, `zoneValidity`, and `status: {count, requiredCount, overflow}`.
The active tuples must contain unique `(zone, cell)` pairs with fractions in `[0, 1]`. Unused
capacity after `status.count` is ignored. Duplicate pairs are not deduplicated: the producer's
unique-pair contract is required. Disjoint tile cores can be combined only after their cell IDs
are mapped into the common input grid; overlapping halo coverage must not be included twice.

## Category and output contract

Categories are uncalibrated `uint32` identifiers in `[0, categoryCapacity)`. Their numeric order
is the stable tie order. Noncontiguous external codes must be explicitly encoded into a dense
ordered dictionary before upload. Zero is a valid class. A validity mask of zero or the explicit
`noDataValue` marks missing support; a valid out-of-range code is an error. Calibration other than
`scale: 1` and `offset: 0` is rejected.

The three category columns have `zoneCount * categoryCapacity` rows, in zone-major order:

| Output | Meaning |
| --- | --- |
| `coverageSums` | Sum of covered cell fractions for a category; this is fractional support, not an integer count |
| `classAreas` | Category coverage multiplied by the absolute affine determinant |
| `proportions` | Category coverage divided by total valid category coverage in the zone |

The remaining columns have `zoneCount` rows:

| Output | Meaning |
| --- | --- |
| `coveredArea` | All represented coverage, including nodata cells |
| `validArea` | Represented area with valid category values |
| `missingArea` | Represented area whose categories are masked or nodata |
| `majority` | Present category with greatest fractional support; greatest category wins exact ties |
| `minority` | Present category with least fractional support; lowest category wins exact ties |
| `validity` | One when coverage is complete and valid category support is positive |

For complete zones, class areas sum to valid area, and valid plus missing area equals covered
area, within float32 rounding. Proportions sum to one when valid support is positive. Absent
categories have zero coverage and area and never win minority. Complete empty or fully missing
zones have zero class areas, NaN proportions, `0xffffffff` winners and zero validity. Their area
outputs still distinguish an empty zone from a zone containing only missing categories.

Overflow, inconsistent active/required counts, malformed active tuples, and covered unmasked
out-of-range category IDs invalidate every zone. Failed polygon triangulation invalidates its
zone. Float32 overflow of a zone's covered area also invalidates that zone. Invalid zones return
NaN in every floating output, `0xffffffff` winners and zero validity. This separates incomplete
or numerically unrepresentable results from explicit missing category area. Input and output
buffers must be distinct; input updates are read on every execution of the same compiled graph.

## Bounds and units

- `categoryCapacity`: 1 through 65534.
- `zoneCount`: positive, with `zoneCount * (categoryCapacity + 1) <= 65535`.
- Coverage tuple capacity: 1 through 1,048,576, including unused capacity.
- Cell area must be positive and representable in float32.

The extra group per zone stores missing support. Stable sorted segmented sums avoid a dense
zone/category/tuple scan. These are bounded experimental workloads, not measured throughput
claims. Category, coverage and area calculations use float32 support sums; tie decisions apply
to these sums. No tolerance merges nearly equal categories.

Areas are squared coordinate units from the affine determinant. By default that is the
determinant of `grid.affine`, fixed at compile time. Pass the optional `parameters` prop (the
per-frame georeference view given to `GPURasterFractionalCoverage`) to read the determinant on
every encoding instead, so `classAreas`, `coveredArea`, `validArea` and `missingArea` follow a
georeference repacked for another cell size. By default geographic degree grids do not
produce square meters; set `cellSizeMode: 'geographic'` (degrees) or `'web-mercator'` (EPSG:3857
meters) for WGS84 ground square meters. Each coverage fraction is then weighted by the ground area
per world area of its row (row edges from `grid.affine`, evaluated at the row center, so
`coverageSums` holds weighted coverage and `proportions` are area proportions). The row edges are
read from `parameters` (origin y and row step y, slots 1 and 5) on every encoding when it is given,
so the ground scale follows a repacked georeference together with the cell area; without
`parameters` they are baked from `grid.affine`. The grid must be axis-aligned. Nonzero packed
cross-axis terms suppress every dependent result for that frame (NaN areas/proportions and
validity zero), and an aligned later submission restores validity. The default `'uniform'` keeps planar coordinate units. Missing area describes nodata inside the supplied raster footprint; it
does not estimate a polygon's area outside that footprint. This API does not add observation
weights, spherical areas, category dictionary discovery, or implicit tile stitching.

## Reference evidence

The checked-in generator `modules/analysis/test/oracles/rs02-fractional-categories.py` pins
exactextract 0.3.0, NumPy 2.2.6 and Shapely 2.1.2. It calls exactextract `frac`, `majority`,
`minority`, `count`, and Cartesian area-weighted count directly. A separate all-valid raster
retains missing-cell coverage. Shapely constructs partial-cell multipart and hole geometries;
it does not compute the expected statistics. Fixtures retain versions and generator hash.

The majority/minority tie rules match the [pinned exactextract implementation](https://github.com/isciences/exactextract/blob/v0.3.0/src/raster_stats.h).
Coverage and observation-weight meanings follow the [official operations reference](https://isciences.github.io/exactextract/operations.html).
The tests compare every category, retain empty classes, verify area conservation, compose actual
GPU polygon coverage with the reducer, and update categories, masks and failure status on one
compiled graph. Tests are executable evidence only after the corresponding GPU run succeeds.

## Executable examples

These examples use explicit synthetic inputs and expose real GPU graph captures.

### GPURasterFractionalCategoricalStatistics

{/* lens:GPURasterFractionalCategoricalStatistics */}

::lens{scenario=gpu-raster.raster-fractional-categorical-statistics rung=graph}

The polygon-composition test compares floating outputs and input coverage tuples with an absolute
plus relative bound of `4 * 2^-23 + 4 * 2^-23 * abs(reference)`. Triangle clipping can introduce
nonbinary intersection ratios even for these dyadic polygon coordinates. The observed hole-zone
class area was `11.999998092651367` versus `12` (two float32 ULP). Direct tuple-fed comparisons
retain an absolute `5e-7` bound; category winners, tuple identities and validity flags remain exact.

Discrete majority/minority ties apply to **represented float32 coverage sums**, not ideal polygon
areas. Clipping roundoff can split an ideal geometric tie: in the updated hole-zone fixture,
exactextract chooses minority category 0 while the GPU composition chose category 3. The composed
test therefore validates each captured GPU coverage tuple against exactextract first, then computes
expected winners independently from those captured input weights and asserts exact category
equality. Each fixture zone/class has at most two contributing cells, allowing an independent
double sum followed by one float32 rounding without copying the GPU reduction tree. The test logs
the captured fractions and class supports when these winners differ from ideal geometry. This is
composition validation under represented weights; direct exactextract tie parity remains covered
by the separate reference-tuple test. No near-tie epsilon changes the API's tie rule.
