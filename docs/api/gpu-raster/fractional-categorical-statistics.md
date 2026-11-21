# GPURasterFractionalCategoricalStatistics

`GPURasterFractionalCategoricalStatistics` summarizes categorical raster values over fractional
zone coverage. It consumes `GPURasterFractionalCoverage` tuples, groups them by zone and category,
and returns class coverage, planar class area, composition and majority/minority categories.
It does not count intersected cells or apply observation weights.

```typescript
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPURasterFractionalCategoricalStatisticsProps} from '@luma-contributors/analysis/raster-analysis';
import {GPURasterFractionalCategoricalStatistics} from '@luma-contributors/analysis/raster-analysis';

// Borrow inputs from the earlier coverage producer in this graph.
declare const grid: GPURasterMetadata;
declare const zoneCount: number;
declare const coverage: GPURasterFractionalCategoricalStatisticsProps['coverage'];
// categories and categoryValidity each cover width * height cells.
declare const categories: GraphDataView<'uint32'>;
declare const categoryValidity: GraphDataView<'uint32'>;
// The three class columns hold zoneCount * 6 rows; other outputs hold zoneCount rows.
declare const coverageSums: GraphDataView<'float32'>;
declare const classAreas: GraphDataView<'float32'>;
declare const proportions: GraphDataView<'float32'>;
declare const coveredArea: GraphDataView<'float32'>;
declare const validArea: GraphDataView<'float32'>;
declare const missingArea: GraphDataView<'float32'>;
declare const majority: GraphDataView<'uint32'>;
declare const minority: GraphDataView<'uint32'>;
declare const validity: GraphDataView<'uint32'>;
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

The input accepts a native uint32 buffer band or sampled uint32 texture band with a selected
channel. Texture gathering uses public `GPURasterTextureToBuffer` with calibration disabled,
preserving every uint32 ID and native nodata sentinel, including values above 2²⁴. Masks and
texture contents are read on each submission. Category capacity remains at most 65,534; larger
external codes require a dense dictionary, while nodata may use any uint32 value.
Other data arguments are graph views over caller-owned buffers. `coverage` contains
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

| Route | Logical groups `zoneCount * (categoryCapacity + 1)` | Coverage tuple capacity | Sort tile size |
| --- | --- | --- | --- |
| Default `mode: 'tiled'` | At most 4,294,967,294 (`2^32 - 2`) | 1 through 4,294,967,294 | `tupleTileSize`: 1 through 1,048,576 |
| Reference `mode: 'exact'` | At most 65,535 | 1 through 1,048,576 | One full sort; an explicit tile size must cover the tuple capacity |

Both routes require `categoryCapacity` from 1 through 65,534 and positive `zoneCount`.
The extra group per zone stores missing support. Tiling bounds one sort allocation, while
caller-owned output bindings and dense group scratch must each fit the device. In particular,
category columns have `zoneCount * categoryCapacity` float32 rows, and segment offsets need
`4 * (groups + 1)` bytes. Each backing buffer must also fit `maxBufferSize`.

The command estimate is `8 + 256 * ceil(tupleCapacity / tupleTileSize) <= 32768`, hence at most
127 tiles per declaration. These are hard logical and allocation bounds, not throughput claims.
Use the side-effect-free planner before graph declaration, with the same props and device limits:

```typescript
import type {Device} from '@luma.gl/core';
import {
  GPURasterFractionalCategoricalStatistics,
  type GPURasterFractionalCategoricalStatisticsProps
} from '@luma-contributors/analysis/raster-analysis';

function planFractionalCategories(props: GPURasterFractionalCategoricalStatisticsProps, device: Device) {
  const plan = GPURasterFractionalCategoricalStatistics.getExecutionPlan(props, device.limits);
  if (!plan.feasible) {
    console.log(plan.constraints.filter(constraint => !constraint.satisfied));
  }
  // plan.route, plan.tupleTileSize, plan.tiles, plan.estimatedCommandNodeCount
  return plan;
}
```

The planner uses the constructor's side-effect-free port, option, format, length and alias checks,
and shares selection, binding/scratch expressions and per-stage device checks with declaration. It declares no graph resources. Invalid option spellings and invalid
explicit tile sizes throw; supported shapes report fit through `feasible` and named constraints.
Exact mode preserves the historical full-sort reduction order. Tiled mode uses the same planar
coverage, merging support in tile order before normalization; changing the tile size can change
float32 sum rounding and near-tie outcomes. No tolerance merges nearly equal categories.
Cell area must be positive and representable in float32.

Public coverage and patch-metrics packets retain the signed physical determinant of the original
affine in word 6. Clipping inverts the packed axes independently. The analysis-core resolver's
coordinate packet instead carries its packed-axes determinant and exposes `affineDeterminant`
separately.

Areas are squared coordinate units from the affine determinant. By default that is the
determinant of `grid.affine`, fixed at compile time. Pass the optional `parameters` prop (the
per-frame georeference view given to `GPURasterFractionalCoverage`) to read the determinant on
every encoding instead, so `classAreas`, `coveredArea`, `validArea` and `missingArea` follow a
georeference repacked for another cell size. By default geographic degree grids do not
produce square meters; set `cellSizeMode: 'geographic'` (degrees) or `'web-mercator'` (EPSG:3857
meters) for WGS84 ground square meters. Each coverage fraction is then weighted by the ground area
per world area of its row using the grid-derived WGS84 midpoint model (the product of
the two local ground scales at the row midpoint, rather than an integrated geodesic cell area, so
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


For large absolute coordinates, pack coverage with
`getGPURasterFractionalCoverageParameterValues(grid, {localOrigin: [originX, originY]})`
and subtract that origin from the original binary64 polygon vertices before float32 upload.
Pass this same georeference as `parameters`. This overload returns ten float32 words: the legacy
first eight words (word 7 remains ignored padding), followed by layout discriminator `2` and the
absolute Y offset in words 8 and 9. The explicit layout restores absolute Y for geographic and
Web Mercator row-area weighting. The historical `(grid, target?)` overload still writes eight words,
preserves destination identity across JavaScript realms, and ignores nonzero legacy padding.
A reused destination for the options overload must have at least ten words.
