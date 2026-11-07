# Kepler analysis bindings

These adapters append existing luma contributors to a caller-owned command graph. They do not
submit work, read GPU results, create a Kepler dataset, or silently fall back to another backend.
The host owns result publication and uses the dataset registry to reject stale completions.

`addKeplerAnalysisRecipe(graph, recipe, bindings)` resolves a serialized recipe against explicitly
provided GPU columns. Saved operations include standardization, classification, excess risk, spatial
rates, hot spots, Local Moran, Quantile LISA, interpolation, DBSCAN, OLS and bounded Dorling
cartograms. Named helpers expose contributor outputs, inference metadata and compact GPU
diagnostics. Each operation records its own numerical controls and topology policy.

```ts
const result = addKeplerAnalysisRecipe(graph, {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'income-z-score',
  source: {datasetId: 'tracts', revision: 1},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'standardize',
    column: 'income',
    method: 'z-score',
    variance: 'sample'
  }
}, {
  source: {datasetId: 'tracts', revision: 1},
  columns: {income: incomeView},
  validityMask,
  outputs: {values: resultView}
});
```

All bound numeric columns are explicitly Float32. This adapter does not convert Int64, Float64 or
opaque category IDs. Standardization preserves input chunks when allocating its output. Rate and
classification contributors require a packed view; callers must request packing explicitly before
using multi-chunk inputs. The router reports this limitation instead of concatenating batches.

`population: 'all'` uses `validityMask`, ignoring the display selection. A selected recipe requires a
saved selection revision and a `mask` already intersected with input validity. Externally supplied spatial
weights require both a matching weights revision and the dataset revision to which they belong.

## Point weights and projected coordinates

`spatial-rate-points` records `{events, populations, weights, scale, islandPolicy}`. Rates, point hot
spots, Local Moran and Quantile LISA share `KeplerPointWeightsSettings`. Choose one coordinate
contract explicitly:

```ts
const geographic = {
  longitudeColumn: 'longitude', latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'
} as const;

const projected = {
  xColumn: 'easting', yColumn: 'northing',
  coordinateSpace: 'planar', metric: 'native', units: 'meters',
  crs: 'EPSG:32618', bounds: [300000, 4400000, 700000, 4800000]
} as const;

const nearest = {kind: 'knn', ...projected, k: 8} as const;
const distanceBand = {
  kind: 'distance-band', ...geographic, radiusMeters: 4000, neighborCapacity: 65536
} as const;
```

Geographic columns contain Float32 degrees inside longitude ±180° and latitude ±90°; distances are
spherical meters. Projected columns already contain Float32 XY meters in the declared CRS;
`metric: 'native'` means Euclidean distance in that plane. The adapter never reprojects or verifies
that the declared CRS matches source values. Bounds are inclusive and source-wide, not a viewport.
`parseKeplerPointWeightsSettings` normalizes bounds/radius to Float32 and rejects unsafe squared
distances. Large coordinate origins can erase small separations; declaring meters does not remove
Float32 precision loss. Analysis XY roles do not replace geographic geometry used by a map.

`k` is 1–32. Distance bands require a positive radius and an explicit total CSR slot capacity of
1–1,048,576; their source limit is 65,536 rows. Read `status.overflow` and `status.requiredCount`
before publishing. Insufficient capacity produces an incomplete topology, not a valid truncated
neighborhood. Permutation consumers separately cap active row degree at 64 and report inference
overflow; they do not silently discard excess neighbors.

`addKeplerPointWeights(graph, props)` builds a directed binary, no-self graph across all valid source
coordinates. For geographic settings pass `{longitude, latitude}` views; for projected settings pass
`{x, y}` views. Both forms take `source`, `settings`, optional `coordinateValidityMask`, and optional
caller-owned CSR/status/coordinate-mask outputs. Columns must be explicitly packed Float32 views.
The returned `weights`, `coordinateMask`, immutable settings and source identity can be reused by
other graph consumers. Compact status counts source nulls, nonfinite coordinates and coordinates
outside the declared bounds. Numeric analysis validity and selection never change this topology:
filtering observations removes existing neighbors without finding replacements. Invalid coordinates
produce undefined rates; valid isolated rows follow the declared island policy.

To reuse completed weights, import their buffers into the consumer graph and supply
`bindings.pointWeights: {source, settings, weights, coordinateMask, status}`. Validation checks source
revision, roles, metric/CRS/bounds, neighborhood settings, row counts, binary/no-self provenance and
graph ownership. The caller encodes topology before its first consumer and retains it through all
consumers. Supplied buffers are borrowed. Generated topology owns a 48-byte parameter buffer; call
`destroyParameters()` after dependent work completes. Borrowed topology adds no parameter allocation.

`hot-spots-points` records `{column, weights, minimumNeighbors, selfWeight, multipleTesting, islands,
inference: 'analytic-two-sided'}`. It uses `GPUHotSpotAnalysis` directly on the original observations,
with the same fixed source-wide point topology as rates. The selected population removes
observations from that graph without replacing neighbors. `islands` explicitly selects `'exclude'`
or `'keep'` for global moments; isolated rows never receive a score. Results retain contributor
inference metadata, source-aligned z-scores, unadjusted p-values, corrected significance bins and
eligible-neighbor counts. Compact status distinguishes invalid coordinates from included finite
observations with undefined statistics. No point-to-cell aggregation or grid topology is inferred.

The example host caches completed point topology across supported point analyses by source revision and
the full coordinate descriptor. The first exact search can approach quadratic work on densely
clustered inputs; this cache avoids rebuilding for population or inference changes but does not
change the initial search's complexity.

## Bounded ordinary kriging

`addKeplerPointKriging` wraps `GPUKriging` for local ordinary kriging at north-up raster cell centers.
The saved operation is `kriging-points`, with an explicit value `column` and geographic-only
`KeplerKrigingSettings`. Coordinates are Float32 longitude/latitude degrees; search radius and
variogram range are spherical meters. Supply the variogram explicitly—this adapter does not fit it,
perform cross-validation, model observation error, or construct confidence intervals.

```ts
const result = addKeplerAnalysisRecipe(graph, {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'temperature-kriging',
  source: {datasetId: 'stations', revision: 1},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'kriging-points',
    column: 'temperature',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    grid: {bounds: [-74.2, 40.5, -73.7, 41], width: 64, height: 64},
    searchRadiusMeters: 20000,
    neighborCount: 12,
    minimumNeighbors: 3,
    variogram: {model: 'spherical', nugget: 0.1, sill: 10, rangeMeters: 15000}
  }
}, {
  source: {datasetId: 'stations', revision: 1},
  columns: {longitude: longitudeView, latitude: latitudeView, temperature: temperatureView},
  validityMask: numericValidity,
  coordinateValidityMask,
  outputs: {kriging: {values: predictionOutput, variance: varianceOutput, validity: cellValidity}}
});
```

The source limit is 4,096 rows. The output is at most 16,384 cells, with each dimension 1–256.
`grid.bounds` is `[west, south, east, north]`; row zero is the north edge and columns advance east.
Cell centers are prediction locations, not area-averaged block supports. The finite positive search
radius expands the admitted grid: this expanded domain must not reach the poles or cross the
antimeridian. The same conservative geographic admission applies to IDW. Selection changes the
contributing samples while preserving the grid placement. Null, nonfinite and out-of-range source
coordinates and nonfinite values are excluded on the GPU without another source upload.

`neighborCount` is 1–16, and `minimumNeighbors` is 1–`neighborCount`. Models are `spherical`,
`exponential` or `gaussian`, with a positive effective `rangeMeters`. `sill` is the partial sill;
`nugget` and `sill` must be nonnegative finite Float32 values with a finite positive total. Both are
in squared source-value units. The nugget represents microscale variation, not measurement error.
Positive parameters must remain representable after Float32 normalization.

The core retains nearest source rows in deterministic distance/source-row order. Duplicate positions
within that retained neighborhood are averaged, but duplicates consume slots before coalescing;
`minimumNeighbors` counts distinct positions. An exact hit returns the retained duplicate mean and
variance zero even below the minimum count. Singular or poorly conditioned solves produce no data;
only tiny negative variance from rounding is clamped. These are the contributor's numerical policies,
not a GeoDa equivalence or a precision guarantee for tightly clustered geographic coordinates.

| Output | Meaning |
| --- | --- |
| `values` | Float32 prediction per raster cell, in source-value units |
| `variance` | Float32 ordinary kriging variance per cell, in squared source-value units |
| `validity` | Uint32 cell mask requiring both finite prediction and finite nonnegative variance |
| `status.validSourceCount` | Included finite sample rows under the declared population |
| `status.validCellCount` | Cells passing the final prediction/variance gate |
| `status.noDataCellCount` | All cells failing that final gate |
| `status.invalidCount` | Core cells rejected for invalid settings or insufficient distinct neighbors |
| `status.uncertainCount` | Core cells rejected for an unreliable solve or variance |

The wrapper writes NaN to both prediction and variance when the final gate fails. Final arithmetic
overflow can add no-data cells beyond the core categories; do not derive `noDataCellCount` by adding
`invalidCount` and `uncertainCount`. The contributor does not expose retained-neighbor counts or
per-cell reason codes, so the adapter does not fabricate an IDW-style samples column. Variance is
model-based prediction uncertainty, not observed error or an automatically calibrated interval.

Direct callers provide packed `longitude`, `latitude` and `values` views, numeric/population `mask`
and optional `coordinateValidityMask`. The result includes grid metadata, source/selection
provenance, source `effectiveMask`, outputs and compact status. Recipe output `values` falls back to
`bindings.outputs.values`; every prediction, variance and validity row remains a raster cell, not a
source observation. Use raster rendering, bounded cell inspection or a correctly georeferenced
single-band export rather than source-row colors or tables. Prediction and variance exports must
identify which metric and units they contain.

The adapter owns a 48-byte parameter buffer. Retain it, borrowed source buffers and supplied output
buffers until submitted work completes, then call `destroyParameters()`. Omitted outputs and scratch
belong to the compiled graph. No CPU solve or hidden GPU readback is performed by the shared helper.

## Bounded Dorling cartograms

`addKeplerDorlingCartogram` wraps `GPUDorlingCartogram` on original source rows. It accepts only the
explicit projected-meter contract above, with packed `x`, `y` and `values` views. The saved operation
spreads `KeplerDorlingSettings` under `kind: 'dorling-cartogram'`:

```ts
const result = addKeplerAnalysisRecipe(graph, {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'population-circles',
  source: {datasetId: 'districts', revision: 1},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'dorling-cartogram',
    ...projected,
    valueColumn: 'population',
    areaScale: 100,          // square projected meters per value unit
    maximumIterations: 100, // 0..256; zero diagnoses the initial layout
    overlapTolerance: 1,    // meters of residual penetration
    attraction: 0.01        // pull toward original centers per pass, 0..1
  }
}, {
  source: {datasetId: 'districts', revision: 1},
  columns: {easting: xView, northing: yView, population: populationView},
  validityMask: combinedSourceValidity,
  outputs: {dorling: {positions: positionOutput, radii: radiusOutput}}
});
```

The source limit is 1,024 rows, with at most 256 synchronous all-pairs relaxation passes. Radius is
`sqrt(value * areaScale / pi)`. Zero values remain stationary zero-radius points; negative, missing,
nonfinite, out-of-bounds and unselected rows emit NaN positions/radii while retaining source IDs.
The core also excludes input coordinate magnitudes and radii above 1e15. Source uploads and row order
remain unchanged. Selection recomputes the layout, rather than merely hiding circles afterward.

Direct helper inputs separate source `validityMask` from selected-population `mask`; the all-source
population ignores `mask`. Recipe bindings combine numeric and coordinate validity. Direct callers
can provide `effectiveMask` storage; recipe callers use `outputs.dorling.effectiveMask`. The mask
reports admitted source rows. Geometry remains a separate output rather than a source color override.

| Output | Format and meaning |
| --- | --- |
| `positions` | Float32 XY pair per original row, in declared projected meters |
| `radii` | Float32 projected radius per row; falls back to generic `outputs.values` in recipe bindings |
| `sourceIds` | Uint32 original row index, including excluded rows |
| `metrics` | One Float32 pair: maximum residual penetration, maximum displacement from original center |
| `status` | Four Uint32 words: iterations used, converged, budget exhausted, excluded row count |

Exhausting the budget does not invalidate finite output geometry. Publish its residual overlap and
exhausted status; stationary overlapping circles are not convergence. Settings are static graph
constants, so `parameterByteLength` is zero and `destroyParameters()` is a no-op. Omitted outputs and
scratch belong to the compiled graph. `estimateKeplerDorlingByteLength(rowCount)` conservatively
budgets GPU output/scratch, excluding borrowed sources, CPU graph metadata and driver overhead.

The shared adapter accepts arbitrary declared projected-meter CRSs. The example host exposes those
results in its table, but enables a map overlay only for declared `EPSG:3857` with a compatible Web
Mercator viewport. That host conversion keeps positions and radii in the projected plane; projected
meters are not ground meters. Other CRS layouts must not be fed to a longitude/latitude layer or
silently reinterpreted as Web Mercator. No CRS conversion, GeoDa coordinate parity or equal-area map
claim is implied by this adapter.

## Output ownership and execution

Caller-owned output views are required for readback and for outputs used outside the command graph.
When outputs are omitted, graph-owned transient views are suitable for downstream nodes in the same
graph. Call `destroyParameters()` only after the compiled graph and all dependent work have finished;
classification, hot spots and point weights retain parameter buffers. Bindings without parameters
return a no-op method with the same name.

`serializeKeplerAnalysisRecipe` and `restoreKeplerAnalysisRecipe` store no GPU resources. Restoration
requires matching dataset, selection and externally supplied weights revisions. Point-weight recipes
reconstruct weights from their source and saved settings. Unknown versions/properties are rejected;
restoring a reference does not reconstruct a missing selection or externally supplied weights dataset.

The execution router sends arbitrary SQL to a supplied DuckDB executor and statistical operations to
luma or an explicitly supplied analysis backend. It does not equate DuckDB with GeoDa. A failed GPU
execution is never retried silently. Cancelled completions are passed to `discardResult` so allocated
outputs can be released. `createKeplerDuckDBExecutor` wraps a caller-owned DuckDB connection, exposes
explicit Arrow registration and reports CPU Arrow results; GPU uploads/readbacks remain separate.
