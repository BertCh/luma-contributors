# Units, coordinates, time and no-data conventions

## Summary

I counted 498 public GPU producers. The library already has good contracts for spatial frames, time encodings, raster affines and validity. The largest risks occur when a caller carries data from one of those contracts into another.

- A point-cloud terrain raster passed to hydrology can acquire both a false elevation of zero and a north–south displacement. Matching dimensions, row order and TypeScript types do not catch either error.
- At a UTM northing of 4,600,000 m, the documented fractional-coverage upload can turn a half-cell strip into a full-cell strip before the shader starts. Putting the grid origin beside the polygon is insufficient.
- Spatial validation accepts a declared degree or survey-foot frame labelled `meters`. The compatibility helper also accepts metric results whose numerical units differ by 1,000.
- The most valuable pre-alpha change is to make the existing descriptors travel with prepared data. Start with a point-cloud-to-raster adapter that supplies the affine and the correct validity plane, and a spatial resolver that checks declared frame units.

I ran host packers, CPU twins, graph-declaration probes and a lens comparison model. Device effects below are marked `SOURCE` and have a proposed confirming probe.

## Census

`S/` means `modules/analysis/src/`, `T/` means `modules/analysis/test/`, and `D/` means `docs/api/`. `KI` means the newer copy in `/Users/robertchristie/Documents/GitHub/luma-contributors-ki`. Counts are from the review checkout.

I enumerated actual exports through the TypeScript API. I deduplicated classes re-exported by multiple barrels, included inherited `getCommandNodes`, and counted properties on the resolved first constructor argument. The census core was:

```js
const program = ts.createProgram(familyIndexFiles, {
  noEmit: true, target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.ESNext, moduleResolution: ts.ModuleResolutionKind.Bundler,
  skipLibCheck: true
});
const checker = program.getTypeChecker();
for (const file of familyIndexFiles) {
  const entry = program.getSourceFile(file);
  for (const exported of checker.getExportsOfModule(checker.getSymbolAtLocation(entry))) {
    const symbol = exported.flags & ts.SymbolFlags.Alias
      ? checker.getAliasedSymbol(exported) : exported;
    const decl = symbol.valueDeclaration;
    if (!decl || !ts.isClassDeclaration(decl) || !decl.name?.text.startsWith('GPU')) continue;
    if (!checker.getTypeAtLocation(decl.name).getProperty('getCommandNodes')) continue;
    const signature = checker.getTypeOfSymbolAtLocation(symbol, decl).getConstructSignatures()[0];
    const arg = signature?.getParameters()[0];
    const names = arg ? checker.getTypeOfSymbolAtLocation(arg, decl).getProperties().map(p => p.name) : [];
    // Keep local declarations once by source filename and class name. Tally names by source family.
  }
}
```

`familyIndexFiles` was the list of `modules/analysis/src/*/index.ts`. `Time port` counts `times` or `timestamps`. The last four columns count the literal top-level prop names. A nested band validity, `timeWindow`, date-axis descriptor or packed setting is outside those columns.

| Family | Producers | `spatialContext` | Time port | `temporalContext` | `dateAxis` | `grid` | `mask` | `noDataValue` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| analysis-core | 6 | 3 | 0 | 0 | 0 | 0 | 2 | 0 |
| computation-record | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dataframe-analysis | 34 | 11 | 9 | 5 | 0 | 0 | 27 | 0 |
| geospatial-analysis | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| gpu-network | 41 | 9 | 0 | 0 | 0 | 0 | 1 | 0 |
| gpu-spatial-analysis | 270 | 146 | 32 | 29 | 0 | 11 | 62 | 1 |
| gpu-terrain | 46 | 0 | 0 | 0 | 0 | 44 | 0 | 0 |
| pointcloud-analysis | 11 | 1 | 0 | 0 | 0 | 7 | 0 | 0 |
| raster-analysis | 86 | 5 | 0 | 0 | 10 | 40 | 48 | 11 |
| testing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **498** | **175** | **41** | **34** | **10** | **102** | **140** | **12** |

Three distinctions matter when interpreting those counts.

| Convention measured | Count and examples |
|---|---|
| Coordinate storage | 148 producers expose at least one of `positions`, `points`, `origins`, `queryPositions`. In the printed types of those ports, 121 contain literal `float32x2` and 10 contain literal `float32x3`. Aliases and generics make these lower bounds on format support. Seven of the XYZ matches are point-cloud producers. The same storage format can mean quite different things. |
| Geographic representation | `GPUDatumGridTransform` takes longitude degrees, latitude degrees, height, with optional low words. `GPULocalFrameProjection` accepts float32 pairs or binary64 bit patterns in `uint32x4`. Five spherical predicates/overlays use unit XYZ directions while declaring a longitude-latitude spatial context. They are `GPUSphericalConvexContains`, `GPUSphericalConvexIntersection`, `GPUSphericalPolygonContains`, `GPUSphericalPolygonOverlay`, and `GPUSphericalArcOverlay`. The latter three were also checked in KI. |
| Relative projected coordinates | `rg -n 'localOrigin\??: readonly \[number, number\]' S/raster-analysis` finds four public settings declarations covering six producers: `GPUDistanceField`, `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings`, `GPURasterSampling`, `GPURasterProfile`. Point clouds instead have `makePointCloudLocalPositions` for XYZ. Both models subtract in host binary64 before upload. UNI-03 identifies a missing sibling. |
| Distance and area | `rg -n 'readonly measures: GPUMeasureDescription' S` finds two declarations, `GPUGeometryMeasures` and `GPUCellMeasures`. They distinguish native squared units, square metres and square kilometres. Network edge costs remain in the caller’s cost unit. Terrain cell sizes can be uniform world units, geographic degrees converted by row, or Web Mercator metres converted to ground metres. A generic `distance` name alone does not establish metres. |
| Time | Of the 41 direct time-port producers, 33 expose `temporalContext` and eight do not. The eight are named below. The ten raster `dateAxis` producers use a separate meaningful distinction between dated observations and slice indices. Network routing takes `departure` or `window`, so appears outside the direct-time count. |
| No-data | The 12 direct sentinel props belong to `GPUFocalStatistics` and 11 raster producers named below. Raster bands additionally support native raw sentinels and an independent nonzero-valid plane before calibration. Point-cloud Surfaces instead publishes zero plus coverage bits. Arrow time uses exact words plus a separate validity vector. Neither can safely be interpreted by looking at the numeric value alone. |
| Wrap | `rg` over the 498 producer implementation files for `antimeridian\|wrapLongitude\|longitude branch` matched files defining 42 producers. This is a mention count. Actual policies include canonical longitude, continuous longitude branches for datum-grid sampling, offsets that the point-buffer caller must wrap, and padded viewport min/max. UNI-18 demonstrates the consequence at a public host boundary. |

## Proposed convention

Every spatial or temporal port should declare its **logical quantity, physical representation, unit or source of units, and missing-value rule**. Use `Degrees`, `Radians`, `Seconds`, `Milliseconds`, `Meters`, `SquareMeters`, `Pixels`, `Index` or `Count` for fixed-unit scalar fields, and state the governing context for native-unit fields. Check the resolved public types against a small per-port table, then test each adapter with an asymmetric fixture that distinguishes swapped axes, changed origins, changed units and a valid zero from a missing value.

## Findings

| ID | Kind | Reach | Change | Cost | Where | One line |
|---|---|---|---|---|---|---|
| UNI-01 | W | ORDINARY | ADDITIVE | S | Point-cloud Surfaces → terrain | Passing coverage as raster validity creates a false downhill receiver. |
| UNI-02 | W | ORDINARY | ADDITIVE | S | Spatial frame resolution | A degree or survey-foot frame can be declared and accepted as metres. |
| UNI-03 | W | ORDINARY | ADDITIVE | M | Fractional coverage | Absolute projected coordinates can double a valid parcel fraction during upload. |
| UNI-04 | W | ORDINARY | ADDITIVE | S | Point-cloud grid → raster grid | The same `origin` property denotes opposite corners on the default grids. |
| UNI-05 | W | ORDINARY | ADDITIVE | S | Spatial compatibility | Default metres compare compatible with explicit kilometres. |
| UNI-06 | W | ORDINARY | ADDITIVE | S | Point buffer | The library’s usual geographic spelling silently selects the planar kernel in JavaScript. |
| UNI-07 | W | ORDINARY | BREAKING | S | Viewport bounds | Three ordinary XY points silently become two XYZ points. |
| UNI-08 | W | ORDINARY | ADDITIVE | M | Arrow time → relative time | Conversion can turn a missing timestamp into a genuine instant. |
| UNI-09 | W | ORDINARY | ADDITIVE | M | Relative CRS frames | Two origins 10 km apart can receive the same compatibility identity. |
| UNI-10 | W | ORDINARY | ADDITIVE | M | Computation records → lens comparison | Rectangular cells cannot retain both dimensions in a raster capture. |
| UNI-11 | W | UNUSUAL | ADDITIVE | S | Solar position and irradiance | An impossible latitude yields a plausible sun position and positive irradiance. |
| UNI-12 | W | UNUSUAL | ADDITIVE | S | Fractional coverage affine | The packer publishes a nonzero determinant alongside singular packed axes. |
| UNI-13 | W | UNUSUAL | ADDITIVE | S | Local-frame arrays | Forward conversion drops a trailing coordinate and inverse conversion fabricates a zero. |
| UNI-14 | W | EXTREME | ADDITIVE | S | Relative-time preparation | The precision guard can certify collapsed or infinite timestamps. |
| UNI-15 | C | ORDINARY | ADDITIVE | S | Trajectory preparation → Kalman | Two returned time columns need different units and epochs. |
| UNI-16 | D | ORDINARY | ADDITIVE | S | Network time routing | Rebasing departures also requires rebasing the cost model. |
| UNI-17 | D | ORDINARY | ADDITIVE | S | First-run time window | “Share an epoch and units” still allows a common upload that erases every event. |
| UNI-18 | E | ORDINARY | ADDITIVE | S | Geographic viewport | Canonical dateline neighbours fit the whole world, and unwrapped positions can disappear. |
| UNI-19 | C | ORDINARY | ADDITIVE | S | Angle-bearing settings | `azimuth` can mean compass degrees, mathematical radians, an RVT direction or a bin index. |

### UNI-01: Point-cloud coverage can create a false drainage path

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-surfaces.ts:141,343`; `S/raster-analysis/raster-grid/gpu-raster-grid.ts:61,119`; `S/gpu-terrain/hydrology/cpu-terrain-flow-operation.ts:128`. The KI point-cloud guide at `D/pointcloud-analysis.md:247,600` invites terrain composition and explains the bit test separately for seeding Ground.

**Now.** From those pages a reasonable composition is:

```ts
const dem = makeGPURasterBand(terrain, rasterGrid, {validity: coverage});
```

**Evidence — EXECUTED.** I fed `computeTerrainFlowOnCPU` a two-cell DEM `[0,5]`. The first cell had a non-ground return and the second had ground. Raw coverage `[1,3]` produced filled elevations `[0,5]`, D8 directions `[0,16]`, and accumulation `[2,1]`. Ground validity `[0,1]` produced `[NaN,5]`, `[4294967295,0]`, and `[NaN,1]`. The first result drains the valid cell west into missing terrain. Setting `noDataValue: 0` would also discard genuine sea-level ground.

**Wanted.** Expose the example’s existing `createGroundSeedCoverageNode` logic as a reusable point-cloud operation, or let Surfaces optionally write `terrainValidity`. Document `surface` as requiring bit 1 and `terrain` and `canopy` as requiring bit 2. Put the conversion in the terrain handoff example. This adds a concrete cross-family failure to `KNOWN(RT-109)`.

**In-repository callers that change.** None required. The showcase can reuse the public implementation at `examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support.ts:52`. **Value and cost.** DEM users avoid false basins and receivers. S.

### UNI-02: Declared frame units can contradict the planar metric

**Where.** `S/analysis-core/contracts/spatial-context.ts:145,178,291`; `spatial-frame-crs.ts:53,70`; `S/gpu-spatial-analysis/trajectory-estimation/cpu-trajectory-kalman-operation.ts:65`. `GPUGeometryMeasures` derives output labels from the context at `gpu-geometry-measures.ts:622`.

**Evidence — EXECUTED.** I ran the same two-observation Kalman input with positions `(0,0),(1,0)` and timestamps `[0,1]`. An explicit longitude-latitude context threw `does not support longitude-latitude coordinates; project positions to a planar frame first`. An omitted context returned final x `0.6999999881`. This declaration returned the same number:

```ts
{coordinateSpace: 'planar', metric: 'native', units: 'meters',
 frame: createGPUSpatialFrameFromCRS('EPSG:4326')}
```

The frame itself reported `{name:'degrees', degreesPerUnit:1}`. A UTM `+units=us-ft` frame with `metersPerUnit: 0.3048006096012192` also passed as `meters`. A second probe found that the provider-free constructor returns metres for `+proj=latlong`, `lonlat` and `latlon`, and no unit for `longlat`. With resolved math.gl metadata, all four returned degrees.

**Wanted.** Reject a declared metre metric that contradicts a known frame unit. For planar operators measuring ground distance or area, reject a known angular frame with a projection hint. Return angular units for all four supported geographic PROJ spellings. Raw unlabelled XY remains inherently ambiguous, which the first-use example should say explicitly. This extends `KNOWN(CORE-15)` beyond the fixed kilometre case.

**In-repository callers that change.** None need a signature edit. **Value and cost.** GIS users get an error at construction instead of a result labelled metres or square metres. S.

### UNI-03: A nearby absolute grid origin does not preserve parcel fractions

**Where.** `S/raster-analysis/gpu-raster-fractional-coverage.ts:109,329,341`; `S/analysis-core/contracts/raster-grid.ts:327,365`; KI `D/gpu-raster/expansion.md:146`. Contrast the usable `localOrigin` setting on `GPUDistanceField`, `GPURasterSampling` and `GPURasterProfile`.

**Now.** The guide says to keep the grid origin near the geometry. Consider a one-metre cell with its origin at northing `4_600_000` and a full-width strip spanning `originY + 0.25` to `originY + 0.75`. The origin is already at the cell corner.

**Evidence — EXECUTED.** `Float32Array.of(4600000.25,4600000.75)` gave `[4600000,4600001]`. Its height is `1`; the source height is `0.5`. Subtracting first gave `[0.25,0.75]`. `resolveGPURasterContext(...,{grid,localOrigin:[0,4600000]})` packed a zero Y origin and preserved unit steps. The public coverage packer accepts only `(grid,target?)`.

**SOURCE and GPU probe.** Run the single rectangle on a 1×1 grid using absolute uploads and then a common local origin. Read `fractions`, `count` and `zoneValidity`. The absolute upload describes a full cell, the local upload half a cell. The host probe proves the information loss independently of triangulation.

**Wanted.** Provide an explicit origin-relative coverage packing route and show subtraction from the original binary64 vertices before conversion. Keep absolute latitude available to geographic and Web Mercator area consumers that reuse the georeference. Passing a rebased Y origin to a latitude-dependent area calculation would introduce a second error.

**In-repository callers that change.** None for the additive route. **Value and cost.** Parcel and land-cover workflows preserve fractional area at common projected magnitudes. M including downstream area tests.

### UNI-04: Point-cloud and raster grids disagree about which corner `origin` names

**Where.** `S/pointcloud-analysis/operations/gpu-pointcloud-surfaces.ts:44,104`; `pointcloud-grid-index.ts:148,167`; `S/raster-analysis/raster-grid/gpu-raster-grid.ts:29,44`; KI `D/pointcloud-analysis.md:247`. The point-cloud convention is shared by `GPUPointCloudSurfaces`, `GPUPointCloudHeight`, `GPUPointCloudGround`, and `GPUPointCloudSMRF`.

**Now.** The following is structurally valid, and both defaults put the northern row first:

```ts
const cloudGrid = {width: 2, height: 2, cellSize: 10, origin: [100, 200] as const};
const rasterGrid = makeGPURasterMetadata(cloudGrid);
```

**Evidence — EXECUTED.** The raster helper returned `[10,0,100,0,-10,200]`, covering Y `180..200`. The cloud covers Y `200..220`. The raster needs `[10,0,100,0,-10,220]`. **SOURCE:** there is also a boundary distinction after correcting the origin. At Y `210`, the default cloud assigns `height-1-floor(1) = 0`; raster snapping from the northern edge assigns row `1`. A GPU probe should rasterize points at Y `205,210,215` and compare cell IDs against distance-field seed snapping on the corrected grid.

**Wanted.** Add `getRasterMetadataFromPointCloudGrid(grid)`. For southward rows add `height * cellSize` to the minimum Y; northward rows already start at minimum Y. Document the edge tie separately from array alignment. Combine this helper with UNI-01 in one runnable handoff.

**In-repository callers that change.** None required. **Value and cost.** Users can overlay a derived DEM in its real location. S.

### UNI-05: Coordinate compatibility can certify incompatible metric results

**Where.** `S/analysis-core/contracts/spatial-context.ts:412,446,506,519,555,644`; `S/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation.ts:143`.

**Evidence — EXECUTED.** For `(0°,0°) → (1°,0°)`, `computeGeodesicPairsOnCPU` returned `111195.078125` with omitted units, `111.19508361816406` with `units:'kilometers'`, and `0.01745329238474369` with `sphereRadius:1`. `getGPUSpatialContextCompatibility(defaultContext, otherContext)` returned `{compatible:true}` for both alternatives.

**Now.** `assertCompatibleGPUSpatialContexts(...)` can pass immediately before comparing a threshold or displaying two lengths in one unit. Coordinate-only consumers do legitimately share longitude-latitude rows across these metrics.

**Wanted.** Add a metric-result compatibility check, or an explicit `purpose:'metric-results'` mode. Resolve effective length units, sphere radius and ellipsoid before comparing. Preserve coordinate-only checks for ports carrying positions. Test omitted radius against both explicit `6371008.8` and explicit `1`.

**In-repository callers that change.** None for a new check. **Value and cost.** Callers composing numeric measurements can distinguish compatible coordinates from comparable values. S.

### UNI-06: The standard geographic spelling selects planar point-buffer math in JavaScript

**Where.** `S/geospatial-analysis/point-buffer/gpu-point-buffer.ts:89,119,230,259`; `D/geospatial.md:55`. Of 498 direct producers, `GPUPointBuffer` alone exposes the top-level constructor prop `coordinateSystem` in the resolved-props census.

**Evidence — EXECUTED, graph declaration only.** With one centre, four vertices and valid graph views, `coordinateSystem:'geographic'` declared the spherical shader and `'planar'` declared the planar shader. Passing `coordinateSystem:'longitude-latitude'` also successfully declared the **planar** shader. I inspected the emitted source for its radius-times-sine expression and geographic latitude branch. TypeScript catches the literal typo in a typed object; JSON and JavaScript callers reach it.

**SOURCE and GPU probe.** A four-vertex, 100 m buffer at `(0,0)` should have a north offset near `0.000899°`. The selected planar expression instead writes `100` in the north component.

**Wanted.** Validate the existing discriminant at construction. Add a `spatialContext` input for composition with the rest of the library, preserving the existing explicit spellings. Reject conflicting declarations. `KNOWN(DOC-102)` concerns a different class whose constructor already rejects a mismatched system.

**In-repository callers that change.** None for validation of invalid input or an additive context. **Value and cost.** A familiar vocabulary mistake cannot turn metres into degrees. S.

### UNI-07: Viewport bounds silently reinterpret valid XY data as XYZ

**Where.** `S/geospatial-analysis/viewport-bounds/geographic-viewport-bounds.ts:18,33,36`.

**Evidence — EXECUTED.** Three XY points, with every coordinate individually valid under either interpretation:

```ts
const xy = Float32Array.of(-30,40, -29,41, -28,42);
getGeographicViewportBounds([xy]);    // [-33.55, -31.4, 44.55, 43.4]
getGeographicViewportBounds([xy], 2); // [-30.1, 39.9, -27.9, 42.1]
```

Length six satisfies both layouts. The first result is a plausible-looking viewport in the wrong place.

**Wanted.** Require `positionSize: 2 | 3`. Do not infer an ambiguous stride. The existing packed trip buffers should pass `3` explicitly.

**In-repository callers that change.** **8 of 13 actual calls**, counted with `rg -lF 'getGeographicViewportBounds('` and TypeScript call-expression arity. Four are in `T/geospatial-analysis/geographic-viewport-bounds.node.spec.ts`, two in `modules/kepler/test/trips-browser-fixture.node.spec.ts`, one in `modules/kepler/test/resident-layers.node.spec.ts`, and one in `examples/deck/kepler-compute/src/source-publication.ts:530`. **Value and cost.** Every caller specifies the one fact the helper cannot recover. S, breaking, recommended before alpha.

### UNI-08: The relative-time adapter loses Arrow nullness

**Where.** `S/analysis-core/graph-support/arrow-analytics-normalization.ts:360`; `arrow-analytics-time-upload.ts:42`; `S/dataframe-analysis/time-words-relative/gpu-time-words-relative.ts:63,139,179,192`.

**Evidence — EXECUTED.** Preparing a nullable Arrow Int64 column `[0n,null,1n]` as milliseconds returned words `[0,0, 0,0, 1,0]` and validity `[1,0,1]`. `GPUTimeWordsRelative` accepts timestamps, temporal context, origin and output. There is no validity input. The advertised output context accurately describes the units but cannot identify the missing row.

**SOURCE and GPU probe.** Bind these three rows with origin `0n`. The conversion kernel reads only the words and writes their difference from the origin. Read the three outputs and distinguish the valid epoch-zero row from the missing row. The expected null-preserving result is `[0,NaN,1]` or `[0,unused,1]` accompanied by `[1,0,1]`.

**Wanted.** Add an optional chunk-aligned input mask and specify missing relative-time output, preferably NaN. Show the validity handoff in the Arrow-to-relative example and state which downstream algorithms require compacting missing rows. `KNOWN(DPC-123)` identified the ingress test gap; the additional defect here is a named adapter with no port for the information that ingress already supplies.

**In-repository callers that change.** None required for non-null input. **Value and cost.** Null observations cannot acquire a real date during conversion. M including a chunked composition fixture.

### UNI-09: Relative projected origins are absent from CRS frame identity

**Where.** `S/analysis-core/contracts/spatial-frame-crs.ts:29,59`; `spatial-context.ts:578`; `D/gpu-spatial-analysis/spatial-frames.md:132`.

**Evidence — EXECUTED.** I constructed two frames by spreading `createGPUSpatialFrameFromCRS('EPSG:32633',{precision:'float32-origin-relative'})`, setting `anchor:'relative-origin'`, and using origins `[500000,4600000]` and `[500000,4610000]`. The contexts compared `{compatible:true}`. Both local coordinates `[0,0]` have the same numerical value and represent locations 10 km apart.

**Now.** The CRS factory accepts a precision but has no origin option. The docs describe putting an origin in the descriptor and passing it as `destinationOrigin`. The caller must independently invent a distinct ID.

**Wanted.** Accept `origin` and `anchor` in the factory. Include the effective storage origin in the derived ID for relative-origin frames, or compare origins when both frames explicitly declare that anchor. Document that this origin uses **target CRS units**; a local projection centre uses longitude-latitude degrees.

**In-repository callers that change.** None for the factory overload. **Value and cost.** Independently prepared projected tiles can be checked before joining. M including checking consumers of explicit IDs.

### UNI-10: A raster capture cannot describe a rectangular cell

**Where.** `S/computation-record/computation-record-types.ts:684,690,701`; `modules/lenses/src/computation-views/views/compare-view.ts:56,75`.

**Evidence — EXECUTED.** I created two 2×2 grids with origin `[0,40]`, equal 10 m column steps, and row steps of −10 m and −20 m. Their first-row centres are Y `35` and `30`. The capture type supplies one scalar `cellSize`. Recording `10`, the common origin, southward rows and projected metres for both gave `COMPARE_VIEW.satisfies(context) → {ok:true}` and cellwise deltas `[1,1,1,1]` for values `[1,1,1,1]` versus `[2,2,2,2]`. There is no second cell dimension or affine field in the typed raster capture.

**Wanted.** Add optional six-coefficient `affine`, `pixelInterpretation` and frame/CRS identity. Compare effective sample transforms before cellwise subtraction. Define the existing scalar form as square, axis-aligned cells for older records. Until a producer can record the transform, it should present a rectangular or rotated result in pixel space.

**In-repository callers that change.** None for existing square-cell records. The comparison model and record writers for non-square grids gain work. **Value and cost.** A comparison cannot subtract measurements from different ground footprints while presenting them as aligned. M.

### UNI-11: Solar scalar and time-table paths accept impossible latitudes

**Where.** `S/gpu-terrain/terrain-illumination/solar-position.ts:94,98,143`; `gpu-solar-irradiance.ts:107,141`; `cpu-solar-position-operation.ts:61`.

**Evidence — EXECUTED.** At `2026-06-21T12:00:00Z`, longitude `0`, latitude `91`, `getSolarPosition` returned azimuth `179.547286°`, altitude `22.476891°`. The one-minute irradiance table returned a finite row with direct normal irradiance `684.0847`. `computeSolarPositionOnCPU` for the same coordinate returned azimuth `NaN`, altitude `NaN`, daylight `0`.

**Wanted.** Validate finite longitude and latitude in `[-90,90]` for a scalar request, and validate the table’s common location before allocating or mutating a target. Preserve the dense twin’s row-level missing-data behavior. **In-repository callers that change.** None for valid locations. **Value and cost.** A coordinate-order mistake involving a longitude over 90° cannot become a credible irradiance estimate. S.

### UNI-12: A nonsingular binary64 affine can become a singular public parameter buffer

**Where.** `S/raster-analysis/gpu-raster-fractional-coverage.ts:115`; `S/analysis-core/contracts/raster-grid.ts:373`; `S/raster-analysis/raster-patches/gpu-raster-patch-metrics.ts:71`.

**Evidence — EXECUTED.** For `[1,1,0,1,1+2**-25,0]`, the coverage packer returned `[0,0,1,1,1,1,2.9802322387695312e-8,0]`. Both packed axis vectors are identical. `getGPURasterPatchMetricsParameterValues` rejected the same grid with `grid.affine relative to localOrigin must be finite and invertible in float32`.

**Wanted.** Route coverage through the existing packed-axis validation before publishing the target. Test a reused target for no mutation on rejection. This identifies a specific remaining migration and runnable counterexample under `KNOWN(STD-06)`. **In-repository callers that change.** None for representable transforms. **Value and cost.** Coverage cannot publish an internally contradictory geotransform. S.

### UNI-13: Local-frame array conversion accepts an incomplete coordinate

**Where.** `S/analysis-core/contracts/local-frame.ts:129,133,144`; compare `S/pointcloud-analysis/operations/pointcloud-local-origin.ts:80` and the viewport stride check at `geographic-viewport-bounds.ts:33`.

**Evidence — EXECUTED.** At origin `[0,0]`, `projectArray([0,0,1])` returned `Float32Array [0,0]`; `unprojectArray([0,0,1])` returned `Float64Array [0,0,0]`. The first drops a value; the second manufactures a zero in its place. This concerns the **input** after `KNOWN(CORE-101)` fixed short destination arrays.

**Wanted.** Check `input.length % stride === 0` and `xys.length % 2 === 0` before writing. Name the stride and input length in the error. **In-repository callers that change.** None with complete rows. **Value and cost.** Truncated coordinates cannot silently change output cardinality. S.

### UNI-14: Guarded relative times can contain infinity, collapse to zero, or carry an invalid epoch

**Where.** `S/gpu-spatial-analysis/space-time-tests/space-time-parameters.ts:225,249`; `space-time-bands.ts:126,151`; `S/analysis-core/contracts/temporal-context.ts:113`.

**Evidence — EXECUTED.**

| Call | Actual |
|---|---|
| `getGPUSpaceTimeGuardedRelativeTimes([0,1e40],1e35)` | `times=[0,Infinity]`, reported resolution `6.490371073168535e32` |
| `getGPUSpaceTimeGuardedRelativeTimes([0,1e-46],1e-46)` | `times=[0,0]`, reported resolution `1.044048714879764e-53` |
| Same helper with `[1,2]`, width `4`, nanoseconds, input epoch `2n**63n-1n` | Returned `epoch=9223372036854775808n`; the public temporal resolver immediately rejected it |

**Wanted.** Check the values after float32 conversion, floor nonzero spacing at the actual subnormal quantum `2**-149`, and validate the completed context before returning it. **In-repository callers that change.** None for outputs that satisfy the advertised guard. **Value and cost.** Consumers can trust the library’s explicitly guarded preparation path. S.

### UNI-15: Prepared trajectory times and exact words need different metadata

**Where.** `S/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations.ts:111,125`; `cpu-trajectory-kalman-operation.ts:94,297`; `D/gpu-network/network-movement-coverage.md:328`.

**Evidence — EXECUTED.** With timestamps `[1000500n,1001500n]`, epoch `1000500n` and `unitsPerSecond:1000n`, preparation returned `times=[0,1]`, absolute words `[1000500,0,1001500,0]`, and one top-level epoch and tick rate. In the same two-fix Kalman calculation, interpreting `times` as seconds gave final x velocity `5.0093770`. Applying the source millisecond unit to `times` gave `0.00556875`. Millisecond **words** gave `5.0093770`. Additionally, mutating the supplied origin changed `prepared.projection.origin` after preparation.

**Wanted.** Describe each output separately: `times` contains elapsed seconds; `timeWords` contains absolute source ticks. Supply the word context when the rate maps to a supported unit, and preserve the float stream’s absolute origin as exact source ticks plus rate. In this example the origin is `1000.5` seconds, which the integer-seconds context cannot express. Copy the projection origin. Expose a metre-valued spatial descriptor that composes with `GPUSpatialContext`; the current metadata says `'metres'`, while the spatial API accepts `'meters'`.

**In-repository callers that change.** None for additive metadata. **Value and cost.** A user can switch to exact words without guessing which quantities require rescaling. S.

### UNI-16: Rebasing a network departure changes the affine cost intercept

**Where.** `S/gpu-network/network-optimization/gpu-network-time-routing.ts:68,78`; `cpu-network-time-routing-operation.ts:152,183`; `S/gpu-network/network-time-geography/cpu-network-travel-time-percentiles-operation.ts:99,122`; KI `D/gpu-network/network-time-geography-mob09.md:32`.

**Evidence — EXECUTED.** One edge with duration `10 + 0.5*t`, departure `100`, gave travel time `60`. Following the relative-time hint and changing departure to `0` while keeping costs `[10,0.5]` gave `10`. Rebasing costs to `[60,0.5]` restored `60`. All three CPU percentile runs reported `[valid,converged]=[1,1]`.

**Wanted.** Put the conversion beside the `departure` and `window` examples. For `t′=t−origin`, set `base′=base+slope*origin`; slope and durations retain their units. For a piecewise arrival table, subtract the origin from **both** departure and arrival columns. Choose an origin consistent with the nonnegative relative-time domain. **In-repository callers that change.** None. **Value and cost.** Timetable users can improve float32 precision without changing their routing problem. S.

### UNI-17: The first-run time filter omits the distinction that makes real timestamps work

**Where.** `D/README.md:185`; `S/dataframe-analysis/time-window-filter/time-window-parameters.ts:13,39,114`; `time-window-classify-node.ts:164,173`. The detailed dataframe guide does explain `timestampsLow`.

**Now.** Replacing the tutorial’s three toy numbers with three events one second apart is the natural first edit:

```ts
const t = Date.UTC(2026, 10, 19);
const timestamps = Float32Array.of(t, t + 1000, t + 2000);
const window = getGPUTimeWindowFilterParameterValues({start: t + 1000, end: t + 1000});
```

**Evidence — EXECUTED.** All three upload values were `1795046375424`. The exact window packed as high `1795046375424`, low `25576`. `splitTimestamps` instead returned low words `[24576,25576,26576]`. **SOURCE and GPU probe:** run this three-row window with and without `timestampsLow`. The single-float comparison supplies zero for each missing low lane; expected intended selection is row `[1]` with the split input.

**Wanted.** Add one sentence and the two-line relative-time alternative at the first-run example: subtract `t` **before** constructing the `Float32Array`, and subtract it from both window endpoints. Link `splitTimestamps` for callers retaining absolute values. This is independent of the packed-endpoint validation in `KNOWN(DPC-108)`. **In-repository callers that change.** None. **Value and cost.** The tutorial continues to work when users substitute real event times. S.

### UNI-18: The viewport helper has no compact dateline fit

**Where.** `S/geospatial-analysis/viewport-bounds/geographic-viewport-bounds.ts:14,36,50`; `examples/deck/kepler-compute/src/source-publication.ts:530`.

**Evidence — EXECUTED.** `getGeographicViewportBounds([Float32Array.of(179,5,-179,5)],2)` returned longitude bounds `[-180,180]` for points 2° apart. A single continuous-branch point `[181,5]` returned `[180,180]` in longitude, which contains neither `181` nor its canonical longitude `-179`. The TSDoc discloses the absence of unwrapping. The second example shows why simply clipping incoming bounds is insufficient for outputs from another operator.

**Wanted.** Add a branch-aware fit returning an explicit centre longitude and span, or west/east with an explicit wrap flag. Require callers of the current bounded helper to supply canonical longitudes. Document which result to use with an API requiring ordered `west <= east`. `KNOWN(GEO-104)` concerns group geometry bounds; this is a separate host viewport implementation and consumer. **In-repository callers that change.** None for a new helper. **Value and cost.** Dateline-adjacent trips open at a useful scale. S.

### UNI-19: Angle names conceal three different references and one index space

**Where and COUNT.** `rg -n '^  (\w*[Aa]zimuth\w*|straightAngle|uTurnAngle|angleCost)\??: number;' S` exposes these distinct public input conventions:

| Classes | Current field | Actual meaning |
|---|---|---|
| `GPUTerrainDerivatives`, `GPUReliefShading`, `GPUSolarShadowMask`, `GPUTerrainCastShadow` | `azimuthDegrees` | Clockwise from north; terrain grid orientation matters |
| `GPUTerrainHorizon` | `anisotropyAzimuthDegrees` | RVT counterclockwise from north; `90` weights west |
| `GPUVariogram` | `azimuthOffset` | Undirected sector edge, radians counterclockwise from +X |
| `GPUNetworkLineGraph` | `straightAngle`, `uTurnAngle`, `angleCost` | Radian thresholds and cost per radian |
| `GPUPointHorizonProfile`, `GPUPointHorizonVisibility` | `firstAzimuth`, `azimuthSpan` | Integer sector index and sector count |

Sources are `S/gpu-terrain/terrain-illumination/terrain-horizon-parameters.ts:79`, `S/dataframe-analysis/pair-statistics/variogram-parameters.ts:38`, `S/gpu-network/network-line-graph/network-line-graph-parameters.ts:11,22`, and `S/gpu-terrain/point-horizon/point-horizon-march.ts:343`. The RVT exception is accurately documented in KI `D/gpu-terrain/operators.md:1368`.

**Evidence — EXECUTED.** A degree-minded `straightAngle:30,uTurnAngle:150` packed as `30,150` radians; `azimuthOffset:90` packed as `90` radians. These are valid numbers and survive construction. At the default 720 point-horizon sectors, sector index `90` denotes 45°.

**Wanted.** Introduce explicit aliases `straightAngleRadians`, `uTurnAngleRadians`, `turnCostPerRadian`, `sectorOffsetRadians`, `rvtDirectionDegrees`, `firstAzimuthIndex` and `azimuthBinCount`. State the zero axis, rotation and period for directional sectors. Preserve RVT numeric compatibility. Also distinguish `grid north` from `true north` where solar azimuth is copied into terrain lighting: a rotated uniform raster uses the grid’s up direction while the solar calculation produces a geographic compass bearing.

**In-repository callers that change.** None for additive aliases. **Value and cost.** Users can choose the right field from autocomplete and keep the bin count out of angle arithmetic. S for names, docs and conflicting-alias checks.

## Departures by name

- **Direct time ports without a top-level `temporalContext`**, eight of 41: `GPUCrossVariogram`, `GPUSpaceTimeVariogram`, `GPUTemporalReduction`, `GPUTimeWindowFilter`, `GPULineSimplification`, `GPUKnoxTest`, `GPUMantelTest`, `GPUMobilityIndicators`. The KI copies of `GPULineSimplification` and `GPUMantelTest` retain this shape. A prop may be unnecessary when only ordering matters. The documentation should say whether the consumer uses order, tick differences, durations or absolute instants.
- **Direct sentinel owners**, 12: `GPUFocalStatistics`, `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings`, `GPURasterReclassify`, `GPUWeightedOverlay`, `GPURasterCellStatistics`, `GPURasterConditional`, `GPURasterArithmetic`, `GPURasterSampling`, `GPURasterProfile`, `GPURasterStretch`. The legacy raw-view sentinel and a band’s native pre-calibration sentinel have different scopes. A mechanical check should preserve that distinction.
- **Grid and coverage handoffs:** `GPUPointCloudSurfaces`, `GPUPointCloudHeight`, `GPUPointCloudGround`, `GPUPointCloudSMRF`, `makeGPURasterMetadata`, `makeGPURasterBand`, `GPURasterFractionalCoverage`, and the raster variant of `GPUComputationCaptureGeometry`. UNI-01, 03, 04, 10 and 12 identify the exact incompatible fields.
- **Frame, tuple and wrap boundaries:** `createGPUSpatialFrameFromCRS`, `resolveGPUSpatialContext`, `getGPUSpatialContextCompatibility`, `createGPULocalFrame().projectArray`, `.unprojectArray`, `GPUPointBuffer`, and `getGeographicViewportBounds`.
- **Time, location and directional settings:** `GPUTimeWordsRelative`, `getGPUSpaceTimeRelativeTimes`, `getGPUSpaceTimeGuardedRelativeTimes`, `prepareGPUTrajectoryObservations`, `getSolarPosition`, `getGPUSolarIrradianceSunTable`, `GPUNetworkTimeRouting`, `GPUNetworkTravelTimePercentiles`, and the nine angle consumers in UNI-19 plus `GPUTerrainHorizon`. The angle table gives each field and convention.

## Quick wins

- [ ] Reject a known frame/metric unit contradiction and recognize all four supported geographic PROJ aliases.
- [ ] Validate `GPUPointBuffer.coordinateSystem` and the scalar solar location before graph or table construction.
- [ ] Publish the point-cloud grid conversion and ground-bit validity extraction alongside each other.
- [ ] Delegate fractional-coverage packed-affine validation to the raster context helper.
- [ ] Reject incomplete local-frame rows before writing either destination.
- [ ] Validate the actual packed relative times and their resulting epoch before returning from guarded preparation.
- [ ] Add the three-event Unix-millisecond example and the one-edge affine-cost rebasing example.
- [ ] Label the two prepared trajectory time streams individually and copy the returned projection origin.

## Larger proposals

1. **Carry frame and validity through common adapters.** Start with one real journey: classified points → DEM → flow → recorded raster. The fixture should contain a non-ground-only cell, a valid zero elevation, a non-square cell, a nonzero projected origin and one point exactly on a cell edge. This tests five public boundaries with independently predictable answers. M.
2. **Make projected-frame identity describe stored coordinates.** Add an origin-aware CRS factory and a separate result-metric comparison. Existing coordinate-only consumers can keep their current compatibility check. M.
3. **Require viewport stride before alpha.** The migration is eight call edits, including one application call. Keeping an implicit three-component default saves one integer per call and makes ambiguity undetectable for common array lengths. I recommend the breaking change now. S.
4. **Extend the record schema with full raster placement.** Interpret older records under their existing square-cell convention. Use the affine and pixel interpretation for new captures, overlays and comparisons. M.
5. **Extend the existing public API checker with unit contracts.** Build on `KNOWN(ERG-104)` and `KNOWN(DIR-109)`. Inspect resolved props, nested bands and exported parameter packers. The checks should distinguish a scalar angle, a view containing angles, a sector index, an elapsed duration and an absolute timestamp. A suffix-only source scan misses exactly the cross-family failures demonstrated here. M.

## Not read

I read the local-frame implementation, both spatial-frame factories, geographic viewport helper, relative-time preparation, time-window packing and trajectory observation preparation in full. I traced the relevant paths through spatial compatibility, fractional coverage, point-cloud rasterization, terrain flow, solar position, Kalman filtering, network time percentiles, and record comparison. I sampled remaining families through public types, their guides and the named convention searches.

I checked the synthesis, area reports, API/catalog and direction reports, decision list, and staging ownership list. For staging-owned material used above, I checked the newer KI point-cloud, raster, terrain and network docs, the shared raster operation-band implementation, terrain-flow packer, and the named spatial producer declarations. The census describes the review checkout as read. I did not independently audit every shader, catalog record, lens capture, CPU twin or geodetic model, and I ran no GPU, browser, build or network operation.