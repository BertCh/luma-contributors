---
title: "GPU Terrain concepts"
description: "Conventions and execution model of GPU Terrain contributors."
---


# GPU Terrain concepts

Start with the [checked raster alignment and cost-unit example](./README.md#check-alignment-before-extracting-bands):
construct `new GPURaster({metadata, bands})`, call `assertCompatibleWith` before
`getBand('red')`, and inspect `getPixelWorldPosition(0,0)`. One-column shifts are rejected.
A GDAL reader takes geotransform `[GT0,GT1,GT2,GT3,GT4,GT5]`, for example
`[700,20,0,350,0,-10]`. A library `GPURasterMetadata.affine` takes
`[GT1,GT2,GT0,GT4,GT5,GT3]`, hence `[20,0,700,0,-10,350]` for that same grid.
Both consumers place area-pixel center `(0,0)` at `[710,345]`:
`[700 + 20/2, 350 - 10/2]`. Pass each tuple only to its corresponding consumer.
Factory cell sizes are magnitudes and
`rowDirection` sets the row sign. A CRS does not select `cellSizeMode`: the one-degree unit-friction
costs are `[0,1]` by default versus `[0,111319.4921875]` in geographic mode; 10 m projected
cells with friction `[1,3,1]` give `[0,20,40]`.


## Conventions

- **Raster grids.** Grid-based terrain contributors take `grid: GPURasterMetadata` (`makeGPURasterMetadata` builds
  an axis-aligned one); its width and height size the kernels. Cell sizes and the latitude band
  are never typed twice: each packer that needs them takes the frame's grid as its second
  argument, `getGPU<Class>ParameterValues(settings, grid)`, and reads them from `grid.affine`
  in world units (meters for `'uniform'`, EPSG:3857 meters for `'web-mercator'`, degrees for
  `'geographic'`, see `GPURasterCellSizeMode`).
- **`parameters` in, `output` out.** The packed per-frame view is `parameters` (a second view of
  another format is `uint32Parameters`). Outputs live in one `output` bag; the validity output is
  `output.outputValidity` (a `validity` prop is always an input mask) and one-row status words are
  in `output.status` (`converged`, `iterationCount`, `overflow`). `makeGPURasterBand(view, grid)`
  turns one contributor's output view into the next contributor's input band.
- **Compile-time limits say so.** Values that size kernels are named `maximum*`
  (`maximumRadius`, `maximumFillIterations`); the value actually used each frame (`radius`, a
  distance) is in `parameters`.
- **Graph views in, graph views out.** Inputs and outputs are `GraphDataView`, `GraphVectorView`, or
  `GraphTextureView` objects created on the target graph. Outputs are always caller-owned; scratch
  storage is graph transients that die with `compiled.destroy()`. Contributors never compile, encode,
  submit, or read back.
- **Per-frame values never recompile.** Viewports, thresholds, radii, observer positions, budgets,
  and region shapes are read from storage views, usually a `GPUParameterBuffer` that the
  application rewrites with `write()`. Lengths, capacities, grid sizes, and which optional views
  exist are compile-time topology; each prop's TSDoc says which category it belongs to.
- **Bounded results report overflow on the GPU.** Compact ID lists use `GPUCompactOutput`
  (`ids` and a `status` with `count`, `overflow` and optional `requiredCount`). `status.count` is clamped
  to `ids.length` and can be an indirect draw instance count; `status.overflow` is rewritten every
  encoding.
- **Stable IDs.** Result IDs are the caller's `sourceIds[row]` (or tile IDs) when given and zero-based
  rows otherwise. Node and transient IDs are `${id}-<step>`, so two instances in one graph need
  different `id` props.
- **One import per buffer.** Import a buffer once (for example with `graph.importGPUData()` or
  `GPUParameterBuffer.importToGraph()`) and pass the returned view to every contributor that reads
  it, so the graph tracks hazards on one logical handle.
- **Shared contracts.** Four shapes mean the same on every analysis page:
  - **Status.** Every GPU-written status word sits in the `status` member of the bag that owns it:
    `output.status.overflow`, `output.status.converged`, and `output.status.code` for model fits.
  - **Grid.** Grid-based raster and terrain contributors take `grid: GPURasterMetadata`, and their packers read
    the cell size from it (see [GPU Terrain conventions](#conventions)).
  - **Network.** Network contributors take `topology: GPUNetworkTopology` (`offsets`, `neighbors`,
    `weights`, optional `nodePositions`). Its CSR part can be passed as `GPUSpatialWeights` (see
    [GPU Network conventions](../gpu-network/concepts.md#conventions)).
  - **Polygons.** Polygons use the four-level `GPUPolygonGeometryPort`: `kind: 'polygons'`,
    `positions`, `featureOffsets`, `polygonOffsets` and `ringOffsets`. Contributors with a single
    geometry take the same fields flat, without `kind` (`GPUPolygonGeometryFields`). See
    [Shared ports](../gpu-spatial-analysis/concepts.md#shared-ports-status-and-planning).

## Heights and datums

Declare height datums when combining DEM elevations with absolute observer or catalogue heights.

Absolute heights and DEM heights must share one vertical datum, and observer positions must share the DEM grid's horizontal datum. Declare both (`verticalDatum` on the band or `elevationVerticalDatum` on the operator, the operator prop taking precedence; `observerVerticalDatum` on the operator, `datum` on contexts). The library checks the declarations and never converts. Heights above the ground (`heightReference: 'ground'`) need no conversion. For a few observers, convert them on the CPU in binary64: h_orthometric = h_ellipsoidal − N(lat, lon), with N taken from the DEM's own geoid model (PROJ, pyproj or GeographicLib). For many observers or targets, convert the DEM on the GPU: resample N onto the DEM grid, add it with `GPURasterBandMath('add')`, and declare `'ellipsoidal'`. Convert horizontal datums (NAD27, GCJ-02) on the CPU before upload.

| Source | Vertical datum to declare |
| --- | --- |
| SRTM/NASADEM/AW3D30 | EGM96 (`'EPSG:5773'`) |
| Copernicus GLO-30 | EGM2008 (`'EPSG:3855'`) |
| USGS 3DEP | NAVD88 (`'EPSG:5703'`) |
| Terrarium / Mapbox Terrain-RGB | "above sea level", unspecified geoid — leave `verticalDatum` undeclared |
| GNSS receivers | `'ellipsoidal'` |
| drone metadata | vendor-dependent, verify before declaring |

- **Absolute inputs.** `GPUPointHorizonProfile`, `GPUPointHorizonVisibility`, `GPUTerrainViewshed`,
  `GPUTerrainLineOfSight`, and `GPUTerrainCumulativeViewshed` accept `heightReference: 'absolute'`
  (default: `'ground'`). Their `observerVerticalDatum` is checked against the elevation band's
  `verticalDatum`; declaring it in ground mode throws. `GPUTerrainPeakSnap` checks
  `candidateVerticalDatum` against the band's `verticalDatum` for absolute `candidateHeights`;
  declaring it without `candidateHeights` throws.
- **Compatibility.** A mismatch between two declared vertical datums throws; either undeclared
  passes. Comparison trims whitespace and is case-insensitive, with no aliases.
- **Absolute observers.** Set the observer height explicitly: the `observerHeight` default of 1.7
  is meaningless in absolute mode. The observer component of LOS `pairHeights` and cumulative
  `observerHeights` also becomes absolute. Observers must be inside the grid but may sit over
  no-data ground. Targets stay ground-relative; an eye below the ground hides most targets.
- **Conversion.** There is no geoid-undulation input: the contract performs no CRS conversion,
  GPU Project rejects geoid grids, and a constant undulation can be silently wrong across a DEM.
  Use `GPURasterBandMath` with `'add'` or `'subtract'` and an undulation raster on the same grid;
  a band `offset` is appropriate only when the correction really is constant.


## GPU execution model

For tiled terrain, a halo stage must bound every source read and preserve the sampling geometry
when the window moves. `GPUTerrainHorizon.getHaloStage()` provides the march bound
`maximumRadius + 1`. For sweep it provides `maximumRadius` with explicit `parameterLayout: 'sweep'`.
That layout packs a common `sweepSourceGrid`, `sweepReferenceCellSize` and per-frame `sweepOrigin`
in 24 parameters. Both algorithms require this shared source-row layout for latitude-dependent
halos, including distance cutoffs. View padding never selects a layout. The origin is measured in
overview pixels relative to the shared source grid; do not infer downsampling from a level number.
Placement updates reuse the compiled shader. March adds only integer displacements to the pixel
origin, keeping fractions separate so crop offsets cannot change validity corners by rounding.
Horizon construction captures declared options by property access, including inherited and
non-enumerable getters. In particular, `cellSizeMode: 'geographic'` requires the shared source-row
layout for a halo regardless of how the property is defined; changing the caller's option later
does not change that validated requirement.
`GPUTerrainCastShadow` uses the same shared-source model with its own 32-value sweep parameter
layout. Its `getHaloStage()` returns the radius only with explicit `parameterLayout: 'sweep'`;
the standard 16-value window-local model refuses halo assembly. Use the same source grid,
reference cells and fixed radius across windows, and pack each window's origin per frame.
`GPUTerrainWeissLandforms.getHaloStage()` provides `localRadius` only with local
standardization. Global Weiss statistics and geomorphons with per-frame anisotropic cell sizes
have no complete fixed local bound. Their advisory `localRadius` (Weiss) and `searchRadius`
(geomorphons) are not halo stages. Keep only the core outputs of a correctly expanded tile.

How terrain contributors run on the device: the command graph they declare, the kernels it
compiles to, and the resources each pass reads and writes, drawn from a recorded real run.

{/* lens:execution-model */}

A local operator is one kernel over every cell; a propagating one (fill, flats, accumulation) is an
unrolled loop of gated rounds whose indirect dispatch drops to zero once a round changes nothing.

::lens{scenario=gpu-terrain.terrain-derivatives rung=graph caption="A local operator: two Sobel passes feed one fused shade kernel, each one invocation per cell."}
::lens{scenario=gpu-terrain.terrain-derivatives rung=kernel caption="One shade invocation per cell; the dispatch is the grid, not a loop."}

## Grid metrics and texture-shading tiles

Curvature, geomorphons, vector ruggedness, Weiss landforms, relief shading, horizon, cast shadow,
viewshed, line of sight, cumulative viewshed, summits, peak snap, flow field and point-horizon
models use perpendicular pixel-axis lengths. Their constructors and public grid-taking per-frame
packers reject shear. Each packed pixel-axis length must also stay positive and finite after
float32 conversion: even an axis-aligned length of `1e-46` (rounds to zero) or `1e39` (rounds to
infinity) is rejected before target writes. Uniform mode still accepts genuinely rotated orthogonal
affines. Cumulative viewshed uses the line-of-sight packer; both point-horizon packers validate their own grid before
packing. These checks happen before the packer changes its target. Full-affine distance operators
and uniform patch metrics retain their own supported shear semantics.

`GPUTextureShading.requiredHalo` bounds source reads only when expanded tile origins share the
full cascade's decimation phase. Use `getCommandNodesForTile(graph, expandedOrigin)` for an explicit
checked route: the expanded window's integer column and row, relative to one common source-pixel
origin, must each be divisible by the largest scale `2 ** levelScaleLog2[levelCount - 1]`.
It validates both axes before declaring nodes. Supply at least `requiredHalo` source pixels around
the retained core and keep the same cascade settings across tiles. The radius-only halo assembler
does not read or validate origins. With `downsampleLevels: false`, the largest scale is 1 and any
integer expanded origin is supported. Physical source borders retain replicated-edge sampling.


`GPUPointHorizonVisibility` exposes `estimatedCommandNodeCount` and rejects a total estimate
above 32,768 command nodes per contributor at construction. The estimate includes the target
chunks (at least one, even for zero targets), the chunk-table writer, elevation preparation and
any locally built pyramid; a shared pyramid's construction belongs to its producer. Increase
`targetsPerDispatch` (default 65,536) or partition targets to stay within this total budget.


The point-horizon `web-mercator` projection requires an axis-aligned EPSG:3857 window with square
pixels, columns increasing east and rows increasing south. Both contributors and their parameter
packers reject rotated or mirrored axes: the projection's latitude depends only on row. The planar
route retains its existing orthogonal-grid support.
