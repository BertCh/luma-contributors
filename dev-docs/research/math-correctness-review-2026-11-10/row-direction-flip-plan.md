# Row direction flip plan (raster-fields-05)

Closing pass, 2026-11-11. Read-only study: no source file was changed. Paths are relative to the
repository root unless they start with `raster-analysis/`, which is under `modules/analysis/src`.
Line numbers are those of the working tree on 2026-11-11.

## Summary

- Eight classes are named in the finding. Only four of them receive a grid with an affine today:
  `GPUIsolines`, `GPUIsobands`, `GPUIsobandRings` (the raster) and `GPULineIntegralConvolution` (its
  output raster only). The vector fields of the four flow classes arrive as a bounds tuple
  `[originX, originY, cellWidth, cellHeight]` with no affine, and `GPUFiniteTimeLyapunovExponent`
  receives no placement at all.
- 108 constructor call sites in executable code: 3 inside `modules/analysis/src`, 73 in specs, 6 in
  lenses, 11 in website runs, 15 in examples, 0 in kepler. With the design recommended below, 76
  keep their output bit for bit, 14 draw a mirrored picture until the caller is edited (10 of them
  reverse rows on the CPU today and can drop that), 15 spec call sites fail an assertion until
  their grid is declared `rowDirection: 'north'`, and 3 spec call sites are expected to pass either
  way.
- No oracle fixture has to be regenerated. Three lens records, their 16 twins and 3 website view
  models have to be, because the three contour lenses change their input.
- No class of the flow family negates `v`. Three other defects were found on the way (section 7).

## 1. What the terrain classes did

`git diff HEAD -- modules/analysis/src/gpu-terrain` adds `resolveTerrainRowDirection(id, grid,
rowDirection)` in `gpu-terrain/terrain-curvature/terrain-geomorphometry-utils.ts:41`. It reads only
`grid.affine[4]`: negative means `'south'` (row 0 is the northern row), positive means `'north'`,
zero means the affine states nothing and the explicit value or `'south'` is used. An explicit
`rowDirection` that disagrees with a non-zero row term throws at construction. Seven classes and
the point-horizon march call it once in the constructor (`this.rowDirection = ...`) and compile the
result into the kernel as a constant (`getRasterRowNorthSignWGSL`, `analysis-core/raster-grid.ts:188`).
Nothing is decided per frame and no packer changed.

The affine-following raster siblings do it differently: `raster-sampling/raster-sampling-parameters.ts:59-99`
reads the whole affine in the packer and writes a signed row reciprocal and the y of the row 0
edge, so the direction is a per-frame parameter. Both precedents exist in the library.

## 2. The eight classes

### 2.1 `GPUIsolines`

- Receives: `grid: GPURasterMetadata` (required; `validateRasterGrid` at `isolines/gpu-isolines.ts:176`
  requires six finite affine values, so there is no caller without an affine) and a per-frame
  `extent: [minX, minY, maxX, maxY]`. Only `grid.width` and `grid.height` are read.
- Row to y: the packer writes `minY` and a positive cell height
  (`isolines/isolines-parameters.ts:72-75`); the kernel places a vertex at
  `minY + (gridY + 0.5) * cellHeight` (`gpu-isolines.ts:992` and `:996`, inside `getEdgeCrossing`,
  `:970`). The corner layout is `v0 (cx, cy)`, `v3 (cx, cy + 1)` (`isolines/isolines-utils.ts:125-128`).
- Orientation depends on it too: a segment starts on a high-to-low exit edge of the
  counter-clockwise walk in grid space (`isolines-utils.ts:141-148`), which keeps the high side on
  the left only while rows advance toward larger y. Segments and their edge ids are written at
  `gpu-isolines.ts:417-428`.
- Smallest change: resolve the direction once in the constructor from `grid.affine[4]`. For
  `'south'`, (a) the y expression becomes `minY + (f32(HEIGHT) - 0.5 - gridY) * cellHeight` in the two
  lines above, and (b) the scatter swaps `start` and `end` and the two edge ids, so the documented
  "high side on the left (x right, y up)" stays true in world space. The stitching pass reads the
  stored `[startEdge, endEdge]` pairs (`gpu-isolines.ts:545-548`), so it needs no change. The packer
  keeps its layout. Shared vertices stay bit-identical between neighbouring cells because both
  cells evaluate the same expression on the same canonical edge.
- North-up grid (negative row term): output changes. Example: `makeGPURasterMetadata({width: 2,
  height: 2})`, values `[10, 10, 0, 0]` (row 0 first), level 2.5, extent `[0, 0, 2, 2]`. Today the
  one segment is `[1.5, 1.25, 0.5, 1.25]` (row 0 drawn at the bottom). Following the affine it is
  `[0.5, 0.75, 1.5, 0.75]`: row 0 at the top, and the endpoints swapped so the high side (north)
  is still on the left.
- Grid with a positive row term (`rowDirection: 'north'`, or the literal `[1, 0, 0, 0, 1, 0]`):
  unchanged, provided the generated WGSL for this case is left byte-identical.

### 2.2 `GPUIsobands`

- Receives the same `grid` and extent. Packer: `isolines/isobands-parameters.ts:85-90`.
- Row to y: `getWorldPosition`, `isolines/isobands-wgsl.ts:62-66` (y at `:64`), used for every corner
  and crossing.
- Orientation: triangles are documented counter-clockwise and boundary edges keep the band on the
  left. They are written by `writeTriangle` (`isolines/gpu-isobands.ts:418`) and `writeEdge` (`:543`).
- Smallest change: the same resolved direction; for `'south'` the y expression uses
  `params[7] - (gridY + 0.5) * params[9]` (`maxY` is already packed in slot 7), `writeTriangle`
  writes `a, c, b` and `writeEdge` writes `end, start`. `bandClasses` is per sample in raster order
  and does not change.
- North-up grid: output changes (mirrored, windings preserved in world space). Positive row term:
  unchanged.
- A second writer of this parameter layout exists: `createIsochronesBandParametersNode`
  (`gpu-network/network-isochrones/network-isochrones-passes.ts:33-67`) fills the isoband parameters
  on the GPU. It stays valid if the layout stays as it is, which is one reason to keep the packers
  unchanged.

### 2.3 `GPUIsobandRings`

- Receives `grid` and passes it to an inner `GPUIsobands` (`isolines/gpu-isoband-rings.ts:180-189`);
  ring assembly runs with `interiorSide: 'left'` (`:198`).
- Smallest change: none beyond forwarding an optional `rowDirection`. Because `GPUIsobands` swaps
  its edges for `'south'`, the band stays on the left and `interiorSide: 'left'` stays right. Shells
  stay counter-clockwise in world space.

### 2.4 `GPULineIntegralConvolution`

- Receives: `grid` for the output raster (size only is read, `flow-texture/gpu-line-integral-convolution.ts:146`,
  `:182`), `fieldWidth` and `fieldHeight`, and two per-frame bounds tuples `fieldExtent` and
  `outputExtent` (`flow-texture/line-integral-convolution-parameters.ts:19-22`). The field has no affine.
- Row to y, output: pixel centre at `:297`, visited pixel at `:321` and its noise index at `:326`;
  the noise is keyed by storage `(column, row)` at `:213`.
- Row to y, field: the shared sampler, `vector-field-common/vector-field-sampling.ts:43`
  (`local = (position - extent.xy) / extent.zw`) and `readFieldVelocity` (`:33-36`).
- `v` and the advection direction: `getDirection` (`:270-280`) normalises the sample and the steps
  at `:316` and `:320` add it to the world position. `v` is the velocity toward larger y. Nothing
  depends on the row order except which row the sampler reads.
- Smallest change: resolve the output row direction from `grid.affine[4]` in the constructor; for
  `'south'` the row used at `:297` and the row index at `:326` become `HEIGHT - 1 - row`. The field
  stays on the bounds rule.
- North-up output grid: the output raster rows are reversed (row 0 becomes the largest y). The
  noise is keyed by storage row, so the texture is not the mirror image of today's; it is a
  different realisation of the same statistics. Positive row term: unchanged.

### 2.5 `GPUStreamlines`

- Receives: `fieldWidth`, `fieldHeight`, and per-frame `fieldExtent` and `gridExtent`
  (`flow-texture/streamlines-parameters.ts:27-33`). No grid, no affine. The occupancy grid is
  internal and its cell sizes must be positive (`:63-66`).
- Row to y: the shared sampler only (`flow-texture/gpu-streamlines.ts:380`, `:389`). Advection:
  `:432-444`, in world space. Output is world points.
- Smallest change: none. Bounds only, so today's output is kept.

### 2.6 `GPUParticleAdvection`

- Receives: `fieldWidth`, `fieldHeight`, per-frame `fieldExtent` and optional `spawnBounds`
  (`particle-advection/particle-advection-parameters.ts:20-37`). No grid.
- Row to y: the shared sampler (`particle-advection/gpu-particle-advection.ts:312`). Advection:
  `:338-342`, in world space; respawn at `:358`.
- Smallest change: none. Bounds only.

### 2.7 `GPULagrangianFlowMap`

- Receives: `fieldWidth`, `fieldHeight`, per-frame `fieldExtent`
  (`lagrangian-diagnostics/gpu-lagrangian-flow-map.ts:41-47`), seeds as world positions, optional
  `regions` raster on the field layout.
- Row to y: the shared sampler for both slices (`:327-328`), `isInsideField` (`:345-349`) and
  `getRegion` (`:353-361`), all through the same `local` expression. Advection: `:398-408`.
- Smallest change: none. Bounds only.

### 2.8 `GPUFiniteTimeLyapunovExponent`

- Receives: `positions` of a `seedWidth x seedHeight` lattice and two positive spacings. No
  placement at all.
- Row order enters only through `gradientY = (north - south) / (rows * spacingY)`
  (`lagrangian-diagnostics/gpu-finite-time-lyapunov-exponent.ts:189`). Reversing the rows negates
  `gradientY` exactly, which negates `b` (`:191`) and leaves `a`, `d` and `b * b` unchanged, so the
  exponent (`:193-194`) at a given seed is the same. The class has no row direction to follow.
- Smallest change: none in the kernel. Its documentation is wrong (section 7, defect 1).

### 2.9 Adjacent: `GPUTerrainFlowField`

Not one of the eight, but it is the only place where the sign of `v` depends on the row order. It
writes velocities in the raster frame, `v` along increasing rows
(`gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.ts:86-92`, `:133-135`), and does not call
`resolveTerrainRowDirection`. On a north-up grid its `v` points south. It matches the flow family
only when the particles run in raster coordinates or the raster is stored south-first, which is
what its three callers do. The flip does not change it; the rule should stay written down.

### 2.10 Are there callers with no affine?

For the three contour classes and the LIC output: no, the grid is required and validated. The
nearest case is a row term of exactly 0, which should keep today's rule (row 0 at the smallest y).
For the fields of the flow family: every caller, because there is no way to pass an affine.

## 3. Call sites

Counted: `new <Class>(` in executable code. Code shown to the reader as a string (6 in
`website/src/catalog/entries/terrain-hydrology-flow.ts`, 8 in four showcase `*.scene.ts` files) and
6 in `docs/api` are listed separately. `modules/kepler` has none. Website previews do not construct
any of the eight classes.

| Where | Contours | LIC | Streamlines, advection, flow map, FTLE | Total |
| --- | --- | --- | --- | --- |
| `modules/analysis/src` | 3 | 0 | 0 | 3 |
| `modules/analysis/test` (32 files) | 34 | 9 | 30 | 73 |
| `modules/lenses/src` | 3 | 1 | 2 | 6 |
| `website/src/gpu/runs` | 4 | 2 | 5 | 11 |
| `examples/deck` | 6 | 2 | 7 | 15 |
| Total | 50 | 14 | 44 | 108 |

Outcome with the recommended design (section 5):

| Outcome | Count | Which |
| --- | --- | --- |
| Unchanged | 76 | All 44 field-only sites; 3 internal; 3 explorer contours; 1 lens LIC; 25 spec sites that only construct or audit |
| Picture mirrored, CPU row reversal to remove | 10 | 3 lens contours, 4 website contours, 3 showcase contours |
| Output raster rows reversed, grid to re-declare | 4 | 2 website LIC, 1 explorer LIC, 1 showcase LIC |
| Spec assertion breaks | 15 | 12 contour, 3 LIC |
| Expected to pass either way (not run) | 3 | `isolines-isobands-consistency.spec.ts` (2), `isobands-benchmark.spec.ts` (1) |

### 3.1 `modules/analysis/src`

| Call site | Grid passed | Compensation | After the flip |
| --- | --- | --- | --- |
| `gpu-network/network-isochrones/gpu-network-isochrones.ts:697` (`GPUIsobands`), `:729` (`GPUIsobandRings`) | Literal affine `[1, 0, 0, 0, 1, 0]` (`:629-634`), row term positive | None; its own rasteriser writes row 0 at `minY` | Unchanged |
| `raster-analysis/isolines/gpu-isoband-rings.ts:180` | The caller's grid | None | Follows the caller |

### 3.2 `modules/lenses/src`

All six are in `computation-lenses/scenarios/gpu-raster/`.

| Call site | Grid or extent | Compensation today | After the flip |
| --- | --- | --- | --- |
| `isolines.example.ts:49`, data from `isolines.lens.ts:41-45` | North-up DEM grid (negative row term) | Rows reversed on the CPU (`southUp`); input descriptor says `rowDirection: 'north'` (`isolines.lens.ts:169`) | Mirrored. Remove the reversal, upload `terrain.elevation`, change the descriptor and the comments |
| `isobands.example.ts:45`, `isobands.lens.ts:41-45`, `:126` | Same | Same | Same |
| `isoband-rings.example.ts:46`, `isoband-rings.lens.ts:41-45`, `:126` | Same | Same | Same |
| `line-integral-convolution.example.ts:55`, `line-integral-convolution.lens.ts:48-63` | Output grid built with `rowDirection: 'north'`; field bounds `[west, south, cell, cell]` | Field rows reversed on the CPU (`:61`); `v` not negated | Unchanged |
| `streamlines.example.ts:57`, `streamlines.lens.ts:42-48` | Bounds only | Field rows reversed on the CPU | Unchanged |
| `particle-advection.example.ts:65`, `particle-advection.lens.ts:44-50` | Bounds only | Field rows reversed on the CPU | Unchanged |

`gpu-raster/computation-pack/operators/isolines.ts:44-56` binds the formula to the WGSL names
`segments`, `t`, `level`, `a`, `b`; the kernel edit must keep them.
`gpu-raster/computation-pack/operators/vector-field.ts:39` states "row 0 at the smallest y", which
stays true.

### 3.3 `website/src`

| Call site | Grid or extent | Compensation today | After the flip |
| --- | --- | --- | --- |
| `gpu/runs/entries/terrain-hydrology-flow/isolines.ts:40`, `:61` | `makeGPURasterMetadata({width, height})`, north-up (`shared/iso.ts:50`); extent `[0, 0, w, h]` | A second, row-reversed copy of the field (`valuesBottomUp`, `shared/iso.ts:38-46`) | Mirrored. Upload `valuesTopDown` to the operation and delete `valuesBottomUp` |
| `.../isobands.ts:38` | Same | Same (`:31`) | Same |
| `.../isoband-rings.ts:65` | Same | Same (`:59`) | Same |
| `.../line-integral-convolution.ts:35` | Output grid north-up by default (`:39`); both extents `[0, 0, c, c]` | Field kept north-first and `v` negated (`shared/vector-field.ts:52-56`, mode `'raster'`): the run works in a y-down frame | Output rows reversed against the field. Declare the output grid `rowDirection: 'north'`, or move the run to world coordinates |
| `.../terrain-flow-field.ts:58` | Output grid north-up by default (`:62-65`); terrain grid `'north'` (`:31-36`) | None, raster frame | Output rows reversed. Declare the output grid `rowDirection: 'north'` |
| `.../streamlines.ts:37`, `.../particle-advection.ts:58` | Bounds only | Field rows reversed on the CPU (`shared/vector-field.ts:52`, mode `'world'`) | Unchanged |
| `gpu/runs/entries/raster-embeddings-flow/lagrangian-flow-map.ts:68` | Bounds only | Field rows reversed (`_lib/wind.ts:31`) | Unchanged |
| `.../lagrangian-finite-time-lyapunov-exponent.ts:52`, `:64` | Bounds only | Field rows reversed; the FTLE raster is reversed again for drawing (`addRowFlip`, `:73`) | Unchanged |

`catalog/entries/terrain-hydrology-flow.ts:1612`, `:1700`, `:1786`, `:1891`, `:1968`, `:2047` are
usage strings. They compile nothing and stay correct.

### 3.4 `examples/deck`

| Call site | Grid or extent | Compensation today | After the flip |
| --- | --- | --- | --- |
| `spatial-analysis-explorer/modes/contours-mode.ts:464`, `:507`, `:627` | `flippedGrid`, built with `rowDirection: 'north'` (`:233-239`) | The analysis runs in a y-negated frame and the layers scale y by -1 (`:222-229`) | Unchanged. The y-negated frame can be removed later |
| `spatial-analysis-showcase/src/scenes/hydrology/contours.compute.ts:228`, `:270`, `:365` | `grid.rasterGrid`, north-up (`b15-common.ts:112-117`) | Elevation reversed on the CPU (`:123-126`) | Mirrored. Pass `southFirstGrid`, already in scope at `:110`, to the three constructors and two packers (`:523-525`) |
| `spatial-analysis-explorer/modes/flow-field-mode.ts:261` | Output grid north-up by default (`:266`); output extent from the south-west corner (`:390-395`) | Field rows reversed (`:818-830`) | Output rows reversed. Declare the grid `rowDirection: 'north'` |
| `spatial-analysis-showcase/src/scenes/raster/wind-flow.compute.ts:371` | Output grid north-up by default (`:377`); output extent from the south (`:493-497`) | Field rows reversed (`:146-147`) | Same edit |
| `flow-field-mode.ts:210`, `:294`; `wind-flow.compute.ts:290`, `:335`; `drainage-mode.ts:360`; `terrain-flow.compute.ts:207`; `ocean-drifters-vs-model.compute.ts:265` | Bounds only | Field rows reversed on the CPU in every one | Unchanged |

`drainage-mode.ts:317` passes a north-up grid to `GPUTerrainFlowField` over south-first data. In
`'uniform'` mode the class does not read the row direction, so the result is right, but the grid
is mis-declared.

### 3.5 `modules/analysis/test`

Every spec that builds a grid for a contour class or for LIC uses
`makeGPURasterMetadata({width, height})` or `makeGPURasterMetadata(scene)`, which is north-up, and
then asserts the smallest-y rule. None passes a negative cell height to a flow class.

Assertion breaks, cured by `rowDirection: 'north'` in the grid (fixtures stay as they are):

| Spec (under `modules/analysis/test/gpu-raster/`) | Grid line | Sites |
| --- | --- | --- |
| `isolines/gpu-isolines.spec.ts` | 72 | 1 |
| `isolines/gpu-isobands.spec.ts` | 100 | 2 |
| `isolines/gpu-isoband-rings.spec.ts` | 108, 181 | 1 |
| `isolines/isolines-saddle-rule.spec.ts` | 64 | 2 |
| `isolines/parity-wp4-isolines.parity.spec.ts` | 36 | 1 |
| `isolines/parity-wp4-stitched.parity.spec.ts` | 38 | 1 |
| `isolines/parity-wp4-isobands.parity.spec.ts` | 32 | 1 |
| `isolines/parity-wp4-isoband-edge-modes.parity.spec.ts` | 31 | 1 |
| `isolines/parity-wp4-isoband-rings.parity.spec.ts` | 29 | 1 |
| `isolines/parity-wp4-ring-modes.parity.spec.ts` | 40 | 1 |
| `flow-texture/gpu-line-integral-convolution.spec.ts` | 104 | 1 |
| `flow-texture/parity-wp4-lic.parity.spec.ts` | 53 | 1 |
| `flow-texture/parity-wp4-lic-modes.parity.spec.ts` | 84 | 1 |

Expected to pass either way: `isolines/isolines-isobands-consistency.spec.ts:59` (compares the two
classes with each other) and `isolines/isobands-benchmark.spec.ts:47`. Give them the same grid
edit so they keep testing what they test today.

Unchanged (construct only, or audit the kernels): `isolines/gpu-isolines.node.spec.ts` (6),
`isolines/gpu-isobands.node.spec.ts` (8), `isolines/raster-input-mask.node.spec.ts` (2),
`flow-texture/gpu-line-integral-convolution.node.spec.ts` (5),
`analysis-core/kernel-audit/cases-raster.ts` (4 of its 6 sites; the other 2 are field-only).

Unchanged (field only, 30 sites): the streamlines, particle-advection and Lagrangian specs,
`analysis-core/kernel-audit/cases-mob06-lagrangian-diagnostics.ts`, and
`gpu-terrain/terrain-flow-field/gpu-terrain-flow-field.spec.ts:389`.

Packer-only node specs (`local-origin-precision.node.spec.ts`, the `parity-wp4-*-fixtures`
specs): unchanged as long as the packers keep their layout.

## 4. Committed artefacts

| Artefact | Regenerate? | Why |
| --- | --- | --- |
| `docs/lens-records/gpu-raster.{isolines,isobands,isoband-rings}.json` | Yes (3) | Input row order, input descriptor, segment and ring order and the kernel text all change |
| `docs/lens-records/twins/gpu-raster.{isolines,isobands,isoband-rings}~*.json` | Yes (16) | Derived from those records |
| `website/src/lens/generated/gpu-raster.{isolines,isobands,isoband-rings}.json` | Yes (3) | Derived from those records |
| `docs/lens-records/gpu-raster.{line-integral-convolution,particle-advection,streamlines}.json`, 17 twins, 3 view models | No, on one condition | Records embed kernel WGSL. They stay valid only if the WGSL generated for a positive row term, and the shared sampler, are byte-identical to today's |
| `docs/lens-records/gpu-network.network-isochrones.json`, 6 twins, 1 view model | No, on the same condition | The record embeds the isoband kernels |
| `website/public/runs/{isolines,isobands,isoband-rings,line-integral-convolution,terrain-flow-field}-{dark,light}.webp` | Check, probably no | The run edits are meant to keep the picture. Contours can move by a float32 rounding step. Compare one poster before deciding |
| Oracle fixtures under `modules/analysis/test/oracles/fixtures/` | No | Specs declare the grid the fixtures were generated for |
| `modules/analysis/test/oracles/parity-wp4-showcase-modes.json` | Yes (1) | An inventory of showcase call expressions (`grid: rasterGrid` at lines 386, 462, 469). Rerun `parity-wp4-showcase-modes.py` after the showcase edit. No spec reads it |

## 5. Recommended design

Row order becomes a construction-time fact, resolved the way the terrain classes resolve it. Every
class that takes a `grid` reads the sign of `grid.affine[4]`: `GPUIsolines`, `GPUIsobands`,
`GPUIsobandRings`, and `GPULineIntegralConvolution` for its output raster. A negative row term
puts row 0 at the largest y; a positive one, or exactly 0, keeps today's rule. An optional
`rowDirection` prop may restate the direction and throws when it contradicts a non-zero row term,
with the terrain message. The per-frame extents keep their meaning as bounds with positive sizes,
and the packers keep their layouts, so the direction has one source and the isochrone parameter
writer is untouched. The contour kernels swap segment, edge and triangle order for `'south'`, so
"high side on the left" and counter-clockwise shells hold in world space for both directions.
Field rasters that arrive as a bounds tuple (LIC, streamlines, particle advection, flow map) keep
the smallest-y rule and `v` stays the velocity toward larger y; the documentation says so without
the word "deferred". `GPUFiniteTimeLyapunovExponent` needs no rule. The generated WGSL for a
positive row term stays byte-identical, which keeps four lens records valid.

Optional, in the same change or later: a compile-time `fieldRowDirection` on the four field
classes (one branch in `readFieldVelocity`), so that the 16 call sites that reverse field rows on
the CPU (10 separate field preparations) can stop.

### Order of edits

All of steps 2 to 8 must be in the tree together; between them pictures are mirrored.

1. Specs first, failing: the 2 x 2 isoline case of section 2.1, the same raster through
   `GPUIsobands` (triangle winding) and `GPUIsobandRings` (shell winding), an LIC output on a
   north-up grid, the contradiction throw, and a row term of 0.
2. `analysis-core/raster-grid.ts`: a resolver that takes the fallback as an argument;
   `resolveTerrainRowDirection` becomes a call to it with `'south'`. `raster-analysis` must not
   import from `gpu-terrain`.
3. `isolines/gpu-isolines.ts`, `isolines/isobands-wgsl.ts`, `isolines/gpu-isobands.ts`,
   `isolines/gpu-isoband-rings.ts`: the changes of sections 2.1 to 2.3, class and prop documentation.
4. `flow-texture/gpu-line-integral-convolution.ts`: output row direction. Its packer rejects
   non-positive cell sizes.
5. Documentation of the other flow classes and packers; the three field packers validate their
   extents as the flow-map packer does (section 7, defect 2); the FTLE paragraph is corrected.
6. Specs: `rowDirection: 'north'` in the grids of the 15 spec files of section 3.5 (13 that break,
   2 that should keep their meaning); `isolines-oracle.ts`,
   `isobands-oracle.ts` and `oracles/line-integral-convolution-cpu.ts` take the direction for the
   new cases.
7. `yarn build` (the website reads the analysis `dist`), then the lenses (3 lens files, 3 example
   files), the record regeneration, and the website (`shared/iso.ts`, the three contour runs, the
   two LIC grids).
8. Examples: `contours.compute.ts`, `wind-flow.compute.ts`, `flow-field-mode.ts`; rerun the
   showcase inventory generator.
9. `docs/api/gpu-raster/operations-analysis.md` (lines 118-126, 788-806, 894, 913, 1220-1221),
   `docs/api/gpu-raster/lagrangian-diagnostics-mob06.md` (lines 25-29), `behaviour-changes.md`, and
   the row of roadmap 12.4.

### Size

| Area | Files |
| --- | --- |
| `modules/analysis/src` with code changes | 7 (resolver, terrain helper, 4 contour files, LIC) |
| `modules/analysis/src` documentation or validation only | 10 (3 packers, the sampler, 4 flow classes, 2 contour packers) |
| `modules/analysis/test` | about 20 (15 spec files with grid edits, 3 twins, 1 or 2 new specs) |
| `modules/lenses/src` | 6 |
| `website/src` | 6 |
| `examples/deck` | 3 |
| Documentation | 5 |
| Regenerated artefacts | 23 (3 records, 16 twins, 3 view models, 1 inventory) |

About 57 hand-edited files, most of them one line.

## 6. Alternative considered: a signed cell height

The flow kernels compute everything from `(position - origin) / cell`, so a tuple
`[originX, maxY, cellWidth, -cellHeight]` already describes a north-up field or output raster
without any kernel change. I ran the CPU twins, which mirror the WGSL, on a 12 x 9 field stored
both ways (script: scratchpad `row-direction/signed-row-step-check.ts`):

- sampler: 4,000 positions, no validity mismatch, largest difference 9.5e-7;
- LIC with the field given both ways: 2,592 pixels, identical values;
- LIC with a negative output cell height: speeds equal to the mirrored baseline within 4.8e-7;
- particle advection, 200 particles, 30 frames: same respawns, positions within 7.6e-6.

Two visible differences: the half-open edge moves (`y = minY` is inside with a positive step,
`y = maxY` with a negative one), and default spawn bounds come out as `[minX, maxY, maxX, minY]`,
which the kernel handles. This was not run on a GPU.

Three packers accept such a tuple today without saying so, and the flow-map packer rejects it.
The alternative is cheaper in code, but it gives LIC two sources for its output direction (the
grid and the sign) with no place to compare them at construction, and it makes the flow extents
mean something different from the contour extent. The recommendation therefore closes this door
(step 5) instead of documenting it. If the lead prefers the signed tuple, the changes are: accept
a non-zero `cellHeight` in the flow-map packer, document the tuple as `[c, f, a, e]` of an
axis-aligned affine, and leave LIC's `grid` as a size.

## 7. Defects noticed (not fixed)

1. **`GPUFiniteTimeLyapunovExponent` documentation states a requirement that does not exist.**
   The class documentation (`gpu-finite-time-lyapunov-exponent.ts:100-103`) and
   `lagrangian-diagnostics-mob06.md:27-29`, both written in the first pass, say positions from a
   north-up seed grid "must be reordered by the caller". The exponent does not depend on the row
   order (section 2.8). Input: a 2 x 2 seed lattice with spacing 1 mapped by `(x, y) -> (x + y, y)`,
   `integrationTime` 1. Rows south-first: `gradientX = (1, 0)`, `gradientY = (1, 1)`, `a = 1`,
   `b = 1`, `d = 2`. Rows north-first: `gradientY = (-1, -1)`, `b = -1`. Both give
   `largest = 1.5 + sqrt(1.25) = 2.618` and an exponent of 0.4812 at every seed. Output row `r`
   belongs to seed row `r` in either order.
2. **The four field packers do not validate the same things.**
   `getGPULagrangianFlowMapParameterValues` rejects a non-positive or non-finite cell size
   (`gpu-lagrangian-flow-map.ts:80-86`). `getGPULineIntegralConvolutionParameterValues`,
   `getGPUStreamlinesParameterValues` (for `fieldExtent`) and `getGPUParticleAdvectionParameterValues`
   check nothing. Input: `getGPUParticleAdvectionParameterValues({fieldExtent: [0, 0, 0, 1],
   timeStep: 1}, [4, 4])` returns normally; in the kernel `local.x` is then infinite or NaN, every
   sample is invalid and every particle respawns each frame. LIC with the same tuple writes NaN
   to every pixel. The same three accept `[10, 127, 2, -3]` and place the field north-up, which
   the flow-map packer refuses (section 6).
3. **Showcase snippets name the wrong package.** The code shown to the reader in
   `examples/deck/spatial-analysis-showcase/src/scenes/hydrology/contours.scene.ts:438-441`,
   `raster/wind-flow.scene.ts:490`, `hydrology/terrain-flow.scene.ts:382` and
   `earth/ocean-drifters-vs-model.scene.ts:449` imports these classes from
   `'@luma.gl/graph/gpu-raster'`. Published `@luma.gl/graph` 10.0.0-alpha.12 does not export them;
   they come from `@luma-contributors/analysis/raster-analysis`.

Checked and found consistent: none of LIC, streamlines, particle advection or the flow map negates
`v`. All four use the one sampler and add the sample to the world position.

## 8. Not verified

- Nothing was run on a GPU. The per-class statements come from reading the kernels; section 6 comes
  from the CPU twins.
- The three "expected to pass either way" spec sites were not run against a changed kernel.
- Whether the posters change by a visible amount.
- How the lens harness (`raster-lens-harness`) uses `grid` for thumbnails once the contour lenses
  stop reversing rows. I read the three lens files and their examples, not the harness.
- Website and example pictures were judged from the code, not rendered.
