# Live operation runs

A **run** is the live picture of one catalog entry: it builds the entry's real operation from
`@luma-contributors/analysis` into a command graph, runs it on the GPU in the browser, and draws the
buffers the operation wrote with luma.gl. Sliders are bound to the operation's real props, so the
picture answers the reader. One run is about 30 to 80 lines.

```
src/gpu/runs/
  define-run.ts    defineOperationRun(spec), layer descriptors, RunBuilder
  interaction.ts   pointer gesture contract, glyphs, bindAxes and aimAround
  pointer-overlay.ts  draws a gesture's glyphs and the probe spotlight
  pointer-driver.ts   the same gesture behaviour for the hand-written demos
  pick.ts          CPU lookups under the pointer: cell, point, segment, polygon, chart
  inputs.ts        observation samples, scenario inputs and upload helpers
  record-demo.ts   shared by the computation-record runs: a recorded relaxation, graph boxes, small multiples
  layers.ts        luma.gl layers the demo kit lacks (polygons, typed edges, charts)
  registry.ts      getRunScene / getRunParameters / hasRun, by entry id
  harness.ts       browser side of the verification harness
  entries/<group>/<entry-id>.ts              the scene (imports luma.gl and the operation)
  entries/<group>/<entry-id>.parameters.ts   the sliders (no luma.gl imports)
```

`<group>` is the catalog file name (`spatial-statistics`, `raster-bands-indices`, ...) and
`<entry-id>` is the `id` in `src/catalog/entries/<group>.ts`. The file name **is** the id.

## Writing a run

1. Read the operation's source in `modules/analysis/src` and a GPU test in `modules/analysis/test`
   that constructs real inputs for it. Adapt the test; keep the entry's own `example` in mind.
2. Create `entries/<group>/<id>.parameters.ts` (skip it only if the run has no sliders).
3. Create `entries/<group>/<id>.ts` with `export default defineOperationRun({...})`.
4. Verify (below), read the screenshots in `site3/runs-report/<group>/`, fix.

### Parameters file

```ts
import type {GpuSceneParameter} from '../../../types';

export const parameters: GpuSceneParameter[] = [
  {id: 'bandwidth', label: 'bandwidth', min: 1.5, max: 12, step: 0.25, default: 6, unit: '%'},
  {id: 'kernel', label: 'kernel', min: 0, max: 2, step: 1, default: 0,
   options: ['gaussian', 'quartic', 'epanechnikov']}
];
```

The file must export `parameters` and import nothing but types. Prerendered pages read it through
`getRunParameters(id)` so the slider strip is in the static HTML; the scene (luma.gl, the analysis
operation) loads lazily. Rules:

- **Bind sliders to real props.** Use the prop name as `id` and `label` (for example `normalize`,
  `epsilon`, `facilityCount`), so the reader sees the same name as in the Parameters table. A
  parameter that exists only to vary the data is labelled `Data: noise`.
- Enumerated props use `options` (value is the index). Booleans are `options: ['false', 'true']`.
- Two to four parameters. Every slider must visibly change the picture (the verifier warns about
  inert ones). Defaults must give a good-looking picture.

### The spec

```ts
defineOperationRun<Inputs, Outputs>({
  parameters,                                  // from the sibling file
  createInputs({random, world, aspect, seed}) { return inputs; },
  build(builder, inputs, values) { return {outputs, update}; },
  draw: [...layers] | ({inputs, outputs, values, theme, world}) => [...layers],
  rebuildOn: ['kernel'],                        // optional, see below
  caption: 'One line.'                          // optional
});
```

**`createInputs`** returns small CPU typed arrays. Related operations sample the same checked-in
extracts in `../data`, using a deterministic seed where subsampling is needed. The drawing world is
`{width: aspect, height: 1}`, with x right and y up. Keep geographic inputs in the operation's
coordinate system and use a common transform for layers drawn together.

**`build`** runs once at start (and again on rebuild). It receives a `RunBuilder`:

- `builder.upload(id, format, typedArray)` creates a storage buffer and imports it into the graph.
- `builder.allocate(id, format, length)` creates a zero-filled output buffer of `length` rows.
- `builder.graph.add(new TheOperation({...}))` adds the real operation. Pass `buffer.view` wherever
  the operation wants a `GraphDataView`.
- Both return a `RunBuffer`: `{buffer, view, format, length, write(typedArray)}`.
- Formats: `float32`, `uint32`, `sint32`, and `float32x2|x3|x4`, `uint32x2|x3|x4`. Positions are
  `float32x2`, edges `uint32x2`, segments `float32x4`.

`build` returns `{outputs, update?}`. `outputs` is any object of `RunBuffer`s that `draw` shows (inputs
you want to draw go here too). `update(values)` writes the current slider values into buffers (the
operation's per-frame parameter view, a threshold, congestion-scaled costs) and is called before every
run. Writing buffers is cheap; **prefer per-frame props**: the operation re-runs without recompiling.
Props that the operation fixes at construction (a kernel, a method, `maximumIterations`) are
compile-time; list their parameter ids in `rebuildOn` and read them from `values` inside `build`.
Without an `update`, every parameter change rebuilds. The graph runs once per change, not per frame.

**`afterRun`** (optional, async) is for operations whose output is not a buffer the graph wrote but
something read back from it or recorded about it: a snapshot arena, a described command graph, a CPU
summary from a readback ring. It runs after the first run and coalesces later runs as described below.
It receives `{device, compiled, inputs, outputs, values, upload, allocate}` and returns any value, which
`draw` receives as `extra`. Draw it by uploading CPU data with `state.upload(id, format, typedArray)` (same
formats as `builder.upload`) and putting the returned `RunBuffer`s in `extra`. Those buffers belong to
that result and are destroyed when the next one replaces it, so create them inside `afterRun`, every
time. One `afterRun` call is in flight at a time: runs that finish meanwhile get one follow-up call
for the latest run, so under a moving pointer it is called less often than the graph runs, and the
run that ends a burst always gets its call. The framework drops a result whose build a rebuild
replaced and keeps the previous `extra` until the new one is ready. The harness `settle()` waits up to
400 animation frames for runs, rebuilds, `afterRun`, peeks and pointer settlement. If work is still
pending, it logs a console warning, reads the ranges of the latest committed buffers, draws that
state and resolves even though pending results may be missing from the capture. That final range
read (including any follow-up) is still awaited without a frame bound. Destruction stops settlement
without further drawing. Runs without `afterRun` are unchanged (`extra` is `undefined`).

```ts
defineOperationRun({
  build(builder, inputs, values) {
    // Resources that must be cleaned up with the build (arena, readback ring): hand them over.
    const arena = builder.own(new GPUSnapshotArena(builder.device, {slots, capacity: values.capacity}));
    // ... add the operation and a GPUSnapshotTap to builder.graph ...
    return {outputs: {arena}};          // no `update`: every slider rebuilds, so each run has a fresh arena
  },
  async afterRun({outputs, upload}) {
    const captures = await outputs.arena.read();             // async readback of what the run did
    return {frames: upload('frames', 'float32', flatten(captures))};
  },
  draw: ({extra}) => [{layer: 'chart', kind: 'bars', values: extra.frames}]
});
```

`builder.own(resource)` hands any object with `destroy()` to the framework; it is destroyed with the
build's buffers on rebuild and on destroy. A stateful resource that a run consumes (an arena that
`read()` closes) needs a rebuild per run, so leave `update` out. Examples: the six
`computation-*` and `snapshot-*` runs and `region-statistics-readback`. A build that owns a resource
is not kept by `buildCache`; pass `builder.own(resource, {reusable: true})` for one no run changes
(an operation that owns a constant plan buffer, as in `geometry-processing/local-frame-projection`).

**`draw`** lists layers in paint order: input context first, then the operation's output. Use a
function when layers depend on `values` (ranges, counts, which ramp).

### Pointer interaction

Every run declares what the pointer **is** for its operation. On a card the gesture starts on
hover; on the function page the same gesture runs on the hero and its values move the sliders.
Without `interaction` a card falls back to scrubbing one slider along x, which the verifier reports.

```ts
defineOperationRun({
  ...,
  interaction: {
    kind: 'place',                 // place | aim | reach | level | window | swipe | probe | drag | time | pad
    hint: 'move the observer',     // two or three words, shown before the reader's first move
    rest: [0.3, 0.55],             // pointer position (0..1 across the canvas, y down) of the poster
    resolve(pointer, {inputs, outputs, values, world, pixelSize, peek}) {
      return {values, settle, glyphs, readout, curtain, focus, rerun};   // all optional
    }
  }
});
```

`resolve` is called at most once per frame with `pointer.x`, `pointer.y` in drawing-world units (y
up; degrees with `geographicWindow`) and `pointer.unitX`, `pointer.unitY` across the canvas (y down).
The position is the pointer itself, except for a short glide from `rest` when the pointer first
arrives. `resolve` is also called again, at the same position, after a slider change, after a
rebuild (the buffers in `outputs` are then new: compare `state.build` before skipping a rewrite of
a render-only buffer) and when a `peek` readback arrives. It must be cheap. What it returns:

| field | effect |
| --- | --- |
| `values` | Parameter values by id, applied exactly like a slider: `update` writes them and the operation re-runs. Only per-frame parameters. A compile-time id here fails verification |
| `settle` | Compile-time (`rebuildOn`) parameter values. Applied only after the pointer has rested, so the graph never recompiles under a moving pointer. Snap to at most five values and set `buildCache: 5` on the spec so revisits are instant |
| `glyphs` | Marks drawn over the picture in the ink color with a halo: `crosshair`, `dot`, `ring` (world radius), `box`, `line`, `arrow`, `curtain`, `polyline`. Positions are world units. They are also drawn in the poster, at `rest` |
| `readout` | One short line for the chip, built from real prop names, values and units. Cards truncate at about 34 characters, so lead with what changes |
| `curtain` | World x for a swipe; read it in `draw` as `interaction.curtain` and give layers `clip: {side: 'left' \| 'right', x}` |
| `focus` | The hovered element, for `draw` (`interaction.focus`) |
| `rerun` | `true` when `resolve` changed something `update` reads (see "Pointer-owned inputs") |
| `spotlight` | A bright circle `{at, radius}` in world units; the rest of the picture dims while the reader points. A `probe` gets one around its glyphs automatically; `null` turns it off |

`draw` receives `interaction` (the latest result) and `pointer`.

Rules:

- The pointer drives a **real prop or a real input** of the operation, or reveals structure the
  operation computed. Never a `Data:` noise knob.
- `rest` must reproduce the default parameter values, so the poster, the first live frame and the
  sliders agree. The verifier warns when it does not.
- Use both axes when y has a meaning. One gesture, no modes.
- Snap to cells, nodes and features, and use `chooseSticky` so the choice does not flicker.
- Draw what the hand controls: a handle, a ring of the radius, the box, the curtain.

**Shorthands** (`../../interaction`): `bindAxes(parameters, {kind, hint, x, y})` maps the canvas to
one or two parameters (inner 90% spans the range; `{parameter, scale: 'logarithmic'}` for props that
span decades) and derives `rest` from their defaults. `aimAround(parameters, {hint, angle,
distance})` maps the compass angle around an anchor to one parameter and the distance to another.
Both derive `rest` for you.

**Pickers** (`../../pick`), all on CPU arrays the run already holds: `getRasterCell`,
`getRasterCellCenter`, `createPointIndex` (`nearest`, `within`), `chooseSticky`,
`findNearestSegment`, `findNearestOnPolyline`, `findNearestEdge`, `findPolygonAt` (triangle mesh),
`isInsideRing`, `getChartPosition`, `getChartWorldX`. Build indexes once in `createInputs`.

**Reading the operation's output.** `peek(outputs.buffer)` returns the latest CPU copy of a buffer
(a typed array, or `null` until the first readback) and subscribes it: it is read back after every
run and `resolve` is called again when it arrives. Use it for readouts (`visible 34%`) and for
probes that follow GPU results (the links, label or path of the hovered element).

**Pointer-owned inputs.** When the pointer is an input that has no slider (one extra query point, a
dragged vertex, a source cell), keep it in a mutable field of `inputs`, have `update` write the
buffer from that field, and return `rerun: true` from `resolve` after changing it:

```ts
createInputs: ctx => ({...createPoints(ctx), query: [0.5, 0.5] as [number, number]}),
build(builder, inputs) {
  const query = builder.allocate('query', 'float32x2', 1);
  ...
  return {outputs: {query, ...}, update: () => query.write(Float32Array.from(inputs.query))};
},
interaction: {
  kind: 'place', hint: 'move the query point', rest: [0.5, 0.5],
  resolve(pointer, {inputs}) {
    inputs.query = [pointer.x, pointer.y];
    return {rerun: true, glyphs: [{shape: 'crosshair', at: inputs.query}], readout: '...'};
  }
}
```

Patterns by gesture, with a finished example of each:

| gesture | how | example |
| --- | --- | --- |
| `place` | cell or node under the pointer to position parameters, or a pointer-owned input | `terrain-relief-light/viewshed.ts` |
| `aim` | `aimAround` | `terrain-relief-light/slope-aspect-hillshade.ts` |
| `reach` | distance from the pointer to the nearest input geometry (`findNearestSegment`) or to an anchor becomes the radius, distance or tolerance; draw a ring or a tick | |
| `level` | the input value under the pointer (read from the CPU inputs) becomes the threshold or level | |
| `window` | the pointer is the center of the box, mask or kernel footprint | |
| `swipe` | `curtain` plus `clip` on the before and after layers, both on one fixed `range` | `raster-bands-indices/fill-nodata.ts` |
| `probe` | pick the hovered element, read its links from `peek` or the inputs, draw them as glyphs | `regression-weights/neighbor-search.ts` |
| `drag` | pointer-owned input: one site, vertex or feature follows the pointer | |
| `time` | `bindAxes` with `x` on the time or iteration parameter | |
| `pad` | `bindAxes` with `x` and `y` on two real props | |

### Layer descriptors

All layers accept `alpha`, `context`, `color` (solid RGBA 0-1). `context: true` marks an **input**
layer: it is drawn muted, and the harness's inputs-only render shows only these. The output layer
must not be `context`, otherwise the verifier fails the run ("output is not drawn").

Value layers (`raster`, `dots`, `segments`, `polygons`, `chart` bars) also take `ramp`
(`'sequential'`, `'diverging'`, `'categorical'`, `'cyclic'`, and `'grey'` for illumination and
hillshade, shadow to lit in both themes; default categorical for `uint32` buffers), `range`
(`[min, max]` or `'auto'`, the default, which reads the buffer back after each run; diverging auto is
symmetric about zero), `logarithmic` (the ramp takes `log2(max(value, 1))`, so it suits counts of 1 and more and `range` is then in log2 units; values under 1 all map to the low end), `bands`, `fade`. Invalid values (NaN, Infinity, unreached
`0xffffffff`) draw in the muted color or transparent. `uint32` values wrap into 8 categories.

| layer | required | draws |
| --- | --- | --- |
| `raster` | `values`, `columns`, `rows` (row 0 at the top) | a grid; `smoothing`, `hillshade`, `elevation` |
| `dots` | `positions` (float32x2); `values?`, `count?`, `radius?`, `sizeByValue?` | circles, solid or by value; with `fade`, low values disappear |
| `segments` | `nodes`, `edges` (uint32x2); `values?` per node | network edges, colored by node value or label |
| `lines` | `segments` (float32x4); `count?` (number or uint32 buffer) | solid segment list |
| `paths` | `nodes`, `ids`, `count` | consecutive node ids as a path |
| `triangles` | `positions`, `triangles`, `count` | triangle wireframe (Delaunay) |
| `polygons` | `positions`, `indices`, `owners`, `values`, `triangleCount` | filled polygons by value |
| `chart` | `kind` (`bars`, `line`, `scatter`), `values`; `xValues?`, `xRange?`, `rectangle?` | scalar and table outputs (histograms, series, scatter) |

`count` may be a GPU-written `uint32` buffer, so the picture shows exactly what the operation
produced. Charts plot a buffer's rows; set `range` for the y axis or leave `'auto'`.

### Text

Pictures can carry labels: a `text` layer, drawn on the GPU from a glyph atlas, so posters and live
pictures show the same words in both themes (`src/gpu/runs/text/`). Add `{layer: 'text', ...}` to
`draw`; the helpers in `../../text` build the usual cases and take the same rectangle as the layer
they label (use `getChartRectangle(world, fractions)` for a `chart` layer's `rectangle`):

| helper | for |
| --- | --- |
| `createChartAxes({rectangle, x, y, references})` | ticks and a short title with its unit; `.grid` goes in its own text layer before the chart, the rest after it |
| `createBarValueLabels({rectangle, values, range})` | a number above each bar (`values` is a CPU copy: `afterRun`, `peek` or the inputs) |
| `createLegend({title, entries})` | a corner legend for classed maps |
| `createMatrixTicks({rectangle, rows, columns})` | row and column names of a matrix |
| `mergeText(a, b)` | several of the above in one layer |

Rules: text is `context: false` by default and is hidden in the inputs-only render; it never counts as
drawn output (the verifier requires another non-context layer). Label only what a reader needs
(few labels: a card is 265 CSS pixels wide, 111 on a phone), take names, values and units from the
operation, and never put information only in a label (the readout says it too). Labels are sized for
the card and thin themselves when crowded. Keep them away from where the gesture draws glyphs and
away from the bottom 30 pixels of a card, where the hover hint covers the picture until the reader
moves. A narrow bar chart names its bars up the bars (`regression-weights/shared/labels.ts`). The
text of a `draw` function runs every render, so cheap CPU copies only; take numbers from `afterRun`.

### Input data

See the [website data notice](../../../../THIRD-PARTY-DATA.md#website-demo-extracts) for sample
sizes, dates, units, preparation scripts and provider terms. `createElevationRaster` and
`createImageryBands` serve different geographic regions. `createTracts` returns Chicago boundaries,
triangles, demographic estimates and rook contiguity. The network and trajectory helpers provide
San Francisco streets and vessel reports. Other helpers sample point observations, time series or
lidar returns. For new extracts, register the data asset in `website/build/demo-data-sources.ts`
and document how to reproduce it. Identify scenario modifications in the caption or provenance.

`uploadSpatialWeights(builder, csr)` gives a `GPUSpatialWeights`. `uploadNetworkTopology(builder,
network, costs?)` gives `{topology, weights}`. Use the `Data:` label prefix for sliders that change
input data.

## Verify

```bash
cd website
node scripts/verify-runs.mjs <entry-id>...            # specific runs
node scripts/verify-runs.mjs --group <group>          # a whole group
node scripts/verify-runs.mjs --themes dark,light      # also render the light theme
```

It starts a harness server (`scripts/harness.vite.config.ts`, port 3301; `--port` to change; reuses
one that is already up), opens `run-harness.html` on real WebGPU, and checks per run: starts on
WebGPU; no GPU validation errors or console errors; canvas not blank; at least one slider changes
the picture (inert sliders are listed as warnings); the picture differs from the inputs-only render.
For the pointer gesture it visits five positions: the picture must change, the graph must not
recompile, and returning to rest must restore the first picture. It warns about a missing
`interaction`, fewer than three distinct pictures, a missing readout and a `rest` that does not
reproduce the defaults.
It writes `.reports/runs/<group>.json`, a contact sheet `<group>.png`, pointer sheets
`<group>-pointer-<page>.png` (eight runs per page, each at three pointer positions with its
readouts), and full-size
`<group>/<id>-<theme>.png` and `<group>/<id>-pointer-<0|1|2>.png`. **Read the screenshots**: passing
checks do not mean the picture is clear or that the gesture explains the operation.
Also typecheck: `npx tsc -p website/tsconfig.json --noEmit`.

Wiring into pages is done by the site, not by runs: `hasRun(id)`, `getRunScene(id)` (a loader for
`<GpuCanvas scene>`) and `getRunParameters(id)` come from `registry.ts`.

## Annotated examples

### 1. Raster: `raster-bands-indices/raster-arithmetic.ts`

Per-frame props only (`update` writes the operation's parameter buffer), so sliders never recompile.

```ts
export default defineOperationRun({
  parameters,
  createInputs: ({random, world}) => createImageryBands(world), // Sentinel-2 NIR, red, ...
  build(builder, imagery) {
    const nearInfrared = builder.upload('nir', 'float32', imagery.nearInfrared);
    const red = builder.upload('red', 'float32', imagery.red);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8); // per-frame props
    const result = builder.allocate('result', 'float32', imagery.values.length);
    builder.graph.add(
      new GPURasterArithmetic({
        cellCount: imagery.values.length,
        a: nearInfrared.view, b: red.view,
        parameters: parameterBuffer.view, output: {values: result.view}
      })
    );
    return {
      outputs: {nearInfrared, result},
      // Sliders -> the operation's real parameter values.
      update: current => parameterBuffer.write(getGPURasterArithmeticParameterValues({...}))
    };
  },
  draw: ({inputs, outputs, values}) => [
    {layer: 'raster', context: true, values: outputs.nearInfrared, columns: inputs.columns, rows: inputs.rows},
    {layer: 'raster', values: outputs.result, columns: inputs.columns, rows: inputs.rows,
     ramp: values.operation === 0 ? 'diverging' : 'sequential', smoothing: true} // the output
  ]
});
```

### 2. Polygon values: `regression-weights/spatial-lag.ts`

`normalize` is a compile-time prop (`rebuildOn`), the noise slider only rewrites an input buffer.

```ts
build(builder, tracts, values) {
  const weights = uploadSpatialWeights(builder, tracts.contiguity);      // GPUSpatialWeights
  const noisy = builder.allocate('values', 'float32', tracts.count);
  const lag = builder.allocate('lag', 'float32', tracts.count);
  builder.graph.add(new GPUSpatialLag({
    values: noisy.view, weights, normalize: values.normalize === 1, output: lag.view
  }));
  return {outputs: {noisy, lag, vertices: ..., indices: ..., owners: ..., centers: ...},
          update: current => noisy.write(addNoise(tracts.values, current.noise))};
},
rebuildOn: ['normalize'],
draw: ({inputs, outputs, values}) => [
  {layer: 'polygons', positions: outputs.vertices, indices: outputs.indices, owners: outputs.owners,
   values: outputs.lag, triangleCount: inputs.triangleCount, range: [0, 1.2]},        // the output
  {layer: 'lines', context: true, segments: outputs.outline, count: inputs.outlineCount},
  {layer: 'dots', positions: outputs.centers, values: outputs.noisy, range: [0, 1.2]} // the input
]
```

### 3. Network: `network-routing-access/network-service-areas.ts`

Per-frame views (`facilityCount`, `costLimit`) are one-row buffers written by `update`; the output is
categorical (`uint32` assignments), so edges take the facility colors with no ramp settings.

```ts
build(builder, {network, facilityNodes}) {
  const {topology, weights} = uploadNetworkTopology(builder, network);
  const facilityCount = builder.allocate('facility-count', 'uint32', 1);
  const assignments = builder.allocate('assignments', 'uint32', network.nodeCount);
  builder.graph.add(new GPUNetworkServiceAreas({
    topology, facilities: builder.upload('facilities', 'uint32', facilityNodes).view,
    facilityCount: facilityCount.view, output: {assignments: assignments.view}, ...
  }));
  return {outputs: {assignments, nodes: ..., edges: ...},
          update: values => { facilityCount.write(Uint32Array.of(values.facilityCount)); ... }};
},
draw: ({outputs}) => [
  {layer: 'segments', context: true, nodes: outputs.nodes, edges: outputs.edges},       // bare roads
  {layer: 'segments', nodes: outputs.nodes, edges: outputs.edges, values: outputs.assignments}
]
```

## Notes

- Runs need WebGPU and throw otherwise; `<GpuCanvas>` then keeps the static fallback.
- Keep runs small: under about 20k elements, a few milliseconds per run. Do not animate unless
  the operation is the animation; the framework redraws only on change.
- `src/gpu/demos/kit.ts` is shared with the flagship demos. Do not change it; ask the framework owner.
- Identifiers are spelled out (`facilityCount`, not `facCnt`), SPDX header on every file.

## Static snapshots and page checks

Function pages and catalog cards show their own operation's output before the live canvas starts.

- `node scripts/snapshot-runs.mjs [entry id...]` renders the catalog's live scenes, including the
  flagship demos, in both themes on WebGPU. Each capture uses the page's input seed, default
  parameters and a fixed reduced-motion frame. The script checks for errors and blank images before
  writing `public/runs/<entry id>-<theme>.webp` (640 px wide, WebP quality 0.62). Regenerate and
  commit both posters when the scene or its source data changes.
- `node scripts/check-all-pages.mjs` serves `dist/client` and opens every `/fn/<id>` page, the home
  page and the how-it-works pages in headless Chromium: the live canvas must reach
  `data-ready=true`, nothing may log an error, and moving a slider must change the picture.
