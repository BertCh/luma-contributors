# Live operation runs

A **run** is the live picture of one catalog entry: it builds the entry's real operation from
`@luma-contributors/analysis` into a command graph, runs it on the GPU in the browser, and draws the
buffers the operation wrote with luma.gl. Sliders are bound to the operation's real props, so the
picture answers the reader. One run is about 30 to 80 lines.

```
src/gpu/runs/
  define-run.ts    defineOperationRun(spec), layer descriptors, RunBuilder
  inputs.ts        seeded input generators and upload helpers
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

**`createInputs`** returns seeded CPU data (plain typed arrays, at most about 20k elements). Use the
generators in `inputs.ts` and the context's `random` (seeded) and `world` (`{width: aspect, height: 1}`;
all positions live in it, x right, y up).

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
summary from a readback ring. It runs after every run (the first one included) with
`{device, compiled, inputs, outputs, values, upload, allocate}` and returns any value, which `draw`
receives as `extra`. Draw it by uploading CPU data with `state.upload(id, format, typedArray)` (same
formats as `builder.upload`) and putting the returned `RunBuffer`s in `extra`. Those buffers belong to
that result and are destroyed when the next one replaces it, so create them inside `afterRun`, every
time. The framework drops the result of a run that a newer run or a rebuild superseded, keeps the
previous `extra` until the new one is ready, and the harness `settle()` waits for it. Runs without
`afterRun` are unchanged (`extra` is `undefined`).

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
`computation-*` and `snapshot-*` runs and `region-statistics-readback`.

**`draw`** lists layers in paint order: input context first, then the operation's output. Use a
function when layers depend on `values` (ranges, counts, which ramp).

### Layer descriptors

All layers accept `alpha`, `context`, `color` (solid RGBA 0-1). `context: true` marks an **input**
layer: it is drawn muted, and the harness's inputs-only render shows only these. The output layer
must not be `context`, otherwise the verifier fails the run ("output is not drawn").

Value layers (`raster`, `dots`, `segments`, `polygons`, `chart` bars) also take `ramp`
(`'sequential'`, `'diverging'`, `'categorical'`; default categorical for `uint32` buffers), `range`
(`[min, max]` or `'auto'`, the default, which reads the buffer back after each run; diverging auto is
symmetric about zero), `logarithmic`, `bands`, `fade`. Invalid values (NaN, Infinity, unreached
`0xffffffff`) draw in the muted color or transparent. `uint32` values wrap into 8 categories.

| layer | required | draws |
| --- | --- | --- |
| `raster` | `values`, `columns`, `rows` (row 0 at the top) | a grid; `smoothing`, `hillshade`, `elevation` |
| `dots` | `positions` (float32x2); `values?`, `count?`, `radius?` | circles, solid or by value |
| `segments` | `nodes`, `edges` (uint32x2); `values?` per node | network edges, colored by node value or label |
| `lines` | `segments` (float32x4); `count?` (number or uint32 buffer) | solid segment list |
| `paths` | `nodes`, `ids`, `count` | consecutive node ids as a path |
| `triangles` | `positions`, `triangles`, `count` | triangle wireframe (Delaunay) |
| `polygons` | `positions`, `indices`, `owners`, `values`, `triangleCount` | filled polygons by value |
| `chart` | `kind` (`bars`, `line`, `scatter`), `values`; `xValues?`, `xRange?`, `rectangle?` | scalar and table outputs (histograms, series, scatter) |

`count` may be a GPU-written `uint32` buffer, so the picture shows exactly what the operation
produced. Charts plot a buffer's rows; set `range` for the y axis or leave `'auto'`.

### Input generators (`inputs.ts`)

`createSmoothField`, `createFractalRaster` (DEM), `createImageryBands` (green, red, NIR, SWIR and a
land-cover map), `createUniformPoints`, `createClusteredPoints` (with labels), `createKnnWeights`,
`createDistanceBandWeights`, `createTracts` (contiguous hexagon-like polygons with values, triangle
fans, outline and rook contiguity CSR), `createNetwork` (street grid CSR and drawing edges),
`createTrajectories`, `createPolylines`, `createTimeSeries`, `createPointCloud`, `createHexCells`.
Upload helpers: `uploadSpatialWeights(builder, csr)` gives a `GPUSpatialWeights`,
`uploadNetworkTopology(builder, network, costs?)` gives `{topology, weights}`. Need something else?
Write a small local generator in your entry file (same seeded `random`); do not edit `inputs.ts`.
Use the `Data:` label prefix for sliders that change input data.

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
It writes `site3/runs-report/<group>.json`, a contact sheet `<group>.png`, and full-size
`<group>/<id>-<theme>.png`. **Read the screenshots**: passing checks do not mean the picture is clear.
Also typecheck: `npx tsc -p website/tsconfig.json --noEmit`.

Wiring into pages is done by the site, not by runs: `hasRun(id)`, `getRunScene(id)` (a loader for
`<GpuCanvas scene>`) and `getRunParameters(id)` come from `registry.ts`.

## Annotated examples

### 1. Raster: `raster-bands-indices/raster-arithmetic.ts`

Per-frame props only (`update` writes the operation's parameter buffer), so sliders never recompile.

```ts
export default defineOperationRun({
  parameters,
  createInputs: ({random, world}) => createImageryBands(random, world), // seeded NIR, red, ...
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

Function pages show a run's own output before the live canvas starts, so prerendered HTML is never a
generic stand-in.

- `node scripts/snapshot-runs.mjs [entry id...]` renders every run on real WebGPU in both themes and
  writes `public/runs/<entry id>-<theme>.webp` (640 px wide, WebP quality 0.62). Commit the files.
  Run it again after changing a run's picture. `EntryPreview` uses them as the fallback of any entry
  with a run, and `catalog.css` shows the one that matches the page theme.
- `node scripts/check-all-pages.mjs` serves `dist/client` and opens every `/fn/<id>` page, the home
  page and the how-it-works pages in headless Chromium: the live canvas must reach
  `data-ready=true`, nothing may log an error, and moving a slider must change the picture.
