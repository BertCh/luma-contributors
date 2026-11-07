// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// defineOperationRun: a declarative live picture of one analysis operation. The author lists seeded
// inputs, how to build the real operation into a command graph, and which output buffers to draw
// as which layers. The framework owns the device plumbing: buffers, compile, run on parameter
// change, value ranges, layers, theme, resize and cleanup. See ./README.md.

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import type {GpuScene, GpuSceneInstance, GpuSceneParameter} from '../types';
import {readParameterValues} from '../demos/parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  PathLayer,
  RasterLayer,
  SegmentListLayer,
  TriangleEdgeLayer,
  type DemoLayer,
  type DemoTheme,
  type DemoView,
  type LookOptions,
  type ValueFormat
} from '../demos/kit';
import {ChartLayer, EdgeLayer, MeshLayer} from './layers';

// ---------------------------------------------------------------------------------------------
// Buffers

/** GPU storage formats a run can upload or allocate. */
export type RunFormat = Extract<
  GPUVectorFormat,
  | 'float32'
  | 'uint32'
  | 'sint32'
  | 'float32x2'
  | 'float32x3'
  | 'float32x4'
  | 'uint32x2'
  | 'uint32x3'
  | 'uint32x4'
>;

type RunArray = Float32Array | Uint32Array | Int32Array;

const FORMAT_LAYOUT: Record<RunFormat, {components: number; create: (length: number) => RunArray}> =
  {
    float32: {components: 1, create: length => new Float32Array(length)},
    uint32: {components: 1, create: length => new Uint32Array(length)},
    sint32: {components: 1, create: length => new Int32Array(length)},
    float32x2: {components: 2, create: length => new Float32Array(length)},
    float32x3: {components: 3, create: length => new Float32Array(length)},
    float32x4: {components: 4, create: length => new Float32Array(length)},
    uint32x2: {components: 2, create: length => new Uint32Array(length)},
    uint32x3: {components: 3, create: length => new Uint32Array(length)},
    uint32x4: {components: 4, create: length => new Uint32Array(length)}
  };

/** A storage buffer plus the typed view of it that operations take as input or output. */
export type RunBuffer<Format extends RunFormat = RunFormat> = {
  readonly id: string;
  readonly format: Format;
  /** Rows (elements of `format`). */
  readonly length: number;
  readonly buffer: Buffer;
  /** Pass this to the operation's props. */
  readonly view: GraphDataView<Format>;
  /** Overwrites the contents (same element count); call from `update` to change an input. */
  write(data: RunArray): void;
};

/** Everything `build` can use to create buffers and add operations to the graph. */
export type RunBuilder = {
  readonly device: Device;
  readonly graph: GPUCommandGraph<void>;
  /** Uploads `data` as a storage buffer of `format` and imports it into the graph. */
  upload<Format extends RunFormat>(id: string, format: Format, data: RunArray): RunBuffer<Format>;
  /** Allocates a zero-filled buffer for an operation output. */
  allocate<Format extends RunFormat>(id: string, format: Format, length: number): RunBuffer<Format>;
  /**
   * Hands a resource the run must clean up (a snapshot arena, a readback ring) to the framework.
   * It is destroyed together with this build's buffers, on rebuild and on destroy.
   */
  own<Resource extends {destroy(): void}>(resource: Resource): Resource;
};

class RunBufferSet {
  readonly buffers: Buffer[] = [];
  readonly resources: {destroy(): void}[] = [];

  createBuilder(device: Device, graph: GPUCommandGraph<void>): RunBuilder {
    const wrap = <Format extends RunFormat>(
      id: string,
      format: Format,
      buffer: Buffer,
      length: number
    ): RunBuffer<Format> => {
      this.buffers.push(buffer);
      return {
        id,
        format,
        length,
        buffer,
        view: importBufferView(graph, id, buffer, format, length),
        write: data => buffer.write(data)
      };
    };
    return {
      device,
      graph,
      upload: (id, format, data) =>
        wrap(
          id,
          format,
          createStorageBuffer(device, data),
          data.length / FORMAT_LAYOUT[format].components
        ),
      allocate: (id, format, length) =>
        wrap(
          id,
          format,
          createEmptyStorageBuffer(device, length * FORMAT_LAYOUT[format].components),
          length
        ),
      own: resource => {
        this.resources.push(resource);
        return resource;
      }
    };
  }

  destroy(): void {
    for (const resource of this.resources) resource.destroy();
    this.resources.length = 0;
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers.length = 0;
  }
}

/** Creates a typed array for `format` with room for `length` rows. */
export function createRunArray(format: RunFormat, length: number): RunArray {
  const layout = FORMAT_LAYOUT[format];
  return layout.create(length * layout.components);
}

// ---------------------------------------------------------------------------------------------
// Layer descriptors

export type RampKind = 'sequential' | 'diverging' | 'categorical' | 'cyclic';

type ValueLook = {
  /** Default: categorical for uint32 values, sequential for float32. */
  ramp?: RampKind;
  /** Value range mapped to the ramp. `'auto'` (default) reads the buffer back after each run. */
  range?: readonly [number, number] | 'auto';
  logarithmic?: boolean;
  /** Quantize the ramp into this many bands. */
  bands?: number;
  /** Fade alpha from transparent at the low end to opaque at the high end. */
  fade?: boolean;
  /** Up to 16 RGB colors for categorical values (value rounds to the color index). */
  categories?: LookOptions['categories'];
};

type CommonLook = {
  alpha?: number;
  /** Input layer: drawn muted, and the only layers shown in the harness's inputs-only render. */
  context?: boolean;
  /** Solid RGBA (0-1) instead of the ramp, for layers without values. */
  color?: readonly [number, number, number, number];
};

type AnyBuffer = RunBuffer<RunFormat>;

export type RasterRunLayer = {
  layer: 'raster';
  values: AnyBuffer;
  columns: number;
  rows: number;
  smoothing?: boolean;
  hillshade?: number;
  elevation?: number;
} & ValueLook &
  CommonLook;

export type DotsRunLayer = {
  layer: 'dots';
  positions: AnyBuffer;
  /** Optional per-dot values; solid color without it. */
  values?: AnyBuffer;
  /** Draw only the first `count` dots. Default: all positions. */
  count?: number;
  radius?: number;
} & ValueLook &
  CommonLook;

export type SegmentsRunLayer = {
  layer: 'segments';
  /** Node positions (float32x2). */
  nodes: AnyBuffer;
  /** Edges as node-index pairs (uint32x2). */
  edges: AnyBuffer;
  /** Per-node values; solid color without it. */
  values?: AnyBuffer;
  radius?: number;
} & ValueLook &
  CommonLook;

export type LinesRunLayer = {
  layer: 'lines';
  /** Segments as (x0, y0, x1, y1) (float32x4). */
  segments: AnyBuffer;
  /** Segment count: a GPU-written uint32 buffer (row 0) or a constant. Default: all rows. */
  count?: AnyBuffer | number;
  radius?: number;
} & CommonLook;

export type PathsRunLayer = {
  layer: 'paths';
  nodes: AnyBuffer;
  /** Node ids along the path (uint32). */
  ids: AnyBuffer;
  /** Id count: GPU-written uint32 buffer (row 0) or a constant. */
  count: AnyBuffer | number;
  radius?: number;
} & CommonLook;

export type TrianglesRunLayer = {
  layer: 'triangles';
  positions: AnyBuffer;
  /** Triangle vertex ids, three per triangle (uint32 rows or uint32x3 rows). */
  triangles: AnyBuffer;
  /** Triangle count: GPU-written uint32 buffer (row 0) or a constant. */
  count: AnyBuffer | number;
  radius?: number;
} & CommonLook;

export type PolygonsRunLayer = {
  layer: 'polygons';
  positions: AnyBuffer;
  /** Triangle vertex ids (uint32 rows, three per triangle). */
  indices: AnyBuffer;
  /** Owner polygon row per triangle (uint32). */
  owners: AnyBuffer;
  /** Value per polygon. */
  values: AnyBuffer;
  triangleCount: number;
} & ValueLook &
  CommonLook;

export type ChartRunLayer = {
  layer: 'chart';
  kind: 'bars' | 'line' | 'scatter';
  /** Bars and line: value per position; scatter: y. */
  values: AnyBuffer;
  /** Scatter only: x per point (float32). */
  xValues?: AnyBuffer;
  /** Scatter x range. Default [0, 1]. */
  xRange?: readonly [number, number];
  /** Plot rectangle as fractions of the view (x0, y0, x1, y1; y up). Default leaves a margin. */
  rectangle?: readonly [number, number, number, number];
  radius?: number;
} & ValueLook &
  CommonLook;

export type RunLayer =
  | RasterRunLayer
  | DotsRunLayer
  | SegmentsRunLayer
  | LinesRunLayer
  | PathsRunLayer
  | TrianglesRunLayer
  | PolygonsRunLayer
  | ChartRunLayer;

// ---------------------------------------------------------------------------------------------
// Spec

export type ParameterValues = Record<string, number>;

export type RunInputContext = {
  seed: number;
  random: () => number;
  /** Drawing world: width = canvas aspect ratio, height = 1. Generate data inside it. */
  world: {width: number; height: number};
  aspect: number;
};

export type RunBuild<Outputs> = {
  /** Named buffers that `draw` can show. */
  outputs: Outputs;
  /**
   * Writes the current parameter values into input or parameter buffers (cheap). When present,
   * parameter changes call it and re-run the graph instead of rebuilding; only ids in `rebuildOn`
   * rebuild. Called once after `build` as well.
   */
  update?: (values: ParameterValues) => void;
};

export type RunDrawState<Inputs, Outputs, Extra = undefined> = {
  inputs: Inputs;
  outputs: Outputs;
  values: ParameterValues;
  theme: DemoTheme;
  world: {width: number; height: number};
  /** What `afterRun` returned for the latest run; `undefined` for runs without `afterRun`. */
  extra: Extra;
};

/** What `afterRun` receives: the finished run, and builders for the buffers `draw` will show. */
export type RunAfterState<Inputs, Outputs> = {
  device: Device;
  /** The compiled graph that just ran (describe it, read its preflight, encode it again). */
  compiled: CompiledGPUCommandGraph<void>;
  inputs: Inputs;
  outputs: Outputs;
  /** Slider values of this run (a copy). */
  values: ParameterValues;
  /**
   * Uploads CPU data (an async readback, a decoded record) for `draw`. The buffers belong to this
   * result and are destroyed when the next `afterRun` result replaces it.
   */
  upload: RunBuilder['upload'];
  allocate: RunBuilder['allocate'];
};

export type OperationRunSpec<Inputs, Outputs, Extra = undefined> = {
  /** Slider parameters, bound to real props. Import them from the sibling `.parameters.ts`. */
  parameters?: readonly GpuSceneParameter[];
  /** Seeded CPU inputs, at most ~20k elements. */
  createInputs(context: RunInputContext): Inputs;
  /** Uploads inputs, allocates outputs and adds the real operation: `builder.graph.add(...)`. */
  build(builder: RunBuilder, inputs: Inputs, values: ParameterValues): RunBuild<Outputs>;
  /**
   * Optional asynchronous post-run hook, for operations whose output is not a buffer but a
   * readback or a record of the run (a snapshot arena, a command-graph description, a CPU
   * summary). Called after every run, with the finished graph; its result is handed to `draw` as
   * `extra`. Buffers made through `state.upload` are drawn like any other. Results of superseded
   * runs are dropped, and `draw` keeps the previous `extra` until the new one is ready. Runs
   * without `afterRun` behave exactly as before.
   */
  afterRun?: (state: RunAfterState<Inputs, Outputs>) => Promise<Extra> | Extra;
  /** Layers to draw, in order: input context first, then the operation's output. */
  draw:
    | readonly RunLayer[]
    | ((state: RunDrawState<Inputs, Outputs, Extra>) => readonly RunLayer[]);
  /**
   * Parameter ids that are compile-time props (change the graph itself). Default: every parameter
   * when `build` returns no `update`, none otherwise.
   */
  rebuildOn?: readonly string[];
  /** One line shown by the verification harness. */
  caption?: string;
};

/** The scene a run produces. The sliders live in the sibling `.parameters.ts` file. */
export type RunScene = GpuScene & {
  parameters: readonly GpuSceneParameter[];
  caption?: string;
};

/** Debug handle on the scene instance, used by the verification harness. */
export type RunDebug = {
  /** Draws only the `context` layers (the inputs) when true. */
  setInputsOnly(isInputsOnly: boolean): void;
  /** Resolves when the last run and value-range readback have finished and the frame is drawn. */
  settle(): Promise<void>;
  getRunCount(): number;
  getRebuildCount(): number;
  getLayerKinds(): string[];
};

export type RunInstance = GpuSceneInstance & {debug: RunDebug};

// ---------------------------------------------------------------------------------------------
// Implementation

const RAMP_MODES: Record<RampKind, 0 | 1 | 2 | 3> = {
  sequential: 0,
  diverging: 1,
  categorical: 2,
  cyclic: 3
};
const DEFAULT_RECTANGLE = [0.07, 0.14, 0.95, 0.88] as const;
const INVALID_LIMIT = 1e20;

function getValueFormat(buffer: AnyBuffer): ValueFormat {
  return buffer.format === 'uint32' ? 'uint32' : 'float32';
}

function readRange(bytes: Uint8Array, format: ValueFormat): [number, number] | null {
  const values =
    format === 'uint32'
      ? new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2)
      : new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2);
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value) || Math.abs(value) >= INVALID_LIMIT) continue;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return minimum <= maximum ? [minimum, maximum] : null;
}

type Resolved = {layer: DemoLayer; look: LookOptions; instanceCount?: number};

type CachedLayer = {signature: unknown[]; layer: DemoLayer; owned: Buffer[]};

/** Builds the scene function for one operation. */
export function defineOperationRun<Inputs, Outputs, Extra = undefined>(
  spec: OperationRunSpec<Inputs, Outputs, Extra>
): RunScene {
  const parameters = spec.parameters ?? [];

  const scene: GpuScene = async context => {
    const {device} = context;
    assertWebGPU(device);
    const theme = getDemoTheme(context.theme);
    const values = readParameterValues(parameters, context.parameters);
    const aspect = context.width / context.height;
    const world = {width: aspect, height: 1};
    const view: DemoView = {
      worldWidth: world.width,
      worldHeight: world.height,
      width: context.width,
      height: context.height
    };
    const inputs = spec.createInputs({
      seed: context.seed,
      random: makeRandom(context.seed),
      world,
      aspect
    });

    type Built = {
      runner: GraphRunner;
      buffers: RunBufferSet;
      result: RunBuild<Outputs>;
    };
    const createBuilt = async (): Promise<Built> => {
      const buffers = new RunBufferSet();
      let result: RunBuild<Outputs> | undefined;
      const runner = await GraphRunner.create(device, 'run-operation', graph => {
        result = spec.build(buffers.createBuilder(device, graph), inputs, values);
      });
      // result is set synchronously by the callback above.
      return {runner, buffers, result: result as RunBuild<Outputs>};
    };

    let current = await createBuilt();
    const destroyBuilt = (built: Built) => {
      built.runner.destroy();
      built.buffers.destroy();
    };

    // Layer cache: one entry per descriptor index, rebuilt when its buffers change.
    const layerCache = new Map<number, CachedLayer>();
    const disposeLayers = () => {
      for (const cached of layerCache.values()) {
        cached.layer.destroy();
        for (const buffer of cached.owned) buffer.destroy();
      }
      layerCache.clear();
    };
    const ranges = new Map<Buffer, [number, number]>();
    const wantedRanges = new Map<Buffer, ValueFormat>();
    let dummyValues: Buffer | null = null;
    const getDummyValues = () => (dummyValues ??= createEmptyStorageBuffer(device, 4));

    let drawLayersList: Resolved[] = [];
    let isInputsOnly = false;
    let rebuildCount = 0;
    let runCount = 0;
    let isDestroyed = false;
    let needsRun = true;
    let needsRebuild = false;
    let framesToDraw = 3;
    let rebuildPromise: Promise<void> | null = null;
    let kinds: string[] = [];

    const rebuildIds = new Set(spec.rebuildOn ?? []);

    const getRange = (layer: ValueLook, buffer: AnyBuffer | undefined): [number, number] => {
      if (layer.range && layer.range !== 'auto') return [layer.range[0], layer.range[1]];
      if (!buffer) return [0, 1];
      wantedRanges.set(buffer.buffer, getValueFormat(buffer));
      const measured = ranges.get(buffer.buffer) ?? [0, 1];
      if (layer.ramp === 'diverging') {
        const extent = Math.max(Math.abs(measured[0]), Math.abs(measured[1]), 1e-9);
        return [-extent, extent];
      }
      return measured;
    };

    const getLook = (
      layer: RunLayer,
      buffer: AnyBuffer | undefined,
      rangeBuffer: AnyBuffer | undefined = buffer
    ): LookOptions => {
      const isContext = Boolean(layer.context);
      const valued = layer as ValueLook & CommonLook;
      const format = buffer ? getValueFormat(buffer) : 'float32';
      const ramp = valued.ramp ?? (format === 'uint32' ? 'categorical' : 'sequential');
      const solid =
        valued.color ??
        ([...(isContext ? theme.muted : theme.ink), isContext ? 0.55 : 0.85] as const);
      const contextRamp = [0, 1, 2, 3, 4, 5].map(step => {
        const amount = step / 5;
        return [
          theme.background[0] + (theme.muted[0] - theme.background[0]) * amount,
          theme.background[1] + (theme.muted[1] - theme.background[1]) * amount,
          theme.background[2] + (theme.muted[2] - theme.background[2]) * amount
        ] as const;
      });
      const hasValues = Boolean(buffer);
      return {
        mode: hasValues && !valued.color ? RAMP_MODES[ramp] : -1,
        range: getRange(valued, rangeBuffer),
        alpha: layer.alpha ?? (isContext ? 0.9 : 1),
        solid,
        ramp:
          isContext && hasValues ? contextRamp : ramp === 'diverging' ? theme.diverging : undefined,
        categories: isContext && hasValues ? [theme.muted] : valued.categories,
        logarithmic: valued.logarithmic,
        bands: valued.bands,
        fade: valued.fade
      };
    };

    const getCountBuffer = (count: AnyBuffer | number | undefined, fallback: number) => {
      if (count && typeof count !== 'number') return {buffer: count.buffer, owned: []};
      const buffer = createStorageBuffer(device, Uint32Array.of(count ?? fallback));
      return {buffer, owned: [buffer]};
    };

    const resolveLayer = (layer: RunLayer, index: number): Resolved => {
      const buffers = collectBuffers(layer);
      const signature: unknown[] = [layer.layer, ...buffers];
      const cached = layerCache.get(index);
      let layerObject: DemoLayer;
      if (
        cached &&
        cached.signature.length === signature.length &&
        cached.signature.every((item, position) => item === signature[position])
      ) {
        layerObject = cached.layer;
      } else {
        if (cached) {
          cached.layer.destroy();
          for (const buffer of cached.owned) buffer.destroy();
        }
        const created = createLayer(layer);
        layerObject = created.layer;
        layerCache.set(index, {signature, layer: created.layer, owned: created.owned});
      }
      return resolveLook(layer, layerObject);
    };

    const createLayer = (layer: RunLayer): {layer: DemoLayer; owned: Buffer[]} => {
      switch (layer.layer) {
        case 'raster':
          return {
            layer: new RasterLayer(
              device,
              layer.values.buffer,
              layer.columns,
              layer.rows,
              getValueFormat(layer.values)
            ),
            owned: []
          };
        case 'dots':
          return {
            layer: new DotLayer(
              device,
              layer.positions.buffer,
              layer.values?.buffer ?? getDummyValues(),
              layer.count ?? layer.positions.length,
              layer.values ? getValueFormat(layer.values) : 'float32'
            ),
            owned: []
          };
        case 'segments':
          return {
            layer: new EdgeLayer(
              device,
              layer.nodes.buffer,
              layer.edges.buffer,
              layer.values?.buffer ?? getDummyValues(),
              layer.edges.length,
              layer.values ? getValueFormat(layer.values) : 'float32'
            ),
            owned: []
          };
        case 'lines': {
          const count = getCountBuffer(layer.count, layer.segments.length);
          const capacity = layer.segments.length;
          return {
            layer: new SegmentListLayer(device, layer.segments.buffer, count.buffer, capacity),
            owned: count.owned
          };
        }
        case 'paths': {
          const count = getCountBuffer(layer.count, layer.ids.length);
          return {
            layer: new PathLayer(
              device,
              layer.nodes.buffer,
              layer.ids.buffer,
              count.buffer,
              Math.max(layer.ids.length - 1, 1)
            ),
            owned: count.owned
          };
        }
        case 'triangles': {
          const triangleCapacity =
            layer.triangles.format === 'uint32x3'
              ? layer.triangles.length
              : Math.floor(layer.triangles.length / 3);
          const count = getCountBuffer(layer.count, triangleCapacity);
          return {
            layer: new TriangleEdgeLayer(
              device,
              layer.positions.buffer,
              layer.triangles.buffer,
              count.buffer,
              triangleCapacity
            ),
            owned: count.owned
          };
        }
        case 'polygons':
          return {
            layer: new MeshLayer(
              device,
              layer.positions.buffer,
              layer.indices.buffer,
              layer.owners.buffer,
              layer.values.buffer,
              layer.triangleCount,
              getValueFormat(layer.values)
            ),
            owned: []
          };
        case 'chart': {
          const count = layer.values.length;
          return {
            layer: new ChartLayer(
              device,
              layer.kind,
              layer.values.buffer,
              layer.xValues?.buffer ?? getDummyValues(),
              count,
              getValueFormat(layer.values)
            ),
            owned: []
          };
        }
      }
    };

    const resolveLook = (layer: RunLayer, layerObject: DemoLayer): Resolved => {
      switch (layer.layer) {
        case 'raster': {
          const look = getLook(layer, layer.values);
          return {
            layer: layerObject,
            look: {
              ...look,
              mode: layer.color ? -1 : look.mode,
              smoothing: layer.smoothing,
              hillshade: layer.hillshade,
              elevation: layer.elevation
            }
          };
        }
        case 'dots':
          return {
            layer: layerObject,
            look: {...getLook(layer, layer.values), radius: layer.radius ?? 2.4, hillshade: 0.5}
          };
        case 'segments':
          return {
            layer: layerObject,
            look: {...getLook(layer, layer.values), radius: layer.radius ?? 1.4}
          };
        case 'lines':
        case 'paths':
        case 'triangles':
          return {
            layer: layerObject,
            look: {
              ...getLook(layer, undefined),
              radius: layer.radius ?? 1.2,
              solid: layer.color ?? [
                ...(layer.context ? theme.muted : theme.ink),
                layer.context ? 0.6 : 0.9
              ]
            }
          };
        case 'polygons':
          return {layer: layerObject, look: getLook(layer, layer.values)};
        case 'chart': {
          const chart = layerObject as ChartLayer;
          const fractions = layer.rectangle ?? DEFAULT_RECTANGLE;
          chart.setPlot(
            [
              fractions[0] * world.width,
              fractions[1] * world.height,
              fractions[2] * world.width,
              fractions[3] * world.height
            ],
            layer.xRange ?? [0, 1]
          );
          // Only bars are colored by value; line and scatter use the buffer for the y range.
          const look = getLook(
            layer,
            layer.kind === 'bars' ? layer.values : undefined,
            layer.values
          );
          return {
            layer: layerObject,
            look: {...look, radius: layer.radius ?? (layer.kind === 'scatter' ? 3 : 1.6)}
          };
        }
      }
    };

    const refreshRanges = async () => {
      const wanted = [...wantedRanges];
      const measured = await Promise.all(
        wanted.map(async ([buffer, format]) => {
          try {
            return [buffer, readRange(await buffer.readAsync(), format)] as const;
          } catch {
            return [buffer, null] as const;
          }
        })
      );
      if (isDestroyed) return;
      let changed = false;
      for (const [buffer, range] of measured) {
        if (!range) continue;
        const previous = ranges.get(buffer);
        if (!previous || previous[0] !== range[0] || previous[1] !== range[1]) changed = true;
        ranges.set(buffer, range);
      }
      if (changed) framesToDraw = 3;
    };

    // Post-run results (`afterRun`): the latest committed value, its buffers, and the run in flight.
    let extra: Extra | undefined;
    let extraBuffers: RunBufferSet | null = null;
    let afterGeneration = 0;
    let pendingAfter: Promise<void> | null = null;

    const getLayerList = (): readonly RunLayer[] =>
      typeof spec.draw === 'function'
        ? spec.draw({
            inputs,
            outputs: current.result.outputs,
            values,
            theme,
            world,
            extra: extra as Extra
          })
        : spec.draw;

    const render = () => {
      drawLayersList = [];
      kinds = [];
      getLayerList().forEach((layer, index) => {
        if (isInputsOnly && !layer.context) return;
        kinds.push(layer.layer);
        drawLayersList.push(resolveLayer(layer, index));
      });
      for (const resolved of drawLayersList) resolved.layer.update(view, theme, resolved.look);
      drawLayers(
        device,
        theme,
        drawLayersList.map(resolved => resolved.layer)
      );
    };

    const runOperation = () => {
      current.result.update?.(values);
      current.runner.run();
      runCount++;
    };

    /** Runs `afterRun` for the build that just ran; a newer run or rebuild supersedes it. */
    const refreshExtra = (): Promise<void> => {
      const afterRun = spec.afterRun;
      if (!afterRun) return Promise.resolve();
      const generation = ++afterGeneration;
      const built = current;
      const staged = new RunBufferSet();
      const task = (async () => {
        let result: Extra;
        try {
          const builder = staged.createBuilder(
            device,
            new GPUCommandGraph<void>(device, {id: 'run-after'})
          );
          result = await afterRun({
            device,
            compiled: built.runner.compiled,
            inputs,
            outputs: built.result.outputs,
            values: {...values},
            upload: builder.upload,
            allocate: builder.allocate
          });
        } catch (error) {
          staged.destroy();
          // A rebuild or newer run may have destroyed what this one was reading.
          if (isDestroyed || generation !== afterGeneration) return;
          throw error;
        }
        if (isDestroyed || generation !== afterGeneration) {
          staged.destroy();
          return;
        }
        extraBuffers?.destroy();
        extraBuffers = staged;
        extra = result;
        framesToDraw = 3;
        scheduler.request();
      })();
      // The caller handles `task`'s rejection; this copy only tracks that a run is in flight.
      const tracked: Promise<void> = task
        .catch(() => undefined)
        .finally(() => {
          if (pendingAfter === tracked) pendingAfter = null;
        });
      pendingAfter = tracked;
      return task;
    };

    const rebuild = () => {
      rebuildPromise ??= (async () => {
        needsRebuild = false;
        const next = await createBuilt();
        if (isDestroyed) {
          destroyBuilt(next);
          return;
        }
        disposeLayers();
        ranges.clear();
        wantedRanges.clear();
        afterGeneration++;
        destroyBuilt(current);
        current = next;
        rebuildCount++;
        needsRun = true;
        framesToDraw = 3;
      })().finally(() => {
        rebuildPromise = null;
        if (needsRebuild) rebuild();
      });
    };

    const draw = () => {
      if (isDestroyed) return;
      if (needsRebuild) rebuild();
      if (needsRun && !rebuildPromise) {
        needsRun = false;
        runOperation();
        framesToDraw = 3;
        render();
        void refreshRanges();
        // biome-ignore lint/suspicious/noConsole: a failed post-run hook must be visible
        void refreshExtra().catch(error => console.error(error));
        return;
      }
      render();
    };
    const scheduler = new DrawScheduler(draw);

    // First run, so the first frame already shows the operation's output.
    runOperation();
    needsRun = false;
    await refreshExtra();
    render();
    await refreshRanges();

    const instance: RunInstance = {
      render() {
        if (framesToDraw > 0 || needsRun || needsRebuild) {
          framesToDraw = Math.max(framesToDraw - 1, 0);
          scheduler.flush();
        }
      },
      resize(width, height) {
        view.width = width;
        view.height = height;
        framesToDraw = 3;
        scheduler.request();
      },
      setParameters(update) {
        Object.assign(values, update);
        const hasUpdate = Boolean(current.result.update);
        const changesGraph = Object.keys(update).some(id =>
          hasUpdate ? rebuildIds.has(id) : true
        );
        if (changesGraph && parameters.length > 0) needsRebuild = true;
        needsRun = true;
        scheduler.request();
      },
      destroy() {
        isDestroyed = true;
        scheduler.cancel();
        disposeLayers();
        if (dummyValues) dummyValues.destroy();
        extraBuffers?.destroy();
        destroyBuilt(current);
      },
      debug: {
        setInputsOnly(flag) {
          isInputsOnly = flag;
          framesToDraw = 3;
          scheduler.request();
        },
        async settle() {
          scheduler.cancel();
          const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
          for (
            let attempt = 0;
            attempt < 400 && (rebuildPromise || needsRebuild || needsRun || pendingAfter);
            attempt++
          ) {
            draw();
            await nextFrame();
          }
          render();
          await refreshRanges();
          render();
          await nextFrame();
        },
        getRunCount: () => runCount,
        getRebuildCount: () => rebuildCount,
        getLayerKinds: () => kinds
      }
    };
    return instance;
  };

  return Object.assign(scene, {parameters, caption: spec.caption});
}

function collectBuffers(layer: RunLayer): unknown[] {
  switch (layer.layer) {
    case 'raster':
      return [layer.values, layer.columns, layer.rows];
    case 'dots':
      return [layer.positions, layer.values, layer.count];
    case 'segments':
      return [layer.nodes, layer.edges, layer.values];
    case 'lines':
      return [layer.segments, layer.count];
    case 'paths':
      return [layer.nodes, layer.ids, layer.count];
    case 'triangles':
      return [layer.positions, layer.triangles, layer.count];
    case 'polygons':
      return [layer.positions, layer.indices, layer.owners, layer.values, layer.triangleCount];
    case 'chart':
      return [layer.kind, layer.values, layer.xValues];
  }
}
