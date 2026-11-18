// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// defineOperationRun: a declarative live picture of one analysis operation. The author lists seeded
// inputs, how to build the real operation into a command graph, and which output buffers to draw
// as which layers. The framework owns the device plumbing: buffers, compile, run on parameter
// change, value ranges, layers, theme, resize and cleanup. See ./README.md.

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
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
  getDemoTheme,
  getWorldPosition,
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
import {
  type ParameterPatch,
  type RunInteraction,
  type RunInteractionResult,
  type RunInteractionState,
  type RunPeek,
  type RunPointer
} from './interaction';
import {ChartLayer, EdgeLayer, MeshLayer} from './layers';
import {getGlyphMetrics, getPixelOfWorld, PointerOverlay} from './pointer-overlay';
import type {TextRunLayer} from './text';
import {loadFontPixels} from './text/font';
import {TextLayer} from './text/text-layer';

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
   * It is destroyed together with this build's buffers, on rebuild and on destroy. A build that
   * owns a resource is not kept by `buildCache`, because a run may consume it (an arena that
   * `read()` closes); pass `{reusable: true}` for a resource no run changes (an operation's
   * constant plan buffer), which keeps the build eligible for the cache.
   */
  own<Resource extends {destroy(): void}>(
    resource: Resource,
    options?: {reusable?: boolean}
  ): Resource;
};

class RunBufferSet {
  readonly buffers: Buffer[] = [];
  readonly resources: {destroy(): void}[] = [];
  /** Owned resources a run may consume: a build holding any cannot be reused from the cache. */
  consumableResourceCount = 0;

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
      own: (resource, options) => {
        this.resources.push(resource);
        if (!options?.reusable) this.consumableResourceCount++;
        return resource;
      }
    };
  }

  destroy(): void {
    for (const resource of this.resources) resource.destroy();
    this.resources.length = 0;
    this.consumableResourceCount = 0;
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

export type RampKind = 'sequential' | 'diverging' | 'categorical' | 'cyclic' | 'grey';

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
  /**
   * Draw only the part of the layer left or right of world x. A swipe gesture gives the input
   * layer `{side: 'left', x: curtain}` and the output layer `{side: 'right', x: curtain}`.
   */
  clip?: {side: 'left' | 'right'; x: number};
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
  /** Scale each dot's radius with its value, from 30% at the low end of `range` to all of it. */
  sizeByValue?: boolean;
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
  | ChartRunLayer
  | TextRunLayer;

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
  /** What `interaction.resolve` returned for the current pointer (empty without an interaction). */
  interaction: RunInteractionResult;
  /** The pointer the interaction last resolved, or null without an interaction. */
  pointer: RunPointer | null;
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
  /** Draw geographic analysis output directly, in [west, south, east, north] degrees. */
  geographicWindow?: readonly [number, number, number, number];
  /** Seeded CPU inputs, at most ~20k elements. */
  createInputs(context: RunInputContext): Inputs;
  /** Uploads inputs, allocates outputs and adds the real operation: `builder.graph.add(...)`. */
  build(builder: RunBuilder, inputs: Inputs, values: ParameterValues): RunBuild<Outputs>;
  /**
   * Optional asynchronous post-run hook, for operations whose output is not a buffer but a
   * readback or a record of the run (a snapshot arena, a command-graph description, a CPU
   * summary), with the finished graph; its result is handed to `draw` as `extra`. Buffers made
   * through `state.upload` are drawn like any other. One call is in flight at a time: runs that
   * finish meanwhile get one follow-up call for the latest run when it returns, so under a moving
   * pointer it is called less often than the graph runs, and every run that ends a burst gets
   * its call. A rebuild drops the result of a call on the old build, and `draw` keeps the
   * previous `extra` until the new one is ready. Runs without `afterRun` behave exactly as before.
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
  /**
   * The pointer gesture of this picture: what the pointer means for the operation. Without it the
   * card falls back to scrubbing one slider along x. See ./README.md, "Pointer interaction".
   */
  interaction?: RunInteraction<Inputs, Outputs>;
  /**
   * Keep up to this many compiled builds, keyed by the values of the compile-time parameters, so
   * returning to a value the reader already visited does not recompile. Leave it out for runs that
   * hand stateful resources to `builder.own` (resources owned with `{reusable: true}` are fine).
   */
  buildCache?: number;
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
  /**
   * Waits up to 400 animation frames for runs, rebuilds, `afterRun`, peeks and pointer settlement.
   * If work remains, warns and uses the latest committed state, which may omit pending results.
   * Then waits for its value-range read (including any follow-up), draws and resolves. The range
   * read has no frame bound. Stops without further drawing if the run is destroyed.
   */
  settle(): Promise<void>;
  getRunCount(): number;
  getRebuildCount(): number;
  getLayerKinds(): string[];
  /** The interaction's current readout line. */
  getReadout(): string | null;
  /** Parameter values the interaction returns at its rest position (for the verifier). */
  getRestValues(): ParameterPatch;
  /** Rebuilds caused by `values` the interaction returned: must stay 0 (use `settle`). */
  getPointerRebuildCount(): number;
  /** Ids of every parameter the interaction has written through `values` or `settle`. */
  getPointerParameterIds(): string[];
};

export type RunInstance = GpuSceneInstance & {debug: RunDebug};

// ---------------------------------------------------------------------------------------------
// Implementation

const RAMP_MODES: Record<RampKind, 0 | 1 | 2 | 3> = {
  sequential: 0,
  diverging: 1,
  categorical: 2,
  cyclic: 3,
  grey: 0
};
/** Shadow to lit, the same in both themes: for illumination and hillshade buffers. */
const GREY_RAMP = [0.04, 0.2, 0.4, 0.6, 0.8, 0.98].map(level => [level, level, level] as const);
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

type Resolved = {
  layer: DemoLayer;
  look: LookOptions;
  instanceCount?: number;
  clip?: {side: 'left' | 'right'; x: number};
};

/** Time constant of the glide from the rest position to a pointer that has just arrived. */
const POINTER_GLIDE_MILLISECONDS = 45;
/** How long the pointer must rest before compile-time (`settle`) values are applied. */
const SETTLE_MILLISECONDS = 140;

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
    if (spec.geographicWindow) {
      const [west, south, east, north] = spec.geographicWindow;
      view.origin = [west, south];
      view.worldWidth = east - west;
      view.worldHeight = north - south;
      view.longitudeScale = Math.cos(((north + south) * Math.PI) / 360);
    }
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
      try {
        const runner = await GraphRunner.create(device, 'run-operation', graph => {
          result = spec.build(buffers.createBuilder(device, graph), inputs, values);
        });
        // result is set synchronously by the callback above.
        return {runner, buffers, result: result as RunBuild<Outputs>};
      } catch (error) {
        buffers.destroy();
        throw error;
      }
    };

    const rebuildIds = new Set(spec.rebuildOn ?? []);
    /**
     * Identifies a build by the values of the parameters that are compiled into it: the
     * `rebuildOn` ids when the build has an `update`, otherwise every parameter.
     */
    const getBuildKey = (): string =>
      parameters
        .filter(parameter => !current.result.update || rebuildIds.has(parameter.id))
        .map(parameter => values[parameter.id])
        .join(',');

    let current = await createBuilt();
    let currentKey = getBuildKey();
    const builtCache = new Map<string, Built>();
    const destroyBuilt = (built: Built) => {
      built.runner.destroy();
      built.buffers.destroy();
    };
    /** Keeps the outgoing build for reuse when the spec asks for a cache, otherwise destroys it. */
    const retireBuilt = (key: string, built: Built) => {
      const capacity = spec.buildCache ?? 0;
      // A build that owns stateful resources (an arena a run consumes) cannot be reused.
      if (capacity <= 0 || builtCache.has(key) || built.buffers.consumableResourceCount > 0) {
        destroyBuilt(built);
        return;
      }
      builtCache.set(key, built);
      while (builtCache.size > capacity) {
        const [oldestKey, oldest] = builtCache.entries().next().value as [string, Built];
        builtCache.delete(oldestKey);
        destroyBuilt(oldest);
      }
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
    // Keyed weakly so a cached build keeps its measured ranges and destroyed buffers drop theirs.
    const ranges = new WeakMap<Buffer, [number, number]>();
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
    /** True from a build swap until its first run: its buffers do not hold this state yet. */
    let isAwaitingFirstRun = false;
    /** Counts builds, so a gesture can tell that its render-only buffers are new. */
    let buildNumber = 0;
    let kinds: string[] = [];

    // -------------------------------------------------------------------------------------------
    // Pointer interaction

    const interaction = spec.interaction;
    const pointerState = {
      /** Where the gesture rests: the poster. Position across the canvas, 0 to 1, y down. */
      rest: [0.5, 0.5] as [number, number],
      /** Where the pointer is; null until it first arrives and after a slider takes over. */
      target: null as [number, number] | null,
      /** The position the interaction is resolved at: the pointer, or on its way to it. */
      effective: [0.5, 0.5] as [number, number],
      /** True while `effective` glides from where it was to a pointer that has just arrived. */
      isGliding: false,
      lastTime: 0,
      result: {} as RunInteractionResult,
      pointer: null as RunPointer | null
    };
    let readout: string | null = null;
    let restValues: ParameterPatch = {};
    let settleTimer: ReturnType<typeof setTimeout> | undefined;
    let pendingSettle: ParameterPatch | null = null;
    let pointerRebuildCount = 0;
    /** False once a slider has changed a value: the pointer may not write values until it moves. */
    let pointerOwnsValues = true;
    /** Runs started by readbacks since the pointer or a slider last moved; stops a feedback loop. */
    let peekDrivenRunCount = 0;
    const pointerParameterIds = new Set<string>();

    // CPU copies of buffers the interaction asked to `peek`, refreshed after each run.
    type PeekEntry = {format: RunFormat; data: RunArray | null};
    const peeks = new Map<Buffer, PeekEntry>();
    /** Bumped on every rebuild: a readback of the previous build's buffers is discarded. */
    let peekEpoch = 0;
    let pendingPeek: Promise<void> | null = null;
    let isPeekInFlight = false;
    let isPeekQueued = false;
    let hasNewPeek = false;
    const peek: RunPeek = wanted => {
      const buffer = wanted.buffer as Buffer;
      let entry = peeks.get(buffer);
      if (!entry) {
        entry = {format: wanted.format as RunFormat, data: null};
        peeks.set(buffer, entry);
        hasNewPeek = true;
      }
      return entry.data;
    };
    /**
     * Reads the peeked buffers back and resolves the interaction again. One readback is in flight
     * at a time; a request that arrives meanwhile is queued, so a moving pointer cannot starve it.
     */
    const refreshPeeks = (): Promise<void> => {
      hasNewPeek = false;
      if (peeks.size === 0) return Promise.resolve();
      if (isPeekInFlight) {
        isPeekQueued = true;
        return pendingPeek ?? Promise.resolve();
      }
      isPeekInFlight = true;
      const epoch = peekEpoch;
      const task = (async () => {
        const wanted = [...peeks];
        const read = await Promise.all(
          wanted.map(async ([buffer, entry]) => {
            try {
              const bytes = await buffer.readAsync();
              if (isDestroyed) return null;
              const copy = bytes.buffer.slice(
                bytes.byteOffset,
                bytes.byteOffset + bytes.byteLength
              );
              if (entry.format.startsWith('uint32')) return new Uint32Array(copy);
              if (entry.format.startsWith('sint32')) return new Int32Array(copy);
              return new Float32Array(copy);
            } catch {
              return null;
            }
          })
        );
        if (isDestroyed || epoch !== peekEpoch) return;
        wanted.forEach(([, entry], index) => {
          entry.data = read[index];
        });
        // The interaction can now answer from the operation's output: resolve again. A slider's
        // values stay as the reader set them.
        applyPointer({isPeekRefresh: true, applyValues: pointerOwnsValues});
        framesToDraw = Math.max(framesToDraw, 2);
        scheduler.request();
      })();
      const tracked: Promise<void> = task
        .catch(() => undefined)
        .finally(() => {
          isPeekInFlight = false;
          if (pendingPeek === tracked) pendingPeek = null;
          if (isPeekQueued && !isDestroyed) {
            isPeekQueued = false;
            void refreshPeeks();
          }
        });
      pendingPeek = tracked;
      return tracked;
    };

    const getInteractionState = (): RunInteractionState<Inputs, Outputs> => ({
      inputs,
      outputs: current.result.outputs,
      values,
      world,
      pixelSize: getGlyphMetrics(view).pixelSize,
      toUnit: point => {
        const [pixelX, pixelY] = getPixelOfWorld(view, point[0], point[1]);
        return [pixelX / view.width, pixelY / view.height];
      },
      peek,
      build: buildNumber
    });

    // Glyphs and the probe spotlight, drawn over the picture.
    const overlay = new PointerOverlay(device);
    /** True while the reader is pointing: the spotlight belongs to probing, not to the poster. */
    let isProbing = false;
    const setOverlay = (result: RunInteractionResult) => {
      const spotlight =
        result.spotlight === undefined
          ? interaction?.kind === 'probe'
            ? 'auto'
            : null
          : result.spotlight;
      overlay.set(result.glyphs ?? [], view, spotlight);
      if (context.reducedMotion) overlay.setSpotlightOn(isProbing);
    };

    /** Applies parameter values (from a slider or the pointer) and schedules the run or rebuild. */
    const changeValues = (update: ParameterPatch) => {
      Object.assign(values, update);
      const hasUpdate = Boolean(current.result.update);
      const changesGraph = Object.keys(update).some(id => (hasUpdate ? rebuildIds.has(id) : true));
      if (changesGraph && parameters.length > 0) needsRebuild = true;
      needsRun = true;
      scheduler.request();
    };
    /** The entries of `patch` that differ from the current values, or null when none do. */
    const getChanged = (patch: ParameterPatch | undefined): ParameterPatch | null => {
      if (!patch) return null;
      const changed: ParameterPatch = {};
      let hasChange = false;
      for (const [id, value] of Object.entries(patch)) {
        if (Number.isFinite(value) && values[id] !== value) {
          changed[id] = value;
          hasChange = true;
        }
      }
      return hasChange ? changed : null;
    };
    const applySettle = () => {
      clearTimeout(settleTimer);
      const changed = getChanged(pendingSettle ?? undefined);
      pendingSettle = null;
      if (!changed) return;
      changeValues(changed);
      context.reportParameters?.(changed);
    };

    /** Resolves the interaction at the damped pointer position and applies what it returns. */
    const applyPointer = (
      options: {applyValues?: boolean; isPeekRefresh?: boolean; isAfterRebuild?: boolean} = {}
    ) => {
      if (!interaction || isDestroyed) return;
      // While a rebuild is pending the values already describe the next build but `outputs` and
      // the peeked arrays still belong to the old one. The gesture resumes when the swap is done.
      if ((needsRebuild || rebuildPromise) && !options.isAfterRebuild) return;
      const [unitX, unitY] = pointerState.effective;
      const [x, y] = getWorldPosition(view, [unitX * view.width, unitY * view.height]);
      const pointer: RunPointer = {x, y, unitX, unitY};
      const result = interaction.resolve(pointer, getInteractionState()) ?? {};
      pointerState.pointer = pointer;
      pointerState.result = result;
      for (const id of Object.keys(result.values ?? {})) pointerParameterIds.add(id);
      for (const id of Object.keys(result.settle ?? {})) pointerParameterIds.add(id);
      if (options.applyValues !== false) {
        let changed = getChanged(result.values);
        // Values derived from the output can chase themselves: readback, new value, run, readback.
        // A few rounds may converge; after that the readbacks stop driving runs.
        if (changed && options.isPeekRefresh && ++peekDrivenRunCount > 3) changed = null;
        if (changed) {
          const wasRebuilding = needsRebuild;
          changeValues(changed);
          if (needsRebuild && !wasRebuilding) {
            // Compile-time parameters belong in `settle`, never in `values`.
            pointerRebuildCount++;
          }
          context.reportParameters?.(changed);
        }
        clearTimeout(settleTimer);
        if (getChanged(result.settle)) {
          pendingSettle = {...result.settle};
          settleTimer = setTimeout(applySettle, SETTLE_MILLISECONDS);
        } else {
          // Back on the value that is already live: nothing is left to settle.
          pendingSettle = null;
        }
      }
      // A peek refresh re-resolves against new output; it must not start another run by itself.
      if (result.rerun && !options.isPeekRefresh) {
        needsRun = true;
      }
      // When a slider owns the values and has moved one away from what this pointer position
      // gives, the gesture's glyphs and readout would contradict the picture: hide them.
      const isContradicted =
        options.applyValues === false &&
        Object.entries({...result.values, ...result.settle}).some(
          ([id, value]) => Number.isFinite(value) && values[id] !== value
        );
      setOverlay(isContradicted ? {} : result);
      const nextReadout = isContradicted ? null : (result.readout ?? null);
      if (nextReadout !== readout) {
        readout = nextReadout;
        context.reportReadout?.(readout);
      }
      framesToDraw = Math.max(framesToDraw, 2);
      // A buffer peeked for the first time is read now, unless a run is about to refresh them all.
      if (hasNewPeek && !needsRun) void refreshPeeks();
    };
    /**
     * Jumps the resolved position to the pointer. With `flushSettle` it also applies pending
     * compile-time values now (verification); otherwise their debounce keeps running.
     */
    const finishPointer = (flushSettle = false) => {
      if (!interaction) return;
      overlay.setSpotlightOn(isProbing);
      pointerState.isGliding = false;
      const target = pointerState.target;
      if (
        target &&
        (target[0] !== pointerState.effective[0] || target[1] !== pointerState.effective[1])
      ) {
        pointerState.effective = [target[0], target[1]];
        applyPointer();
      }
      if (flushSettle && pendingSettle) applySettle();
    };
    /**
     * Moves the resolved position to the pointer; called every animation frame. A pointer that has
     * just arrived is reached by a short glide from the rest position; after that the position is
     * the pointer itself, so handles stay exactly under the cursor.
     */
    const advancePointer = (time: number) => {
      const elapsed = Math.min(100, Math.max(0, time - pointerState.lastTime));
      pointerState.lastTime = time;
      const target = pointerState.target;
      if (!interaction) return;
      if (overlay.fadeSpotlight(isProbing, elapsed)) framesToDraw = Math.max(framesToDraw, 1);
      if (!target) return;
      const [x, y] = pointerState.effective;
      const deltaX = target[0] - x;
      const deltaY = target[1] - y;
      if (deltaX === 0 && deltaY === 0) return;
      if (!pointerState.isGliding || Math.hypot(deltaX, deltaY) < 0.004) {
        pointerState.isGliding = false;
        pointerState.effective = [target[0], target[1]];
      } else {
        const amount = 1 - Math.exp(-elapsed / POINTER_GLIDE_MILLISECONDS);
        pointerState.effective = [x + deltaX * amount, y + deltaY * amount];
      }
      applyPointer();
    };

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
          isContext && hasValues
            ? contextRamp
            : ramp === 'diverging'
              ? theme.diverging
              : ramp === 'grey'
                ? GREY_RAMP
                : undefined,
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
      return {...resolveLook(layer, layerObject), clip: layer.clip};
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
        case 'text':
          // Draws again once the lazily loaded atlas arrives (only when it was not preloaded).
          return {
            layer: new TextLayer(device, () => {
              framesToDraw = 3;
              scheduler.request();
            }),
            owned: []
          };
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
            look: {
              ...getLook(layer, layer.values),
              radius: layer.radius ?? 2.4,
              hillshade: 0.5,
              sizeByValue: layer.sizeByValue
            }
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
        case 'text':
          // Laid out in TextLayer.update, which reruns the layout only when the text or view changes.
          (layerObject as TextLayer).setText(layer, {
            displayWidth: context.displayWidth,
            context: layer.context,
            alpha: layer.alpha
          });
          return {layer: layerObject, look: {}};
      }
    };

    // Batch auto ranges through one reusable staging slot when they fit the device's buffer limit.
    // Larger batches use separate reads so their combined size does not limit readable inputs.
    let rangeRing: GPUReadbackRing | null = null;
    /** Copies the wanted buffers into the staging slot and measures each one's range. */
    const readRanges = async (): Promise<(readonly [Buffer, [number, number] | null])[]> => {
      if (isDestroyed) return [];
      const wanted: {
        buffer: Buffer;
        format: ValueFormat;
        byteOffset: number;
        byteLength: number;
      }[] = [];
      let byteLength = 0;
      for (const [buffer, format] of wantedRanges) {
        // Buffers of a replaced `afterRun` result are destroyed: stop measuring them.
        if (buffer.destroyed) {
          wantedRanges.delete(buffer);
          continue;
        }
        const size = Math.floor(buffer.byteLength / 4) * 4;
        if (size === 0) continue;
        wanted.push({buffer, format, byteOffset: byteLength, byteLength: size});
        byteLength += size;
      }
      if (wanted.length === 0) return [];
      if (
        byteLength > device.limits.maxBufferSize ||
        wanted.some(source => (source.buffer.usage & Buffer.COPY_SRC) === 0)
      ) {
        return Promise.all(
          wanted.map(async source => {
            try {
              const bytes = await source.buffer.readAsync();
              if (isDestroyed) return [source.buffer, null] as const;
              return [source.buffer, readRange(bytes, source.format)] as const;
            } catch {
              return [source.buffer, null] as const;
            }
          })
        );
      }
      if (!rangeRing || rangeRing.byteLength < byteLength) {
        rangeRing?.destroy();
        rangeRing = null;
        rangeRing = new GPUReadbackRing(device, {
          id: 'run-ranges',
          byteLength,
          slotCount: 1
        });
      }
      // One read is in flight at a time, so the single slot is free here.
      const ticket = rangeRing.tryAcquire();
      if (!ticket) return [];
      let isEncoded = false;
      let hasStartedRead = false;
      try {
        if (isDestroyed) return [];
        const commandEncoder = device.createCommandEncoder({id: 'run-ranges-encoder'});
        try {
          for (const source of wanted) {
            commandEncoder.copyBufferToBuffer({
              sourceBuffer: source.buffer,
              destinationBuffer: ticket.buffer,
              destinationOffset: source.byteOffset,
              size: source.byteLength
            });
          }
          const commandBuffer = commandEncoder.finish();
          ticket.markEncoded({byteOffset: 0, byteLength});
          isEncoded = true;
          device.submit(commandBuffer);
        } finally {
          commandEncoder.destroy();
        }
        hasStartedRead = true;
        const bytes = await ticket.read();
        if (isDestroyed) return [];
        return wanted.map(
          source =>
            [
              source.buffer,
              readRange(
                bytes.subarray(source.byteOffset, source.byteOffset + source.byteLength),
                source.format
              )
            ] as const
        );
      } finally {
        if (isEncoded && !hasStartedRead) {
          // Submission can throw after queuing work. Drain the encoded slot before reusing it.
          const discarded = ticket.read();
          ticket.cancel();
          await discarded.catch(() => undefined);
        } else {
          // Unsubmitted reservations release immediately; read() releases mapped slots itself.
          ticket.cancel();
        }
      }
    };
    /** The range read in flight, including the follow-up it owes to requests made meanwhile. */
    let pendingRanges: Promise<void> | null = null;
    let isRangeStale = false;
    let needsExtraRanges = false;
    /**
     * Measures the auto ranges of the drawn buffers. One read is in flight at a time; a request
     * made meanwhile marks it stale and exactly one follow-up read of the latest state happens
     * when it finishes. The returned promise resolves after that follow-up, so `await` always
     * sees the ranges of the latest run.
     */
    const refreshRanges = (): Promise<void> => {
      if (pendingRanges) {
        isRangeStale = true;
        return pendingRanges;
      }
      const task = (async () => {
        try {
          do {
            isRangeStale = false;
            await applyRanges();
          } while (isRangeStale && !isDestroyed);
        } finally {
          // Cleared in the same step as the last stale check, so no request can fall between.
          pendingRanges = null;
        }
      })();
      pendingRanges = task;
      return task;
    };
    const applyRanges = async () => {
      let measured: (readonly [Buffer, [number, number] | null])[];
      try {
        measured = await readRanges();
      } catch {
        // The device or a source buffer went away while the read was in flight.
        return;
      }
      if (isDestroyed) return;
      let changed = false;
      for (const [buffer, range] of measured) {
        if (!range) continue;
        const previous = ranges.get(buffer);
        if (!previous || previous[0] !== range[0] || previous[1] !== range[1]) changed = true;
        ranges.set(buffer, range);
      }
      if (changed) {
        framesToDraw = 3;
        // Without an animation loop (reduced motion) nothing else would draw the new range.
        scheduler.request();
      }
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
            extra: extra as Extra,
            interaction: pointerState.result,
            pointer: pointerState.pointer
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
      const glyphLayers = overlay.getLayers(view, theme);
      const renderPass = device.beginRenderPass({
        clearColor: [...theme.background] as [number, number, number, number]
      });
      // Read after the pass has acquired its framebuffer, which may resize the drawing buffer.
      const [bufferWidth, bufferHeight] = device.getDefaultCanvasContext().getDrawingBufferSize();
      let isClipped = false;
      for (const resolved of drawLayersList) {
        if (resolved.clip && Number.isFinite(resolved.clip.x)) {
          // Scissor in framebuffer pixels: the layer keeps only one side of the curtain.
          const curtain = Math.min(
            bufferWidth,
            Math.max(
              0,
              Math.round((getPixelOfWorld(view, resolved.clip.x, 0)[0] / view.width) * bufferWidth)
            )
          );
          const left = resolved.clip.side === 'left' ? 0 : curtain;
          const width = resolved.clip.side === 'left' ? curtain : bufferWidth - curtain;
          if (width <= 0) continue;
          renderPass.setParameters({scissorRect: [left, 0, width, bufferHeight]});
          isClipped = true;
        } else if (isClipped) {
          renderPass.setParameters({scissorRect: [0, 0, bufferWidth, bufferHeight]});
          isClipped = false;
        }
        resolved.layer.draw(renderPass);
      }
      if (isClipped) renderPass.setParameters({scissorRect: [0, 0, bufferWidth, bufferHeight]});
      for (const layer of glyphLayers) layer.draw(renderPass);
      renderPass.end();
      device.submit();
      if (needsExtraRanges) {
        // Resolving the new layers above registers their auto-range buffers before this read.
        needsExtraRanges = false;
        void refreshRanges();
      }
    };

    const runOperation = () => {
      current.result.update?.(values);
      current.runner.run();
      runCount++;
    };

    /**
     * Runs `afterRun` once for `built`. Its result is committed unless a rebuild has replaced the
     * build meanwhile (the result could describe destroyed buffers).
     */
    const runAfter = async (built: Built): Promise<void> => {
      const afterRun = spec.afterRun;
      if (!afterRun) return;
      const generation = ++afterGeneration;
      const staged = new RunBufferSet();
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
        // A rebuild may have destroyed what this one was reading.
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
      needsExtraRanges = true;
      framesToDraw = 3;
      scheduler.request();
    };

    /** The build whose newest run still needs an `afterRun` pass after the one in flight. */
    let queuedAfterBuilt: Built | null = null;
    /**
     * Runs `afterRun` for the build that just ran. One pass is in flight at a time: a run that
     * finishes meanwhile queues exactly one follow-up pass for the latest state, which starts when
     * the pass in flight has committed. A rebuild drops the pass in flight and its queued
     * follow-up; the new build's first run asks again. The returned promise resolves after the
     * follow-up, so `await` sees the `extra` of the latest run.
     */
    const refreshExtra = (): Promise<void> => {
      if (!spec.afterRun || isDestroyed) return Promise.resolve();
      if (pendingAfter) {
        queuedAfterBuilt = current;
        return pendingAfter;
      }
      const task = (async () => {
        let built: Built | null = current;
        let failure: {error: unknown} | null = null;
        try {
          while (built && !isDestroyed) {
            queuedAfterBuilt = null;
            try {
              await runAfter(built);
            } catch (error) {
              failure = {error};
            }
            if (isDestroyed) return;
            built = queuedAfterBuilt && queuedAfterBuilt === current ? queuedAfterBuilt : null;
          }
        } finally {
          // Cleared in the same step as the last queue check, so no request can fall between.
          queuedAfterBuilt = null;
          pendingAfter = null;
        }
        if (failure) throw failure.error;
      })();
      // The first caller handles `task`'s rejection; later callers wait on this copy.
      pendingAfter = task.catch(() => undefined);
      return task;
    };

    const rebuild = () => {
      rebuildPromise ??= (async () => {
        needsRebuild = false;
        const nextKey = getBuildKey();
        const cached = builtCache.get(nextKey);
        if (cached) builtCache.delete(nextKey);
        const next = cached ?? (await createBuilt());
        if (isDestroyed) {
          destroyBuilt(next);
          return;
        }
        disposeLayers();
        wantedRanges.clear();
        peeks.clear();
        peekEpoch++;
        buildNumber++;
        // Drops the `afterRun` pass in flight and its queued follow-up: they describe the old build.
        afterGeneration++;
        queuedAfterBuilt = null;
        retireBuilt(currentKey, current);
        current = next;
        currentKey = nextKey;
        if (!cached) rebuildCount++;
        needsRun = true;
        isAwaitingFirstRun = true;
        // The new build has new buffers: the gesture subscribes its readbacks again and rewrites
        // whatever it draws from them.
        // (Unless the values have already moved on to yet another build.)
        if (interaction && !needsRebuild) {
          applyPointer({applyValues: pointerOwnsValues, isAfterRebuild: true});
        }
        framesToDraw = 3;
      })()
        .catch(error => {
          // Device teardown can race a queued compilation after a card releases its live slot.
          if (!isDestroyed) {
            // biome-ignore lint/suspicious/noConsole: surface asynchronous graph errors
            console.error('Could not rebuild operation', error);
          }
        })
        .finally(() => {
          rebuildPromise = null;
          if (isDestroyed) return;
          if (needsRebuild) rebuild();
          else scheduler.request(); // Also present the completed build in reduced-motion mode.
        });
    };

    const draw = () => {
      if (isDestroyed) return;
      if (needsRebuild) rebuild();
      // A build that was just swapped in is drawn only after it has run.
      if (isAwaitingFirstRun && rebuildPromise) return;
      if (needsRun && !rebuildPromise) {
        needsRun = false;
        isAwaitingFirstRun = false;
        runOperation();
        framesToDraw = 3;
        render();
        void refreshRanges();
        void refreshPeeks();
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
    if (interaction) {
      // The picture starts at the gesture's rest position: the poster.
      const rest =
        typeof interaction.rest === 'function'
          ? interaction.rest(getInteractionState())
          : (interaction.rest ?? [0.5, 0.5]);
      pointerState.rest = [rest[0], rest[1]];
      pointerState.effective = [rest[0], rest[1]];
      // Keep slider values the page restored (a theme change restarts the scene with them).
      const isUntouched = parameters.every(parameter => values[parameter.id] === parameter.default);
      pointerOwnsValues = isUntouched;
      applyPointer({applyValues: isUntouched});
      restValues = {...pointerState.result.values, ...pointerState.result.settle};
      if (needsRun) {
        needsRun = false;
        runOperation();
        await refreshExtra();
      }
      await refreshPeeks();
      if (needsRun) {
        needsRun = false;
        runOperation();
        await refreshExtra();
      }
    }
    // A picture with text waits for the font atlas (a lazy chunk) so its first frame, and the
    // poster, already show the labels. Runs without text never load it.
    if (getLayerList().some(layer => layer.layer === 'text')) await loadFontPixels();
    render();
    await refreshRanges();

    const instance: RunInstance = {
      // Prefer an adjustable uniform over a compile-time setting for pointer scrubbing.
      scrubParameter:
        parameters.find(parameter => parameter.cardScrub) ??
        parameters.find(
          parameter => current.result.update && !rebuildIds.has(parameter.id) && !parameter.options
        ),
      gesture: interaction
        ? {
            kind: interaction.kind,
            hint: interaction.hint,
            rest: [pointerState.rest[0], pointerState.rest[1]]
          }
        : undefined,
      render(time) {
        advancePointer(time);
        if (framesToDraw > 0 || needsRun || needsRebuild) {
          framesToDraw = Math.max(framesToDraw - 1, 0);
          scheduler.flush();
        }
      },
      resize(width, height) {
        view.width = width;
        view.height = height;
        framesToDraw = 3;
        if (interaction) applyPointer({applyValues: false});
        scheduler.request();
      },
      setParameters(update) {
        if (interaction) {
          // A slider now owns the values: stop following the pointer until it moves again.
          pointerState.target = null;
          isProbing = false;
          pointerOwnsValues = false;
          peekDrivenRunCount = 0;
          clearTimeout(settleTimer);
          pendingSettle = null;
        }
        changeValues(update);
        // Glyphs, readouts and render-only buffers the gesture derives from the values follow them.
        if (interaction) applyPointer({applyValues: false});
      },
      setPointer: interaction
        ? position => {
            // Off the picture (but still on the card): hold the last state.
            if (!position) return;
            // A pointer that arrives (first hover, or after a slider took over) glides in.
            if (!pointerState.target && !context.reducedMotion) pointerState.isGliding = true;
            isProbing = true;
            pointerOwnsValues = true;
            peekDrivenRunCount = 0;
            pointerState.target = [
              Math.min(1, Math.max(0, position[0] / view.width)),
              Math.min(1, Math.max(0, position[1] / view.height))
            ];
            if (context.reducedMotion) finishPointer();
            else framesToDraw = Math.max(framesToDraw, 1);
            scheduler.request();
          }
        : undefined,
      resetPointer: interaction
        ? () => {
            // Straight back to rest: a glide would write every value on the way over the defaults.
            pointerState.target = null;
            pointerState.isGliding = false;
            pointerState.effective = [pointerState.rest[0], pointerState.rest[1]];
            isProbing = false;
            pointerOwnsValues = true;
            peekDrivenRunCount = 0;
            overlay.setSpotlightOn(false);
            applyPointer();
            scheduler.request();
          }
        : undefined,
      destroy() {
        isDestroyed = true;
        scheduler.cancel();
        clearTimeout(settleTimer);
        disposeLayers();
        overlay.destroy();
        if (dummyValues) dummyValues.destroy();
        extraBuffers?.destroy();
        rangeRing?.destroy();
        destroyBuilt(current);
        for (const built of builtCache.values()) destroyBuilt(built);
        builtCache.clear();
      },
      debug: {
        setInputsOnly(flag) {
          isInputsOnly = flag;
          framesToDraw = 3;
          scheduler.request();
        },
        async settle() {
          if (isDestroyed) return;
          scheduler.cancel();
          const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
          finishPointer(true);
          const hasPendingWork = () =>
            Boolean(
              rebuildPromise ||
                needsRebuild ||
                needsRun ||
                pendingAfter ||
                pendingPeek ||
                pendingSettle
            );
          for (let attempt = 0; attempt < 400 && hasPendingWork(); attempt++) {
            // A readback can arm the settle debounce again: apply it rather than wait it out.
            if (pendingSettle && !rebuildPromise) applySettle();
            draw();
            await nextFrame();
            if (isDestroyed) return;
          }
          if (hasPendingWork()) {
            // biome-ignore lint/suspicious/noConsole: make captures of unfinished work visible
            console.warn(
              'Operation run did not settle within 400 animation frames; drawing the latest committed state.'
            );
          }
          render();
          await refreshRanges();
          if (isDestroyed) return;
          render();
          await nextFrame();
        },
        getRunCount: () => runCount,
        getRebuildCount: () => rebuildCount,
        getLayerKinds: () => kinds,
        getReadout: () => readout,
        getRestValues: () => restValues,
        getPointerRebuildCount: () => pointerRebuildCount,
        getPointerParameterIds: () => [...pointerParameterIds]
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
    case 'text':
      return [];
  }
}
