// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  createGPUComputationCaptures,
  createGPUComputationRecord,
  describeGPUCommandGraph,
  getGPUComputationAdapter,
  validateGPUComputationRecord,
  validateGPUComputationSidecars,
  type GPUAnalysisChoice,
  type GPUComputationCapture,
  type GPUComputationCaptureElementType,
  type GPUComputationCaptureGeometry,
  type GPUComputationFormulaRef,
  type GPUComputationKeySpace,
  type GPUComputationRelation
} from '@luma-contributors/analysis/computation-record';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {mergeGPUCoreFormulaRefsForGraphs} from '../../../computation-views/packs/gpu-core/formulas';
import type {LensScenarioResult} from '../../types';
import {
  addLensViewSnapshot,
  getLensRecordSourceText,
  LensCommandGraph,
  readLensRunMetrics,
  readLensViewSnapshot,
  submitTimedLensGraph
} from '../../lens-view-snapshot';

/** Storage formats a lens can upload or capture. */
export type RasterLensFormat = 'float32' | 'uint32' | 'sint32';

/** How one captured view appears in the record. */
export type RasterLensCaptureSpec = {
  /** Role in the element's input → output figure; captures without a role are intermediates. */
  role?: 'input' | 'output' | 'context' | 'intermediate';
  /**
   * `'raster'` draws the view on the lens grid (one value per cell). Vector outputs pass explicit
   * point, line or polygon geometry with their offsets and values capture names.
   */
  geometry?: 'raster' | GPUComputationCaptureGeometry;
  /** Reader-facing meaning, such as `elevation (m)` or `settledAt round`. */
  meaning?: string;
  /** The single answer the card figure points at (output captures only). */
  highlight?: {key: number; label: string};
  /**
   * Picks the highlight from the captured final values (finite values only): the cell with the
   * largest or smallest value, labelled from that value. Never chosen by a view.
   */
  highlightExtreme?: {pick: 'max' | 'min'; label: (value: number, key: number) => string};
  /** Key space id; raster captures default to `cells`. */
  keySpace?: string;
  /** Values per key (2 for vec2 coordinates). */
  components?: number;
  /** File the final value under per-datum stamps instead of snapshots (settledAt rounds). */
  stamp?: boolean;
};

/** Upload, output and capture helpers handed to a scenario's `build`. */
export type RasterLensSession = {
  device: Device;
  /** Recording graph the example adds its contributor to. */
  graph: LensCommandGraph;
  /** Uploads values as an imported, capturable storage view and optionally captures it. */
  input<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    values: ArrayLike<number>,
    spec?: RasterLensCaptureSpec | false
  ): GraphDataView<Format>;
  /** Creates a zero-filled imported output view (default one value per grid cell) and captures it. */
  output<Format extends RasterLensFormat>(
    name: string,
    format: Format,
    spec?: RasterLensCaptureSpec | false,
    length?: number
  ): GraphDataView<Format>;
  /** Captures an existing packed view under `name`. */
  capture(name: string, view: GraphDataView, spec?: RasterLensCaptureSpec): void;
  /** Destroys `resource` after the run (parameter buffers created by the example). */
  track(resource: {destroy(): void} | undefined): void;
};

/** Properties of {@link runRasterLens}. */
export type RunRasterLensProps = {
  device: Device;
  /** Scenario id; also the record id. */
  scenarioId: string;
  /** Entry id, also the record domain: `gpu-terrain` or `gpu-raster`. */
  domain: string;
  /** Human label of the computation graph. */
  label: string;
  /** Lens grid: shape and cell size of every `'raster'` capture. */
  grid: GPURasterMetadata;
  /** `projected` (metres) or `geographic` (degrees). */
  coordinateModel: 'projected' | 'geographic';
  /** Horizontal unit of the grid, such as `m` or `deg`. */
  unit: string;
  /** Optional local frame identity, retained on captures for spatial comparison. */
  frameId?: string;
  /** CRS identity when the dataset supplies it beside the grid. */
  crs?: string;
  /** Dataset id and version stamped into the run id. */
  dataset: {id: string; version: string};
  /** Analyst choices the run actually used. */
  choices?: readonly GPUAnalysisChoice[];
  /** Formula refs (the pack's curated formula for this element). */
  formulas?: readonly GPUComputationFormulaRef[];
  /** Extra key spaces (observers, vertices, particles, zones); `cells` is always present. */
  keySpaces?: readonly GPUComputationKeySpace[];
  relations?: readonly GPUComputationRelation[];
  /** Adds the contributor and declares captures. */
  build(session: RasterLensSession): Promise<void>;
};

const CAPTURE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST;
const ELEMENT_TYPES: Record<RasterLensFormat, GPUComputationCaptureElementType> = {
  float32: 'f32',
  uint32: 'u32',
  sint32: 'i32'
};

type RegisteredCapture = {name: string; view: GraphDataView; spec: RasterLensCaptureSpec};

/**
 * Runs one raster or terrain lens scenario on the real device: builds a recording graph, lets the
 * scenario add its element through the example code, compiles, describes, executes once (in steps
 * when key frames are requested) and returns the validated record with its captures sidecar.
 */
export async function runRasterLens(props: RunRasterLensProps): Promise<LensScenarioResult> {
  const {device, grid} = props;
  const cellCount = grid.width * grid.height;
  const graph = new LensCommandGraph(device, {id: props.scenarioId.replace(/\./g, '-')});
  const owned: {destroy(): void}[] = [];
  const captures: RegisteredCapture[] = [];
  const register = (name: string, view: GraphDataView, spec: RasterLensCaptureSpec | false) => {
    if (spec === false) return;
    // Capture names are the record's join keys; a duplicate would silently shadow a figure layer.
    if (captures.some(capture => capture.name === name)) {
      throw new Error(`${props.scenarioId}: duplicate capture ${name}`);
    }
    captures.push({name, view, spec});
  };
  const importValues = <Format extends RasterLensFormat>(
    name: string,
    format: Format,
    data: ArrayBufferView
  ): GraphDataView<Format> => {
    const buffer = device.createBuffer({id: `${graph.id}-${name}`, data, usage: CAPTURE_USAGE});
    owned.push(buffer);
    const handle = graph.importBuffer(
      {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    return graph.createDataView(handle, {format, length: data.byteLength / 4});
  };
  const session: RasterLensSession = {
    device,
    graph,
    input: (name, format, values, spec = {role: 'input', geometry: 'raster'}) => {
      const view = importValues(name, format, toTypedArray(format, values));
      register(name, view, spec);
      return view;
    },
    output: (name, format, spec = {role: 'output', geometry: 'raster'}, length = cellCount) => {
      const view = importValues(name, format, toTypedArray(format, new Array(length).fill(0)));
      register(name, view, spec);
      return view;
    },
    capture: (name, view, spec = {}) => register(name, view, spec),
    track: resource => {
      if (resource) owned.push(resource);
    }
  };
  try {
    await props.build(session);
    const snapshot = addLensViewSnapshot(graph, device, {
      id: graph.id,
      views: Object.fromEntries(captures.map(capture => [capture.name, capture.view])),
      elementTypes: Object.fromEntries(
        captures.map(capture => [
          capture.name,
          ELEMENT_TYPES[capture.view.format as RasterLensFormat] ?? 'u32'
        ])
      )
    });
    const compiled = await graph.compileAsync();
    try {
      const description = describeGPUCommandGraph(compiled, {
        label: props.label,
        domain: props.domain,
        graph,
        sourceText: getLensRecordSourceText(props.scenarioId)
      });
      const record = createGPUComputationRecord({
        id: props.scenarioId,
        adapter: getGPUComputationAdapter(device),
        domains: [props.domain],
        graphs: [description],
        axes: {
          keySpaces: [
            {
              id: 'cells',
              label: 'grid cells',
              size: cellCount,
              indexSpace: '2d',
              shape: [grid.width, grid.height]
            },
            ...(props.keySpaces ?? [])
          ],
          relations: props.relations ?? []
        },
        formulas: mergeGPUCoreFormulaRefsForGraphs([description], props.formulas),
        choices: props.choices,
        datasetVersion: `${props.dataset.id}@${props.dataset.version}`
      });
      const graphId = description.graph.id;
      const submission = submitTimedLensGraph(device, compiled, {
        parameters: undefined,
        coalesceComputePasses: false
      });
      // Node timings are present only when the device wrote timestamps; never estimated.
      const metrics = await readLensRunMetrics(submission, {
        runId: record.runId,
        executionId: record.executionId,
        graphId,
        taps: 1
      });
      const readback = await readLensViewSnapshot(snapshot, {graphId, runId: record.runId});
      const sidecar = arrangeCaptures(props, captures, readback);
      const captureSidecar = createGPUComputationCaptures({
        runId: record.runId,
        executionId: record.executionId,
        snapshots: sidecar.snapshots,
        stamps: sidecar.stamps
      });
      assertValid(props.scenarioId, validateGPUComputationRecord(record));
      assertValid(
        props.scenarioId,
        validateGPUComputationSidecars(record, {metrics: [metrics], captures: captureSidecar})
      );
      return {...record, metrics: [metrics], captures: captureSidecar};
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const resource of owned) resource.destroy();
  }
}

/** Applies roles and geometry to the end-of-run readbacks and files stamps. */
export function arrangeCaptures(
  props: RunRasterLensProps,
  registered: readonly RegisteredCapture[],
  raw: readonly GPUComputationCapture[]
): {snapshots: GPUComputationCapture[]; stamps: GPUComputationCapture[]} {
  const snapshots: GPUComputationCapture[] = [];
  const stamps: GPUComputationCapture[] = [];
  for (const capture of registered) {
    const frames = raw.filter(candidate => candidate.name === capture.name);
    const finalFrame = frames.at(-1);
    // The snapshot holds one frame per requested view.
    if (!finalFrame) throw new Error(`${props.scenarioId}: no capture for ${capture.name}`);
    const decorate = (frame: GPUComputationCapture): GPUComputationCapture => ({
      ...frame,
      ...(capture.spec.role && {role: capture.spec.role}),
      ...(capture.spec.meaning && {meaning: capture.spec.meaning}),
      ...(capture.spec.highlight && {highlight: capture.spec.highlight}),
      ...(capture.spec.components && {components: capture.spec.components}),
      ...((capture.spec.keySpace || capture.spec.geometry === 'raster') && {
        keySpace: capture.spec.keySpace ?? 'cells'
      }),
      ...(capture.spec.geometry && {
        geometry:
          capture.spec.geometry === 'raster'
            ? getRasterLensGeometry(props)
            : capture.spec.geometry.kind === 'raster'
              ? getRasterLensGeometry(props, capture.spec.geometry)
              : capture.spec.geometry
      })
    });
    const extreme = capture.spec.highlightExtreme;
    if (extreme) {
      const key = findExtreme(finalFrame.values, extreme.pick);
      if (key !== undefined) {
        capture.spec = {
          ...capture.spec,
          highlight: {key, label: extreme.label(finalFrame.values[key], key)}
        };
      }
    }
    if (capture.spec.stamp) {
      stamps.push(decorate(finalFrame));
    } else {
      snapshots.push(decorate(finalFrame));
    }
  }
  return {snapshots, stamps};
}

function findExtreme(values: ArrayLike<number>, pick: 'max' | 'min'): number | undefined {
  let best: number | undefined;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    if (best === undefined || (pick === 'max' ? value > values[best] : value < values[best])) {
      best = index;
    }
  }
  return best;
}

function toTypedArray(format: RasterLensFormat, values: ArrayLike<number>): ArrayBufferView {
  switch (format) {
    case 'float32':
      return Float32Array.from(values);
    case 'uint32':
      return Uint32Array.from(values);
    default:
      return Int32Array.from(values);
  }
}

function assertValid(
  scenarioId: string,
  report: {valid: boolean; issues: readonly {level: string; path: string; message: string}[]}
): void {
  if (report.valid) return;
  const errors = report.issues
    .filter(issue => issue.level === 'error')
    .map(issue => `${issue.path}: ${issue.message}`);
  throw new Error(`${scenarioId}: invalid record\n${errors.join('\n')}`);
}

/** Records full ground placement; scalar-only renderers use pixel space for nonsquare grids. */
export function getRasterLensGeometry(
  props: Pick<RunRasterLensProps, 'grid' | 'coordinateModel' | 'unit' | 'frameId' | 'crs'>,
  explicit?: Extract<GPUComputationCaptureGeometry, {kind: 'raster'}>
): GPUComputationCaptureGeometry {
  const [columnX, rowX, originX, columnY, rowY, originY] = props.grid.affine;
  const square = columnX > 0 && rowX === 0 && columnY === 0 && Math.abs(rowY) === columnX;
  const shift = props.grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const reference = props.grid.coordinateReferenceSystem;
  const crs =
    props.crs ??
    reference?.authority ??
    reference?.wellKnownText ??
    (reference?.projectionJson
      ? JSON.stringify(canonicalizeIdentity(reference.projectionJson))
      : undefined);
  if (explicit)
    return {
      ...explicit,
      shape: [...explicit.shape] as const,
      ...(explicit.affine ? {affine: [...explicit.affine] as const} : {}),
      ...(explicit.origin ? {origin: [...explicit.origin] as const} : {}),
      ...(explicit.crs === undefined && crs ? {crs} : {}),
      ...(explicit.frameId === undefined && props.frameId ? {frameId: props.frameId} : {})
    };
  return {
    kind: 'raster',
    shape: [props.grid.width, props.grid.height],
    affine: [...props.grid.affine],
    pixelInterpretation: props.grid.pixelInterpretation,
    coordinateModel: props.coordinateModel,
    unit: props.unit,
    ...(crs ? {crs} : {}),
    ...(props.frameId ? {frameId: props.frameId} : {}),
    ...(square
      ? {
          cellSize: columnX,
          rowDirection: rowY > 0 ? 'north' : 'south',
          origin: [originX + shift * columnX, originY + shift * rowY] as const
        }
      : {})
  };
}

/** Object-key order carries no CRS meaning; array order still does. */
function canonicalizeIdentity(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalizeIdentity);
  if (value !== null && typeof value === 'object')
    return Object.fromEntries(
      Object.entries(value)
        .sort(([first], [second]) => first.localeCompare(second))
        .map(([key, item]) => [key, canonicalizeIdentity(item)])
    );
  return value;
}
