// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared by the computation-record runs (computation-recorder, computation-record,
// computation-metrics, computation-captures, snapshot-arena, snapshot-tap). Their subject is a
// recording of another run, so they all record the same small real analysis: a relaxation that
// smooths noisy tract values with GPUSpatialLag, one iteration after another, and measures how
// much each iteration changed them. This file builds that graph (optionally through a
// GPUComputationRecorder and with snapshot taps) and draws records: command-graph boxes and
// edges, small multiples of captured buffers, and bars. Drawing is CPU geometry that `afterRun`
// uploads, because a record is data, not a GPU buffer.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import {
  createGPUKernelAnnotation,
  GPUSnapshotTap,
  type GPUComputationEdge,
  type GPUComputationGraph,
  type GPUComputationNode,
  type GPUComputationNodeAnnotation,
  type GPUComputationRecordScope,
  type GPUComputationRecorder,
  type GPUSnapshotArena
} from '@luma-contributors/analysis/computation-record';
import {GPUSpatialLag} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder, RunLayer} from './define-run';
import {
  createTracts,
  uploadSpatialWeights,
  type Random,
  type TractsInput,
  type World
} from './inputs';

// ---------------------------------------------------------------------------------------------
// The recorded analysis

/** Tracts for the recorded relaxation, with the world they were generated in. */
export type RecordTracts = TractsInput & {world: World};

/** Roughly fifty noisy tracts: enough to see smoothing, few enough to draw many small copies. */
export function createRecordTracts(random: Random): RecordTracts {
  const world = {width: 1.25, height: 1};
  return {...createTracts(random, world, 7, 0.7), world};
}

/** One snapshot tap request: which buffers go into which arena slots. */
export type RelaxationTap = {
  arena: GPUSnapshotArena;
  /** Buffers keyed by arena slot name. */
  views: Readonly<Record<string, RunBuffer<'float32'>>>;
  label?: string;
  iteration?: number;
};

/** Buffers of one relaxation iteration. */
export type RelaxationIteration = {
  iteration: number;
  /** The smoothed values after this iteration. */
  state: RunBuffer<'float32'>;
  /** Absolute change from the previous iteration. */
  residual: RunBuffer<'float32'>;
  /** Signed difference from the initial values, when requested. */
  deviation?: RunBuffer<'float32'>;
  /** The last node of the iteration; a tap placed after it follows it. */
  lastNodeId: string;
};

export type RelaxationOptions = {
  iterations: number;
  /** When set, every node goes through the recorder, so the described graph carries annotations. */
  recorder?: GPUComputationRecorder;
  /** Scope the recorder wraps around the nodes of an iteration. */
  scope?: (iteration: number) => GPUComputationRecordScope | undefined;
  /** Also compute the signed difference from the initial values. */
  deviation?: boolean;
  /** Returns the tap to add after the iteration's last node. */
  tap?: (iteration: RelaxationIteration) => RelaxationTap | undefined;
};

export type Relaxation = {
  /** The initial noisy values (state 0). */
  initial: RunBuffer<'float32'>;
  iterations: RelaxationIteration[];
};

const RESIDUAL_BODY =
  'residualOutput[residualOutputOffset + index] = abs(residualCurrent[residualCurrentOffset + index] - residualPrevious[residualPreviousOffset + index]);';
const DEVIATION_BODY =
  'deviationOutput[deviationOutputOffset + index] = deviationCurrent[deviationCurrentOffset + index] - deviationInitial[deviationInitialOffset + index];';

/**
 * Builds the relaxation into the run's graph: per iteration a real GPUSpatialLag (row-standardized
 * rook neighbors), a residual kernel and optionally a deviation kernel and a snapshot tap.
 */
export function buildRelaxation(
  builder: RunBuilder,
  tracts: RecordTracts,
  options: RelaxationOptions
): Relaxation {
  const {graph} = builder;
  const {recorder} = options;
  const weights = uploadSpatialWeights(builder, tracts.contiguity);
  const initial = builder.upload('state-0', 'float32', tracts.values);
  const addNode = (node: GPUCommandNode<void>, annotation?: GPUComputationNodeAnnotation) => {
    if (recorder) recorder.addNode(graph, node, annotation);
    else graph.add(node);
  };
  const addKernel = (
    id: string,
    operation: string,
    body: string,
    bindings: {name: string; buffer: RunBuffer<'float32'>; access: 'read' | 'read_write'}[],
    iteration: number
  ) => {
    const node = createWGSLKernelNode(graph, {
      id,
      operation,
      bindings: bindings.map(({name, buffer, access}) => ({
        name,
        view: buffer.view,
        type: 'f32',
        access
      })),
      invocationCount: tracts.count,
      body
    });
    addNode(
      node,
      recorder &&
        createGPUKernelAnnotation({
          op: operation,
          moduleId: `relaxation/${operation.toLowerCase()}`,
          source: body,
          workgroupSize: [64, 1, 1],
          workgroups: [Math.ceil(tracts.count / 64), 1, 1],
          invocationCount: tracts.count,
          bindings: bindings.map(({name, buffer, access}) => ({
            name,
            view: buffer.view,
            access
          })),
          loop: {id: 'relaxation', iteration: iteration - 1, maximumIterations: options.iterations},
          dedupName: operation.toLowerCase()
        })
    );
  };

  const iterations: RelaxationIteration[] = [];
  let previous = initial;
  for (let iteration = 1; iteration <= options.iterations; iteration++) {
    const state = builder.allocate(`state-${iteration}`, 'float32', tracts.count);
    const residual = builder.allocate(`residual-${iteration}`, 'float32', tracts.count);
    const deviation = options.deviation
      ? builder.allocate(`deviation-${iteration}`, 'float32', tracts.count)
      : undefined;
    const previousState = previous;
    const addIteration = () => {
      const lag = new GPUSpatialLag({
        id: `relax-${iteration}`,
        values: previousState.view,
        weights,
        normalize: true,
        output: state.view
      });
      for (const node of lag.getCommandNodes(graph)) {
        addNode(node, {
          loop: {id: 'relaxation', iteration: iteration - 1, maximumIterations: options.iterations}
        });
      }
      addKernel(
        `residual-${iteration}`,
        'Residual',
        RESIDUAL_BODY,
        [
          {name: 'residualPrevious', buffer: previousState, access: 'read'},
          {name: 'residualCurrent', buffer: state, access: 'read'},
          {name: 'residualOutput', buffer: residual, access: 'read_write'}
        ],
        iteration
      );
      if (deviation) {
        addKernel(
          `deviation-${iteration}`,
          'Deviation',
          DEVIATION_BODY,
          [
            {name: 'deviationInitial', buffer: initial, access: 'read'},
            {name: 'deviationCurrent', buffer: state, access: 'read'},
            {name: 'deviationOutput', buffer: deviation, access: 'read_write'}
          ],
          iteration
        );
      }
    };
    const scope = options.scope?.(iteration);
    if (recorder && scope) recorder.withScope(scope, addIteration);
    else addIteration();

    const entry: RelaxationIteration = {
      iteration,
      state,
      residual,
      deviation,
      lastNodeId: deviation ? `deviation-${iteration}` : `residual-${iteration}`
    };
    iterations.push(entry);
    const tap = options.tap?.(entry);
    if (tap) {
      graph.add(
        new GPUSnapshotTap({
          id: `tap-${iteration}`,
          arena: tap.arena,
          views: Object.fromEntries(
            Object.entries(tap.views).map(([name, buffer]) => [name, buffer.view])
          ),
          afterNode: entry.lastNodeId,
          label: tap.label,
          iteration: tap.iteration,
          recorder
        })
      );
    }
    previous = state;
  }
  return {initial, iterations};
}

// ---------------------------------------------------------------------------------------------
// Shapes: CPU geometry for the layers the framework already draws

export type Rectangle = {x0: number; y0: number; x1: number; y1: number};

type ShapeBuffers = {
  positions: RunBuffer<'float32x2'>;
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  triangleCount: number;
};

/** Scale and offset that fit the tract mesh extent (it overhangs the world a little) into `rectangle`. */
function fitTracts(tracts: RecordTracts, rectangle: Rectangle) {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let vertex = 0; vertex < tracts.vertices.length; vertex += 2) {
    minimumX = Math.min(minimumX, tracts.vertices[vertex]);
    maximumX = Math.max(maximumX, tracts.vertices[vertex]);
    minimumY = Math.min(minimumY, tracts.vertices[vertex + 1]);
    maximumY = Math.max(maximumY, tracts.vertices[vertex + 1]);
  }
  const width = rectangle.x1 - rectangle.x0;
  const height = rectangle.y1 - rectangle.y0;
  const scale = Math.min(width / (maximumX - minimumX), height / (maximumY - minimumY));
  return {
    scale,
    offsetX: rectangle.x0 + (width - (maximumX - minimumX) * scale) / 2 - minimumX * scale,
    offsetY: rectangle.y0 + (height - (maximumY - minimumY) * scale) / 2 - minimumY * scale
  };
}

/** Filled shapes (one value each) and line segments, accumulated on the CPU. */
export class ShapeBatch {
  private readonly positions: number[] = [];
  private readonly indices: number[] = [];
  private readonly owners: number[] = [];
  private readonly shapeValues: number[] = [];
  private readonly segments: number[] = [];

  get shapeCount(): number {
    return this.shapeValues.length;
  }

  get segmentCount(): number {
    return this.segments.length / 4;
  }

  /** A filled rectangle with one value. */
  addRectangle(rectangle: Rectangle, value: number): void {
    const base = this.positions.length / 2;
    const owner = this.shapeValues.length;
    this.positions.push(
      rectangle.x0,
      rectangle.y0,
      rectangle.x1,
      rectangle.y0,
      rectangle.x1,
      rectangle.y1,
      rectangle.x0,
      rectangle.y1
    );
    this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    this.owners.push(owner, owner);
    this.shapeValues.push(value);
  }

  /** A rectangle outline as four segments. */
  addOutline(rectangle: Rectangle): void {
    const {x0, y0, x1, y1} = rectangle;
    this.addSegment(x0, y0, x1, y0);
    this.addSegment(x1, y0, x1, y1);
    this.addSegment(x1, y1, x0, y1);
    this.addSegment(x0, y1, x0, y0);
  }

  addSegment(x0: number, y0: number, x1: number, y1: number): void {
    this.segments.push(x0, y0, x1, y1);
  }

  /** A segment with a small arrow head at its end. */
  addArrow(x0: number, y0: number, x1: number, y1: number, headSize: number): void {
    this.addSegment(x0, y0, x1, y1);
    const angle = Math.atan2(y1 - y0, x1 - x0);
    for (const side of [-1, 1]) {
      const headAngle = angle + Math.PI + side * 0.45;
      this.addSegment(
        x1,
        y1,
        x1 + Math.cos(headAngle) * headSize,
        y1 + Math.sin(headAngle) * headSize
      );
    }
  }

  /** The tract polygons fitted into `rectangle`, each tract colored by `values[tract]`. */
  addTractMap(tracts: RecordTracts, rectangle: Rectangle, values: ArrayLike<number>): void {
    const fit = fitTracts(tracts, rectangle);
    const vertexBase = this.positions.length / 2;
    const ownerBase = this.shapeValues.length;
    for (let vertex = 0; vertex < tracts.vertices.length; vertex += 2) {
      this.positions.push(
        fit.offsetX + tracts.vertices[vertex] * fit.scale,
        fit.offsetY + tracts.vertices[vertex + 1] * fit.scale
      );
    }
    for (const index of tracts.triangleIndices) this.indices.push(vertexBase + index);
    for (const owner of tracts.triangleOwners) this.owners.push(ownerBase + owner);
    for (let tract = 0; tract < tracts.count; tract++) this.shapeValues.push(values[tract] ?? 0);
  }

  /** The tract boundaries fitted into `rectangle` as segments. */
  addTractOutline(tracts: RecordTracts, rectangle: Rectangle): void {
    const fit = fitTracts(tracts, rectangle);
    for (let segment = 0; segment < tracts.outlineCount; segment++) {
      this.addSegment(
        fit.offsetX + tracts.outline[segment * 4] * fit.scale,
        fit.offsetY + tracts.outline[segment * 4 + 1] * fit.scale,
        fit.offsetX + tracts.outline[segment * 4 + 2] * fit.scale,
        fit.offsetY + tracts.outline[segment * 4 + 3] * fit.scale
      );
    }
  }

  /** Uploads the filled shapes. The value buffer is `uint32` (categorical) or `float32`. */
  uploadShapes(
    upload: RunBuilder['upload'],
    id: string
  ): ShapeBuffers & {values: RunBuffer<'float32'>; categories: RunBuffer<'uint32'>} {
    // Layers need at least one triangle; an empty batch gets a degenerate one.
    const isEmpty = this.positions.length === 0;
    const positions = isEmpty ? [0, 0, 0, 0, 0, 0] : this.positions;
    const indices = isEmpty ? [0, 1, 2] : this.indices;
    const owners = isEmpty ? [0] : this.owners;
    const values = isEmpty ? [0] : this.shapeValues;
    return {
      positions: upload(`${id}-positions`, 'float32x2', Float32Array.from(positions)),
      indices: upload(`${id}-indices`, 'uint32', Uint32Array.from(indices)),
      owners: upload(`${id}-owners`, 'uint32', Uint32Array.from(owners)),
      triangleCount: indices.length / 3,
      values: upload(`${id}-values`, 'float32', Float32Array.from(values)),
      categories: upload(`${id}-categories`, 'uint32', Uint32Array.from(values))
    };
  }

  /** Uploads the segments as `float32x4` rows. */
  uploadSegments(upload: RunBuilder['upload'], id: string): RunBuffer<'float32x4'> {
    const segments = this.segments.length === 0 ? [0, 0, 0, 0] : this.segments;
    return upload(id, 'float32x4', Float32Array.from(segments));
  }
}

// ---------------------------------------------------------------------------------------------
// Drawing a described command graph

export type GraphBox = Rectangle & {
  node: GPUComputationNode;
  column: number;
  row: number;
};

export type GraphLayout = {
  boxes: GraphBox[];
  /** Distinct predecessor/successor pairs, with the strongest kind of the pair. */
  edges: {from: GraphBox; to: GraphBox; kind: GPUComputationEdge['kind']}[];
};

/**
 * Lays a described graph out in columns by dependency depth, left to right. Boxes sit in the
 * given area; rows in a column are centered on the area.
 */
export function layoutGraph(graph: GPUComputationGraph, area: Rectangle): GraphLayout {
  const depth = new Map<string, number>();
  const predecessors = new Map<string, string[]>();
  for (const edge of graph.edges) {
    predecessors.set(edge.to, [...(predecessors.get(edge.to) ?? []), edge.from]);
  }
  // Nodes are listed in scheduled order, so predecessors come first.
  for (const node of graph.nodes) {
    let level = 0;
    for (const predecessor of predecessors.get(node.id) ?? []) {
      level = Math.max(level, (depth.get(predecessor) ?? 0) + 1);
    }
    depth.set(node.id, level);
  }
  const columnCount = Math.max(1, ...[...depth.values()].map(level => level + 1));
  const rowsPerColumn = new Array<number>(columnCount).fill(0);
  const placed = graph.nodes.map(node => {
    const column = depth.get(node.id) ?? 0;
    return {node, column, row: rowsPerColumn[column]++};
  });
  const rowCount = Math.max(1, ...rowsPerColumn);
  const cellWidth = (area.x1 - area.x0) / columnCount;
  const cellHeight = (area.y1 - area.y0) / rowCount;
  const boxWidth = Math.min(cellWidth * 0.74, 0.16);
  const boxHeight = Math.min(cellHeight * 0.62, 0.075);
  const centerY = (area.y0 + area.y1) / 2;
  const boxes: GraphBox[] = placed.map(({node, column, row}) => {
    const centerX = area.x0 + (column + 0.5) * cellWidth;
    const rowCenter = centerY + ((rowsPerColumn[column] - 1) / 2 - row) * cellHeight;
    return {
      node,
      column,
      row,
      x0: centerX - boxWidth / 2,
      x1: centerX + boxWidth / 2,
      y0: rowCenter - boxHeight / 2,
      y1: rowCenter + boxHeight / 2
    };
  });
  const byId = new Map(boxes.map(box => [box.node.id, box]));
  const pairs = new Map<string, GraphLayout['edges'][number]>();
  for (const edge of graph.edges) {
    const from = byId.get(edge.from);
    const to = byId.get(edge.to);
    if (!from || !to) continue;
    const key = `${edge.from}\0${edge.to}`;
    const existing = pairs.get(key);
    if (!existing || edge.kind === 'explicit') pairs.set(key, {from, to, kind: edge.kind});
  }
  return {boxes, edges: [...pairs.values()]};
}

/** Draws the edges of a layout as arrows from the right of a box to the left of the next. */
export function addGraphEdges(
  batch: ShapeBatch,
  layout: GraphLayout,
  filter: (edge: GraphLayout['edges'][number]) => boolean = () => true
): void {
  for (const edge of layout.edges) {
    if (!filter(edge)) continue;
    const startX = edge.from.x1;
    const startY = (edge.from.y0 + edge.from.y1) / 2;
    const endX = edge.to.x0;
    const endY = (edge.to.y0 + edge.to.y1) / 2;
    batch.addArrow(startX, startY, endX, endY, Math.min(0.012, (endX - startX) * 0.5));
  }
}

/** True for the tap nodes a GPUSnapshotTap records as observation taps. */
export function isObservationTap(node: GPUComputationNode): boolean {
  return node.attrs['observationTap'] === true;
}

/** A small deterministic category for a string, so the same namespace keeps its color. */
export function getTextCategory(text: string): number {
  let hash = 0;
  for (const character of text) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  return hash % 8;
}

/** A described graph uploaded for drawing: filled boxes, dependency arrows and the layout. */
export type Diagram = {
  layout: GraphLayout;
  shapes: ReturnType<ShapeBatch['uploadShapes']>;
  edges: RunBuffer<'float32x4'>;
  edgeCount: number;
};

/** Lays out `graph` in `area` and uploads boxes (value from `getValue`) and edge arrows. */
export function createDiagram(
  upload: RunBuilder['upload'],
  id: string,
  graph: GPUComputationGraph,
  area: Rectangle,
  getValue: (node: GPUComputationNode) => number,
  edgeFilter?: (edge: GraphLayout['edges'][number]) => boolean
): Diagram {
  const layout = layoutGraph(graph, area);
  const boxes = new ShapeBatch();
  for (const box of layout.boxes) boxes.addRectangle(box, getValue(box.node));
  const edges = new ShapeBatch();
  addGraphEdges(edges, layout, edgeFilter);
  return {
    layout,
    shapes: boxes.uploadShapes(upload, `${id}-boxes`),
    edges: edges.uploadSegments(upload, `${id}-edges`),
    edgeCount: edges.segmentCount
  };
}

/** Edge arrows then categorical boxes of a diagram. `context` mutes both (an input graph). */
export function getDiagramLayers(
  diagram: Diagram,
  options: {
    context?: boolean;
    categories?: readonly (readonly [number, number, number])[];
  } = {}
): RunLayer[] {
  return [
    {
      layer: 'lines',
      context: options.context,
      segments: diagram.edges,
      count: diagram.edgeCount,
      radius: 1
    },
    {
      layer: 'polygons',
      context: options.context,
      positions: diagram.shapes.positions,
      indices: diagram.shapes.indices,
      owners: diagram.shapes.owners,
      values: diagram.shapes.categories,
      triangleCount: diagram.shapes.triangleCount,
      categories: options.categories
    }
  ];
}

/** Namespace to category: no namespace is 0, every other namespace gets the next category. */
export function createNamespaceCategories(
  graph: GPUComputationGraph
): (node: GPUComputationNode) => number {
  const ranks = new Map<string, number>();
  for (const node of graph.nodes) {
    if (node.namespace && !ranks.has(node.namespace)) ranks.set(node.namespace, ranks.size + 1);
  }
  return node => (node.namespace ? 1 + (((ranks.get(node.namespace) ?? 1) - 1) % 7) : 0);
}

/** The rectangle of cell (`column`, `row`) in a grid inside `area`; row 0 is at the top. */
export function getGridCell(
  area: Rectangle,
  columns: number,
  rows: number,
  column: number,
  row: number,
  gap = 0.008
): Rectangle {
  const cellWidth = (area.x1 - area.x0) / columns;
  const cellHeight = (area.y1 - area.y0) / rows;
  return {
    x0: area.x0 + column * cellWidth + gap,
    x1: area.x0 + (column + 1) * cellWidth - gap,
    y1: area.y1 - row * cellHeight - gap,
    y0: area.y1 - (row + 1) * cellHeight + gap
  };
}

/** Arena slots of the relaxation: the state, how much it changed, and its drift from the start. */
export const RELAXATION_SLOT_NAMES = ['state', 'residual', 'deviation'] as const;

/** Splits arena captures into per-slot lists of frames, in capture order. */
export function groupCapturesBySlot(
  captures: readonly {name: string; values: ArrayLike<number>; iteration?: number}[]
): Map<string, {values: ArrayLike<number>; iteration?: number}[]> {
  const bySlot = new Map<string, {values: ArrayLike<number>; iteration?: number}[]>();
  for (const capture of captures) {
    const frames = bySlot.get(capture.name) ?? [];
    frames.push({values: capture.values, iteration: capture.iteration});
    bySlot.set(capture.name, frames);
  }
  return bySlot;
}
