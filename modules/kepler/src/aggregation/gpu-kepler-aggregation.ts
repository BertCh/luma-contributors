// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {Buffer, type CommandEncoder, type Device} from '@luma.gl/core';
import {GPUData, GPUVector, type VertexList} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUMask,
  type CompiledGPUCommandGraph
} from '@luma-contributors/graph/gpu-core';
import {
  GPUPointDensity,
  getGPUPointDensityHexagonCenter,
  getGPUPointDensityHexagonGridSize,
  type GPUPointDensityStatistic
} from '@luma-contributors/graph/gpu-spatial-analysis';

export type GPUKeplerAggregationTopology =
  | {
      kind: 'grid';
      /** Lower-left boundary, in the same planar units as positions. */
      origin: readonly [number, number];
      cellSize: readonly [number, number];
      gridSize: readonly [number, number];
    }
  | {
      kind: 'hexagon';
      /** Inclusive clipping domain; its lower-left corner is hexagon (0, 0)'s center. */
      bounds: readonly [number, number, number, number];
      /** Pointy-top circumradius, in the same planar units as positions. */
      radius: number;
      /** d3 follows native hexbin assignment; the compatibility default is cube rounding. */
      assignment?: 'cube' | 'd3';
    };

export type GPUKeplerAggregationLayout = {
  readonly kind: 'grid' | 'hexagon';
  readonly bounds: readonly [number, number, number, number];
  readonly gridSize: readonly [number, number];
  readonly cellCount: number;
  readonly cellSize?: readonly [number, number];
  readonly radius?: number;
  readonly assignment?: 'cube' | 'd3';
};

export type GPUKeplerAggregationProps = {
  device: Device;
  id?: string;
  /** Borrowed planar float32 coordinates. Project/rebase longitude/latitude before this boundary. */
  positions: GPUVector<'float32x2'>;
  weights?: GPUVector<'float32'>;
  /** Nonzero accepts a source row. Chunk boundaries may differ from positions. */
  selectionMask?: GPUVector<'uint32'>;
  /** Zero excludes a source row from every statistic, including count. */
  validity?: GPUVector<'uint32'>;
  topology: GPUKeplerAggregationTopology;
  statistic?: GPUPointDensityStatistic;
};

export type GPUKeplerAggregationPolygons = {
  positions: GPUVector<VertexList<'float32x4'>>;
  rowIndices: GPUVector<VertexList<'uint32'>>;
  indices: GPUVector<VertexList<'uint32'>>;
  /** Cell counts double as the resident nonempty-cell mask. */
  selectionMask: GPUVector<'uint32'>;
};

/**
 * Resident grid/hex aggregation over source rows, using GPUPointDensity's existing kernels.
 *
 * Counts include selected finite in-bounds positions. Zero and negative weights are accepted;
 * nonfinite weights contribute zero but still count. Supply validity to exclude missing weights
 * from both numerator and denominator. Means are sums/counts, with zero for an empty cell.
 * Deterministic sums use GPUPointDensity's documented fixed-point quantization, not f64 sums.
 *
 * Output row identity is always row * columns + column, including empty cells. No member list is
 * materialized. GPUColumnLayer/GPUGridCellLayer can borrow centers and values directly; their
 * current radii use pixels. getPolygonGeometry supplies exact planar footprints for polygon
 * rendering. This is an explicit luma contract, not a claim of Kepler/d3 binning parity.
 */
export class GPUKeplerAggregation {
  readonly layout: GPUKeplerAggregationLayout;
  readonly centers: GPUVector<'float32x2'>;
  readonly values: GPUVector<'float32'>;
  readonly counts: GPUVector<'uint32'>;
  readonly sums?: GPUVector<'float32'>;
  readonly means?: GPUVector<'float32'>;
  /** One resident word: nonzero reports an in-bounds point outside the allocated hex lattice. */
  readonly overflow: GPUVector<'uint32'>;
  /** The adapter performs no readbacks. Caller reads of outputs are caller-accounted. */
  readonly readbackByteLength = 0;
  readonly memberLookup = 'not-materialized' as const;
  private readonly device: Device;
  private readonly graph: GPUCommandGraph;
  private readonly ownedBuffers: Buffer[] = [];
  private compiled?: CompiledGPUCommandGraph;
  private polygons?: GPUKeplerAggregationPolygons;
  private destroyed = false;

  constructor(props: GPUKeplerAggregationProps) {
    this.layout = getKeplerAggregationLayout(props.topology);
    this.device = props.device;
    const id = props.id ?? 'kepler-aggregation';
    validateInputs(props, this.layout);
    this.graph = new GPUCommandGraph(this.device, {id});
    try {
      const {cellCount, gridSize, bounds, radius, kind} = this.layout;
      const centerValues = new Float32Array(cellCount * 2);
      for (let row = 0; row < gridSize[1]; row++) {
        for (let column = 0; column < gridSize[0]; column++) {
          const center = getCellCenter(this.layout, column, row);
          centerValues.set(center, (row * gridSize[0] + column) * 2);
        }
      }
      this.centers = this.createVector(`${id}-centers`, 'float32x2', cellCount, centerValues);
      this.values = this.createVector(`${id}-values`, 'float32', cellCount);
      this.counts = this.createVector(`${id}-counts`, 'uint32', cellCount);
      this.overflow = this.createVector(`${id}-overflow`, 'uint32', 1);
      if (props.weights) {
        this.sums = this.createVector(`${id}-sums`, 'float32', cellCount);
        this.means = this.createVector(`${id}-means`, 'float32', cellCount);
      }
      const masks = [props.selectionMask, props.validity].flatMap((vector, index) =>
        vector ? [this.graph.importGPUVector(`${id}-mask-${index}`, vector)] : []
      );
      // GPUPointDensity requires one packed mask. This explicit intermediate never repacks source
      // columns and GPUMask supports independently chunked inputs, including empty chunks.
      const combinedMask = masks.length
        ? this.createVector(`${id}-combined-mask`, 'uint32', props.positions.length)
        : undefined;
      const mask = combinedMask
        ? this.graph.importGPUData(`${id}-combined-mask`, combinedMask.data[0])
        : undefined;
      if (mask) this.graph.add(new GPUMask({id: `${id}-mask`, inputs: masks, output: mask}));
      this.graph.add(
        new GPUPointDensity({
          id,
          positions: this.graph.importGPUVector(`${id}-positions`, props.positions),
          weights: props.weights
            ? this.graph.importGPUVector(`${id}-weights`, props.weights)
            : undefined,
          mask,
          bounds,
          gridSize,
          binning: kind,
          hexagonRadius: radius,
          hexagonAssignment: this.layout.assignment,
          statistic: props.statistic ?? 'count',
          sumAccumulation: 'workgroup',
          output: {
            values: this.graph.importGPUData(`${id}-values`, this.values.data[0]),
            counts: this.graph.importGPUData(`${id}-counts`, this.counts.data[0]),
            sums: this.sums ? this.graph.importGPUData(`${id}-sums`, this.sums.data[0]) : undefined,
            means: this.means
              ? this.graph.importGPUData(`${id}-means`, this.means.data[0])
              : undefined,
            status: {overflow: this.graph.importGPUData(`${id}-overflow`, this.overflow.data[0])}
          }
        })
      );
    } catch (error) {
      for (const buffer of this.ownedBuffers) buffer.destroy();
      throw error;
    }
  }

  /** Reuses the compiled plan and outputs after borrowed mask/weight/position contents change. */
  encode(commandEncoder: CommandEncoder): void {
    this.assertAlive();
    if (commandEncoder.device !== this.device)
      throw new Error('Aggregation encoder device mismatch');
    this.prepare();
    this.compiled!.encode(commandEncoder, {parameters: undefined});
  }

  /** Compiles/allocates scratch without encoding or submitting, for host admission accounting. */
  prepare(): void {
    this.assertAlive();
    this.compiled ??= this.graph.compile();
  }

  /**
   * Lazily creates one immutable triangulated footprint per aggregate. Row indices are aggregate
   * IDs, never source point IDs. The caller may borrow this object directly for GPUSolidPolygonLayer.
   * Geometry is generated from fixed topology only; source rows and aggregate values are not read.
   */
  getPolygonGeometry(): GPUKeplerAggregationPolygons {
    this.assertAlive();
    if (this.polygons) return this.polygons;
    const {cellCount, gridSize, kind, radius, cellSize} = this.layout;
    const verticesPerCell = kind === 'grid' ? 4 : 6;
    const indicesPerCell = (verticesPerCell - 2) * 3;
    // Each polygon attribute must fit one storage binding and signed list offsets.
    if (
      cellCount * verticesPerCell * 16 > this.device.limits.maxStorageBufferBindingSize ||
      cellCount * verticesPerCell > 0x7fffffff
    )
      throw new KeplerAllocationError('capacity', {
        message: 'Aggregation polygon geometry exceeds binding capacity'
      });
    const positions = new Float32Array(cellCount * verticesPerCell * 4);
    const rowIndices = new Uint32Array(cellCount * verticesPerCell);
    const indices = new Uint32Array(cellCount * indicesPerCell);
    for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
      const center = getCellCenter(
        this.layout,
        cellIndex % gridSize[0],
        Math.floor(cellIndex / gridSize[0])
      );
      for (let vertexIndex = 0; vertexIndex < verticesPerCell; vertexIndex++) {
        const angle = (vertexIndex * 2 * Math.PI) / verticesPerCell + Math.PI / 2;
        const corner =
          kind === 'grid'
            ? [
                center[0] + (vertexIndex === 0 || vertexIndex === 3 ? -0.5 : 0.5) * cellSize![0],
                center[1] + (vertexIndex < 2 ? -0.5 : 0.5) * cellSize![1]
              ]
            : [center[0] + Math.cos(angle) * radius!, center[1] + Math.sin(angle) * radius!];
        const outputIndex = cellIndex * verticesPerCell + vertexIndex;
        positions.set([corner[0], corner[1], 0, 1], outputIndex * 4);
        rowIndices[outputIndex] = cellIndex;
      }
      for (let triangleIndex = 0; triangleIndex < verticesPerCell - 2; triangleIndex++) {
        const vertexBase = cellIndex * verticesPerCell;
        indices.set(
          [vertexBase, vertexBase + triangleIndex + 1, vertexBase + triangleIndex + 2],
          cellIndex * indicesPerCell + triangleIndex * 3
        );
      }
    }
    this.polygons = {
      positions: this.createListVector(
        'aggregation-polygon-positions',
        'vertex-list<float32x4>',
        cellCount,
        verticesPerCell,
        positions
      ),
      rowIndices: this.createListVector(
        'aggregation-polygon-rows',
        'vertex-list<uint32>',
        cellCount,
        verticesPerCell,
        rowIndices
      ),
      indices: this.createListVector(
        'aggregation-polygon-indices',
        'vertex-list<uint32>',
        cellCount,
        indicesPerCell,
        indices,
        true
      ),
      selectionMask: this.counts
    };
    return this.polygons;
  }

  /** Owned outputs/topology and compiled scratch, excluding borrowed source columns. */
  get accountedResidentByteLength(): number {
    return this.destroyed
      ? 0
      : this.ownedBuffers.reduce((total, buffer) => total + buffer.byteLength, 0) +
          (this.compiled?.stats.physicalTransientResourceBytes ?? 0);
  }

  /** Call after all dependent GPU commands finish. Source buffers remain caller-owned. */
  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.compiled?.destroy();
    for (const buffer of this.ownedBuffers) buffer.destroy();
  }

  private createVector<Format extends 'float32' | 'uint32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number,
    values?: Float32Array | Uint32Array
  ): GPUVector<Format> {
    const byteLength = length * (format === 'float32x2' ? 8 : 4);
    const buffer = this.createBuffer(id, byteLength, values);
    return new GPUVector({
      type: 'data',
      name: id,
      format,
      ownsData: false,
      data: [new GPUData({buffer, format, length, ownsBuffer: false})]
    });
  }

  private createListVector<Format extends VertexList<'float32x4' | 'uint32'>>(
    id: string,
    format: Format,
    length: number,
    valuesPerRow: number,
    values: Float32Array | Uint32Array,
    indexBuffer = false
  ): GPUVector<Format> {
    const buffer = this.createBuffer(id, values.byteLength, values, indexBuffer);
    return new GPUVector({
      type: 'data',
      name: id,
      format,
      ownsData: false,
      data: [
        new GPUData({
          buffer,
          format,
          length,
          valueLength: length * valuesPerRow,
          valueOffsets: Int32Array.from({length: length + 1}, (_, index) => index * valuesPerRow),
          ownsBuffer: false
        })
      ]
    });
  }

  private createBuffer(
    id: string,
    byteLength: number,
    data?: Float32Array | Uint32Array,
    indexBuffer = false
  ): Buffer {
    const buffer = this.device.createBuffer({
      id,
      ...(data ? {data} : {byteLength: Math.max(4, byteLength)}),
      usage:
        Buffer.STORAGE |
        Buffer.VERTEX |
        (indexBuffer ? Buffer.INDEX : 0) |
        Buffer.COPY_SRC |
        Buffer.COPY_DST
    });
    this.ownedBuffers.push(buffer);
    return buffer;
  }

  private assertAlive(): void {
    if (this.destroyed) throw new Error('Aggregation has been destroyed');
  }
}

/**
 * Validates fixed topology without a GPU. Grid max boundaries are inclusive in the last cell.
 * Cell centers and polygons are computed in float64 while the binning kernel uses the float32
 * `bounds` and `gridSize`, so a point within about one float32 ULP of a cell edge may be binned
 * into the neighboring cell of the polygon that contains it. This is the accepted tolerance.
 */
export function getKeplerAggregationLayout(
  topology: GPUKeplerAggregationTopology
): GPUKeplerAggregationLayout {
  let layout: GPUKeplerAggregationLayout;
  if (topology.kind === 'grid') {
    const {origin, cellSize, gridSize} = topology;
    if (
      !origin.every(Number.isFinite) ||
      !cellSize.every(value => Number.isFinite(value) && value > 0)
    )
      throw new Error('Grid origin and cell size must be finite and cell size positive');
    layout = {
      kind: 'grid',
      cellSize: [...cellSize],
      gridSize: [...gridSize],
      bounds: [
        origin[0],
        origin[1],
        origin[0] + cellSize[0] * gridSize[0],
        origin[1] + cellSize[1] * gridSize[1]
      ],
      cellCount: gridSize[0] * gridSize[1]
    };
  } else {
    const gridSize = getGPUPointDensityHexagonGridSize(topology.bounds, topology.radius);
    layout = {
      kind: 'hexagon',
      bounds: [...topology.bounds],
      radius: topology.radius,
      ...(topology.assignment ? {assignment: topology.assignment} : {}),
      gridSize,
      cellCount: gridSize[0] * gridSize[1]
    };
  }
  if (
    !layout.gridSize.every(value => Number.isSafeInteger(value) && value > 0) ||
    !Number.isSafeInteger(layout.cellCount) ||
    layout.cellCount > 0xffffffff ||
    !layout.bounds.every(value => Number.isFinite(Math.fround(value)))
  )
    throw new Error('Aggregation topology exceeds supported finite float32/uint32 range');
  if (
    layout.kind === 'grid' &&
    [0, 1].some(
      axis =>
        Math.fround(layout.bounds[axis] + layout.cellSize![axis]) <=
          Math.fround(layout.bounds[axis]) ||
        Math.fround(layout.bounds[axis + 2] - layout.cellSize![axis]) >=
          Math.fround(layout.bounds[axis + 2])
    )
  )
    throw new Error('Grid cells collapse in float32; rebase source coordinates');
  if (
    layout.radius !== undefined &&
    (!Number.isFinite(Math.fround(layout.radius)) || Math.fround(layout.radius) <= 0)
  )
    throw new Error('Hexagon radius exceeds positive float32 range');
  Object.freeze(layout.bounds);
  Object.freeze(layout.gridSize);
  if (layout.cellSize) Object.freeze(layout.cellSize);
  return Object.freeze(layout);
}

function getCellCenter(
  layout: GPUKeplerAggregationLayout,
  column: number,
  row: number
): [number, number] {
  return layout.kind === 'hexagon'
    ? getGPUPointDensityHexagonCenter(
        column,
        row,
        layout.bounds[0],
        layout.bounds[1],
        layout.radius!
      )
    : [
        layout.bounds[0] + (column + 0.5) * layout.cellSize![0],
        layout.bounds[1] + (row + 0.5) * layout.cellSize![1]
      ];
}

function validateInputs(
  props: GPUKeplerAggregationProps,
  layout: GPUKeplerAggregationLayout
): void {
  if (props.device.type !== 'webgpu') throw new Error('Aggregation requires WebGPU');
  if (layout.cellCount * 12 > props.device.limits.maxStorageBufferBindingSize)
    throw new KeplerAllocationError('capacity', {
      message: 'Aggregation topology exceeds binding capacity'
    });
  const hasMask = Boolean(props.selectionMask || props.validity);
  const intermediateRowStride =
    layout.kind === 'grid' && hasMask ? 8 : props.weights || layout.kind === 'hexagon' ? 4 : 0;
  if (
    props.positions.length > 0xffffffff ||
    props.positions.length * intermediateRowStride > props.device.limits.maxStorageBufferBindingSize
  )
    throw new KeplerAllocationError('capacity', {
      message: 'Aggregation working set exceeds intermediate binding capacity'
    });
  if (props.statistic && props.statistic !== 'count' && !props.weights)
    throw new Error('Weighted statistic requires weights');
  for (const [vector, format] of [
    [props.positions, 'float32x2'],
    [props.weights, 'float32'],
    [props.selectionMask, 'uint32'],
    [props.validity, 'uint32']
  ] as const) {
    if (!vector) continue;
    if (vector.format !== format || vector.length !== props.positions.length)
      throw new Error('Aggregation input format or row count mismatch');
    for (const chunk of vector.data) {
      if (
        chunk.buffer.device !== props.device ||
        chunk.byteStride !== (format === 'float32x2' ? 8 : 4)
      )
        throw new Error('Aggregation inputs must be packed on the same device');
    }
  }
}
