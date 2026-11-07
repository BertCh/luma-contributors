// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Polygon fixtures and drawing helpers shared by the geometry-construction runs. Polygons use the
// four-level packed layout the operations read: features own polygons own rings own vertices, with
// open rings (the closing edge is implicit), counter-clockwise shells and clockwise holes.

import type {GPUPolygonOverlayOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {addKernelPass} from './kernel';
import {
  addPackedSegments,
  createPackedSegments,
  type PackedDrawing,
  type PackedDrawingOptions
} from './lines';

export type World = {width: number; height: number};
export type Random = () => number;

export type PolygonSet = {
  featureCount: number;
  positions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
};

/** Accumulates polygons ring by ring into the packed layout. */
export class PolygonBuilder {
  private readonly positions: number[] = [];
  private readonly ringOffsets = [0];
  private readonly polygonOffsets = [0];
  private readonly featureOffsets = [0];

  /** Adds one feature holding one polygon made of `rings` (shell first, then holes). */
  addFeature(rings: readonly (readonly (readonly number[])[])[]): void {
    for (const ring of rings) {
      for (const [x, y] of ring) this.positions.push(x, y);
      this.ringOffsets.push(this.positions.length / 2);
    }
    this.polygonOffsets.push(this.ringOffsets.length - 1);
    this.featureOffsets.push(this.polygonOffsets.length - 1);
  }

  build(): PolygonSet {
    return {
      featureCount: this.featureOffsets.length - 1,
      positions: Float32Array.from(this.positions),
      featureOffsets: Uint32Array.from(this.featureOffsets),
      polygonOffsets: Uint32Array.from(this.polygonOffsets),
      ringOffsets: Uint32Array.from(this.ringOffsets)
    };
  }
}

/** Ring edges of a polygon set as segments (x0, y0, x1, y1), closing edges included. */
export function createRingSegments(set: PolygonSet): Float32Array {
  return createPackedSegments(set.positions, set.ringOffsets, true);
}

/** A star-shaped ring around (centerX, centerY), counter-clockwise. */
export function createBlobRing(
  random: Random,
  centerX: number,
  centerY: number,
  radius: number,
  vertexCount = 16,
  irregularity = 0.25
): number[][] {
  const phase = random() * Math.PI * 2;
  return Array.from({length: vertexCount}, (_, vertex) => {
    const angle = phase + (vertex / vertexCount) * Math.PI * 2;
    const spike = 1 - irregularity * random();
    return [centerX + Math.cos(angle) * radius * spike, centerY + Math.sin(angle) * radius * spike];
  });
}

export type BlobOptions = {
  rows?: number;
  columns?: number;
  vertexCount?: number;
  irregularity?: number;
  /** Fraction of the cell a blob spans. */
  fill?: number;
  /** Shift of every blob, in cells. */
  shiftX?: number;
  shiftY?: number;
  hasHoles?: boolean;
};

/** Blobs on a grid of cells, one feature each (optionally with a clockwise hole). */
export function createBlobSet(random: Random, world: World, options: BlobOptions = {}): PolygonSet {
  const rows = options.rows ?? 2;
  const columns = options.columns ?? Math.max(2, Math.round(rows * world.width));
  const cellWidth = world.width / columns;
  const cellHeight = world.height / rows;
  const builder = new PolygonBuilder();
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const centerX = (column + 0.5 + (options.shiftX ?? 0)) * cellWidth;
      const centerY = (row + 0.5 + (options.shiftY ?? 0)) * cellHeight;
      const radius = Math.min(cellWidth, cellHeight) * 0.5 * (options.fill ?? 0.8);
      const shell = createBlobRing(
        random,
        centerX,
        centerY,
        radius,
        options.vertexCount,
        options.irregularity
      );
      const rings = [shell];
      if (options.hasHoles) {
        rings.push(
          Array.from({length: 8}, (_, vertex) => {
            const angle = -(vertex / 8) * Math.PI * 2;
            return [
              centerX + Math.cos(angle) * radius * 0.3,
              centerY + Math.sin(angle) * radius * 0.3
            ];
          })
        );
      }
      builder.addFeature(rings);
    }
  }
  return builder.build();
}

export type CoverageOptions = {
  columns?: number;
  rows?: number;
  /** Intermediate vertices on every cell edge. */
  edgeVertexCount?: number;
  /** Sideways noise of those vertices, in cell widths. */
  wiggle?: number;
  /** Also wiggle the outer boundary. */
  wiggleBoundary?: boolean;
};

export type Coverage = PolygonSet & {
  columns: number;
  rows: number;
  /** District label per cell: blocks of 2 x 2 cells. */
  labels: Uint32Array;
};

/**
 * A polygon coverage: a grid of cells whose neighbors share every boundary vertex exactly (the
 * wiggly edges are generated once per edge and walked in opposite directions by the two cells).
 */
export function createCoverage(
  random: Random,
  world: World,
  options: CoverageOptions = {}
): Coverage {
  const rows = options.rows ?? 4;
  const columns = options.columns ?? Math.max(3, Math.round(rows * world.width));
  const edgeVertexCount = options.edgeVertexCount ?? 5;
  const wiggle = options.wiggle ?? 0.12;
  const isWiggly = (isInterior: boolean) => isInterior || Boolean(options.wiggleBoundary);
  const margin = 0.03;
  const cellWidth = (world.width - 2 * margin) / columns;
  const cellHeight = (1 - 2 * margin) / rows;
  const corners: number[][][] = Array.from({length: rows + 1}, (_, row) =>
    Array.from({length: columns + 1}, (_, column) => [
      margin +
        (column + (column > 0 && column < columns ? (random() - 0.5) * 0.35 : 0)) * cellWidth,
      margin + (row + (row > 0 && row < rows ? (random() - 0.5) * 0.35 : 0)) * cellHeight
    ])
  );
  const makeEdge = (from: number[], to: number[], isInterior: boolean): number[][] => {
    const normalX = -(to[1] - from[1]);
    const normalY = to[0] - from[0];
    return Array.from({length: edgeVertexCount}, (_, step) => {
      const along = (step + 1) / (edgeVertexCount + 1);
      const sideways = isInterior ? (random() - 0.5) * 2 * wiggle : 0;
      return [
        from[0] + (to[0] - from[0]) * along + normalX * sideways,
        from[1] + (to[1] - from[1]) * along + normalY * sideways
      ];
    });
  };
  // Horizontal edge (row, column) runs corner (row, column) to (row, column + 1); vertical edge
  // (row, column) runs corner (row, column) to (row + 1, column).
  const horizontal = Array.from({length: rows + 1}, (_, row) =>
    Array.from({length: columns}, (_, column) =>
      makeEdge(corners[row][column], corners[row][column + 1], isWiggly(row > 0 && row < rows))
    )
  );
  const vertical = Array.from({length: rows}, (_, row) =>
    Array.from({length: columns + 1}, (_, column) =>
      makeEdge(
        corners[row][column],
        corners[row + 1][column],
        isWiggly(column > 0 && column < columns)
      )
    )
  );
  const builder = new PolygonBuilder();
  const labels: number[] = [];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const ring = [
        corners[row][column],
        ...horizontal[row][column],
        corners[row][column + 1],
        ...vertical[row][column + 1],
        corners[row + 1][column + 1],
        ...[...horizontal[row + 1][column]].reverse(),
        corners[row + 1][column],
        ...[...vertical[row][column]].reverse()
      ];
      builder.addFeature([ring]);
      labels.push(Math.floor(row / 2) * Math.ceil(columns / 2) + Math.floor(column / 2));
    }
  }
  return {...builder.build(), columns, rows, labels: Uint32Array.from(labels)};
}

// ---------------------------------------------------------------------------------------------
// Buffers and drawing

export type PolygonBuffers = {
  positions: RunBuffer<'float32x2'>;
  featureOffsets: RunBuffer<'uint32'>;
  polygonOffsets: RunBuffer<'uint32'>;
  ringOffsets: RunBuffer<'uint32'>;
};

/** Uploads a polygon set as the four packed buffers. */
export function uploadPolygonSet(
  builder: RunBuilder,
  prefix: string,
  set: PolygonSet
): PolygonBuffers {
  return {
    positions: builder.upload(`${prefix}-positions`, 'float32x2', set.positions),
    featureOffsets: builder.upload(`${prefix}-feature-offsets`, 'uint32', set.featureOffsets),
    polygonOffsets: builder.upload(`${prefix}-polygon-offsets`, 'uint32', set.polygonOffsets),
    ringOffsets: builder.upload(`${prefix}-ring-offsets`, 'uint32', set.ringOffsets)
  };
}

/** Allocates polygon-shaped output buffers: `vertexCapacity` vertices over `ringCapacity` rings. */
export function allocatePolygonOutput(
  builder: RunBuilder,
  prefix: string,
  vertexCapacity: number,
  ringCapacity: number
): PolygonBuffers {
  return {
    positions: builder.allocate(`${prefix}-positions`, 'float32x2', vertexCapacity),
    featureOffsets: builder.allocate(`${prefix}-feature-offsets`, 'uint32', ringCapacity + 1),
    polygonOffsets: builder.allocate(`${prefix}-polygon-offsets`, 'uint32', ringCapacity + 1),
    ringOffsets: builder.allocate(`${prefix}-ring-offsets`, 'uint32', ringCapacity + 1)
  };
}

/**
 * Adds the passes that turn polygon output into drawable ring edges. `featureCount` is the
 * GPU-written number of output features; the ring count is derived from the offsets on the GPU.
 */
export function addPolygonDrawing(
  builder: RunBuilder,
  id: string,
  polygons: PolygonBuffers,
  featureCount: RunBuffer<'uint32'>,
  options: PackedDrawingOptions = {}
): PackedDrawing {
  const ringCount = builder.allocate(`${id}-ring-count`, 'uint32', 1);
  addKernelPass(builder, {
    id: `${id}-ring-count-pass`,
    invocationCount: 1,
    bindings: [
      {name: 'featureCount', buffer: featureCount, access: 'read'},
      {name: 'featureOffsets', buffer: polygons.featureOffsets, access: 'read'},
      {name: 'polygonOffsets', buffer: polygons.polygonOffsets, access: 'read'},
      {name: 'ringCount', buffer: ringCount, access: 'read_write'}
    ],
    body: /* wgsl */ `
  let features = min(featureCount[featureCountOffset], ${polygons.featureOffsets.length - 1}u);
  let polygonTotal = min(featureOffsets[featureOffsetsOffset + features], ${polygons.polygonOffsets.length - 1}u);
  ringCount[ringCountOffset] = min(polygonOffsets[polygonOffsetsOffset + polygonTotal], ${polygons.ringOffsets.length - 1}u);`
  });
  return addPackedSegments(builder, id, polygons.positions, polygons.ringOffsets, {
    closed: true,
    lineCount: ringCount,
    ...options
  });
}

/** Buffers that draw a triangle list as a filled mesh with the `polygons` layer. */
export type TriangleListDrawing = {
  indices: RunBuffer<'uint32'>;
  owners: RunBuffer<'uint32'>;
  values: RunBuffer<'float32'>;
  triangleCount: number;
};

/** Identity indices over a vertex buffer that already holds a triangle list. */
export function createTriangleListDrawing(
  builder: RunBuilder,
  id: string,
  vertexCount: number
): TriangleListDrawing {
  const triangleCount = Math.floor(vertexCount / 3);
  return {
    indices: builder.upload(
      `${id}-indices`,
      'uint32',
      Uint32Array.from({length: triangleCount * 3}, (_, index) => index)
    ),
    owners: builder.upload(`${id}-owners`, 'uint32', new Uint32Array(triangleCount)),
    values: builder.upload(`${id}-values`, 'float32', new Float32Array(1)),
    triangleCount
  };
}

/** Buffers behind a `GPUPolygonOverlayOutput` (also the output shape of buffer and dissolve). */
export type OverlayResult = {
  polygons: PolygonBuffers;
  count: RunBuffer<'uint32'>;
  overflow: RunBuffer<'uint32'>;
  output: GPUPolygonOverlayOutput;
};

/** Allocates a `GPUPolygonOverlayOutput` with the given capacities. */
export function allocateOverlayOutput(
  builder: RunBuilder,
  prefix: string,
  capacity: {vertices: number; rings: number; boundary: number}
): OverlayResult {
  const polygons = allocatePolygonOutput(builder, prefix, capacity.vertices, capacity.rings);
  const sourceIds = builder.allocate(`${prefix}-source-ids`, 'uint32', capacity.rings);
  const scalar = (name: string) => builder.allocate(`${prefix}-${name}`, 'uint32', 1);
  const count = scalar('count');
  const overflow = scalar('overflow');
  return {
    polygons,
    count,
    overflow,
    output: {
      geometry: {
        kind: 'polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        sourceIds: sourceIds.view
      },
      status: {
        count: count.view,
        requiredCount: scalar('required-count').view,
        overflow: overflow.view,
        candidateOverflow: scalar('candidate-overflow').view
      },
      boundary: {
        endpoints: builder.allocate(`${prefix}-boundary`, 'float32x4', capacity.boundary).view,
        operandIds: builder.allocate(`${prefix}-boundary-operands`, 'uint32', capacity.boundary)
          .view,
        sourceFeatureIds: builder.allocate(
          `${prefix}-boundary-features`,
          'uint32',
          capacity.boundary
        ).view,
        status: {
          count: scalar('boundary-count').view,
          requiredCount: scalar('boundary-required').view,
          overflow: scalar('boundary-overflow').view,
          candidateOverflow: scalar('boundary-candidate-overflow').view
        }
      }
    }
  };
}

/** The operation-facing port (views) of uploaded or allocated polygon buffers. */
export function getPolygonPort(buffers: PolygonBuffers) {
  return {
    kind: 'polygons' as const,
    positions: buffers.positions.view,
    featureOffsets: buffers.featureOffsets.view,
    polygonOffsets: buffers.polygonOffsets.view,
    ringOffsets: buffers.ringOffsets.view
  };
}

/** The same views without `kind`, for operations that take the four fields as separate props. */
export function getPolygonFields(buffers: PolygonBuffers) {
  const {kind: _kind, ...fields} = getPolygonPort(buffers);
  return fields;
}
