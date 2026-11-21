// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Polygon fixtures and drawing helpers shared by the geometry-construction runs. Polygons use the
// four-level packed layout the operations read: features own polygons own rings own vertices, with
// open rings (the closing edge is implicit), counter-clockwise shells and clockwise holes.

import {createShoreRing, SHORE_PROFILE_COUNT} from '../../../../data/hydrography';
import type {GPUPolygonOverlayOutput} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {addKernelPass, getViewBuffer} from './kernel';
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
  const firstProfile = Math.floor(random() * SHORE_PROFILE_COUNT);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const centerX = (column + 0.5 + (options.shiftX ?? 0)) * cellWidth;
      const centerY = (row + 0.5 + (options.shiftY ?? 0)) * cellHeight;
      const radius = Math.min(cellWidth, cellHeight) * 0.5 * (options.fill ?? 0.8);
      const shell = createShoreRing(
        firstProfile + row * columns + column,
        centerX,
        centerY,
        radius,
        options.vertexCount ?? 16,
        Math.min(1, (options.irregularity ?? 0.4) ** 0.6)
      );
      const rings = [shell];
      if (options.hasHoles) {
        const holeRadius =
          0.35 * Math.min(...shell.map(([x, y]) => Math.hypot(x - centerX, y - centerY)));
        rings.push(
          Array.from({length: 8}, (_, vertex) => {
            const angle = -(vertex / 8) * Math.PI * 2;
            return [centerX + Math.cos(angle) * holeRadius, centerY + Math.sin(angle) * holeRadius];
          })
        );
      }
      builder.addFeature(rings);
    }
  }
  return builder.build();
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
  options: PackedDrawingOptions & {
    status: Pick<GPUPolygonOverlayOutput['status'], 'overflow' | 'candidateOverflow'>;
  }
): PackedDrawing {
  const {status, ...drawingOptions} = options;
  const ringCount = builder.allocate(`${id}-ring-count`, 'uint32', 1);
  addKernelPass(builder, {
    id: `${id}-ring-count-pass`,
    invocationCount: 1,
    bindings: [
      {name: 'featureCount', buffer: featureCount, access: 'read'},
      {name: 'featureOffsets', buffer: polygons.featureOffsets, access: 'read'},
      {name: 'polygonOffsets', buffer: polygons.polygonOffsets, access: 'read'},
      {name: 'ringCount', buffer: ringCount, access: 'read_write'},
      {name: 'overflow', buffer: getViewBuffer(status.overflow), access: 'read'},
      {
        name: 'candidateOverflow',
        buffer: getViewBuffer(status.candidateOverflow),
        access: 'read'
      }
    ],
    body: /* wgsl */ `
  if (overflow[overflowOffset] != 0u || candidateOverflow[candidateOverflowOffset] != 0u) {
    ringCount[ringCountOffset] = 0u;
    return;
  }
  let features = min(featureCount[featureCountOffset], ${polygons.featureOffsets.length - 1}u);
  let polygonTotal = min(featureOffsets[featureOffsetsOffset + features], ${polygons.polygonOffsets.length - 1}u);
  ringCount[ringCountOffset] = min(polygonOffsets[polygonOffsetsOffset + polygonTotal], ${polygons.ringOffsets.length - 1}u);`
  });
  return addPackedSegments(builder, id, polygons.positions, polygons.ringOffsets, {
    closed: true,
    lineCount: ringCount,
    ...drawingOptions
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
        ringOffsets: polygons.ringOffsets.view
      },
      status: {
        count: count.view,
        requiredRingCount: scalar('required-count').view,
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
