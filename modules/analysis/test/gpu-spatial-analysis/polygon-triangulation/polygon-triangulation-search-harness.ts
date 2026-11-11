// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Runs `GPUPolygonTriangulation` on a batch of polygons for the randomized search specs.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount,
  type GPUPolygonTriangulationProps
} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {SearchPolygon} from './polygon-triangulation-search-cases';

export type BatchResult = {
  valid: number[];
  /** Real triangles of each polygon, as rows local to the polygon. */
  triangles: number[][][];
};

/** The Earcut port alone. */
export const EARCUT_ONLY: Partial<GPUPolygonTriangulationProps> = {largePolygonMethod: 'none'};

/**
 * The monotone path alone: every polygon is routed to it, and the Earcut fallback pass cannot
 * finish within one step of work, so a valid result is the monotone path's.
 */
export const MONOTONE_ONLY: Partial<GPUPolygonTriangulationProps> = {
  largePolygonMethod: 'monotone',
  monotoneMinimumVertices: 3,
  maximumWork: 1
};

/** Triangulates every polygon of `polygons` in one graph run. */
export async function triangulateBatch(
  device: Device,
  polygons: readonly SearchPolygon[],
  options: Partial<GPUPolygonTriangulationProps>
): Promise<BatchResult> {
  const graph = new GPUCommandGraph(device, {id: 'polygon-triangulation-search'});
  const positions: number[] = [];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  const polygonRows: number[] = [];
  for (const polygon of polygons) {
    polygonRows.push(positions.length / 2);
    for (const ring of polygon.rings) {
      for (const point of ring) {
        positions.push(point[0], point[1]);
      }
      ringOffsets.push(positions.length / 2);
    }
    polygonOffsets.push(ringOffsets.length - 1);
  }
  const vertexCount = positions.length / 2;
  const ringCount = ringOffsets.length - 1;
  const polygonCount = polygons.length;
  const indexCount = getPolygonTriangulationIndexCount(vertexCount, ringCount, polygonCount);
  const featureOffsets = Array.from({length: polygonCount + 1}, (_, index) => index);
  const buffers = {
    positions: createInputBuffer(device, new Float32Array(positions)),
    featureOffsets: createInputBuffer(device, new Uint32Array(featureOffsets)),
    polygonOffsets: createInputBuffer(device, new Uint32Array(polygonOffsets)),
    ringOffsets: createInputBuffer(device, new Uint32Array(ringOffsets)),
    indices: createOutputBuffer(device, indexCount),
    valid: createOutputBuffer(device, polygonCount),
    triangleCount: createOutputBuffer(device, polygonCount),
    triangleOffsets: createOutputBuffer(device, polygonCount + 1)
  };
  try {
    const view = (name: keyof typeof buffers, length: number) =>
      importGraphBuffer(graph, name, buffers[name], 'uint32', length);
    graph.add(
      new GPUPolygonTriangulation({
        positions: importGraphBuffer(
          graph,
          'positions',
          buffers.positions,
          'float32x2',
          vertexCount
        ),
        featureOffsets: view('featureOffsets', polygonCount + 1),
        polygonOffsets: view('polygonOffsets', polygonCount + 1),
        ringOffsets: view('ringOffsets', ringCount + 1),
        output: {
          indices: view('indices', indexCount),
          valid: view('valid', polygonCount),
          triangleCount: view('triangleCount', polygonCount),
          triangleOffsets: view('triangleOffsets', polygonCount + 1)
        },
        ...options
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const indices = await readUint32(buffers.indices, indexCount);
      const valid = await readUint32(buffers.valid, polygonCount);
      const counts = await readUint32(buffers.triangleCount, polygonCount);
      const offsets = await readUint32(buffers.triangleOffsets, polygonCount + 1);
      const triangles = polygons.map((_, polygon) => {
        const result: number[][] = [];
        for (let triangle = 0; triangle < counts[polygon]; triangle++) {
          const at = 3 * (offsets[polygon] + triangle);
          result.push([
            indices[at] - polygonRows[polygon],
            indices[at + 1] - polygonRows[polygon],
            indices[at + 2] - polygonRows[polygon]
          ]);
        }
        return result;
      });
      return {valid: Array.from(valid), triangles};
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of Object.values(buffers)) {
      buffer.destroy();
    }
  }
}
