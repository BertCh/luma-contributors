// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {checkLocallyDelaunay, mulberry32, toTriangleRows} from './delaunay-reference';
import {requireTestDevice} from '../../utils/require-test-device';

// Mixed-scale coordinate sets: one cluster of unit magnitude and one scaled by 2^-span around the
// origin. The incircle predicate is exact up to a 55 bit exponent span; beyond it the shader keeps
// the float sign. Every case must either be an exact (locally Delaunay) triangulation or fail
// explicitly through the invalid count, never silently return an illegal edge.

async function runDelaunay(device: Device, coordinates: Float32Array, triangleCapacity: number) {
  const graph = new GPUCommandGraph(device, {id: 'delaunay-property'});
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, coordinates);
  const triangleBuffer = createOutputBuffer(device, 3 * triangleCapacity);
  buffers.push(positionsBuffer, triangleBuffer);
  const scalar = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const count = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const invalidCount = scalar('invalid-count');
  const duplicateCount = scalar('duplicate-count');
  graph.add(
    new GPUDelaunayTessellation({
      positions: importGraphBuffer(
        graph,
        'positions',
        positionsBuffer,
        'float32x2',
        coordinates.length / 2
      ),
      output: {
        triangles: importGraphBuffer(
          graph,
          'triangles',
          triangleBuffer,
          'uint32x3',
          triangleCapacity
        ),
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view,
          duplicateCount: duplicateCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const result = {
    written,
    required: (await readUint32(requiredCount.buffer, 1))[0],
    didOverflow: (await readUint32(overflow.buffer, 1))[0],
    invalid: (await readUint32(invalidCount.buffer, 1))[0],
    duplicates: (await readUint32(duplicateCount.buffer, 1))[0],
    indices: await readUint32(triangleBuffer, 3 * written)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function createMixedScaleFixture(seed: number, span: number, perCluster: number): Float32Array {
  const random = mulberry32(seed);
  const coordinates: number[] = [];
  for (const scale of [1, 2 ** -span]) {
    for (let index = 0; index < perCluster; index++) {
      coordinates.push(
        Math.fround((random() * 2 - 1) * scale),
        Math.fround((random() * 2 - 1) * scale)
      );
    }
  }
  return Float32Array.from(coordinates);
}

it('GPUDelaunayTessellation is exact or explicitly failed across incircle exponent spans', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const span of [40, 55, 56, 60]) {
    const coordinates = createMixedScaleFixture(7 + span, span, 24);
    const pointCount = coordinates.length / 2;
    const actual = await runDelaunay(device, coordinates, 2 * pointCount);
    const triangles = toTriangleRows(actual.indices);
    const check = checkLocallyDelaunay(coordinates, triangles);
    if (actual.invalid === 0) {
      expect(check.circleViolations, `span ${span} empty circle`).toBe(0);
      expect(check.degenerate, `span ${span} degenerate`).toBe(0);
      expect(check.duplicateEdges, `span ${span} duplicate edges`).toBe(0);
    } else {
      // Explicit failure yields no triangles rather than a partial illegal mesh.
      expect(actual.written, `span ${span} failed output`).toBe(0);
    }
    // Random magnitudes extend a few bits below the nominal scale, so only span 40 is certainly
    // inside the 55 bit exact range; the wider cases must be exact or explicitly failed.
    if (span === 40) {
      expect(actual.invalid, `span ${span} must be exact`).toBe(0);
    }
    if (span === 60) {
      expect(actual.invalid, `span ${span} must fail explicitly`).toBeGreaterThan(0);
    }
  }
});
