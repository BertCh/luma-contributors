// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('square with a hole and a second source has literal polygon offsets and stable source IDs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Three four-edge rings, in leader order: CCW shell, CW hole, CCW shell.
  // Polygon layout drops closing duplicates: 4 vertices per ring; shell 0 owns rings 0,1.
  const endpoints = new Float32Array([
    0, 0, 4, 0, 4, 0, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0, 1, 1, 1, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 1, 1, 1,
    8, 0, 10, 0, 10, 0, 10, 2, 10, 2, 8, 2, 8, 2, 8, 0
  ]);
  for (const sourceIdsPresent of [true, false]) {
    for (const ringGroupsPresent of [false, true]) {
      const buffers: Buffer[] = [];
      const graph = new GPUCommandGraph(device);
      const unsigned = (id: string, length: number) => {
        const buffer = createOutputBuffer(device, length);
        buffer.write(new Uint32Array(length).fill(123));
        buffers.push(buffer);
        return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
      };
      const positions = (id: string) => {
        const buffer = createOutputBuffer(device, 40);
        buffer.write(new Float32Array(40).fill(123));
        buffers.push(buffer);
        return importGraphBuffer(graph, id, buffer, 'float32x2', 20);
      };
      const endpointBuffer = createInputBuffer(device, endpoints);
      buffers.push(endpointBuffer);
      const groups = unsigned('groups', 12);
      groups.buffer.write(new Uint32Array([7, 7, 7, 7, 7, 7, 7, 7, 19, 19, 19, 19]));
      const ringOffsets = unsigned('rings', 6);
      const polygonRings = unsigned('polygon-rings', 6);
      const polygonOffsets = unsigned('polygons', 6);
      const featureOffsets = unsigned('features', 6);
      const sourceIds = unsigned('sources', 5);
      const count = unsigned('count', 1);
      const overflow = unsigned('overflow', 1);
      graph.add(
        new GPUSegmentRingAssembly({
          endpoints: importGraphBuffer(graph, 'endpoints', endpointBuffer, 'float32x4', 12),
          groups: groups.view,
          output: {
            positions: positions('ring-positions'),
            ringOffsets: ringOffsets.view,
            ringGroups: ringGroupsPresent ? unsigned('ring-groups', 5).view : undefined,
            polygons: {
              kind: 'polygons',
              positions: positions('polygon-positions'),
              ringOffsets: polygonRings.view,
              polygonOffsets: polygonOffsets.view,
              featureOffsets: featureOffsets.view,
              sourceIds: sourceIdsPresent ? sourceIds.view : undefined
            },
            status: {count: count.view, overflow: overflow.view}
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled);
        expect(await readUint32(polygonRings.buffer, 6)).toEqual([0, 4, 8, 12, 12, 12]);
        expect(await readUint32(polygonOffsets.buffer, 6)).toEqual([0, 2, 3, 3, 3, 3]);
        expect(await readUint32(featureOffsets.buffer, 6)).toEqual([0, 1, 2, 2, 2, 2]);
        if (sourceIdsPresent)
          expect(await readUint32(sourceIds.buffer, 5)).toEqual([
            7, 19, 0xffffffff, 0xffffffff, 0xffffffff
          ]);
        expect(await readUint32(ringOffsets.buffer, 6)).toEqual([0, 5, 10, 15, 15, 15]);
        expect(await readUint32(count.buffer, 1)).toEqual([3]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      } finally {
        compiled.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  }
});
