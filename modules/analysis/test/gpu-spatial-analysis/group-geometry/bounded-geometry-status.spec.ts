// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGroupConvexHull} from '../../../src/gpu-spatial-analysis/group-geometry/gpu-group-convex-hull';
import {GPUMinimumBounds} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// One square, two triangles, and a nonfinite padding row excluded by its label.
const POSITIONS = new Float32Array([
  0,
  0,
  2,
  0,
  2,
  2,
  0,
  2,
  10,
  0,
  12,
  0,
  10,
  2,
  20,
  0,
  22,
  0,
  20,
  2,
  NaN,
  NaN
]);
const NONE = 0xffffffff;
const COMBINED = new Uint32Array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, NONE]);
const GROUP_ONLY = new Uint32Array([0, 0, 0, 0, NONE, NONE, NONE, NONE, NONE, NONE, NONE]);
const VERTEX_ONLY = new Uint32Array([NONE, NONE, NONE, NONE, 1, 1, 1, 2, 2, 2, NONE]);
const COMPLETE = new Uint32Array([0, 0, 0, NONE, NONE, NONE, NONE, NONE, NONE, NONE, NONE]);
const EMPTY = new Uint32Array(11).fill(NONE);

function createStatusFixture(device: Device, mode: 'hull' | 'bounds', withDetails: boolean) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const labelsBuffer = track(createInputBuffer(device, COMBINED));
  const positions = importGraphBuffer(
    graph,
    'positions',
    track(createInputBuffer(device, POSITIONS)),
    'float32x2',
    11
  );
  const labels = importGraphBuffer(graph, 'labels', labelsBuffer, 'uint32', 11);
  const scalar = (name: string) => {
    const buffer = track(createOutputBuffer(device, 1));
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const overflow = scalar('overflow');
  const groupOverflow = withDetails ? scalar('group-overflow') : undefined;
  const vertexOverflow = mode === 'hull' && withDetails ? scalar('vertex-overflow') : undefined;
  const status = {
    overflow: overflow.view,
    groupOverflow: groupOverflow?.view,
    vertexOverflow: vertexOverflow?.view
  };
  const counts = track(createOutputBuffer(device, 3));
  const sizes = track(createOutputBuffer(device, 3));
  const indices = track(createOutputBuffer(device, 3));
  const bounds = track(createOutputBuffer(device, 12));
  if (mode === 'hull')
    graph.add(
      new GPUGroupConvexHull({
        positions,
        labels,
        groupCount: 3,
        maximumVerticesPerGroup: 3,
        vertexCapacity: 3,
        output: {
          vertexIndices: importGraphBuffer(graph, 'indices', indices, 'uint32', 3),
          counts: importGraphBuffer(graph, 'counts', counts, 'uint32', 3),
          sizes: importGraphBuffer(graph, 'sizes', sizes, 'uint32', 3),
          offsets: importGraphBuffer(
            graph,
            'offsets',
            track(createOutputBuffer(device, 4)),
            'uint32',
            4
          ),
          status
        }
      })
    );
  else
    graph.add(
      new GPUMinimumBounds({
        positions,
        labels,
        groupCount: 3,
        maximumHullVertices: 3,
        output: {
          bounds: importGraphBuffer(graph, 'bounds', bounds, 'float32x4', 3),
          hullSizes: importGraphBuffer(graph, 'sizes', sizes, 'uint32', 3),
          status: {overflow: status.overflow, groupOverflow: status.groupOverflow}
        }
      })
    );
  const compiled = graph.compile();
  return {
    async run(nextLabels: Uint32Array) {
      labelsBuffer.write(nextLabels);
      // Seed statuses to catch unwritten ports, including on an empty subsequent frame.
      for (const scalar of [overflow, groupOverflow, vertexOverflow])
        scalar?.buffer.write(new Uint32Array([NONE]));
      submitGraph(device, compiled, undefined);
      return {
        overflow: (await readUint32(overflow.buffer, 1))[0],
        groupOverflow: groupOverflow ? (await readUint32(groupOverflow.buffer, 1))[0] : undefined,
        vertexOverflow: vertexOverflow
          ? (await readUint32(vertexOverflow.buffer, 1))[0]
          : undefined,
        counts: mode === 'hull' ? await readUint32(counts, 3) : undefined,
        indices: mode === 'hull' ? await readUint32(indices, 3) : undefined,
        sizes: await readUint32(sizes, 3),
        bounds: mode === 'bounds' ? await readFloat32(bounds, 12) : undefined
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

for (const withDetails of [false, true]) {
  it(`hull limits publish boolean and named diagnostics and reset on replay (details=${withDetails})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createStatusFixture(device, 'hull', withDetails);
    try {
      for (const [
        labels,
        expectedOverflow,
        expectedGroup,
        expectedVertex,
        expectedCounts,
        expectedSizes,
        expectedIndices
      ] of [
        [COMBINED, 1, 1, 1, [0, 3, 0], [4, 3, 3], [4, 5, 6]],
        [GROUP_ONLY, 1, 1, 0, [0, 0, 0], [4, 0, 0], [NONE, NONE, NONE]],
        [VERTEX_ONLY, 1, 0, 1, [0, 3, 0], [0, 3, 3], [4, 5, 6]],
        [COMPLETE, 0, 0, 0, [3, 0, 0], [3, 0, 0], [0, 1, 2]],
        [EMPTY, 0, 0, 0, [0, 0, 0], [0, 0, 0], [NONE, NONE, NONE]],
        [COMBINED, 1, 1, 1, [0, 3, 0], [4, 3, 3], [4, 5, 6]]
      ] as const) {
        const result = await fixture.run(labels);
        expect(result.overflow).toBe(expectedOverflow);
        if (withDetails) {
          expect(result.groupOverflow).toBe(expectedGroup);
          expect(result.vertexOverflow).toBe(expectedVertex);
        }
        expect(result.counts).toEqual(expectedCounts);
        expect(result.sizes).toEqual(expectedSizes);
        expect(result.indices).toEqual(expectedIndices);
      }
    } finally {
      fixture.destroy();
    }
  });

  it(`minimum bounds forwards the group limit without changing shape rows (details=${withDetails})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createStatusFixture(device, 'bounds', withDetails);
    try {
      const limited = await fixture.run(COMBINED);
      expect(limited.overflow).toBe(1);
      if (withDetails) expect(limited.groupOverflow).toBe(1);
      expect(limited.sizes).toEqual([4, 3, 3]);
      expect(limited.bounds!.slice(0, 4).every(Number.isNaN)).toBe(true);
      expect(limited.bounds!.slice(4)).toEqual([10, 0, 12, 2, 20, 0, 22, 2]);
      const complete = await fixture.run(COMPLETE);
      expect(complete.overflow).toBe(0);
      if (withDetails) expect(complete.groupOverflow).toBe(0);
      expect(complete.bounds!.slice(0, 4)).toEqual([0, 0, 2, 2]);
      const empty = await fixture.run(EMPTY);
      expect(empty.overflow).toBe(0);
      if (withDetails) expect(empty.groupOverflow).toBe(0);
      expect(empty.sizes).toEqual([0, 0, 0]);
      expect(empty.bounds!.every(Number.isNaN)).toBe(true);
    } finally {
      fixture.destroy();
    }
  });
}
