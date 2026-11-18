// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUBVH, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {runExample} from '../../../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/pair-gather.example';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('the real pair-gather example accepts the headless fixture resource names', async () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const source = readFileSync(
    new URL('./pair-gather-completeness.spec.ts', import.meta.url),
    'utf8'
  );
  const fixtureSource = source.slice(source.indexOf("it('F4 public example"));
  const resourceId = fixtureSource.match(
    /overflow:\s*fixture\.output\('([^']+)',\s*'uint32',\s*1\)/
  )?.[1];
  expect(resourceId).toBeDefined();
  const scalar = (id: string) => createTransientView(graph, id, 'uint32', 1);
  const polygon = (id: string, count: number) => ({
    kind: 'polygons' as const,
    positions: createTransientView(graph, `${id}-positions`, 'float32x2', count * 4),
    featureOffsets: createTransientView(graph, `${id}-features`, 'uint32', count + 1),
    polygonOffsets: createTransientView(graph, `${id}-polygons`, 'uint32', count + 1),
    ringOffsets: createTransientView(graph, `${id}-rings`, 'uint32', count + 1)
  });
  try {
    await expect(
      runExample(graph, {
        parks: polygon('parks', 1),
        tracts: polygon('tracts', 2),
        pairs: {
          leftRows: createTransientView(graph, 'left', 'uint32', 2),
          rightRows: createTransientView(graph, 'right', 'uint32', 2),
          status: {count: scalar('count'), overflow: scalar(resourceId!)}
        },
        parkAreas: createTransientView(graph, 'areas', 'float32', 1),
        spatialContext: {coordinateSpace: 'planar', metric: 'native'},
        candidateCapacity: 1
      })
    ).resolves.toHaveProperty('candidateOverflow');
  } finally {
    device.destroy();
  }
});

it('invalid slotCapacity is diagnosed before incomplete output props are inspected', () => {
  const view = {length: 4, format: 'uint32'} as never;
  expect(
    () =>
      new GPUNearestFeatureWeights({
        neighborRows: view,
        neighborCounts: view,
        slotCapacity: 0,
        output: {} as never
      })
  ).toThrow(/slotCapacity/);
});

for (const leafCapacity of [1, 2]) {
  it(`the BVH recovery fixture counts two source rows even when one is empty (${leafCapacity} leaves)`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const minima = createTransientView(graph, 'minima', 'float32x2', 2);
    const maxima = createTransientView(graph, 'maxima', 'float32x2', 2);
    const count = createTransientView(graph, 'count', 'uint32', 1);
    const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
    const nodeCount = 2 * leafCapacity - 1;
    const bvh = new GPUBVH({
      id: 'recovery-probe',
      strategy: 'level',
      minima,
      maxima,
      leafCapacity,
      nodeMinima: createTransientView(graph, 'node-minima', 'float32x2', nodeCount),
      nodeMaxima: createTransientView(graph, 'node-maxima', 'float32x2', nodeCount),
      nodeChildren: createTransientView(graph, 'children', 'uint32x2', nodeCount),
      leafIds: createTransientView(graph, 'leaf-ids', 'uint32', leafCapacity),
      count,
      overflow
    });
    const memory = createCPUGraphMemory();
    const maximum = Math.fround(3.402823466e38);
    memory.write(minima, Float32Array.of(0, -1, maximum, maximum));
    memory.write(maxima, Float32Array.of(0, 1, -maximum, -maximum));
    try {
      executeCommandNodesOnCPU(bvh.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(count))).toEqual([2]);
      // Empty bounds do not change the compile-time number of source rows.
      expect(Array.from(memory.readUint32(overflow))).toEqual([Number(leafCapacity < 2)]);
    } finally {
      device.destroy();
    }
  });
}
