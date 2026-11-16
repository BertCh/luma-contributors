// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkAccessibility,
  getGPUNetworkAccessibilityParameterValues
} from '../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {GPUNetworkCostMatrix} from '../../src/gpu-network/network-accessibility/gpu-network-cost-matrix';
import {GPUNetworkCostTable} from '../../src/gpu-network/network-cost-table/gpu-network-cost-table';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

function createPackedContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const packed = createOutputBuffer(device, 132);
  const buffers: Buffer[] = [packed];
  // Import the physical allocation once; dispatches address all fields through this handle.
  const storage = importGraphBuffer(graph, 'packed', packed, 'uint32', 132).buffer;
  const input = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const output = <Format extends 'float32' | 'uint32'>(
    format: Format,
    length: number,
    byteOffset: number
  ) => graph.createDataView(storage, {format, length, byteOffset});
  return {graph, packed, buffers, input, output};
}

async function runPackedGraph(
  build: (context: ReturnType<typeof createPackedContext>) => void,
  verify: (packed: Buffer) => Promise<void>
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createPackedContext(device);
  try {
    build(context);
    const compiled = context.graph.compile();
    try {
      await submitGraph(device, compiled, undefined);
      await verify(context.packed);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of context.buffers) buffer.destroy();
  }
}

it('accessibility computes closed-form scores and catchment into packed outputs', async () => {
  await runPackedGraph(
    ({graph, input, output}) => {
      graph.add(
        new GPUNetworkAccessibility({
          costs: input('costs', Float32Array.of(0, 3, 4, 0), 'float32'),
          opportunityWeights: input('supply', Float32Array.of(10, 20), 'float32'),
          parameters: input(
            'parameters',
            getGPUNetworkAccessibilityParameterValues({threshold: 3}),
            'float32'
          ),
          cumulative: output('float32', 2, 0),
          gravity: output('float32', 2, 256),
          catchment: {
            demand: input('demand', Float32Array.of(2, 3), 'float32'),
            output: output('float32', 2, 8)
          }
        })
      );
    },
    async packed => {
      const values = await readFloat32(packed, 132);
      expect(values.slice(0, 2)).toEqual([10, 30]);
      expect(values.slice(64, 66)).toEqual([10, 30]);
      // Facility ratios are 10/(2+3) and 20/3; node 0 sees only facility 0.
      expect(values[2]).toBeCloseTo(2, 5);
      expect(values[3]).toBeCloseTo(2 + 20 / 3, 5);
    }
  );
});

it('cost matrix publishes costs and adjacent status in one physical allocation', async () => {
  await runPackedGraph(
    ({graph, input, output}) => {
      graph.add(
        new GPUNetworkCostMatrix({
          topology: {
            offsets: input('offsets', Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: input('neighbors', Uint32Array.of(1, 0), 'uint32'),
            weights: input('weights', Float32Array.of(3, 4), 'float32')
          },
          startNodes: input('start-nodes', Uint32Array.of(0), 'uint32'),
          laneCount: 1,
          maximumIterations: 2,
          output: {
            costs: output('float32', 2, 0),
            status: {
              converged: output('uint32', 1, 8),
              iterationLimitReached: output('uint32', 1, 256)
            }
          }
        })
      );
    },
    async packed => {
      expect((await readFloat32(packed, 132)).slice(0, 2)).toEqual([0, 3]);
      const status = await readUint32(packed, 132);
      expect(status[2]).toBe(1);
      expect(status[64]).toBe(0);
    }
  );
});

it('cost table publishes packed CSR fields and adjacent overflow with exact costs', async () => {
  await runPackedGraph(
    ({graph, input, output}) => {
      graph.add(
        new GPUNetworkCostTable({
          topology: {
            offsets: input('offsets', Uint32Array.of(0, 1, 2), 'uint32'),
            neighbors: input('neighbors', Uint32Array.of(1, 0), 'uint32'),
            weights: input('weights', Float32Array.of(3, 4), 'float32')
          },
          costLimit: input('cost-limit', Float32Array.of(10), 'float32'),
          laneCount: 1,
          maximumIterations: 2,
          output: {
            offsets: output('uint32', 3, 0),
            targets: output('uint32', 4, 256),
            costs: output('float32', 4, 512),
            status: {overflow: output('uint32', 1, 12)}
          }
        })
      );
    },
    async packed => {
      const words = await readUint32(packed, 132);
      expect(words.slice(0, 4)).toEqual([0, 2, 4, 0]);
      expect(words.slice(64, 68)).toEqual([0, 1, 0, 1]);
      expect((await readFloat32(packed, 132)).slice(128, 132)).toEqual([0, 3, 4, 0]);
    }
  );
});
