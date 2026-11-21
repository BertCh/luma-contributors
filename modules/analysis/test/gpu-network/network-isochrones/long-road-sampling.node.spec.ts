// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {createLongRoadSamplingFixture} from './long-road-sampling-fixture';
import {splatIsochroneRasterOnCPU} from './network-isochrones-oracle';

it('keeps the midpoint when a long road reaches the three-sample cap (WGS-10)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const fixture = createLongRoadSamplingFixture({
    graph,
    input(format, values) {
      const view = createTransientView(
        graph,
        `input-${serial++}`,
        format,
        (values.length * 4) / getGPUVectorFormatInfo(format).byteLength
      );
      memory.write(
        view,
        format.startsWith('uint32') ? Uint32Array.from(values) : Float32Array.from(values)
      );
      return view;
    },
    output(name, format, length) {
      return createTransientView(graph, name, format, length);
    }
  });
  try {
    // Independent row-walking reference agrees with the hand-derived midpoint costs.
    expect(
      Array.from(
        splatIsochroneRasterOnCPU(
          {
            offsets: Uint32Array.of(0, 1, 1),
            neighbors: Uint32Array.of(1),
            weights: Float32Array.of(2)
          },
          Float32Array.of(-(2 ** 30), 0, 2 ** 30, 0),
          Float32Array.of(0, Infinity),
          {
            width: 3,
            height: 1,
            extent: [-1.5, -0.5, 1.5, 0.5],
            mode: 'min',
            bufferRadius: 1,
            walkCostPerUnit: 0,
            maximumBufferPixels: 6,
            maximumSamplesPerEdge: 3,
            unreachedCost: 1e30
          }
        )
      )
    ).toEqual([1, 1, 1]);
    executeCommandNodesOnCPU(fixture.nodes, memory);
    expect(Array.from(memory.readFloat32(fixture.values))).toEqual([1, 1, 1]);
  } finally {
    device.destroy();
  }
});
