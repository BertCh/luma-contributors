// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {createMomentsNode} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels';
import {getShapeMomentTileNodes} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-moment-tiles';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';

useCPUExecutorTestHooks();
for (const route of ['serial', 'tiled'] as const) {
  it(`B4 ${route} retains both unit cross products in one tile`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = {
      id: 'single-tile',
      operation: 'GPUShapeDescriptors',
      featureCount: 1,
      positions: createTransientView(graph, 'positions', 'float32x2', 5),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      centroids: createTransientView(graph, 'centroids', 'float32x2', 1),
      moments: createTransientView(graph, 'moments', 'float32x4', 1)
    };
    const createView = vi.spyOn(graph, 'createDataView');
    const nodes =
      route === 'serial'
        ? [createMomentsNode(graph, props)]
        : getShapeMomentTileNodes(graph, props);
    const tileIndex = nodes.findIndex(node => node.id.endsWith('-tiles'));
    const selected = route === 'serial' ? nodes : nodes.slice(0, tileIndex + 1);
    const run = (disableCompensation: boolean) => {
      const memory = createCPUGraphMemory();
      memory.write(
        props.positions,
        Float32Array.of(0, 0, 4096, 0, 0, 4096, -1 / 4096, 0, 0, -4096)
      );
      memory.write(props.ringOffsets, Uint32Array.of(0, 5));
      memory.write(props.featureOffsets, Uint32Array.of(0, 1));
      memory.write(props.polygonOffsets, Uint32Array.of(0, 1));
      memory.write(props.centroids, Float32Array.of(0, 0));
      const executionNodes = selected.map(node => {
        if (node.type !== 'compute') {
          throw new Error(`${node.id} must be a compute node`);
        }
        const compile: typeof node.compile = context => {
          const createShader = context.device.createShader;
          context.device.createShader = shaderProps =>
            createShader.call(context.device, {
              ...shaderProps,
              source: disableCompensation
                ? shaderProps.source.replace(
                    'let correction = accumulator.y + getAdditionRoundingError(previous, value, total);',
                    'let correction = accumulator.y;'
                  )
                : shaderProps.source
            });
          try {
            return node.compile(context);
          } finally {
            context.device.createShader = createShader;
          }
        };
        return {...node, compile};
      });
      executeGeometryNodesOnCPU(executionNodes, memory);
      if (route === 'serial') return memory.readFloat32(props.moments)[3];
      const tileArea = createView.mock.results.find(
        result => result.type === 'return' && result.value.buffer.id === 'single-tile-tile-area'
      )!.value;
      return memory.readFloat32(tileArea)[0] / 2;
    };
    try {
      // Successive nonzero crosses: 4096^2, 4096/4096, (-1/4096)*(-4096).
      // Area = (2^24 + 1 + 1)/2 = 8388609 exactly, from five binary32-exact vertices.
      const actual = run(false);
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBe(8388609);
      expect(run(true)).toBe(8388608); // Removing compensation must fail the exact area assertion.
    } finally {
      createView.mockRestore();
      device.destroy();
    }
  });
}
