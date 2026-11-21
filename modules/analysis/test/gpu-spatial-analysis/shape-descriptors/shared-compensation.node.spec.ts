// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {createMomentsNode} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-descriptors-kernels';
import {getShapeMomentTileNodes} from '../../../src/gpu-spatial-analysis/shape-descriptors/shape-moment-tiles';
import {
  createLinePrefixNode,
  ADDITION_ROUNDING_ERROR_WGSL
} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

for (const route of ['serial', 'tiled', 'line-prefix'] as const) {
  it(`GEO-131 ${route} includes the shared compensated helper once`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1024);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
    const inputs = {
      id: 'moments',
      operation: 'GPUShapeDescriptors',
      positions,
      ringOffsets: offsets,
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      featureCount: 1,
      centroids: createTransientView(graph, 'centroids', 'float32x2', 1),
      moments: createTransientView(graph, 'moments', 'float32x4', 1)
    };
    try {
      const nodes =
        route === 'serial'
          ? [createMomentsNode(graph, inputs)]
          : route === 'tiled'
            ? getShapeMomentTileNodes(graph, inputs)
            : [
                createLinePrefixNode(graph, {
                  id: 'prefix',
                  operation: 'GPULineSegmentize',
                  positions,
                  lineOffsets: offsets,
                  coordinateSystem: 'planar',
                  radius: 1,
                  rowMeasures: createTransientView(graph, 'measures', 'float32', positions.length)
                })
              ];
      const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
        kernel.source.includes(
          route === 'line-prefix'
            ? 'prefixCarry'
            : route === 'serial'
              ? 'fn addNeumaier'
              : 'fn addMoment'
        )
      )!.source;
      expect(source.match(/fn addCompensatedValue\(/g)).toHaveLength(1);
      expect(source.match(/fn getAdditionRoundingError\(/g)).toHaveLength(1);
      if (route === 'serial') expect(source).toContain('addCompensated(accumulator, value)');
      if (route === 'tiled')
        expect(source).toContain('return addCompensatedValue(accumulator, value)');
      if (route === 'line-prefix')
        expect(source).toContain('measure = addCompensatedValue(measure, segmentLength)');
      if (route === 'line-prefix')
        expect(source).toContain(
          'let carry = addCompensatedValue(vec2<f32>(prefixCarry[0], prefixCarry[1]), total)'
        );
      expect(() => new WgslReflect(source)).not.toThrow();
      expect(ADDITION_ROUNDING_ERROR_WGSL).toContain('fn getAdditionRoundingError');
    } finally {
      device.destroy();
    }
  });
}
