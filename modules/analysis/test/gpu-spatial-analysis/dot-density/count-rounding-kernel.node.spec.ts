// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUDotDensity} from '../../../src/gpu-spatial-analysis/dot-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('GEO-130 values kernel compares the uniform with the fractional part', () => {
  // Independent integer expectation: floor(16385) = 16385, fractional part = 0.
  // Old f32 subtraction rounds 16385 - 0.9994140863418579 to 16384.
  expect(Math.ceil(Math.fround(16385 - 0.9994140863418579))).toBe(16384);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUDotDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 6),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 3),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 3),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 3),
    values: createTransientView(graph, 'values', 'float32', 2),
    parameters: createTransientView(graph, 'parameters', 'uint32', 8),
    output: {
      positions: createTransientView(graph, 'output', 'float32x2', 16385),
      dots: {
        ids: createTransientView(graph, 'ids', 'uint32', 16385),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    }
  });
  try {
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.source.includes('let scaled = sourceValues')
    )!.source;
    expect(source).not.toContain('ceil(scaled - remainder)');
    expect(source).toContain('let integral = floor(scaled)');
    expect(source).toContain('remainder < scaled - integral');
    expect(() => new WgslReflect(source)).not.toThrow();
  } finally {
    device.destroy();
  }
});
