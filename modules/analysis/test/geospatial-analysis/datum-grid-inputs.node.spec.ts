// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUDatumGridTransform} from '../../src/geospatial-analysis/datum-grid-transform';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

it('VAL-04 rejects selectors before vertical reference checks', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = {
      positions: createTransientView(graph, 'input', 'float32x3', 1),
      samples: createTransientView(graph, 'samples', 'float32x3', 4),
      mask: createTransientView(graph, 'mask', 'uint32', 4),
      grids: [],
      operation: 'horizontal' as const,
      output: {
        positions: createTransientView(graph, 'output', 'float32x3', 1),
        status: createTransientView(graph, 'status', 'uint32', 1)
      }
    };
    expect(
      () => new GPUDatumGridTransform({...props, operation: 'horizonal' as 'horizontal'})
    ).toThrow(/operation.*horizonal.*horizontal.*vertical.*combined/);
    expect(
      () =>
        new GPUDatumGridTransform({
          ...props,
          operation: 'combined',
          direction: 'reverse' as 'forward'
        })
    ).toThrow(/direction.*reverse.*forward.*inverse/);
    expect(() => new GPUDatumGridTransform(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});
