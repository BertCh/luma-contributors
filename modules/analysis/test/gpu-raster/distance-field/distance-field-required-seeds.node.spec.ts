// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUDistanceField,
  type GPUDistanceFieldProps
} from '../../../src/raster-analysis/distance-field';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('TYP-11 requires either seed source and reserves IDs and counts for point seeds', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const base = {
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      output: {distances: createTransientView(graph, 'distances', 'float32', 4)}
    };
    const seedPositions = createTransientView(graph, 'points', 'float32x2', 1);
    const seedMask = createTransientView(graph, 'mask', 'uint32', 4);
    const seedIds = createTransientView(graph, 'ids', 'uint32', 1);
    const seedCount = createTransientView(graph, 'count', 'uint32', 1);
    const points: GPUDistanceFieldProps = {...base, seedPositions, seedIds, seedCount};
    const mask: GPUDistanceFieldProps = {...base, seedMask};
    const both: GPUDistanceFieldProps = {...points, seedMask};
    for (const props of [points, mask, both]) {
      expect(() => new GPUDistanceField(props)).not.toThrow();
    }
    expect(() => {
      // @ts-expect-error At least one seed source is required.
      new GPUDistanceField(base);
    }).toThrow(/^distance-field requires seedPositions or seedMask$/);
    expect(() => {
      // @ts-expect-error A mask alone cannot have point IDs.
      new GPUDistanceField({...base, seedMask, seedIds});
    }).toThrow(/^distance-field seedIds and seedCount require seedPositions$/);
    expect(() => {
      // @ts-expect-error A mask alone cannot have a point count.
      new GPUDistanceField({...base, seedMask, seedCount});
    }).toThrow(/^distance-field seedIds and seedCount require seedPositions$/);
  } finally {
    device.destroy();
  }
});
