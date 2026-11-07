// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUDatumGridTransform,
  GPU_DATUM_GRID_TRANSFORM_STATUS
} from '../../src/geospatial-analysis/gpu-datum-grid-transform';
import * as projectionExports from '@luma.gl/graph/gpu-project';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

it('GPU_DATUM_GRID_TRANSFORM_STATUS exposes the documented status bits', () => {
  expect(GPU_DATUM_GRID_TRANSFORM_STATUS).toEqual({
    success: 0,
    invalid: 1,
    noCoverage: 2,
    nodata: 4,
    inverseExhausted: 8
  });
  expect(projectionExports.GPU_DATUM_GRID_TRANSFORM_STATUS).toBe(GPU_DATUM_GRID_TRANSFORM_STATUS);
});

it('GPUDatumGridTransform kernels write the named status values', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const contributor = new GPUDatumGridTransform({
    positions: createTransientView(graph, 'positions', 'float32x3', 4),
    samples: createTransientView(graph, 'samples', 'float32x3', 4),
    validity: createTransientView(graph, 'validity', 'uint32', 4),
    grids: [{width: 2, height: 2, sampleOffset: 0, origin: [0, 0], spacing: [1, 1]}],
    operation: 'horizontal',
    direction: 'inverse',
    output: {
      positions: createTransientView(graph, 'output', 'float32x3', 4),
      status: createTransientView(graph, 'status', 'uint32', 4)
    }
  });
  const [kernel] = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const {invalid, noCoverage, nodata, inverseExhausted} = GPU_DATUM_GRID_TRANSFORM_STATUS;
  expect(kernel.source).toContain(`GridSample(vec3f(0.0), ${invalid}u)`);
  expect(kernel.source).toContain(`GridSample(vec3f(0.0), ${noCoverage}u)`);
  expect(kernel.source).toContain(`GridSample(vec3f(0.0), ${nodata}u)`);
  expect(kernel.source).toContain(`failure = ${inverseExhausted}u`);
});
