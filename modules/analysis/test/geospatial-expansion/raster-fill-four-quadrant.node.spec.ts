// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPURasterFillNoData,
  type GPURasterFillNoDataProps
} from '../../src/raster-analysis/gpu-raster-fill-nodata';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

function setup() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const props: GPURasterFillNoDataProps = {
    grid: {width: 3, height: 3, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
    input: {
      id: 'source',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'source', 'float32', 9)}
    },
    maximumDistance: 256,
    method: 'gdal-four-quadrant',
    output: {
      values: createTransientView(graph, 'values', 'float32', 9),
      validity: createTransientView(graph, 'validity', 'uint32', 9),
      filled: createTransientView(graph, 'filled', 'uint32', 9),
      unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)
    }
  };
  return {graph, props};
}
it('four-quadrant maximum-radius kernels parse within the default binding limit', () => {
  const {graph, props} = setup();
  const kernels = getParsedKernelSources(new GPURasterFillNoData(props).getCommandNodes(graph));
  expect(kernels.size).toBeGreaterThan(0);
  for (const kernel of kernels.values())
    expect(getStorageBindingCount(kernel)).toBeLessThanOrEqual(8);
});
it('four-quadrant fill rejects invalid bounds and unknown method names', () => {
  const {props} = setup();
  for (const maximumDistance of [0, -1, 1.5, 257, Infinity, NaN]) {
    expect(() => new GPURasterFillNoData({...props, maximumDistance})).toThrow();
  }
  // @ts-expect-error Unknown interpolation methods must also fail for JavaScript callers.
  expect(() => new GPURasterFillNoData({...props, method: 'gdal'})).toThrow();
});
