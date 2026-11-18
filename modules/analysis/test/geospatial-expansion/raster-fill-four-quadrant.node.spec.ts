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
      status: {invalidCount: createTransientView(graph, 'unresolved', 'uint32', 1)}
    }
  };
  return {graph, props};
}
it('four-quadrant maximum-radius kernels parse within the default binding limit', () => {
  const {graph, props} = setup();
  const kernels = getParsedKernelSources(new GPURasterFillNoData(props).getCommandNodes(graph));
  expect(kernels.size).toBeGreaterThan(0);
  for (const {source} of kernels.configurations)
    expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
});
it('four-quadrant fill rejects invalid bounds and unknown method names', () => {
  const {props} = setup();
  for (const maximumDistance of [0, -1, 1.5, 257, Infinity, NaN]) {
    expect(() => new GPURasterFillNoData({...props, maximumDistance})).toThrow();
  }
  // @ts-expect-error Unknown interpolation methods must also fail for JavaScript callers.
  expect(() => new GPURasterFillNoData({...props, method: 'gdal'})).toThrow();
});
it('fill no-data validates its named status port', () => {
  const {graph, props} = setup();
  expect(
    () =>
      new GPURasterFillNoData({
        ...props,
        output: {
          ...props.output,
          status: {invalidCount: createTransientView(graph, 'short', 'float32', 1) as never}
        }
      })
  ).toThrow(/^raster-fill-nodata invalidCount must contain one uint32 row/);
  expect(
    () =>
      new GPURasterFillNoData({
        ...props,
        output: {
          ...props.output,
          status: {invalidCount: createTransientView(graph, 'empty', 'uint32', 0)}
        }
      })
  ).toThrow(/^raster-fill-nodata/);
  // A JavaScript caller still passing the old `unresolvedCount` and no status gets the contributor
  // ID and the prop name, not a TypeError from Object.entries.
  const {status: _status, ...outputWithoutStatus} = props.output;
  expect(
    () =>
      new GPURasterFillNoData({
        ...props,
        output: {
          ...outputWithoutStatus,
          unresolvedCount: createTransientView(graph, 'old-count', 'uint32', 1)
        } as never
      })
  ).toThrow(/^raster-fill-nodata output\.status is required/);
  // The count is optional: without it no counting node is built.
  const withCount = setup();
  const withoutCount = setup();
  expect(
    new GPURasterFillNoData({
      ...withoutCount.props,
      output: {...withoutCount.props.output, status: {}}
    }).getCommandNodes(withoutCount.graph).length
  ).toBeLessThan(new GPURasterFillNoData(withCount.props).getCommandNodes(withCount.graph).length);
});
