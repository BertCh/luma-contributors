// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUDistanceFieldParameterValues} from '../../../src/raster-analysis/distance-field/distance-field-parameters';
import {
  getGPURasterProfileParameterValues,
  getGPURasterSamplingParameterValues
} from '../../../src/raster-analysis/raster-sampling/raster-sampling-parameters';
import {getRasterSamplingWGSL} from '../../../src/raster-analysis/raster-sampling/raster-sampling-utils';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';

it('FLT-02 packs the same reciprocal of the canonical float32 step in every route', () => {
  const grid = makeGPURasterMetadata({
    width: 21,
    height: 2,
    cellSize: [0.85, 0.85],
    origin: [10000000, 0]
  });
  const settings = {localOrigin: [10000000, 0] as const};
  const distance = getGPUDistanceFieldParameterValues(settings, grid);
  const sampling = getGPURasterSamplingParameterValues(settings, grid);
  const profile = getGPURasterProfileParameterValues({...settings, spacing: 1}, grid);
  // IEEE float32(.85) = .8500000238418579; float32(1 / that) = 1.1764705181121826.
  expect([distance[4], sampling[6], profile[6]]).toEqual(Array(3).fill(1.1764705181121826));
  // 41 * float32(1 / 41) rounds below one: nearest must correct against the actual edge.
  expect(Math.fround(41 * Math.fround(1 / 41))).toBe(0.9999999403953552);
  expect(getRasterSamplingWGSL({width: 2, height: 2, hasValidity: false})).toContain(
    'getRasterCellIndex(x, minX'
  );
});

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterProfile} from '../../../src/raster-analysis/raster-sampling/gpu-raster-profile';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('WRT-07 infers the minimum column width in either order and checks explicit capacities', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const [positionWidth, valueWidth] of [
      [1, 3],
      [3, 1]
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = {
        grid: makeGPURasterMetadata({width: 1, height: 1}),
        values: createTransientView(graph, 'raster', 'float32', 1),
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
        parameters: createTransientView(graph, 'parameters', 'float32', 12),
        output: {
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          },
          samplePositions: createTransientView(
            graph,
            'samplePositions',
            'float32x2',
            positionWidth
          ),
          sampleValues: createTransientView(graph, 'sampleValues', 'float32', valueWidth)
        }
      };
      expect(new GPURasterProfile(props).sampleCapacity).toBe(1);
      expect(new GPURasterProfile({...props, sampleCapacity: 1}).sampleCapacity).toBe(1);
      expect(() => new GPURasterProfile({...props, sampleCapacity: 2})).toThrow(
        /output.sample(Positions|Values).*2/
      );
    }
  } finally {
    device.destroy();
  }
});

it('WRT-11 status-only profiles omit locate, sample, raster staging and sample scratch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const profile = new GPURasterProfile({
      grid: makeGPURasterMetadata({width: 1, height: 1}),
      values: createTransientView(graph, 'raster', 'float32', 1),
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      sampleCapacity: 1,
      output: {
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    });
    const identifiers = profile.getCommandNodes(graph).map(node => node.id);
    expect(identifiers).not.toContain('raster-profile-locate');
    expect(identifiers).not.toContain('raster-profile-sample');
    expect(identifiers).toContain('raster-profile-finish-counts');
    // A capacity-proportional allocation would violate this 128 MiB device binding limit.
    expect(() =>
      new GPURasterProfile({
        ...profile.props,
        id: 'status-large',
        sampleCapacity: 20000000
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('W3-08 defaults to propagate and rejects retired and unknown tokens', () => {
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  for (const pack of [
    (settings: Parameters<typeof getGPURasterSamplingParameterValues>[0]) =>
      getGPURasterSamplingParameterValues(settings, grid),
    (settings: Parameters<typeof getGPURasterSamplingParameterValues>[0]) =>
      getGPURasterProfileParameterValues({...settings, spacing: 1}, grid)
  ]) {
    expect(pack({noDataPolicy: 'propagate'})[9]).toBe(0);
    expect(() => pack({noDataPolicy: 'strict' as never})).toThrow(
      /noDataPolicy.*strict.*use 'propagate'/
    );
    expect(pack({})[9]).toBe(0);
    expect(pack({noDataPolicy: 'renormalize'})[9]).toBe(1);
    expect(() => pack({noDataPolicy: 'ignore' as never})).toThrow(
      /noDataPolicy.*propagate.*renormalize/
    );
    expect(() => pack({method: 'toString' as never})).toThrow(/method.*nearest.*bilinear.*bicubic/);
  }
});

import {createRequire} from 'node:module';
import {GraphVectorView} from '@luma.gl/graph/gpu-core';
import {validateRasterSamplingView} from '../../../src/raster-analysis/raster-sampling/raster-sampling-utils';
import {validateIsolinesView} from '../../../src/raster-analysis/isolines/isolines-utils';

it('W3-09 rejects foreign-condition chunked vectors and accepts actual packed views', () => {
  const ForeignGraphVectorView = (
    createRequire(import.meta.url)(
      '@luma.gl/graph/gpu-core'
    ) as typeof import('@luma.gl/graph/gpu-core')
  ).GraphVectorView;
  expect(ForeignGraphVectorView).not.toBe(GraphVectorView);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const first = createTransientView(graph, 'first', 'float32', 1);
    const second = createTransientView(graph, 'second', 'float32', 1);
    const vector = new ForeignGraphVectorView({
      id: 'foreign',
      name: 'foreign',
      format: 'float32',
      length: 2,
      valueLength: 2,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [first, second]
    });
    expect(vector).not.toBeInstanceOf(GraphVectorView);
    for (const validate of [validateRasterSamplingView, validateIsolinesView]) {
      expect(() => validate('mixed', 'values', vector, 'float32', 2)).toThrow(
        /single packed view.*chunked vector/
      );
      expect(() => validate('mixed', 'values', first, 'float32', 1)).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});
