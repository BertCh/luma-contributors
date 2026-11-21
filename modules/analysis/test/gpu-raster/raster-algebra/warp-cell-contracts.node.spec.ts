// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPURasterReclassify,
  type GPURasterReclassifyOutput
} from '../../../src/raster-analysis/raster-algebra/gpu-raster-reclassify';
import type {GPURasterCellStatisticsOutput} from '../../../src/raster-analysis/raster-algebra/gpu-raster-cell-statistics';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {getGPURasterCellStatisticsParameterValues} from '../../../src/raster-analysis/raster-algebra/local-operations-parameters';
import {getGPUWeightedOverlayParameterValues} from '../../../src/raster-analysis/raster-algebra/weighted-overlay-parameters';
import {getGPURasterReclassifyParameterValues} from '../../../src/raster-analysis/raster-algebra/reclassify-parameters';

it('B2 supports exact active count with mask, class values and every output at eight bindings', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBuffersPerShaderStage: 8}
  });
  const graph = new GPUCommandGraph(device);
  const contributor = new GPURasterReclassify({
    values: createTransientView(graph, 'input', 'float32', 1),
    breaks: createTransientView(graph, 'breaks', 'float32', 31),
    classValues: createTransientView(graph, 'class-values', 'float32', 32),
    mask: createTransientView(graph, 'mask', 'uint32', 1),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    activeBreakCount: createTransientView(graph, 'active', 'uint32', 1),
    output: {
      classes: createTransientView(graph, 'classes', 'uint32', 1),
      reclassified: createTransientView(graph, 'reclassified', 'float32', 1),
      classCounts: createTransientView(graph, 'counts', 'uint32', 32)
    }
  });
  try {
    const kernels = getParsedKernelSources(contributor.getCommandNodes(graph));
    for (const {source} of kernels.configurations)
      expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
    for (const {source} of kernels.getConfigurations('raster-reclassify-classify'))
      expect(getStorageBindingCount(source)).toBe(8);
  } finally {
    device.destroy();
  }
});

it('B4 ignores inactive table ramps and atomically rejects switching them back to linear', () => {
  const target = new Float32Array(16).fill(77);
  const layer = {weight: 1, breakCount: 2, inputMinimum: 4096, inputMaximum: 4096.0001};
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{...layer, mode: 'table'}]}, target)
  ).not.toThrow();
  expect(target[9]).toBe(1);
  expect(target[13]).toBe(2);
  const before = new Uint8Array(target.buffer).slice();
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{...layer, mode: 'linear'}]}, target)
  ).toThrow(/requested.*encoded/);
  expect(new Uint8Array(target.buffer)).toEqual(before);
});

it('S3 rejects an active ramp reciprocal that can flush, while allowing it in table mode', () => {
  const target = new Float32Array(16).fill(77);
  const before = new Uint8Array(target.buffer).slice();
  const layer = {weight: 1, inputMinimum: 0, inputMaximum: 2 ** 127};
  expect(() => getGPUWeightedOverlayParameterValues({layers: [layer]}, target)).toThrow(
    /normal float32.*scale/
  );
  expect(new Uint8Array(target.buffer)).toEqual(before);
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{...layer, mode: 'table'}]}, target)
  ).not.toThrow();
});

it('W3-10 accepts every minimal terminal selection and rejects empty selections in types', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const float = createTransientView(graph, 'float', 'float32', 1);
  const integer = createTransientView(graph, 'integer', 'uint32', 1);
  const reclassify: GPURasterReclassifyOutput[] = [
    {classes: integer},
    {reclassified: float},
    {classCounts: integer}
  ];
  const cells: GPURasterCellStatisticsOutput[] = [
    {minimum: float},
    {maximum: float},
    {range: float},
    {sum: float},
    {mean: float},
    {standardDeviation: float},
    {majority: float},
    {minority: float},
    {variety: integer},
    {counts: integer}
  ];
  // @ts-expect-error at least one terminal is required
  const emptyReclassify: GPURasterReclassifyOutput = {};
  // @ts-expect-error at least one terminal is required
  const emptyCells: GPURasterCellStatisticsOutput = {};
  // @ts-expect-error B6 computed entries do not establish a required terminal.
  const computedCells: GPURasterCellStatisticsOutput = Object.fromEntries([['mean', float]]);
  const explicitCells = {
    ...Object.fromEntries([['mean', float]]),
    mean: float
  } satisfies GPURasterCellStatisticsOutput;
  expect(reclassify).toHaveLength(3);
  expect(cells).toHaveLength(10);
  expect(emptyReclassify).toEqual({});
  expect(emptyCells).toEqual({});
  expect(computedCells).toEqual(explicitCells);
  device.destroy();
});

it('PCK-08 binds an exact uint32 count for a table beyond float32 integer precision', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPURasterReclassify({
    values: createTransientView(graph, 'input', 'float32', 1),
    breaks: createTransientView(graph, 'breaks', 'float32', 16777217),
    parameters: createTransientView(graph, 'params', 'float32', 4),
    activeBreakCount: createTransientView(graph, 'active', 'uint32', 1),
    output: {classes: createTransientView(graph, 'classes', 'uint32', 1)}
  });
  try {
    expect(
      [...getParsedKernelSources(contributor.getCommandNodes(graph)).values()].join('\n')
    ).toContain('min(params[paramsOffset], MAXIMUM_BREAK_COUNT)');
  } finally {
    device.destroy();
  }
});

it('PCK-08 rejects a rounded compact count using only four words', () => {
  expect(() => getGPURasterReclassifyParameterValues({breakCount: 16777217})).toThrow(
    /breakCount.*exact.*uint32.*16777217/
  );
  expect(() => getGPURasterReclassifyParameterValues({breakCount: 4294967296})).toThrow(
    /breakCount/
  );
  expect(getGPURasterReclassifyParameterValues({breakCount: 16777216})[0]).toBe(16777216);
});

it('PCK-07/16 rejects collapsed ramps and leaves a good packet untouched on any invalid layer', () => {
  const target = new Float32Array(24).fill(77);
  const before = new Uint8Array(target.buffer).slice();
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{weight: 2}, {weight: NaN}]}, target)
  ).toThrow(/weight/);
  expect(new Uint8Array(target.buffer)).toEqual(before);
  expect(() =>
    getGPUWeightedOverlayParameterValues({
      layers: [{weight: 1, inputMinimum: 99.999999, inputMaximum: 100.000001}]
    })
  ).toThrow(/99.999999.*100.000001.*100.*100/);
  const equal = getGPUWeightedOverlayParameterValues({
    layers: [{weight: 1, inputMinimum: 100, inputMaximum: 100}]
  });
  expect(equal[15]).toBe(1); // deliberate equality retains the upper-step rule
});

it('NAM-05 packs strict exactly like propagate and rejects unknown policy words', () => {
  expect(getGPURasterCellStatisticsParameterValues({noDataPolicy: 'strict'})).toEqual(
    getGPURasterCellStatisticsParameterValues({noDataPolicy: 'propagate'})
  );
  expect(() =>
    getGPURasterCellStatisticsParameterValues({noDataPolicy: 'typo' as 'ignore'})
  ).toThrow(/ignore.*propagate.*strict/);
  expect(
    getGPUWeightedOverlayParameterValues({
      layers: [{weight: 1}],
      normalizeWeights: false,
      noDataPolicy: 'strict'
    })
  ).toEqual(
    getGPUWeightedOverlayParameterValues({
      layers: [{weight: 1}],
      normalizeWeights: false,
      noDataPolicy: 'propagate'
    })
  );
  expect(() =>
    getGPUWeightedOverlayParameterValues({
      layers: [{weight: 1}],
      normalizeWeights: false,
      noDataPolicy: 'typo' as 'ignore'
    })
  ).toThrow(/propagate.*strict.*ignore/);
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{weight: 1, mode: 'typo' as 'linear'}]})
  ).toThrow(/mode.*linear.*table.*typo/);
  expect(() =>
    getGPUWeightedOverlayParameterValues({layers: [{weight: 1, closed: 'typo' as 'left'}]})
  ).toThrow(/closed.*left.*right.*typo/);
  expect(() =>
    getGPURasterReclassifyParameterValues({breakCount: 1, closed: 'typo' as 'left'})
  ).toThrow(/closed.*left.*right.*typo/);
});
