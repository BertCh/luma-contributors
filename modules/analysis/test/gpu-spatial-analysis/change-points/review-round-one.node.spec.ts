// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPULocationChangePoints,
  type GPULocationChangePointsProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {
  GPULocationChangePointsOperation,
  type LocationChangePointsOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import {GPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {CPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {
  computeLocationChangePointsOnCPU,
  CPULocationChangePointsOperation
} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createLocationProps(graph: GPUCommandGraph): GPULocationChangePointsProps {
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `review-${serial++}`, format, length);
  return {
    id: 'review-location',
    values: view('float32', 4),
    parameters: view('float32', 2),
    layout: 'cell-major',
    cellCount: 1,
    sliceCount: 4,
    cost: 'mean',
    maximumChangePoints: 1,
    output: {changePointCount: view('uint32', 1), changePoints: view('uint32', 1)}
  };
}

for (const length of [0, 2])
  it(`S1 requires a scalar overflow output, received length=${length}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const props = createLocationProps(graph);
      props.output.status = {overflow: createTransientView(graph, 'overflow', 'uint32', length)};
      expect(() => new GPULocationChangePoints(props)).toThrow(AnalysisValidationError);
      expect(() => new GPULocationChangePoints(props)).toThrow(/^review-location.*status.overflow/);
    } finally {
      device.destroy();
    }
  });

it('S1 validates the parameter destination before writing and retains the caller identity', () => {
  const device = createNullWebGPUDevice();
  try {
    const producer = new GPULocationChangePoints(createLocationProps(new GPUCommandGraph(device)));
    const target = Float32Array.of(99);
    expect(() => producer.getParameterValues({penalty: 1}, target)).toThrow(
      AnalysisValidationError
    );
    expect(() => producer.getParameterValues({penalty: 1}, target)).toThrow(
      /^review-location.*target.*2/
    );
    expect(Array.from(target)).toEqual([99]);
  } finally {
    device.destroy();
  }
});

const LOCATION_PROPS: LocationChangePointsOperationProps = {
  id: 'shape-location',
  values: 'values',
  parameters: 'parameters',
  cellCount: 1,
  sliceCount: 4,
  layout: 'cell-major',
  cost: 'mean',
  maximumChangePoints: 1,
  output: {changePointCount: 'counts', changePoints: 'changes'}
};

for (const override of [
  {sliceCount: 513},
  {cost: 'slope' as const, sliceCount: 257},
  {maximumChangePoints: 129}
])
  it(`S2 validates GPU location limits before lowering and preserves the CPU shape: ${JSON.stringify(override)}`, () => {
    expect(() => new GPULocationChangePointsOperation({...LOCATION_PROPS, ...override})).toThrow(
      AnalysisValidationError
    );
    expect(
      () => new CPULocationChangePointsOperation({...LOCATION_PROPS, ...override})
    ).not.toThrow();
  });

it('S2 validates GPU Pettitt limits before lowering and preserves the CPU shape', () => {
  const props = {
    id: 'shape-pettitt',
    values: 'values',
    cellCount: 1,
    sliceCount: 513,
    layout: 'cell-major' as const,
    output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
  };
  expect(() => new GPUPettittTestOperation(props)).toThrow(AnalysisValidationError);
  expect(() => new CPUPettittTestOperation(props)).not.toThrow();
  expect(() => new GPUPettittTestOperation({...props, sliceCount: 512})).not.toThrow();
});

it('S3 captures inherited CPU option fields and the caller ID before validation', () => {
  const options = Object.create({...LOCATION_PROPS, id: 'prototype-location'});
  // SSE=100 without splitting, zero within each half, plus penalty=1 at the split.
  const result = computeLocationChangePointsOnCPU(
    Float32Array.of(0, 0, 10, 10),
    Float32Array.of(1, 1),
    options
  );
  expect(Array.from(result.changePointCount)).toEqual([1]);
  expect(Array.from(result.changePoints)).toEqual([2]);
  expect(Array.from(result.optimalCost)).toEqual([1]);
  expect(() =>
    computeLocationChangePointsOnCPU(Float32Array.of(1), Float32Array.of(1, 1), options)
  ).toThrow(/^prototype-location.*values/);
});
