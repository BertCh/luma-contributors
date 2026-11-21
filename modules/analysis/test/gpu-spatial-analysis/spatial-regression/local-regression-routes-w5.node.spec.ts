// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('W5-56 preserves automatic thresholds, dense override and explicit exact Gaussian', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `routes-${serial++}`, format, length);
  const base = (rows: number) => ({
    positions: view('float32x2', rows),
    predictors: view('float32', rows),
    predictorCount: 1,
    response: view('float32', rows),
    parameters: view('float32', 3),
    kernel: 'bisquare' as const,
    output: {
      coefficients: view('float32', rows * 2),
      fitted: view('float32', rows),
      status: {codes: view('uint32', rows), iterationCount: view('uint32', rows)}
    }
  });
  const multiscaleBase = (rows: number) => {
    const props = base(rows);
    return {
      ...props,
      output: {...props.output, status: {...props.output.status, iterationCount: view('uint32', 1)}}
    };
  };
  try {
    expect(
      new GPUGeneralizedGeographicallyWeightedRegression({...base(65536), family: 'poisson'}).props
        .indexGridSize
    ).toBeUndefined();
    expect(
      new GPUGeneralizedGeographicallyWeightedRegression({...base(65537), family: 'poisson'}).props
        .indexGridSize
    ).toBeDefined();
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...base(65537),
          family: 'poisson',
          indexGridSize: false
        })
    ).toThrow(/65536/);
    const exact = new GPUGeneralizedGeographicallyWeightedRegression({
      ...base(65537),
      family: 'poisson',
      kernel: 'gaussian',
      bandwidthMode: 'fixed',
      indexGridSize: [2, 2]
    });
    expect(exact.props.indexGridSize).toEqual([2, 2]);
    expect(exact.props.gaussianCutoff).toBeUndefined();
    expect(
      new GPUMultiscaleGeographicallyWeightedRegression(multiscaleBase(1023)).props.indexGridSize
    ).toBeUndefined();
    expect(
      new GPUMultiscaleGeographicallyWeightedRegression(multiscaleBase(1024)).props.indexGridSize
    ).toBeDefined();
    expect(
      new GPUMultiscaleGeographicallyWeightedRegression({
        ...multiscaleBase(1024),
        indexGridSize: false
      }).props.indexGridSize
    ).toBeUndefined();
  } finally {
    device.destroy();
  }
});
