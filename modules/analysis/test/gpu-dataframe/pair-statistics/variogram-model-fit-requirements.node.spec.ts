// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUVariogramModelFit,
  type GPUVariogramModelFitProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeVariogramModelFitOnCPU} from '../../../src/dataframe-analysis/pair-statistics/cpu-variogram-model-fit-operation';

for (const usableBins of [1, 2]) {
  it(`B2 rejects unsupported CPU fit options before eligibility (${usableBins} bins)`, () => {
    const input = {
      distances: [1, 2].slice(0, usableBins),
      semivariances: [1, 2].slice(0, usableBins),
      pairCounts: [1, 1].slice(0, usableBins)
    };
    expect(() => computeVariogramModelFitOnCPU(input, {weighting: 'bogus' as never})).toThrow(
      'variogram-model-fit invalid model or bin topology: weighting bogus must be one of pairs, none, cressie'
    );
    expect(() => computeVariogramModelFitOnCPU(input, {models: ['bogus' as never]})).toThrow(
      'variogram-model-fit invalid model or bin topology: models member bogus must be one of spherical, exponential, gaussian, matern-3/2, matern-5/2'
    );
  });
}

it('ERG-101 names unsupported models and weighting independently and preserves bin diagnostics', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const props: GPUVariogramModelFitProps = {
    id: 'review-fit',
    distances: view('distances', 'float32', 3),
    semivariances: view('semivariances', 'float32', 3),
    pairCounts: view('counts', 'uint32', 3),
    models: ['spherical'],
    weighting: 'pairs',
    output: {
      parameters: view('parameters', 'float32x4', 1),
      diagnostics: view('diagnostics', 'float32x4', 1),
      status: view('status', 'uint32', 1)
    }
  };
  try {
    expect(() => new GPUVariogramModelFit({...props, models: ['unsupported' as never]})).toThrow(
      'review-fit invalid model or bin topology: models member unsupported must be one of spherical, exponential, gaussian, matern-3/2, matern-5/2'
    );
    expect(() => new GPUVariogramModelFit({...props, weighting: 'unsupported' as never})).toThrow(
      'review-fit invalid model or bin topology: weighting unsupported must be one of pairs, none, cressie'
    );
    expect(
      () =>
        new GPUVariogramModelFit({
          ...props,
          semivariances: view('short-semivariances', 'float32', 2)
        })
    ).toThrow(
      'review-fit invalid model or bin topology: semivariances bin count 2 must equal distances bin count 3'
    );
    expect(
      () => new GPUVariogramModelFit({...props, pairCounts: view('long-counts', 'uint32', 4)})
    ).toThrow(
      'review-fit invalid model or bin topology: pairCounts bin count 4 must equal distances bin count 3'
    );
  } finally {
    device.destroy();
  }
});
