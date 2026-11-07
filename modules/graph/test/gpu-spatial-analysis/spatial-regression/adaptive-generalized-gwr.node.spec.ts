// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';

it('adaptive generalized GWR parses all families/kernels with complete-case optional inputs and eight bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `adaptive-${serial++}`, format, length);
  const props = {
    positions: view('float32x2', 24),
    predictors: view('float32', 24),
    predictorCount: 1,
    response: view('float32', 24),
    parameters: view('float32', 2),
    mask: view('uint32', 24),
    offset: view('float32', 24),
    output: {
      coefficients: view('float32', 48),
      fitted: view('float32', 24),
      status: {codes: view('uint32', 24), iterationCount: view('uint32', 24)}
    }
  };
  for (const family of ['poisson', 'logistic'] as const)
    for (const kernel of ['gaussian', 'bisquare'] as const) {
      const contributor = new GPUGeneralizedGeographicallyWeightedRegression({
        ...props,
        id: `${family}-${kernel}`,
        family,
        kernel,
        bandwidthMode: 'adaptive',
        inference: {standardErrors: view('float32', 48), influence: view('float32', 24)},
        bandwidthSelection: {
          bandwidths: [16, 24],
          output: {
            scores: view('float32', 2),
            bandwidth: view('float32', 1),
            status: view('uint32', 1)
          }
        }
      });
      for (const source of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
        expect(new WgslReflect(source.source).storage.length).toBeLessThanOrEqual(8);
      }
    }
  for (const maximumNeighborCount of [0, 129, 1.5])
    expect(
      () =>
        new GPUGeneralizedGeographicallyWeightedRegression({
          ...props,
          family: 'poisson',
          maximumNeighborCount
        })
    ).toThrow();
  device.destroy();
});
