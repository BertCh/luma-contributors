// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
it('advanced random panel and full multiscale inference audit maximal options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `advanced-${serial++}`, format, length);
  try {
    const panel = new GPUSpatialPanelLagMaximumLikelihood({
      weights: {
        offsets: view('uint32', 33),
        neighbors: view('uint32', 992),
        weights: view('float32', 992),
        status: {overflow: view('uint32', 1)}
      },
      periodCount: 32,
      predictorCount: 7,
      predictors: view('float32', 1024 * 7),
      response: view('float32', 1024),
      maximumIterations: 128,
      maximumSearchIterations: 64,
      hausman: {tests: view('float32', 3), status: view('uint32', 1)},
      output: {
        coefficients: view('float32', 10),
        covariance: view('float32', 81),
        summary: view('float32', 6),
        status: view('uint32', 1),
        fitted: view('float32', 1024),
        residuals: view('float32', 1024)
      }
    });
    const multiscale = new GPUMultiscaleGeographicallyWeightedRegression({
      positions: view('float32x2', 64),
      predictors: view('float32', 64 * 7),
      predictorCount: 7,
      response: view('float32', 64),
      parameters: view('float32', 9),
      mask: view('uint32', 64),
      maximumIterations: 128,
      bandwidthCandidates: view('float32', 8 * 32),
      bandwidthCandidateCount: 32,
      inference: {
        varianceConvention: 'smoother-df',
        output: {
          standardErrors: view('float32', 64 * 8),
          effectiveParameters: view('float32', 8),
          summary: view('float32', 4),
          status: view('uint32', 1)
        }
      },
      output: {
        coefficients: view('float32', 64 * 8),
        fitted: view('float32', 64),
        status: {codes: view('uint32', 64), iterationCount: view('uint32', 1)}
      }
    });
    for (const contributor of [panel, multiscale])
      for (const source of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)))
        expect(auditWGSLModule(source.source, source.constants), source.source).toEqual([]);
    expect(
      () =>
        new GPUMultiscaleGeographicallyWeightedRegression({
          ...multiscale.props,
          maximumIterations: 129
        })
    ).toThrow();
  } finally {
    device.destroy();
  }
}, 120000);
