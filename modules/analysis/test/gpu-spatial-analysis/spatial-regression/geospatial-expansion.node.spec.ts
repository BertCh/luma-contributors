// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {it, expect} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  GPULocalSpatialHeteroscedasticity,
  GPUMultivariateLocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/gpu-local-variability';
import {
  GPUCrossVariogram,
  GPUSpaceTimeVariogram
} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';

it('geospatial statistical expansion generates parseable kernels within eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `expansion-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 9),
    neighbors: view('uint32', 16),
    weights: view('float32', 16),
    status: {overflow: view('uint32', 1)}
  };
  const local = {
    weights,
    values: view('float32', 8),
    mask: view('uint32', 8),
    output: {statistic: view('float32', 8), status: view('uint32', 8)}
  };
  const variogram = {
    positions: view('float32x2', 8),
    values: view('float32', 8),
    secondValues: view('float32', 8),
    parameters: view('float32', 2),
    lagCount: 2,
    mask: view('uint32', 8),
    output: {
      semivariances: view('float32', 2),
      pairCounts: view('uint32', 2),
      status: view('uint32', 2)
    }
  };
  const regression = {
    positions: view('float32x2', 8),
    predictors: view('float32', 16),
    response: view('float32', 8),
    predictorCount: 2,
    parameters: view('float32', 4),
    mask: view('uint32', 8),
    maximumIterations: 2,
    output: {
      coefficients: view('float32', 24),
      fitted: view('float32', 8),
      status: {codes: view('uint32', 8), iterationCount: view('uint32', 8)}
    }
  };
  const contributors: GPUCommandNodeProducer[] = [
    new GPULocalSpatialHeteroscedasticity(local),
    new GPUMultivariateLocalGeary({...local, values: view('float32', 16), variableCount: 2}),
    new GPUCrossVariogram(variogram),
    new GPUSpaceTimeVariogram({...variogram, times: view('float32', 8), temporalLagCount: 1}),
    new GPUMarkedPointStatistics({
      positions: view('float32x2', 8),
      parameters: view('float32', 6),
      radiusCount: 2,
      types: view('uint32', 8),
      typePair: [0, 1],
      marks: view('float32', 8),
      intensities: view('float32', 8),
      mask: view('uint32', 8),
      output: {
        statistics: view('float32', 8),
        pairCounts: view('uint32', 2),
        status: view('uint32', 2)
      }
    }),
    new GPUGeneralizedGeographicallyWeightedRegression({
      ...regression,
      family: 'poisson',
      offset: view('float32', 8)
    }),
    new GPUGeneralizedGeographicallyWeightedRegression({
      ...regression,
      id: 'logistic',
      family: 'logistic'
    }),
    new GPUMultiscaleGeographicallyWeightedRegression({
      ...regression,
      bandwidthCandidates: view('float32', 9),
      bandwidthCandidateCount: 3,
      noiseVariance: view('float32', 1),
      output: {
        ...regression.output,
        status: {...regression.output.status, iterationCount: view('uint32', 1)},
        selectedBandwidths: view('float32', 3),
        conditionalStandardErrors: view('float32', 24)
      }
    }),
    new GPUSpatialRegimeRegression({
      predictors: regression.predictors,
      response: regression.response,
      predictorCount: 2,
      regimes: view('uint32', 8),
      regimeCount: 2,
      mask: regression.mask,
      output: {
        coefficients: view('float32', 6),
        fitted: view('float32', 8),
        counts: view('uint32', 2),
        status: view('uint32', 2)
      }
    }),
    new GPUSpatialPanelRegression({
      weights,
      periodCount: 2,
      predictorCount: 1,
      predictors: view('float32', 16),
      response: view('float32', 16),
      output: {
        table: view('float32', 12),
        summary: view('float32', 32),
        status: {code: view('uint32', 1)}
      }
    })
  ];
  for (const contributor of contributors) {
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(sources.length).toBeGreaterThan(0);
    for (const source of sources) {
      const reflection = new WgslReflect(source.source);
      expect(reflection.storage.length, source.id).toBeLessThanOrEqual(8);
    }
  }
  device.destroy();
});

it('local variability recipe rejects output aliases and invalid output lengths before adding nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `boundary-${serial++}`, format, length);
  const values = view('float32', 4);
  const weights = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 4),
    weights: view('float32', 4)
  };
  const props = {weights, values, parameters: view('float32', 16)};
  expect(() => addLocalVariabilityRecipe(graph, {...props, outputs: {losh: values}})).toThrow();
  const shared = view('float32', 4);
  expect(() =>
    addLocalVariabilityRecipe(graph, {...props, outputs: {losh: shared, localI: shared}})
  ).toThrow();
  expect(() =>
    addLocalVariabilityRecipe(graph, {...props, outputs: {losh: view('float32', 3)}})
  ).toThrow();
  device.destroy();
});
