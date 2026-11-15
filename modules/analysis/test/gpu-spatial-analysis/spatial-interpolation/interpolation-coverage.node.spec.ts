// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('interpolation expansion emits bounded parseable GPU kernels under eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const positions = view('float32x2', 16);
  const queries = view('float32x2', 16);
  const observations = view('float32x4', 16);
  const covariance = {model: 'exponential' as const, range: 1, sill: 1};
  const output = () => ({values: view('float32x4', 16), status: view('uint32', 16)});
  const contributors = [
    ...(['simple', 'ordinary', 'universal', 'external-drift', 'co-kriging'] as const).map(
      mode =>
        new GPULocalKriging({
          id: `local-kriging-${mode}`,
          positions,
          observations,
          supports: queries,
          mode,
          covariance,
          sampleDrift: view('float32x4', 16),
          queryDrift: view('float32x4', 16),
          driftCount: 4,
          coregionalization: [1, 0, 0.5, 0.5],
          output: {...output(), weights: view('float32', 256)}
        })
    ),
    new GPULocalRBF({positions, queries, values: view('float32', 16), output: output()}),
    new GPUNaturalNeighbor({
      positions,
      queries,
      values: view('float32', 16),
      output: {...output(), weights: view('float32', 256)}
    }),
    new GPUConditionalGaussianSimulation({
      positions,
      queries,
      observations,
      covariance,
      seed: 1,
      replicateCount: 10,
      output: {
        values: view('float32', 160),
        moments: view('float32x2', 16),
        status: view('uint32', 10)
      }
    }),
    new GPUVariogramModelFit({
      distances: view('float32', 20),
      semivariances: view('float32', 20),
      pairCounts: view('uint32', 20),
      output: {
        parameters: view('float32x4', 3),
        diagnostics: view('float32x4', 3),
        status: view('uint32', 3)
      }
    })
  ];
  for (const contributor of contributors) {
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels) {
      const reflection = new WgslReflect(kernel.source);
      expect(reflection.storage.length).toBeLessThanOrEqual(8);
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    }
  }
  device.destroy();
});

it('bounded interpolation validates model capacities, external drift and output aliasing', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  const positions = view('float32x2', 4);
  const values = view('float32', 4);
  const output = {values: view('float32x4', 4), status: view('uint32', 4)};
  expect(
    () => new GPULocalRBF({positions, values, queries: positions, neighborCount: 33, output})
  ).toThrow();
  expect(
    () =>
      new GPULocalRBF({
        positions,
        values,
        queries: positions,
        kernel: 'thin-plate-spline',
        degree: 0,
        output
      })
  ).toThrow();
  expect(
    () =>
      new GPULocalKriging({
        positions,
        observations: view('float32x4', 4),
        supports: positions,
        mode: 'external-drift',
        covariance: {model: 'spherical', range: 1, sill: 1},
        output
      })
  ).toThrow();
  expect(
    () =>
      new GPUNaturalNeighbor({positions, values, queries: positions, maximumVertexCount: 2, output})
  ).toThrow();
  expect(
    () =>
      new GPUNaturalNeighbor({
        positions,
        values,
        queries: positions,
        output: {...output, weights: values}
      })
  ).toThrow();
  device.destroy();
});
