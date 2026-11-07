// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {GPULocalRBF} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {GPUNaturalNeighbor} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-natural-neighbor';
import {GPUConditionalGaussianSimulation} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation';
import {GPUVariogramModelFit} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram-model-fit';
import {
  auditWGSLModule,
  getWGSLModuleUsage
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createInterpolationBenchmarkCases,
  solveBenchmarkSystem
} from './interpolation-performance-fixtures';

it('interpolation benchmarks cover small and bounded larger shapes with an independent solve oracle', () => {
  const cases = createInterpolationBenchmarkCases();
  expect(cases).toHaveLength(10);
  expect(new Set(cases.map(benchmark => benchmark.name)).size).toBe(10);
  const solution = solveBenchmarkSystem(
    [
      [4, 1],
      [1, 3]
    ],
    [1, 2]
  );
  expect(solution[0]).toBeCloseTo(1 / 11, 12);
  expect(solution[1]).toBeCloseTo(7 / 11, 12);
  expect(() =>
    solveBenchmarkSystem(
      [
        [1, 1],
        [1, 1]
      ],
      [1, 2]
    )
  ).toThrow();
});

it('maximal interpolation options fit core binding and shader scratch budgets', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'float32x4' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `maximal-${serial++}`, format, length);
  const covariance = {model: 'exponential' as const, sill: 1, range: 1};
  const localOutput = () => ({values: view('float32x4', 16), status: view('uint32', 16)});
  const contributors = [
    new GPULocalKriging({
      positions: view('float32x2', 16),
      observations: view('float32x4', 16),
      supports: view('float32x2', 256),
      supportCount: 16,
      neighborCount: 16,
      mode: 'external-drift',
      driftCount: 4,
      sampleDrift: view('float32x4', 16),
      queryDrift: view('float32x4', 16),
      covariance,
      output: {...localOutput(), weights: view('float32', 256)}
    }),
    new GPULocalRBF({
      positions: view('float32x2', 16),
      values: view('float32', 16),
      queries: view('float32x2', 16),
      neighborCount: 16,
      degree: 1,
      output: localOutput()
    }),
    new GPUNaturalNeighbor({
      positions: view('float32x2', 32),
      values: view('float32', 32),
      queries: view('float32x2', 16),
      maximumVertexCount: 64,
      output: {...localOutput(), weights: view('float32', 512)}
    }),
    new GPUConditionalGaussianSimulation({
      positions: view('float32x2', 16),
      observations: view('float32x4', 16),
      queries: view('float32x2', 16),
      covariance,
      seed: 1,
      replicateCount: 128,
      output: {
        values: view('float32', 2048),
        moments: view('float32x2', 16),
        status: view('uint32', 128)
      }
    }),
    new GPUConditionalGaussianSimulation({
      id: 'prepared-maximal-simulation',
      positions: view('float32x2', 128),
      observations: view('float32x4', 128),
      queries: view('float32x2', 128),
      covariance,
      seed: 1,
      replicateCount: 128,
      reusePreparation: true,
      output: {
        values: view('float32', 16384),
        moments: view('float32x2', 128),
        status: view('uint32', 128)
      }
    }),
    ...(['none', 'pairs', 'cressie'] as const).map(
      weighting =>
        new GPUVariogramModelFit({
          id: `maximal-variogram-${weighting}`,
          distances: view('float32', 1025),
          semivariances: view('float32', 1025),
          pairCounts: view('uint32', 1025),
          weighting,
          rangeSteps: 1024,
          maximumIterations: 64,
          output: {
            parameters: view('float32x4', 3),
            diagnostics: view('float32x4', 3),
            status: view('uint32', 3)
          }
        })
    )
  ];
  const budgets = [];
  for (const contributor of contributors) {
    for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const usage = getWGSLModuleUsage(kernel.source, kernel.constants);
      expect(usage.unresolvedLengths).toEqual([]);
      expect(usage.storageBuffers).toBeLessThanOrEqual(8);
      expect(usage.privateBytes).toBeLessThanOrEqual(8192);
      expect(usage.functionBytes).toBeLessThanOrEqual(8192);
      expect(usage.workgroupBytes).toBeLessThanOrEqual(16384);
      budgets.push({operation: kernel.operation, ...usage});
    }
  }
  for (const contributor of contributors) {
    if (contributor instanceof GPUConditionalGaussianSimulation) contributor.destroy();
  }
  console.log(`GEOSPATIAL_INTERPOLATION_SHADER_BUDGETS ${JSON.stringify(budgets)}`);
  device.destroy();
});
