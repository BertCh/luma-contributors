// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSpatialScanStatistic,
  type GPUSpatialScanStatisticProps
} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {GPUExpectationBasedSpaceTimeScan} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-expectation-based-space-time-scan';
import {getPermutationRejectionFloor} from '../../../src/gpu-spatial-analysis/multiple-testing/permutation-p-value-floor';
import {getGPUAnalysisSignificanceAttainability} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/analysis-interpretation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('reports an upper-tail maximum-statistic floor of .05 at 19 draws, including both directions (NUM-05)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `host-${serial++}`, format, length);
  try {
    const scan = new GPUSpatialScanStatistic({
      positions: view('float32x2', 2),
      cases: view('uint32', 2),
      baseline: view('float32', 2),
      timeBuckets: 1,
      maximumPermutations: 19,
      maximumClusters: 1,
      parameters: view('uint32', 8),
      output: {
        clusterIndices: view('uint32', 4),
        clusterStatistics: view('float32', 8),
        statistics: view('float32', 20),
        summary: view('uint32', 4)
      }
    });
    const expectation = new GPUExpectationBasedSpaceTimeScan({
      positions: view('float32x2', 2),
      cases: view('uint32', 2),
      baseline: view('float32', 2),
      timeBuckets: 1,
      maximumNeighbors: 2,
      maximumWindowLength: 1,
      maximumReplicates: 19,
      parameters: view('uint32', 4),
      output: {
        clusterZones: view('uint32', 2),
        clusterStatistics: view('float32', 4),
        statistics: view('float32', 20),
        summary: view('uint32', 5)
      }
    });
    for (const metadata of [
      scan.getInferenceMetadata({seed: 0, permutationCount: 19, rateDirection: 'both'}),
      expectation.getInferenceMetadata({seed: 0, replicateCount: 19})
    ]) {
      expect(metadata.pValueConstruction).toBe('upper-tail-count');
      // Zero exceedances in the single replicate-maxima comparison: (0 + 1)/(19 + 1) = .05.
      const plan = getGPUAnalysisSignificanceAttainability({
        correction: 'none',
        testCount: 1,
        permutationCount: metadata.permutationCount!,
        pValueConstruction: metadata.pValueConstruction,
        significanceLevel: 0.05
      });
      expect(plan.smallestPValue).toBe(0.05);
      expect(plan.attainable).toBe(true);
      expect(
        getPermutationRejectionFloor({
          alternative: metadata.alternative,
          permutationCount: metadata.permutationCount!,
          pValueConstruction: metadata.pValueConstruction,
          rowCount: 1,
          alpha: 0.05
        })
      ).toMatchObject({pValueFloor: 0.05, canReject: true});
    }
    expect(
      getPermutationRejectionFloor({
        alternative: 'two-sided',
        permutationCount: 19,
        pValueConstruction: 'doubled-minimum-tail',
        rowCount: 1,
        alpha: 0.05
      })
    ).toMatchObject({pValueFloor: 0.1, canReject: false});
  } finally {
    device.destroy();
  }
});

it('plans the selected expectation-scan route and actual binding sizes without allocating (SCL-16)', () => {
  const shape = {
    zones: 100,
    timeBuckets: 4,
    maximumNeighbors: 2,
    maximumWindowLength: 4,
    maximumReplicates: 19
  };
  const plan = GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, {
    maxBufferSize: 134217728,
    maxStorageBufferBindingSize: 134217728
  });
  expect(plan.route).toBe('indexed');
  expect(plan.feasible).toBe(true);
  expect(plan.constraints.map(constraint => constraint.received)).toEqual([
    1600, 800, 32000, 1200, 80, 24, 400
  ]);
  expect(plan.batchReplicates).toBe(19);
  expect(plan.estimatedIndexedCandidateSeparationCount).toBe(10000);
  expect(GPUExpectationBasedSpaceTimeScan.limits.length).toBeGreaterThan(0);
  expect(
    GPUExpectationBasedSpaceTimeScan.getExecutionPlan(
      {...shape, zones: 2},
      {maxBufferSize: 134217728, maxStorageBufferBindingSize: 134217728}
    ).route
  ).toBe('full-scan');
  expect(
    GPUExpectationBasedSpaceTimeScan.getExecutionPlan(shape, {
      maxBufferSize: 64,
      maxStorageBufferBindingSize: 64
    }).feasible
  ).toBe(false);
});

it('keeps custom IDs and structured errors for scan shape and selector checks (W2-15)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `validation-${serial++}`, format, length);
  const props: GPUSpatialScanStatisticProps = {
    id: 'custom-scan',
    positions: view('float32x2', 2),
    cases: view('uint32', 2),
    baseline: view('float32', 2),
    maximumPermutations: 1,
    maximumClusters: 1,
    parameters: view('uint32', 8),
    output: {
      clusterIndices: view('uint32', 4),
      clusterStatistics: view('float32', 8),
      statistics: view('float32', 2),
      summary: view('uint32', 4)
    }
  };
  try {
    for (const changed of [
      {timeBuckets: 2},
      {maximumWindowZones: 0},
      {model: 'invalid'},
      {temporalMode: 'invalid'},
      {model: null},
      {temporalMode: null}
    ]) {
      try {
        new GPUSpatialScanStatistic({...props, ...changed} as GPUSpatialScanStatisticProps);
        throw new Error('expected validation');
      } catch (error) {
        expect(error).toBeInstanceOf(AnalysisValidationError);
        expect((error as Error).message).toMatch(/^custom-scan /);
      }
    }
    const scan = new GPUSpatialScanStatistic(props);
    const captured = scan.props.output.summary;
    props.output.summary = view('uint32', 1);
    expect(scan.props.output.summary).toBe(captured);
  } finally {
    device.destroy();
  }
});
