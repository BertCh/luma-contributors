// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {ScalarData, VectorData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUGeoConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals';
import {getGPUConformalPredictionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-nodes';
import {getConformalPredictionRanks} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';
import {executeSpatialAuditKernel} from './spatial-design-audit.test-utils';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

/** Executes the actual generated interval kernel on the CPU, after entry scoring/filtering. */
function executeLocalizedQuantile(
  scores: number[],
  weights: number[],
  selfWeight: number,
  alpha: number,
  loss?: 'overflow' | 'candidateOverflow' | 'complete'
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let viewCount = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `equal-weights-${viewCount++}`, format, length);
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    // wgsl_reflect cannot interpret the kernel builder's phony binding-reference statements.
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const contributor = new GPUGeoConformalPredictionIntervals({
      calibration: {
        observations: view('float32', scores.length),
        predictions: view('float32', scores.length)
      },
      query: {predictions: view('float32', 1)},
      neighbors: {
        offsets: view('uint32', 2),
        indices: view('uint32', scores.length),
        weights: view('float32', scores.length),
        ...(loss ? {status: {[loss]: view('uint32', 1)}} : {})
      },
      parameters: view('float32', 4),
      output: {intervals: view('float32', 2), status: view('uint32', 1)}
    });
    const nodes = contributor.getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('-intervals'))
    );
    expect(kernels).toHaveLength(1);
    const kernel = kernels[0];
    const intervals = new Float32Array(2);
    const status = new Uint32Array(1);
    const values: Record<string, Float32Array | Uint32Array> = {
      parameters: new Uint32Array(
        getGPUConformalPredictionParameterValues({miscoverage: alpha, selfWeight}).buffer
      ),
      entryScores: Float32Array.from(scores),
      entryWeights: new Uint32Array(Float32Array.from(weights).buffer),
      neighborOffsets: Uint32Array.of(0, scores.length),
      queryPredictions: Float32Array.of(0),
      intervals,
      status
    };
    if (loss) values[loss] = Uint32Array.of(loss === 'complete' ? 0 : 1);
    const bindings = Object.fromEntries(
      new WgslReflect(kernel.source).storage.map(binding => {
        expect(values[binding.name], binding.name).toBeDefined();
        return [binding.binding, values[binding.name]];
      })
    );
    const execution = new WgslExec(new WgslParser().parse(kernel.source));
    const evaluateCall = execution._evalCall.bind(execution);
    // The interpreter's uint64 rank evaluation rounds the packed 0.02 boundary to rank 49.
    // Execute that intrinsic through its tested host mirror, with the actual runtime arguments.
    // This tests localized selection/wiring; the headless case executes the complete shader.
    execution._evalCall = (expression, context) => {
      if (expression.name !== 'getConformalPredictionRanks')
        return evaluateCall(expression, context);
      const count = execution.evalExpression(expression.args![0], context);
      const miscoverage = execution.evalExpression(expression.args![1], context);
      if (!(count instanceof ScalarData && miscoverage instanceof ScalarData))
        throw new Error('rank arguments must be scalar');
      const ranks = getConformalPredictionRanks(count.value, miscoverage.value);
      const type = execution.getTypeInfo('vec2u');
      if (!type) throw new Error('missing rank vector type');
      return new VectorData([ranks.lowerRank, ranks.upperRank], type);
    };
    execution.dispatchWorkgroups(
      kernel.entryPoint,
      [1, 1, 1],
      {0: bindings},
      {constants: kernel.constants}
    );
    if (loss) {
      const clean = {intervals: Array.from(intervals), status: Array.from(status)};
      const [suppression] = getGPUCommandNodeKernelSources(
        nodes.filter(node => node.id.endsWith('-input-completeness'))
      );
      executeSpatialAuditKernel(suppression, values);
      const incomplete = {intervals: Array.from(intervals), status: Array.from(status)};
      values[loss][0] = loss === 'complete' ? 1 : 0;
      execution.dispatchWorkgroups(
        kernel.entryPoint,
        [1, 1, 1],
        {0: bindings},
        {constants: kernel.constants}
      );
      executeSpatialAuditKernel(suppression, values);
      expect(Array.from(intervals)).toEqual(clean.intervals);
      expect(Array.from(status)).toEqual(clean.status);
      return incomplete;
    }
    return {intervals: Array.from(intervals), status: Array.from(status)};
  } finally {
    errors.mockRestore();
    device.destroy();
  }
}

it.each([
  'overflow',
  'candidateOverflow',
  'complete'
] as const)('CPL-04 suppresses localized intervals on neighbor %s', loss => {
  // Equal weights including self: full scores [1,9] have rank 2 and width 9.
  // Retaining only score 1 cannot certify the narrower interval [-1,1].
  const result = executeLocalizedQuantile([1], [1], 1, 0.5, loss);
  expect(result.status).toEqual([32]);
  expect(result.intervals).toEqual([NaN, NaN]);
});

it('B2 suppresses an unbounded interval with consistent bits and recovers', () => {
  // Self mass 1 exceeds alpha=.1: no finite score reaches the weighted quantile.
  const result = executeLocalizedQuantile([2, 5], [0.5, 0.25], 1, 0.1, 'candidateOverflow');
  expect(result.intervals).toEqual([NaN, NaN]);
  expect(result.status).toEqual([32]);
});

it.each([
  'requiredCount',
  'converged',
  'iterationLimitReached',
  'unresolvedCount'
] as const)('refuses unsupported neighbor status %s at construction', field => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `required-${serial++}`, format, length);
  try {
    const status: GPUSpatialWeights['status'] = {
      overflow: field === 'requiredCount' ? undefined : view('uint32', 1),
      [field]: view('uint32', 1)
    };
    expect(
      () =>
        new GPUGeoConformalPredictionIntervals({
          id: 'geo-status-test',
          calibration: {observations: view('float32', 2), predictions: view('float32', 2)},
          query: {predictions: view('float32', 1)},
          parameters: view('float32', 4),
          neighbors: {
            offsets: view('uint32', 2),
            indices: view('uint32', 2),
            weights: view('float32', 2),
            status
          },
          output: {intervals: view('float32', 2), status: view('uint32', 1)}
        })
    ).toThrow(new RegExp(`^geo-status-test neighbors\\.status\\.${field}.*omit`));
  } finally {
    device.destroy();
  }
});

it.each([
  1, 0.1, 1e37
])('spatial-4:F4 keeps equal-weight localized intervals unbounded at packed alpha 0.02 (weight %s)', selfWeight => {
  const scores = Array.from({length: 49}, (_, row) => row + 1);
  expect(
    executeLocalizedQuantile(
      scores,
      scores.map(() => selfWeight),
      selfWeight,
      0.02
    )
  ).toEqual({
    intervals: [-Infinity, Infinity],
    status: [3]
  });
});

it('equal-weight localized intervals retain the exact finite order statistic', () => {
  const scores = Array.from({length: 39}, (_, row) => row + 1);
  expect(
    executeLocalizedQuantile(
      scores,
      scores.map(() => 0.1),
      0.1,
      0.25
    )
  ).toEqual({
    intervals: [-30, 30],
    status: [0]
  });
});

it('equal-weight finite ranks count only usable entries', () => {
  expect(
    executeLocalizedQuantile([1, 2, 3, 4, 5, 6, 7, 999], [1, 1, 1, 1, 1, 1, 1, 0], 1, 0.25)
  ).toEqual({
    intervals: [-6, 6],
    status: [0]
  });
});

it('equal-weight rank selection counts tied scores together', () => {
  expect(executeLocalizedQuantile([1, 2, 2, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1, 1], 1, 0.5)).toEqual({
    intervals: [-2, 2],
    status: [0]
  });
});

it('equal neighbor weights different from selfWeight retain the weighted quantile', () => {
  const scores = Array.from({length: 49}, (_, row) => row + 1);
  expect(
    executeLocalizedQuantile(
      scores,
      scores.map(() => 2),
      1,
      0.02
    )
  ).toEqual({
    intervals: [-49, 49],
    status: [0]
  });
});

it('audit-validation weighted ties and a masked tail match the rational CDF oracle', () => {
  // conformal-validation-audit.oracle.py: (1 - alpha) * (sum weights + 1) = 8.25;
  // cumulative mass is 8 at score 5 and 10 at score 7, hence quantile 7.
  expect(
    executeLocalizedQuantile([1, 3, 1, 5, 5, 1, 7, 7, 999], [1, 2, 1, 1, 2, 1, 1, 1, 0], 1, 0.25)
  ).toEqual({intervals: [-7, 7], status: [0]});
});

it('R9 localized nonuniform weights preserve the quantile under a large finite common scale', () => {
  for (const factor of [1, 2 ** 126]) {
    // Total mass including self=10, target=7.5; score 7 has mass 7, score 8 has 9.
    expect(
      executeLocalizedQuantile(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 2].map(weight => factor * weight),
        factor,
        0.25
      )
    ).toEqual({intervals: [-8, 8], status: [0]});
    // Signed zero is an accepted nonnegative self mass; total=9, target=6.75.
    expect(
      executeLocalizedQuantile(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 2].map(weight => factor * weight),
        -0,
        0.25
      )
    ).toEqual({intervals: [-7, 7], status: [0]});
  }
});

it('review 2 R5 localized mass arithmetic preserves uniformly subnormal positive weights', () => {
  for (const factor of [1, 2 ** -149, 2 ** -100, 2 ** 126]) {
    // Total including self=11, target=8.25; cumulative through score 7 is only 8.
    expect(
      executeLocalizedQuantile(
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 2, 2].map(weight => factor * weight),
        factor,
        0.25
      )
    ).toEqual({intervals: [-8, 8], status: [0]});
  }
});

it('review 2 R5 entry filtering preserves positive subnormal bits with flushed float comparisons', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `tiny-weights-${nextView++}`, format, length);
  try {
    const contributor = new GPUGeoConformalPredictionIntervals({
      calibration: {observations: view('float32', 8), predictions: view('float32', 8)},
      query: {predictions: view('float32', 1)},
      neighbors: {
        offsets: view('uint32', 2),
        indices: view('uint32', 8),
        weights: view('float32', 8)
      },
      parameters: view('float32', 4),
      output: {intervals: view('float32', 2), status: view('uint32', 1)}
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-entries'))
    );
    const entryWeights = new Uint32Array(8),
      entryScores = new Float32Array(8);
    // Interpret each entry as lane zero: wgsl_reflect does not reliably execute all
    // lanes after the generated bounds guard. The device case runs all eight slots.
    for (let row = 0; row < 8; row++) {
      const weight = new Uint32Array(1),
        score = new Float32Array(1);
      executeSpatialAuditKernel(
        kernel,
        {
          scoreKeys: new Uint32Array(Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]).buffer),
          foldKeys: new Uint32Array(8),
          neighborIndices: Uint32Array.of(row),
          neighborWeights: new Uint32Array(Float32Array.of((row < 6 ? 1 : 2) * 2 ** -149).buffer),
          entryWeights: weight,
          entryScores: score
        },
        {invocationCount: 1, flushSubnormalComparisons: true}
      );
      entryWeights[row] = weight[0];
      entryScores[row] = score[0];
    }
    expect(Array.from(new Uint32Array(entryWeights.buffer))).toEqual([1, 1, 1, 1, 1, 1, 2, 2]);
    expect(Array.from(entryScores)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  } finally {
    device.destroy();
  }
});
