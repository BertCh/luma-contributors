// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {ScalarData, VectorData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUGeoConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals';
import {getGPUConformalPredictionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-nodes';
import {getConformalPredictionRanks} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

/** Executes the actual generated interval kernel on the CPU, after entry scoring/filtering. */
function executeLocalizedQuantile(
  scores: number[],
  weights: number[],
  selfWeight: number,
  alpha: number
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
        weights: view('float32', scores.length)
      },
      parameters: view('float32', 4),
      output: {intervals: view('float32', 2), status: view('uint32', 1)}
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-intervals'))
    );
    expect(kernels).toHaveLength(1);
    const kernel = kernels[0];
    const intervals = new Float32Array(2);
    const status = new Uint32Array(1);
    const values: Record<string, Float32Array | Uint32Array> = {
      parameters: getGPUConformalPredictionParameterValues({miscoverage: alpha, selfWeight}),
      entryScores: Float32Array.from(scores),
      entryWeights: Float32Array.from(weights),
      neighborOffsets: Uint32Array.of(0, scores.length),
      queryPredictions: Float32Array.of(0),
      intervals,
      status
    };
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
    return {intervals: Array.from(intervals), status: Array.from(status)};
  } finally {
    errors.mockRestore();
    device.destroy();
  }
}

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
