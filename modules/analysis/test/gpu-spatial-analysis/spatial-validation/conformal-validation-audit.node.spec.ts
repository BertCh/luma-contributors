// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {ScalarData, VectorData, WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {GPUConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-conformal-prediction-intervals';
import {getConformalPredictionRanks} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-ranks';
import {getGPUConformalPredictionParameterValues} from '../../../src/gpu-spatial-analysis/spatial-validation/conformal-prediction-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUGeoConformalPredictionIntervals} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals';
import {watchConstructionFields} from './construction-capture-audit.test-utils';
import {readFileSync} from 'node:fs';
import {entries as spaceTimeEntries} from '../../../../../website/src/catalog/entries/space-time-models';

it('review 2 R4 public CV+ descriptions state the fold-design and fitting assumptions', () => {
  // The independent 32-query enumeration below establishes that row exchangeability
  // alone cannot justify a coverage promise for these supported group folds.
  const catalog = spaceTimeEntries.find(entry => entry.id === 'conformal-prediction-intervals')!;
  expect(catalog.explanation).toMatch(/balanced, data-independent folds/);
  expect(catalog.explanation).toMatch(/permutation-invariant fitting/);
  expect(catalog.explanation).toMatch(/arbitrary spatial or group folds/i);
  const source = readFileSync(
    new URL(
      '../../../src/gpu-spatial-analysis/spatial-validation/gpu-geo-conformal-prediction-intervals.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(source).toMatch(/balanced, data-independent folds/);
  expect(source).toMatch(/permutation-invariant fitting/);
  expect(source).toMatch(/arbitrary spatial or group folds/i);
});

it('R6 grouped CV+ can exclude every outcome despite exchangeable observation order', () => {
  let covered = 0;
  const ranks = getConformalPredictionRanks(31, 0.25);
  expect(ranks).toEqual({lowerRank: 8, upperRank: 24});
  for (let query = 0; query < 32; query++) {
    const queryGroup = Math.floor(query / 4);
    // Eight groups of four, outcomes zero, leave-group-out fits. A fit predicts
    // one iff the queried group is represented in its training rows. Every held-out
    // calibration row belongs to the omitted group, hence its residual is zero.
    const candidates = Array.from({length: 32}, (_, row) => row)
      .filter(row => row !== query)
      .map(row => (Math.floor(row / 4) === queryGroup ? 0 : 1))
      .sort((first, second) => first - second);
    expect(candidates.filter(value => value === 0)).toHaveLength(3);
    const lower = candidates[ranks.lowerRank - 1],
      upper = candidates[ranks.upperRank - 1];
    expect([lower, upper]).toEqual([1, 1]);
    covered += Number(lower <= 0 && 0 <= upper);
  }
  expect(covered).toBe(0);
});

it.each([
  false,
  true
])('CV+ selects the exact zero candidate with flushed float comparisons=%s', flushSubnormals => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `cv-key-${nextView++}`, format, length);
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    const contributor = new GPUConformalPredictionIntervals({
      foldCount: 2,
      calibration: {
        observations: view('float32', 8),
        predictions: view('float32', 16),
        folds: view('uint32', 8)
      },
      query: {predictions: view('float32', 2), scales: view('float32', 1)},
      parameters: view('float32', 4),
      output: {
        intervals: view('float32', 2),
        status: view('uint32', 1),
        calibrationCount: view('uint32', 1)
      }
    });
    const [kernel] = getGPUCommandNodeKernelSources(
      contributor.getCommandNodes(graph).filter(node => node.id.endsWith('-intervals'))
    );
    // The rational CV+ oracle's eight usable scores, already sorted within each fold.
    // Lower candidates: [8,8,0,-4,18,14,10,6]; upper: [12,12,20,24,22,26,30,34].
    // At alpha=1/4 the ranks are 2 and 7, giving exactly [0,30].
    const intervals = new Float32Array(2),
      status = new Uint32Array(1);
    const values: Record<string, Float32Array | Uint32Array> = {
      sortedScores: Float32Array.of(1, 1, 5, 7, 1, 3, 5, 7),
      foldOffsets: Uint32Array.of(0, 4, 8),
      queryPredictions: Float32Array.of(10, 20),
      queryScales: Float32Array.of(2),
      parameters: getGPUConformalPredictionParameterValues({miscoverage: 0.25}),
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
    execution._evalCall = (expression, context) => {
      if (expression.name !== 'getConformalPredictionRanks')
        return evaluateCall(expression, context);
      // Isolate value selection from the interpreter's approximate uint64 rank arithmetic.
      // This fixture has n=8, alpha=1/4, hence exact ranks [2,7]; the device case runs both.
      const count = execution.evalExpression(expression.args![0], context);
      const alpha = execution.evalExpression(expression.args![1], context);
      expect(count instanceof ScalarData && count.value).toBe(8);
      expect(alpha instanceof ScalarData && alpha.value).toBe(0.25);
      return new VectorData([2, 7], execution.getTypeInfo('vec2u')!);
    };
    const evaluateBinary = execution._evalBinaryOp.bind(execution);
    execution._evalBinaryOp = (expression, context) => {
      if (flushSubnormals && ['>', '<=', '<', '>='].includes(expression.operator)) {
        const left = execution.evalExpression(expression.left, context);
        const right = execution.evalExpression(expression.right, context);
        if (
          left instanceof ScalarData &&
          right instanceof ScalarData &&
          left.typeInfo.name === 'f32' &&
          right.typeInfo.name === 'f32'
        ) {
          // Model the permitted Metal comparison behavior that reproduced the lead's failure.
          const flush = (value: number) => (Math.abs(value) < 2 ** -126 ? 0 : value);
          const first = flush(left.value),
            second = flush(right.value);
          const result =
            expression.operator === '>'
              ? first > second
              : expression.operator === '<='
                ? first <= second
                : expression.operator === '<'
                  ? first < second
                  : first >= second;
          return new ScalarData(Number(result), execution.getTypeInfo('bool')!);
        }
      }
      return evaluateBinary(expression, context);
    };
    execution.dispatchWorkgroups(
      kernel.entryPoint,
      [1, 1, 1],
      {0: bindings},
      {constants: kernel.constants}
    );
    expect(Array.from(intervals)).toEqual([0, 30]);
    expect(Array.from(status)).toEqual([0]);
  } finally {
    errors.mockRestore();
    device.destroy();
  }
});

it('keeps exact dyadic conformal boundaries and parameter packing', () => {
  // floor(alpha * 8), ceil((1 - alpha) * 8), exact rational arithmetic.
  for (const [alpha, lowerRank, upperRank] of [
    [0.125, 1, 7],
    [0.25, 2, 6],
    [0.5, 4, 4]
  ]) {
    expect(getConformalPredictionRanks(7, alpha)).toEqual({lowerRank, upperRank});
  }
  const target = new Float32Array([9, 9, 9, 9, 17]);
  expect(
    getGPUConformalPredictionParameterValues({miscoverage: 0.25, selfWeight: 0.5}, target)
  ).toBe(target);
  expect(Array.from(target)).toEqual([0.25, 0.5, 0, 0, 17]);
});

it('rejects a missing or non-uint32 calibration count output at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `audit-${nextView++}`, format, length);
  try {
    const props = {
      calibration: {observations: view('float32', 8), predictions: view('float32', 8)},
      query: {predictions: view('float32', 1)},
      parameters: view('float32', 4),
      output: {
        intervals: view('float32', 2),
        status: view('uint32', 1),
        calibrationCount: view('uint32', 1)
      }
    };
    expect(() => new GPUConformalPredictionIntervals(props)).not.toThrow();
    expect(
      () =>
        new GPUConformalPredictionIntervals({
          ...props,
          output: {...props.output, calibrationCount: {...props.output.calibrationCount, length: 0}}
        })
    ).toThrow('calibrationCount');
    expect(
      () =>
        new GPUConformalPredictionIntervals({
          ...props,
          output: {...props.output, calibrationCount: view('float32', 1) as never}
        })
    ).toThrow('calibrationCount');
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('captures conformal construction getters and nested maps once (localized=%s)', localized => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `capture-${nextView++}`, format, length);
  try {
    const calibrationValues = {
      observations: view('float32', 8),
      predictions: view('float32', 8),
      folds: view('uint32', 8),
      scales: view('float32', 8),
      mask: view('uint32', 8)
    };
    const queryValues = {predictions: view('float32', 2), scales: view('float32', 2)};
    const neighborsValues = {
      offsets: view('uint32', 3),
      indices: view('uint32', 8),
      weights: view('float32', 8)
    };
    const outputValues = {
      intervals: view('float32', 4),
      status: view('uint32', 2),
      ...(!localized ? {calibrationCount: view('uint32', 1)} : {})
    };
    const calibration = watchConstructionFields(calibrationValues, 'calibration');
    const query = watchConstructionFields(queryValues, 'query');
    const neighbors = watchConstructionFields(neighborsValues, 'neighbors');
    const output = watchConstructionFields(outputValues, 'output');
    const common = {
      id: 'capture-conformal',
      calibration: calibration.props,
      query: query.props,
      parameters: view('float32', 4),
      output: output.props
    };
    const original = localized
      ? watchConstructionFields({...common, neighbors: neighbors.props}, 'props')
      : watchConstructionFields({...common, foldCount: 1}, 'props');
    const contributor = localized
      ? new GPUGeoConformalPredictionIntervals(original.props as never)
      : new GPUConformalPredictionIntervals(original.props as never);
    const originalObservation = calibrationValues.observations,
      originalIntervals = outputValues.intervals;
    for (const watched of [original, calibration, query, output, ...(localized ? [neighbors] : [])])
      watched.expectReadOnce();
    calibrationValues.observations = view('float32', 1);
    queryValues.scales = view('float32', 1);
    outputValues.intervals = view('float32', 1);
    neighborsValues.weights = view('float32', 1);
    expect(contributor.props.calibration.observations).toBe(originalObservation);
    expect(contributor.props.output.intervals).toBe(originalIntervals);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    for (const watched of [original, calibration, query, output, ...(localized ? [neighbors] : [])])
      watched.expectReadOnce();
  } finally {
    device.destroy();
  }
});
