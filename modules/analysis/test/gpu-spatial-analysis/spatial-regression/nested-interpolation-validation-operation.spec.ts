// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import {GPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createNestedInput,
  createNestedProps,
  createNestedGraphInputs,
  expectNestedRows,
  fixture
} from './nested-interpolation-validation-operation-test-utils';

for (const scene of [
  'ordinary',
  'overlap-and-empty',
  'buffer-exhaustion',
  'nonfinite-response',
  'constant-response'
] as const) {
  it(`compares every nested-validation output through the mixed graph: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = createNestedProps();
    const input = createNestedInput();
    if (scene === 'overlap-and-empty') {
      input.training.fill(0);
      input.testing.fill(0);
      input.training.fill(1, 0, props.rowCount);
      input.testing.fill(1, 0, props.rowCount);
      input.testing[2 * props.rowCount] = 1;
    }
    if (scene === 'buffer-exhaustion') props.innerBufferDistance = 100;
    if (scene === 'nonfinite-response') input.response[0] = NaN;
    if (scene === 'constant-response') {
      input.response.fill(7);
      // Identical candidates give a deliberate first-candidate tie on both devices.
      const candidate = {
        mode: 'ordinary-kriging' as const,
        covariance: {model: 'spherical' as const, range: 0.001, sill: 1}
      };
      props.candidates = [candidate, candidate];
    }
    const results = [];
    for (const Operation of [
      GPUNestedInterpolationValidationOperation,
      CPUNestedInterpolationValidationOperation
    ]) {
      results.push(
        await runOperationGraph(device, {
          inputs: createNestedGraphInputs(input),
          // The helper destroys each operation. Construct a fresh one for every run.
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        })
      );
    }
    const [gpu, cpu] = results;
    expect(Array.from(gpu['selected'])).toEqual(Array.from(cpu['selected']));
    expect(Array.from(gpu['status'])).toEqual(Array.from(cpu['status']));
    // f64 vs f32 local solves and serial metric accumulation: relative 5e-4 plus 2e-5
    // absolute near zero. Integers, count cells and missing markers compare exactly below.
    for (const port of ['scores', 'predictions', 'metrics'])
      expectNestedRows(gpu[port], cpu[port], 5e-4, 2e-5);
    for (let index = 0; index < cpu['scores'].length; index++)
      if (index % 3 !== 0) expect(gpu['scores'][index]).toBe(cpu['scores'][index]);
    for (let index = 0; index < cpu['metrics'].length; index++)
      if (index % 6 < 2) expect(gpu['metrics'][index]).toBe(cpu['metrics'][index]);
    if (scene === 'ordinary') {
      expect(Array.from(cpu['status'])).toEqual([0, 0, 0]);
      expect(Array.from(cpu['selected'])).toEqual(fixture.folds.map(fold => fold.selected));
      expectNestedRows(
        gpu['scores'],
        fixture.folds.flatMap(fold => fold.scores.flat()),
        5e-4,
        2e-5
      );
      expectNestedRows(
        gpu['predictions'],
        fixture.folds.flatMap(fold => fold.predictions),
        5e-4,
        2e-5
      );
      expectNestedRows(
        gpu['metrics'],
        fixture.folds.flatMap(fold => fold.metrics),
        5e-4,
        2e-5
      );
    }
    if (scene === 'overlap-and-empty') expect(Array.from(cpu['status'])).toEqual([7, 10, 6]);
    if (scene === 'buffer-exhaustion') {
      expect(Array.from(cpu['selected'])).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
      expect(Array.from(cpu['status'])).toEqual([6, 6, 6]);
    }
    if (scene === 'nonfinite-response') expect(Array.from(cpu['status'])).toEqual([4, 0, 0]);
    if (scene === 'constant-response') {
      expect(Array.from(cpu['selected'])).toEqual([0, 0, 0]);
      expect(Array.from(cpu['status'])).toEqual([0, 0, 0]);
      for (let fold = 0; fold < props.foldCount; fold++)
        expect(Number.isNaN(cpu['metrics'][fold * 6 + 5])).toBe(true);
    }
  });
}

for (const candidateIndex of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) {
  it(`compares each candidate as the outer predictor, candidate ${candidateIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props = createNestedProps();
    props.candidates = [props.candidates[candidateIndex]];
    const results = [];
    for (const Operation of [
      GPUNestedInterpolationValidationOperation,
      CPUNestedInterpolationValidationOperation
    ])
      results.push(
        await runOperationGraph(device, {
          inputs: createNestedGraphInputs(createNestedInput()),
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        })
      );
    const [gpu, cpu] = results;
    expect(Array.from(gpu['selected'])).toEqual([0, 0, 0]);
    expect(Array.from(cpu['selected'])).toEqual([0, 0, 0]);
    expect(Array.from(gpu['status'])).toEqual([0, 0, 0]);
    expect(Array.from(cpu['status'])).toEqual([0, 0, 0]);
    for (const port of ['scores', 'predictions', 'metrics'])
      expectNestedRows(gpu[port], cpu[port], 5e-4, 2e-5);
    expectNestedRows(
      gpu['predictions'],
      fixture.folds.flatMap(fold => fold.candidatePredictions[candidateIndex]),
      5e-4,
      2e-5
    );
    for (let fold = 0; fold < props.foldCount; fold++) {
      expect(gpu['scores'][fold * 3 + 1]).toBe(20);
      expect(gpu['scores'][fold * 3 + 2]).toBe(0);
      expect(gpu['metrics'][fold * 6]).toBe(10);
      expect(gpu['metrics'][fold * 6 + 1]).toBe(0);
    }
  });
}
