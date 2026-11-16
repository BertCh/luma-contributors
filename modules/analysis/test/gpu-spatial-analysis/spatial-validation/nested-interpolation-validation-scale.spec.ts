// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import {CPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('nested validation above 256 rows reproduces an affine field and the CPU oracle', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 400;
  const folds = 2;
  const positions = new Float32Array(rows * 2);
  const response = new Float32Array(rows);
  const blocks = new Uint32Array(rows);
  const training = new Uint32Array(folds * rows);
  const testing = new Uint32Array(folds * rows);
  for (let row = 0; row < rows; row++) {
    const horizontal = (row % 20) / 20;
    const vertical = Math.floor(row / 20) / 20;
    positions[row * 2] = horizontal;
    positions[row * 2 + 1] = vertical;
    response[row] = 2 + horizontal * 0.5 - vertical * 0.25;
    blocks[row] = Math.floor(row / 20) * 5 + Math.floor((row % 20) / 4);
    for (let fold = 0; fold < folds; fold++) {
      testing[fold * rows + row] = Number(row % 2 === fold);
      training[fold * rows + row] = Number(row % 2 !== fold);
    }
  }
  const run = (
    Operation:
      | typeof GPUNestedInterpolationValidationOperation
      | typeof CPUNestedInterpolationValidationOperation
  ) =>
    runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: positions},
        response: {format: 'float32', values: response},
        blocks: {format: 'uint32', values: blocks},
        training: {format: 'uint32', values: training},
        testing: {format: 'uint32', values: testing}
      },
      operations: [
        new Operation({
          rowCount: rows,
          foldCount: folds,
          innerFoldCount: 2,
          neighborCount: 8,
          seed: 17,
          positions: 'positions',
          response: 'response',
          training: 'training',
          testing: 'testing',
          innerBlocks: 'blocks',
          candidates: [{mode: 'rbf', kernel: 'thin-plate-spline', degree: 1, epsilon: 1}],
          output: {
            selected: 'selected',
            scores: 'scores',
            predictions: 'predictions',
            metrics: 'metrics',
            status: 'status'
          }
        })
      ],
      outputs: ['selected', 'scores', 'predictions', 'metrics', 'status']
    });
  const actual = await run(GPUNestedInterpolationValidationOperation);
  const expected = await run(CPUNestedInterpolationValidationOperation);
  expect(Array.from(actual['status'])).toEqual([0, 0]);
  expect(Array.from(actual['selected'])).toEqual([0, 0]);
  for (let fold = 0; fold < folds; fold++) {
    expect(actual['metrics'][fold * 6]).toBe(200);
    expect(actual['metrics'][fold * 6 + 1]).toBe(0);
    expect(actual['metrics'][fold * 6 + 2]).toBeLessThan(1e-7);
    for (let row = 0; row < rows; row++) {
      const value = actual['predictions'][fold * rows + row];
      if (!testing[fold * rows + row]) expect(value).toBeNaN();
      else {
        expect(Math.abs(value - response[row])).toBeLessThan(1e-3);
        expect(Math.abs(value - expected['predictions'][fold * rows + row])).toBeLessThan(1e-3);
      }
    }
  }
});
