// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {
  GPULineSimplificationOperation,
  type LineSimplificationOperationProps
} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes the full required count and overflow when selection capacity is zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props: LineSimplificationOperationProps = {
    positions: 'positions',
    lineOffsets: 'offsets',
    parameters: 'parameters',
    importance: 'importance',
    rowCount: 2,
    lineCount: 1,
    capacity: 0,
    output: {
      ids: 'ids',
      outputMask: 'mask',
      lineCounts: 'lineCounts',
      lineStarts: 'lineStarts',
      status: {
        count: 'count',
        requiredCount: 'required',
        overflow: 'overflow',
        converged: 'converged',
        iterationCount: 'iterations'
      }
    }
  };
  const outputs = [
    'importance',
    'ids',
    'mask',
    'lineCounts',
    'lineStarts',
    'count',
    'required',
    'overflow',
    'converged',
    'iterations'
  ];
  const expected = {
    importance: [Infinity, Infinity],
    ids: [],
    mask: [1, 1],
    lineCounts: [2],
    lineStarts: [0],
    count: [0],
    required: [2],
    overflow: [1],
    converged: [1],
    iterations: [1]
  };
  for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        positions: {format: 'float32x2', values: new Float32Array([0, 0, 1, 0])},
        offsets: {format: 'uint32', values: new Uint32Array([0, 2])},
        parameters: {
          format: 'float32',
          values: getGPULineSimplificationParameterValues({tolerance: 0})
        }
      },
      operations: [new Operation(props)],
      outputs
    });
    // No arithmetic uncertainty: two infinite endpoint importances, two kept rows and no slots.
    for (const port of outputs)
      expect(Array.from(result[port])).toEqual(expected[port as keyof typeof expected]);
  }
});

for (const method of ['douglas-peucker', 'visvalingam'] as const)
  for (const empty of [false, true]) {
    it(`compares every ${method} output (${empty ? 'empty lines' : 'two lines with suppression and overflow'})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Two adjacent interiors per line exercise a tie, a neighbor update and a second split.
      // Radius 3 serializes these small Visvalingam lines, matching the exact sequential result.
      // The second line is collinear and both middle rows are suppressed.
      const props: LineSimplificationOperationProps = {
        positions: 'positions',
        lineOffsets: 'offsets',
        parameters: 'parameters',
        importance: 'importance',
        rowCount: 10,
        lineCount: 3,
        capacity: 4,
        method,
        output: {
          ids: 'ids',
          outputMask: 'mask',
          lineCounts: 'lineCounts',
          lineStarts: 'lineStarts',
          status: {
            count: 'count',
            requiredCount: 'required',
            overflow: 'overflow',
            converged: 'converged',
            iterationCount: 'iterations'
          }
        }
      };
      const outputs = [
        'importance',
        'ids',
        'mask',
        'lineCounts',
        'lineStarts',
        'count',
        'required',
        'overflow',
        'converged',
        'iterations'
      ];
      const results = [];
      for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              positions: {
                format: 'float32x2',
                values: new Float32Array([
                  99, 99, 0, 0, 1, 1, 2, 1, 3, 0, 10, 0, 11, 0, 12, 0, 13, 0, 88, 88
                ])
              },
              offsets: {
                format: 'uint32',
                values: new Uint32Array(empty ? [1, 1, 1, 1] : [1, 5, 5, 9])
              },
              parameters: {
                format: 'float32',
                values: getGPULineSimplificationParameterValues({method, tolerance: 0.5})
              }
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      for (const port of outputs) {
        expect(results[0][port].length).toBe(results[1][port].length);
        for (let index = 0; index < results[1][port].length; index++) {
          const expected = results[1][port][index];
          const actual = results[0][port][index];
          // Importance allows 3e-6 relative; integer fields and infinite/zero importance are exact.
          if (port !== 'importance' || !Number.isFinite(expected) || expected === 0)
            expect(actual === expected).toBe(true);
          else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * 3e-6);
        }
      }
      expect(results[1]['required'][0]).toBe(empty ? 0 : 5);
    });
  }
