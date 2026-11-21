// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import {CPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const lineCount of [0, 1]) {
  for (const selection of [false, true]) {
    it(`W5-10 empty named forms execute ${lineCount} lines with selection ${selection}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: new Float32Array(0)},
            offsets: {format: 'uint32', values: new Uint32Array(lineCount + 1)},
            ...(selection
              ? {parameters: {format: 'float32' as const, values: new Float32Array(4)}}
              : {})
          },
          operations: [
            new Operation({
              positions: 'positions',
              lineOffsets: 'offsets',
              importance: 'importance',
              rowCount: 0,
              lineCount,
              capacity: 0,
              parameters: selection ? 'parameters' : undefined,
              output: {
                ...(selection
                  ? {ids: 'ids', outputMask: 'mask', lineCounts: 'counts', lineStarts: 'starts'}
                  : {}),
                status: {
                  converged: 'converged',
                  iterationCount: 'iterations',
                  ...(selection
                    ? {
                        count: 'count',
                        overflow: 'overflow',
                        requiredCount: 'required',
                        candidateOverflow: 'candidate'
                      }
                    : {})
                }
              }
            })
          ],
          outputs: [
            'importance',
            'converged',
            'iterations',
            ...(selection
              ? ['ids', 'mask', 'counts', 'starts', 'count', 'overflow', 'required', 'candidate']
              : [])
          ]
        });
        expect(Array.from(result['importance'])).toEqual([]);
        expect(Array.from(result['converged'])).toEqual([1]);
        expect(Array.from(result['iterations'])).toEqual([1]);
        if (selection) {
          for (const port of ['ids', 'mask']) expect(Array.from(result[port])).toEqual([]);
          for (const port of ['counts', 'starts'])
            expect(Array.from(result[port])).toEqual(new Array(lineCount).fill(0));
          for (const port of ['count', 'overflow', 'required', 'candidate'])
            expect(Array.from(result[port])).toEqual([0]);
        }
      }
    });
  }
}
