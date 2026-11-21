// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const chunked of [false, true]) {
  it(`three-row ${chunked ? 'chunked' : 'whole'} chain preserves spare trace words through each loss and recovery`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    let serial = 0;
    const input = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `input-${serial++}`, buffer, format, values.length)
      };
    };
    const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
      const buffer = createOutputBuffer(device, length + 2);
      buffers.push(buffer);
      buffer.write(
        format === 'float32'
          ? Float32Array.from([...new Array(length).fill(77), 98, 99])
          : Uint32Array.of(77, 98, 99)
      );
      return {
        buffer,
        view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length + 2)
      };
    };
    const flags = Array.from({length: chunked ? 4 : 2}, () => input('uint32', Uint32Array.of(0)));
    const offsets = input('uint32', Uint32Array.of(0, 1, 3, 4));
    const whole = {
      offsets: offsets.view,
      neighbors: input('uint32', Uint32Array.of(1, 0, 2, 1)).view,
      weights: input('float32', Float32Array.of(1, 1, 1, 1)).view,
      status: {overflow: flags[0].view, candidateOverflow: flags[1].view}
    };
    const powerTraces = output('float32', 2);
    const powerSums = output('float32', 2);
    const powerTraceStandardErrors = output('float32', 2);
    const infinityNorm = output('float32', 1);
    const approximation = output('uint32', 1);
    const code = output('uint32', 1);
    const converged = output('uint32', 1);
    code.buffer.write(Uint32Array.of(91, 98, 99));
    converged.buffer.write(Uint32Array.of(92, 98, 99));
    const widerStatus = {
      approximation: approximation.view,
      code: code.view,
      converged: converged.view
    };
    const laterOffsets = input('uint32', Uint32Array.of(0, 2, 3));
    const laterNeighbors = input('uint32', Uint32Array.of(0, 2, 1));
    const firstChild = {
      rowStart: 0,
      rowCount: 1,
      weights: {
        offsets: input('uint32', Uint32Array.of(0, 1)).view,
        neighbors: input('uint32', Uint32Array.of(1)).view,
        weights: input('float32', Float32Array.of(1)).view
      }
    };
    const laterChild = {
      rowStart: 1,
      rowCount: 2,
      weights: {
        offsets: laterOffsets.view,
        neighbors: laterNeighbors.view,
        weights: input('float32', Float32Array.of(1, 1, 1)).view,
        status: whole.status
      }
    };
    const gramTraces = chunked ? undefined : output('float32', 4);
    // Hand calculation: s=2, A=W/2, tr(A)=0, tr(A²)=1, 1'A1=2, 1'A²1=1.5.
    // W is symmetric: tr(W'W)=4, tr(W³)=0, tr(W⁴)=8; column sums [1,2,1] give 6.
    const expectations = [
      {result: powerTraces, values: [0, 1]},
      {result: powerSums, values: [2, 1.5]},
      {result: powerTraceStandardErrors, values: [0, 0]},
      {result: infinityNorm, values: [2]},
      ...(gramTraces ? [{result: gramTraces, values: [4, 0, 8, 6]}] : [])
    ];
    try {
      graph.add(
        new GPUSpatialWeightsTraces({
          weights: chunked
            ? {
                rowCount: 3,
                chunks: [firstChild, laterChild],
                status: {overflow: flags[2].view, candidateOverflow: flags[3].view}
              }
            : whole,
          maximumPower: 2,
          mode: 'exact',
          output: {
            powerTraces: powerTraces.view,
            powerSums: powerSums.view,
            powerTraceStandardErrors: powerTraceStandardErrors.view,
            infinityNorm: infinityNorm.view,
            ...(gramTraces ? {gramTraces: gramTraces.view} : {}),
            status: widerStatus
          }
        })
      );
      const compiled = graph.compile();
      try {
        for (const selected of flags) {
          for (const loss of [0, 9, 0]) {
            for (const flag of flags)
              flag.buffer.write(Uint32Array.of(flag === selected ? loss : 0));
            offsets.buffer.write(Uint32Array.from(loss ? [0, 1, 2, 3] : [0, 1, 3, 4]));
            laterOffsets.buffer.write(Uint32Array.from(loss ? [0, 1, 2] : [0, 2, 3]));
            laterNeighbors.buffer.write(Uint32Array.from(loss ? [0, 1, 1] : [0, 2, 1]));
            submitGraph(device, compiled, undefined);
            for (const {result, values} of expectations) {
              const actual = await readFloat32(result.buffer, values.length + 2);
              expect(Array.from(actual.slice(values.length))).toEqual([98, 99]);
              for (const [index, expected] of values.entries()) {
                if (loss) expect(Number.isNaN(actual[index])).toBe(true);
                else {
                  expect(Number.isFinite(actual[index])).toBe(true);
                  // 1e-5 covers float32 reductions on this three-row exact fixture.
                  expect(Math.abs(actual[index] - expected)).toBeLessThanOrEqual(1e-5);
                }
              }
            }
            expect(Array.from(await readUint32(approximation.buffer, 3))).toEqual([0, 98, 99]);
            expect(Array.from(await readUint32(code.buffer, 3))).toEqual([91, 98, 99]);
            expect(Array.from(await readUint32(converged.buffer, 3))).toEqual([92, 98, 99]);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
