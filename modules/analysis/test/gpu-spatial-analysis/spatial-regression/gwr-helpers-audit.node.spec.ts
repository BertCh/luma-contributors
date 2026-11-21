// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {createReadableGWRAuditBuffer} from './gwr-audit-buffer-test-utils';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getStudentTCriticalValue,
  getStudentTTwoSidedTail,
  getStudentTWGSL
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {
  selectGoldenSectionBandwidthWithStatus,
  getGoldenSectionIntervalNodeCount
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {
  GPUCommandGraph,
  createTransientView,
  getViewBindingRange,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {createGoldenSectionIntervalNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const values of [Float32Array.of(-999, NaN, 3), Uint32Array.of(99, 2, 1)]) {
  it(`allocates initialized ${values.constructor.name} audit storage with upload and readback usage`, () => {
    const device = createNullWebGPUDevice();
    const createBuffer = vi.spyOn(device, 'createBuffer');
    const buffer = createReadableGWRAuditBuffer(device, values);
    try {
      // WebGPU copies the result into a staging buffer for readAsync; COPY_SRC is mandatory.
      expect(buffer.usage & Buffer.COPY_SRC).toBe(Buffer.COPY_SRC);
      expect(buffer.usage & Buffer.COPY_DST).toBe(Buffer.COPY_DST);
      expect(buffer.usage & Buffer.STORAGE).toBe(Buffer.STORAGE);
      expect(buffer.byteLength).toBe(values.byteLength);
      expect(createBuffer).toHaveBeenCalledWith(expect.objectContaining({data: values}));
    } finally {
      createBuffer.mockRestore();
      buffer.destroy();
      device.destroy();
    }
  });
}

it('keeps Cauchy tails and quantiles representable beyond the old doubling and squared-distance ranges', () => {
  // T_1 is Cauchy: P(|T_1|>=t)=2 atan(1/t)/pi and its two-sided quantile is cot(pi*tail/2).
  // These forms avoid subtracting near-one probabilities and require no fitted-code oracle.
  const expectedTail = (2 * Math.atan(1e-200)) / Math.PI;
  const actualTail = getStudentTTwoSidedTail(1e200, 1);
  const expectedQuantile = 1 / Math.tan((Math.PI * 1e-25) / 2);
  const actualQuantile = getStudentTCriticalValue(1e-25, 1);
  console.info('Cauchy audit', {actualTail, expectedTail, actualQuantile, expectedQuantile});
  expect(Number.isFinite(actualTail)).toBe(true);
  expect(Math.abs(actualTail / expectedTail - 1)).toBeLessThan(1e-8);
  expect(Number.isFinite(actualQuantile)).toBe(true);
  expect(Math.abs(actualQuantile / expectedQuantile - 1)).toBeLessThan(1e-8);
});

it('evaluates the same extreme Cauchy tail and critical value in emitted WGSL', () => {
  const output = new Float32Array(2);
  const source = `${GET_NAN_WGSL}\n${getStudentTWGSL()}
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1, 1, 1)
fn main() { output[0]=gwrStudentTwoSidedTail(1.0e25,1.0); output[1]=gwrStudentCriticalValue(1.0e-25,1.0); }`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups('main', [1, 1, 1], {
    0: {0: output}
  });
  const expected = [(2 * Math.atan(1e-25)) / Math.PI, 1 / Math.tan((Math.PI * 1e-25) / 2)];
  console.info('WGSL Cauchy audit', Array.from(output), expected);
  for (let index = 0; index < 2; index++) {
    expect(Number.isFinite(output[index])).toBe(true);
    expect(Math.abs(output[index] / expected[index] - 1)).toBeLessThan(1e-4);
  }
});

it('respects search budgets and tie conventions on independent scalar objectives', () => {
  expect(selectGoldenSectionBandwidthWithStatus(1, 5, value => -value, 1)).toEqual({
    bandwidth: 5,
    iterationLimitReached: true
  });
  const tied = selectGoldenSectionBandwidthWithStatus(1, 5, () => 7, 10);
  expect(tied.bandwidth).toBeCloseTo(2.52788, 12);
  expect(tied.iterationLimitReached).toBe(false);
  expect(getGoldenSectionIntervalNodeCount(3, 2)).toBe(36);
});

for (const [integer, enabled, expected] of [
  [false, true, [5, -5, 5, 0, 3, 0, 1]],
  [true, true, [3, -3, 3, 1, 1, 0, 0]],
  [false, false, [NaN, NaN, NaN, 0, 0, 0, 0]]
] as const) {
  it(`executes nested interval-search output/status against a monotone closed form: integer=${integer}, enabled=${enabled}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const memory = new Map<GraphBufferHandle, ArrayBuffer>();
    const bounds = createTransientView(graph, 'bounds', 'float32', 6);
    memory.set(bounds.buffer, Float32Array.of(1, integer ? 3 : 5, 0, 0, 1e-4, 0).buffer);
    const parent = createTransientView(graph, 'parent', 'float32', 7);
    memory.set(parent.buffer, Float32Array.of(0, 0, 0, 0, 0, Number(enabled), 0).buffer);
    const interval = createGoldenSectionIntervalNodes(graph, {
      id: 'interval-audit',
      bounds,
      axis: 0,
      integer,
      maximumIterations: 3,
      parent,
      evaluate: (value, score, active, id) => [
        createWGSLKernelNode(graph, {
          id,
          operation: 'IntervalAuditObjective',
          invocationCount: 1,
          bindings: [
            {name: 'value', view: value, type: 'f32', access: 'read'},
            {name: 'searchState', view: active, type: 'f32', access: 'read'},
            {name: 'score', view: score, type: 'f32', access: 'read_write'}
          ],
          body: 'if (searchState[searchStateOffset + 5u] < 0.5) { return; } score[scoreOffset] = -value[valueOffset]; score[scoreOffset + 1u] = value[valueOffset]; score[scoreOffset + 2u] = 1.0; score[scoreOffset + 3u] = 0.0;'
        })
      ]
    });
    try {
      for (const node of interval.nodes) {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        const bindings = Object.fromEntries(
          record.bindings.map((binding, index) => {
            if (!memory.has(binding.view.buffer))
              memory.set(binding.view.buffer, new ArrayBuffer(binding.view.buffer.byteLength));
            const range = getViewBindingRange(binding.view);
            return [
              index,
              new Uint8Array(memory.get(binding.view.buffer)!, range.offset, range.size)
            ];
          })
        );
        const source = record.source
          .replace(/^\s*_\s*=.*;$/gm, '')
          .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)');
        new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
          'main',
          [...record.workgroups],
          {0: bindings},
          {constants: record.constants}
        );
      }
      const actual = new Float32Array(memory.get(interval.result.buffer)!);
      for (let index = 0; index < 7; index++) {
        if (Number.isNaN(expected[index])) expect(actual[index]).toBeNaN();
        else expect(actual[index]).toBe(expected[index]);
      }
    } finally {
      device.destroy();
    }
  });
}
