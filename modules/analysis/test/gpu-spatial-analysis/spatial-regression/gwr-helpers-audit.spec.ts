// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {getStudentTWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';
import {createGoldenSectionIntervalNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes';
import {
  createGWRCandidateReduction,
  createGWRSearchBoundsReductionNodes,
  createGWRVarianceReductionNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-global-reductions';
import {createOutputBuffer, readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createReadableGWRAuditBuffer} from './gwr-audit-buffer-test-utils';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (name: string, values: number[]) => {
    const buffer = createReadableGWRAuditBuffer(device, Float32Array.from(values));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', values.length)};
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'float32', length)};
  };
  return {graph, input, output, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

for (const [integer, enabled, expected] of [
  [false, true, [5, -5, 5, 0, 3, 0, 1]],
  [true, true, [3, -3, 3, 1, 1, 0, 0]],
  [false, false, [NaN, NaN, NaN, 0, 0, 0, 0]]
] as const) {
  it(`checks every nested interval-search result word: integer=${integer}, enabled=${enabled}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const context = createContext(device);
    const {graph, input, output} = context;
    const bounds = input('bounds', [1, integer ? 3 : 5, 0, 0, 1e-4, 0]);
    const parent = input('parent', [0, 0, 0, 0, 0, Number(enabled), 0]);
    const interval = createGoldenSectionIntervalNodes(graph, {
      id: 'interval-audit',
      bounds: bounds.view,
      axis: 0,
      integer,
      maximumIterations: 3,
      parent: parent.view,
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
    for (const node of interval.nodes) graph.add(node);
    const result = output('result', 7);
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'interval-audit-copy',
        operation: 'IntervalAuditRead',
        invocationCount: 7,
        bindings: [
          {name: 'result', view: interval.result, type: 'f32', access: 'read'},
          {name: 'output', view: result.view, type: 'f32', access: 'read_write'}
        ],
        body: 'output[outputOffset + index] = result[resultOffset + index];'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const actual = await readFloat32(result.buffer);
      for (let index = 0; index < 7; index++) {
        if (Number.isNaN(expected[index])) expect(actual[index]).toBeNaN();
        else expect(actual[index]).toBe(expected[index]);
      }
    } finally {
      compiled.destroy();
      context.destroy();
    }
  });
}

it('evaluates extreme representable Cauchy tails and quantiles on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  const result = context.output('result', 2);
  context.graph.add(
    createWGSLKernelNode(context.graph, {
      id: 'student-t-audit',
      operation: 'GWRStudentTAudit',
      invocationCount: 1,
      bindings: [{name: 'result', view: result.view, type: 'f32', access: 'read_write'}],
      declarations: `${GET_NAN_WGSL}\n${getStudentTWGSL()}`,
      body: 'result[resultOffset] = gwrStudentTwoSidedTail(1.0e25, 1.0); result[resultOffset + 1u] = gwrStudentCriticalValue(1.0e-25, 1.0);'
    })
  );
  const compiled = context.graph.compile();
  try {
    submitGraph(device, compiled);
    const actual = await readFloat32(result.buffer);
    // Exact df=1 Cauchy formulas, also derived in gwr-helpers-audit.node.spec.ts.
    const expected = [6.366197723675814e-26, 6.366197723675813e24];
    for (let index = 0; index < 2; index++) {
      expect(Number.isFinite(actual[index])).toBe(true);
      // log/exp in f32 at exponent ~58: relative 1e-4 covers the transcendental rounding.
      expect(Math.abs(actual[index] / expected[index] - 1)).toBeLessThan(1e-4);
    }
  } finally {
    compiled.destroy();
    context.destroy();
  }
});

it('reduces independent candidate, bounds and centered-variance tiles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  const {graph, input} = context;
  const source = input(
    'candidate-tiles',
    [
      1, 0.5, 1, 0, 3, 2, 1, 0, 0, 3, 3, 0.7, 0, 0, 2, 4, 0.8, 0, 1, 2, 5, 0.9, 1, 0, 4, 6, 1.2, 0,
      0, 4
    ]
  );
  const candidate = createGWRCandidateReduction(graph, {
    id: 'candidate-audit',
    tileCount: 3,
    candidateWidth: 2,
    source: source.view
  });
  for (const node of candidate.createNodes('')) graph.add(node);
  const candidateTotals = context.output('candidate-totals', 10);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'candidate-audit-read',
      operation: 'GWRCandidateAudit',
      invocationCount: 10,
      bindings: [
        {name: 'totals', view: candidate.totals, type: 'f32', access: 'read'},
        {name: 'output', view: candidateTotals.view, type: 'f32', access: 'read_write'}
      ],
      body: 'output[outputOffset + index] = totals[totalsOffset + index * 64u];'
    })
  );
  const bounds = input('bounds', [3, 7, 2, 2, 8, 0, 1, 9, 3]);
  for (const node of createGWRSearchBoundsReductionNodes(graph, {
    id: 'bounds-audit',
    tileCount: 3,
    tileCandidates: bounds.view
  }))
    graph.add(node);
  // Tiles represent [2,4], [], [6,8,10]: n=5, mean=6, total centered squares=40.
  // RSS=4 thus R2=1-4/40=0.9. The empty tile must contribute no observations.
  const statistics = input('statistics', [2, 3, 2, 0, 0, 0, 3, 8, 8]);
  const summary = input('summary', [4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
  for (const node of createGWRVarianceReductionNodes(graph, {
    id: 'variance-audit',
    tileCount: 3,
    statistics: statistics.view,
    summary: summary.view
  }))
    graph.add(node);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const totals = await readFloat32(candidateTotals.buffer);
    const expected = [9, 2.1, 2, 0, 9, 12, 3, 0, 1, 9];
    for (let index = 0; index < expected.length; index++) {
      expect(Number.isFinite(totals[index])).toBe(true);
      expect(Math.abs(totals[index] - expected[index])).toBeLessThan(1e-5);
    }
    const actualBounds = await readFloat32(bounds.buffer);
    expect(actualBounds.slice(0, 3)).toEqual([1, 9, 5]);
    const actualSummary = await readFloat32(summary.buffer);
    expect(Number.isFinite(actualSummary[3])).toBe(true);
    expect(Math.abs(actualSummary[3] - 0.9)).toBeLessThan(1e-6);
  } finally {
    compiled.destroy();
    context.destroy();
  }
});
