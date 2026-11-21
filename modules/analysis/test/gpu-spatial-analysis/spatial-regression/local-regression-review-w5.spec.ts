// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createMGTWRComponentEvaluationNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-bandwidth-search';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const scenario of ['mixed', 'rounded'] as const) {
  it(`B3 B4 component search retains ${scenario} fit-time residual evidence on WebGPU`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const buffers: ReturnType<typeof createInputBuffer>[] = [];
    let serial = 0;
    const input = (values: number[]) => {
      const buffer = createInputBuffer(device, Float32Array.from(values));
      buffers.push(buffer);
      return importGraphBuffer(graph, `review-input-${serial++}`, buffer, 'float32', values.length);
    };
    const outputBuffer = createOutputBuffer(device, 28);
    buffers.push(outputBuffer);
    const output = importGraphBuffer(graph, 'review-output', outputBuffer, 'float32', 28);
    const gate = createWGSLDispatchGate(graph, {
      id: 'review-gate',
      operation: 'Review',
      shapes: [[8], [1]]
    });
    const evaluation = createMGTWRComponentEvaluationNodes(graph, {
      id: 'review-fit',
      rows: 8,
      predictorCount: 1,
      component: 0,
      points: input(
        Array.from({length: 8}, (_, row) => [
          scenario === 'rounded' && row >= 4 ? 2 : 0,
          0,
          0
        ]).flat()
      ),
      predictors: input(Array(8).fill(0)),
      partial: input(
        scenario === 'mixed'
          ? [33554432, -33554432, 1, 1, 1, 1, 1, 1]
          : [1e6, 1e6, 1e6, 1e6, 1e6 + 1 / 16, 1e6 + 1 / 16, 1e6 + 1 / 16, 1e6 + 1 / 16]
      ),
      parameters: input([1, 0, 0, 0]),
      adaptive: false,
      bisquare: false,
      gate
    });
    const fitRecord = getWGSLKernelNodeRecord(graph, evaluation.nodes[0].id)!;
    const fit = fitRecord.bindings.find(binding => binding.name === 'scratch')!.view;
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'arm-review',
        operation: 'Review',
        invocationCount: 1,
        bindings: [gate.binding],
        body: gate.armWGSL
      })
    );
    evaluation.nodes.forEach(node => graph.add(node));
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'copy-review',
        operation: 'Review',
        invocationCount: 28,
        bindings: [
          {name: 'summary', view: evaluation.summary, type: 'f32', access: 'read'},
          {name: 'fit', view: fit, type: 'f32', access: 'read'},
          {name: 'output', view: output, type: 'f32', access: 'read_write'}
        ],
        body: `if (index < 12u) { output[outputOffset + index] = summary[summaryOffset + index]; }
      else if (index < 20u) { output[outputOffset + index] = fit[fitOffset + (index - 12u) * 4u]; }
      else { output[outputOffset + index] = fit[fitOffset + (index - 20u) * 4u + 3u]; }`
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const actual = await readFloat32(outputBuffer, 28);
      if (scenario === 'mixed') {
        for (let row = 0; row < 8; row++) {
          expect(Number.isFinite(actual[12 + row])).toBe(true);
          expect(actual[12 + row]).toBeCloseTo(0.75, 6);
        }
      } else {
        // r=(1/16)*exp(-2)/(1+exp(-2)), RSS=8*r^2; tr(S)=2/(1+exp(-2)).
        expect(Number.isFinite(actual[2])).toBe(true);
        expect(actual[2]).toBeCloseTo(-45.26422157, 4);
        let residualSquares = 0;
        for (let row = 0; row < 8; row++) {
          expect(actual[12 + row]).toBe(1e6 + (row >= 4 ? 1 / 16 : 0));
          expect(Number.isFinite(actual[20 + row])).toBe(true);
          expect(
            Math.abs(actual[20 + row] - (row < 4 ? -0.007450182626 : 0.007450182626))
          ).toBeLessThan(2e-8);
          residualSquares += actual[20 + row] ** 2;
        }
        expect(Math.abs(residualSquares - 0.0004440417693)).toBeLessThan(2e-9);
      }
    } finally {
      compiled.destroy();
      buffers.forEach(buffer => buffer.destroy());
    }
  });
}
