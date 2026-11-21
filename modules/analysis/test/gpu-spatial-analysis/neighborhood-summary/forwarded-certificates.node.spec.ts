// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import {GPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {
  CPUNeighborhoodSummaryOperation,
  computeNeighborhoodSummaryOnCPU
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('N2 pure CPU helper honors the certificate with zero-as-loss polarity and recovers', () => {
  const complete = Uint32Array.of(1);
  for (const certificate of [1, 0, 1]) {
    complete[0] = certificate;
    const result = computeNeighborhoodSummaryOnCPU(
      {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1),
        values: Float32Array.of(10, 20),
        status: {complete}
      },
      {statistics: ['count', 'mean']}
    );
    // One neighbor per row gives [count,mean]=[1,20],[1,10]; loss suppresses both.
    expect(Array.from(result.table)).toEqual(certificate ? [1, 20, 1, 10] : [NaN, NaN, NaN, NaN]);
  }
});

for (const Operation of [GPUNeighborhoodSummaryOperation, CPUNeighborhoodSummaryOperation]) {
  it(`N2 ${Operation.name} refuses a named complete certificate before lowering`, () => {
    const props = {
      id: 'certificate-summary',
      rowCount: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {complete: 'weightsComplete'}
      },
      values: 'values',
      statistics: ['mean'] as const,
      output: {table: 'table'}
    };
    expect(() => new Operation(props)).toThrow(/certificate-summary.*weights.status.complete/);
  });
}

for (const certificate of ['requiredCount', 'iterationLimitReached', 'converged'] as const) {
  for (const chunked of certificate === 'requiredCount' ? [false, true] : [true]) {
    it(`B4 refuses unsupported ${certificate} certificate: chunked=${chunked}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      try {
        const weights = {
          offsets: createTransientView(graph, 'offsets', 'uint32', 3),
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 1),
          weights: createTransientView(graph, 'weights', 'float32', 1)
        };
        const status = {[certificate]: createTransientView(graph, 'certificate', 'uint32', 1)};
        const supplied = chunked
          ? {...weights, chunks: [{weights: {...weights, status}}]}
          : {...weights, status};
        expect(
          () =>
            new GPUNeighborhoodSummary({
              id: 'certificate-summary',
              weights: supplied,
              values: createTransientView(graph, 'values', 'float32', 2),
              statistics: ['mean'],
              output: {table: createTransientView(graph, 'table', 'float32', 2)}
            })
        ).toThrow(new RegExp(`certificate-summary.*${certificate}`));
      } finally {
        device.destroy();
      }
    });
  }
}
