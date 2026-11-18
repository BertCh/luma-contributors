// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPULocalMoran,
  GPUHotSpotAnalysis,
  GPULocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createSpatialAutocorrelationHarness} from './spatial-autocorrelation-harness';

it('the Moran harness passes the neighbor search status to the next consumer', () => {
  const device = createNullWebGPUDevice();
  const stop = new Error('inspected Moran inputs');
  const inspect = vi.spyOn(GPULocalMoran.prototype, 'getCommandNodes').mockImplementation(function (
    this: GPULocalMoran
  ) {
    expect(this.props.weights.status?.overflow).toBeDefined();
    throw stop;
  });
  try {
    expect(() =>
      createSpatialAutocorrelationHarness(device, {
        contributor: 'local-moran',
        scene: {values: new Float32Array([1, 2, 4, 8])},
        weights: {
          kind: 'neighbor-search',
          positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
          parameters: {bounds: [-1, -1, 4, 1], radius: 1.1},
          capacity: 1
        }
      })
    ).toThrow(stop);
  } finally {
    inspect.mockRestore();
    device.destroy();
  }
});

for (const Contributor of [GPULocalMoran, GPUHotSpotAnalysis, GPULocalGeary]) {
  for (const chunked of [false, true]) {
    it(`${Contributor.name} carries candidate-only loss through ${chunked ? 'chunked' : 'packed'} weights`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
        createTransientView(graph, `moran-loss-${serial++}`, format, length);
      const weights = {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 6),
        weights: view('float32', 6)
      };
      const candidateOverflow = view('uint32', 1);
      const status = {candidateOverflow};
      try {
        const output = {
          zScores: view('float32', 4),
          statistic: view('float32', 4),
          status: {candidateOverflow: view('uint32', 1)}
        };
        const inputWeights = chunked
          ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}], status}
          : {...weights, status};
        const props = {
          weights: inputWeights,
          values: view('float32', 4),
          parameters: view('float32', 8),
          output
        };
        const nodes = new Contributor(props).getCommandNodes(graph);
        const sources = getGPUCommandNodeKernelSources(nodes);
        const moments = sources.find(record => record.id.endsWith('-moments-correct'))!;
        expect(moments.source).toContain('candidateOverflow[candidateOverflowOffset] != 0u');
        expect(sources.some(record => record.id.endsWith('-candidate-overflow-status'))).toBe(true);
        for (const record of sources) {
          expect(getStorageBindingCount(record.source), record.id).toBeLessThanOrEqual(8);
        }
        expect(
          () =>
            new Contributor({
              ...props,
              output: {...output, status: {candidateOverflow}}
            })
        ).toThrow(/share buffers/);
        expect(
          () =>
            new Contributor({
              ...props,
              weights: {...inputWeights, status: {candidateOverflow: view('uint32', 0)}}
            })
        ).toThrow(/candidateOverflow/);
        const foreignGraph = new GPUCommandGraph(device);
        const foreignStatus = createTransientView(
          foreignGraph,
          'foreign-candidate-loss',
          'uint32',
          1
        );
        expect(() =>
          new Contributor({
            ...props,
            weights: {...inputWeights, status: {candidateOverflow: foreignStatus}}
          }).getCommandNodes(graph)
        ).toThrow(/target graph/);
      } finally {
        device.destroy();
      }
    });
  }
}
