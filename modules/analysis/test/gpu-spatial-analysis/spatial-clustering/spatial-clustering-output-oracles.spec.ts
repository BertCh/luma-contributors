// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDBSCAN,
  getGPUDBSCANParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createPoisonedOracleGraph, runPoisonedOracleGraph} from './output-oracle-test-utils';

it('publishes zero candidate loss independently of optional cluster summaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two disconnected pairs, self-inclusive minPoints=2: labels 0,0,1,1. The exact
  // grid route loses no candidates even when the *cluster output* capacity is one.
  for (const candidatePresent of [true, false]) {
    for (const rowCount of [4, 0]) {
      const harness = createPoisonedOracleGraph(
        device,
        {
          positions: {
            format: 'float32x2',
            length: rowCount,
            values: Float32Array.of(0, 0, 1, 0, 8, 0, 9, 0).slice(0, rowCount * 2)
          },
          parameters: {
            format: 'float32',
            length: 8,
            values: getGPUDBSCANParameterValues({
              bounds: [-1, -1, 10, 1],
              epsilon: 1.5,
              minimumSamples: 2
            })
          }
        },
        {
          labels: {format: 'uint32', length: rowCount},
          ids: {format: 'uint32', length: 1},
          count: {format: 'uint32', length: 1},
          overflow: {format: 'uint32', length: 1},
          ...(candidatePresent ? {candidateOverflow: {format: 'uint32' as const, length: 1}} : {})
        }
      );
      harness.graph.add(
        new GPUDBSCAN({
          positions: harness.getView('positions'),
          parameters: harness.getView('parameters'),
          gridSize: [4, 1],
          output: {
            labels: harness.getView('labels'),
            clusters: {
              ids: harness.getView('ids'),
              status: {
                count: harness.getView('count'),
                overflow: harness.getView('overflow'),
                ...(candidatePresent
                  ? {candidateOverflow: harness.getView<'uint32'>('candidateOverflow')}
                  : {})
              }
            }
          }
        })
      );
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        expect(result['count']).toEqual([rowCount ? 1 : 0]);
        expect(result['labels']).toEqual(rowCount ? [0, 0, 1, 1] : []);
        expect(result['overflow']).toEqual([rowCount ? 1 : 0]);
        if (rowCount) expect(result['ids']).toEqual([0]);
        if (candidatePresent) expect(result['candidateOverflow']).toEqual([0]);
      }
    }
  }
});

it('GPU clustering keeps labels and original flag after a caller redirects candidate status into labels', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createPoisonedOracleGraph(
    device,
    {
      positions: {format: 'float32x2', length: 4, values: Float32Array.of(0, 0, 1, 0, 8, 0, 9, 0)},
      parameters: {
        format: 'float32',
        length: 8,
        values: getGPUDBSCANParameterValues({
          bounds: [-1, -1, 10, 1],
          epsilon: 1.5,
          minimumSamples: 2
        })
      }
    },
    {
      labels: {format: 'uint32', length: 4},
      ids: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      candidateOverflow: {format: 'uint32', length: 1}
    }
  );
  const view = harness.getView;
  const status = {
    count: view<'uint32'>('count'),
    overflow: view<'uint32'>('overflow'),
    candidateOverflow: view<'uint32'>('candidateOverflow')
  };
  const clusters = {ids: view<'uint32'>('ids'), status};
  const output = {labels: view<'uint32'>('labels'), clusters};
  const contributor = new GPUDBSCAN({
    positions: view('positions'),
    parameters: view('parameters'),
    gridSize: [4, 1],
    output
  });
  status.candidateOverflow = harness.graph.createDataView(output.labels.buffer, {
    format: 'uint32',
    byteOffset: 8,
    length: 1
  });
  clusters.status = {...status};
  output.clusters = {...clusters};
  harness.graph.add(contributor);
  for (const result of await runPoisonedOracleGraph(harness, device)) {
    // Disconnected self-inclusive pairs have labels [0,0,1,1], no candidate loss.
    expect(result['labels']).toEqual([0, 0, 1, 1]);
    expect(result['candidateOverflow']).toEqual([0]);
    expect(result['count']).toEqual([2]);
    expect(result['ids']).toEqual([0, 2]);
    expect(result['overflow']).toEqual([0]);
  }
});
