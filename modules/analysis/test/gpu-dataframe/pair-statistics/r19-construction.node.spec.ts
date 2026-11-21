// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURipley} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('PTS-137 malformed pair inputs name the contributor prop actual size and enforced limit', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `validation-${serial++}`, format, length);
    const common = {
      id: 'ripley-checked',
      positions: view('float32x2', 3),
      parameters: view('float32', 8),
      gridSize: [4, 4] as const
    };
    expect(
      () =>
        new GPURipley({
          ...common,
          radiusCount: 2,
          output: {k: view('float32', 1), l: view('float32', 2), lMinusR: view('float32', 2)}
        })
    ).toThrow('ripley-checked output.k length 1 must be at least radiusCount (2)');
    expect(
      () =>
        new GPUPointPatternIndices({
          ...common,
          id: 'indices-checked',
          quadratGrid: [0, 4],
          output: {quadratCounts: view('uint32', 4)}
        })
    ).toThrow(/indices-checked quadratGrid.*65536/);
    expect(
      () =>
        new GPUPointPatternIndices({
          ...common,
          id: 'indices-checked',
          output: {nearestNeighborDistances: view('float32', 2)}
        })
    ).toThrow(
      'indices-checked output.nearestNeighborDistances length 2 must equal positions length (3)'
    );
    const marked = {
      id: 'marked-checked',
      positions: common.positions,
      parameters: common.parameters,
      radiusCount: 2,
      output: {
        statistics: view('float32', 8),
        pairCounts: view('uint32', 2),
        status: view('uint32', 2)
      }
    };
    expect(() => new GPUMarkedPointStatistics({...marked, marks: view('float32', 2)})).toThrow(
      'marked-checked: marks length 2 must equal positions length (3)'
    );
    expect(
      () =>
        new GPUMarkedPointStatistics({
          ...marked,
          output: {...marked.output, statistics: view('float32', 7)}
        })
    ).toThrow('marked-checked: output.statistics length 7 must equal radiusCount x 4 (8)');
    expect(
      () =>
        new GPUMarkedPointStatistics({
          ...marked,
          typePair: [0, 0x100000000],
          types: view('uint32', 3)
        })
    ).toThrow(/marked-checked: typePair\[1\].*4294967295/);
    for (const typePair of [[1], [1, 2, 3], [], [1, undefined], new Array(2)]) {
      expect(
        () =>
          new GPUMarkedPointStatistics({
            ...marked,
            types: view('uint32', 3),
            typePair: typePair as unknown as [number, number]
          })
      ).toThrow(/marked-checked: typePair/);
    }
  } finally {
    device.destroy();
  }
});
