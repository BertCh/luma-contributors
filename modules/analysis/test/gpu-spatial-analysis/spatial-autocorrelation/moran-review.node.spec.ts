// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPULocalMoran,
  GPUHotSpotAnalysis,
  GPULocalGeary
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {createHotSpotsCompute} from '../../../../../examples/deck/spatial-analysis-shared/hot-spots-analysis';

function createScene() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const weights = {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 16),
    weights: view('float32', 16)
  };
  const values = view('float32', 4);
  const parameters = view('float32', 8);
  const output = {
    zScores: view('float32', 4),
    statistic: view('float32', 4),
    summary: view('float32', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)
  };
  return {device, graph, view, weights, values, parameters, output};
}

for (const Contributor of [GPULocalMoran, GPUHotSpotAnalysis, GPULocalGeary]) {
  it(`R1 ${Contributor.name} refuses a loss flag hidden on the second chunk`, () => {
    const {device, view, values, parameters, output} = createScene();
    try {
      const chunk = () => ({
        offsets: view('uint32', 3),
        neighbors: view('uint32', 3),
        weights: view('float32', 3)
      });
      for (const name of ['overflow', 'candidateOverflow'] as const) {
        const weights = {
          rowCount: 4,
          chunks: [
            {rowStart: 0, rowCount: 2, weights: chunk()},
            {rowStart: 2, rowCount: 2, weights: {...chunk(), status: {[name]: view('uint32', 1)}}}
          ]
        };
        expect(() => new Contributor({weights, values, parameters, output})).toThrow(
          /chunks\[1\].*status.*outer/
        );
      }
    } finally {
      device.destroy();
    }
  });
}

for (const mode of ['tiled', 'reference'] as const) {
  it(`R1 global ${mode} refuses chunk-local candidate loss`, () => {
    const {device, view, values, output} = createScene();
    try {
      const weights = {
        rowCount: 4,
        chunks: [0, 2].map(rowStart => ({
          rowStart,
          rowCount: 2,
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 3),
            weights: view('float32', 3),
            status: rowStart ? {candidateOverflow: view('uint32', 1)} : undefined
          }
        }))
      };
      expect(
        () => new GPUGlobalSpatialStatistics({mode, weights, values, output, statistics: ['moran']})
      ).toThrow(/chunks\[1\].*status.*outer/);
    } finally {
      device.destroy();
    }
  });

  it(`R5 global ${mode} accepts the search's overflow view with spare capacity`, () => {
    const {device, graph, view, weights, values, output} = createScene();
    try {
      const search = new GPUNeighborSearch({
        mode: 'radius',
        gridSize: [2, 2],
        positions: view('float32x2', 4),
        parameters: view('float32', 12),
        output: {
          ...weights,
          status: {overflow: view('uint32', 2), candidateOverflow: view('uint32', 2)}
        }
      });
      expect(() =>
        new GPUGlobalSpatialStatistics({
          mode,
          weights: search.outputWeights,
          values,
          output,
          statistics: ['moran']
        }).getCommandNodes(graph)
      ).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

it('R2 neighbor search clears the candidate flag it exposes to Moran', () => {
  const {device, graph, view, weights} = createScene();
  try {
    const search = new GPUNeighborSearch({
      mode: 'radius',
      gridSize: [2, 2],
      positions: view('float32x2', 4),
      parameters: view('float32', 12),
      output: {
        ...weights,
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      }
    });
    const sources = getGPUCommandNodeKernelSources(search.getCommandNodes(graph));
    expect(
      sources.some(record =>
        /candidateOverflow\[candidateOverflowOffset\s*\+?\s*(?:index)?\]\s*=\s*0u/.test(
          record.source
        )
      ),
      'requested candidate flag has a writer'
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const Contributor of [GPULocalMoran, GPUHotSpotAnalysis]) {
  it(`R7 ${Contributor.name} marks neighbor counts unavailable behind the same loss gate`, () => {
    const {device, graph, view, weights, values, parameters, output} = createScene();
    try {
      const sources = getGPUCommandNodeKernelSources(
        new Contributor({
          weights: {...weights, status: {candidateOverflow: view('uint32', 1)}},
          values,
          parameters,
          output: {...output, neighborCounts: view('uint32', 4)}
        }).getCommandNodes(graph)
      );
      const neighbors = sources.find(record => record.id.includes('-neighbors'))!;
      expect(neighbors.source).toContain('var neighborCount = 0u');
      expect(neighbors.source).toContain('if (isFiniteFloat(centered))');
      expect(neighbors.source).toContain(
        'neighborCounts[neighborCountsOffset + index] = neighborCount'
      );
      const moments = sources.find(record => record.id.endsWith('-moments-correct'))!;
      expect(moments.source).toContain('residualMean = getQuietNaN(index)');
    } finally {
      device.destroy();
    }
  });
}

it('R3 predicate self-join exposes weights with a producer-owned candidate signal', () => {
  const {device, graph, view, weights} = createScene();
  try {
    const points = {kind: 'points' as const, positions: view('float32x2', 4)};
    const join = new GPUSpatialPredicateJoin({
      left: points,
      right: points,
      predicate: 'dwithin',
      distance: 1,
      candidateCapacity: 14,
      output: {
        weights: {...weights, status: {overflow: view('uint32', 1)}},
        status: {overflow: view('uint32', 1)}
      }
    });
    const outputWeights = (
      join as GPUSpatialPredicateJoin & {
        outputWeights?: typeof weights & {status?: {candidateOverflow?: unknown}};
      }
    ).outputWeights;
    expect(outputWeights?.status?.candidateOverflow).toBeDefined();
    const sources = getGPUCommandNodeKernelSources(join.getCommandNodes(graph));
    expect(
      sources.some(record =>
        record.source.includes(
          'weightsCandidateOverflow[weightsCandidateOverflowOffset] = select(0u, 1u, candidateIncomplete)'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

it('R4 the explorer statistic graph imports the search overflow buffer', async () => {
  const device = createNullWebGPUDevice();
  const stop = new Error('inspected explorer inputs');
  const compile = vi
    .spyOn(GPUCommandGraph.prototype, 'compileAsync')
    .mockResolvedValue({destroy() {}, encode() {}} as never);
  const inspect = vi
    .spyOn(GPUHotSpotAnalysis.prototype, 'getCommandNodes')
    .mockImplementation(function (this: GPUHotSpotAnalysis) {
      expect(this.props.weights.status?.overflow).toBeDefined();
      throw stop;
    });
  try {
    await expect(
      createHotSpotsCompute(
        device,
        {
          positions: new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]),
          values: new Float32Array([1, 2, 4, 8]),
          spatialContext: {coordinateSpace: 'planar', metric: 'native'}
        },
        {} as never
      )
    ).rejects.toThrow(stop);
  } finally {
    inspect.mockRestore();
    compile.mockRestore();
    device.destroy();
  }
});

it('R6 contiguity reports intermediate pair loss separately from final overflow', () => {
  const {device, graph, view} = createScene();
  try {
    const producer = new GPUContiguityWeights({
      criterion: 'queen',
      pairCapacity: 1,
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 8),
        ringOffsets: view('uint32', 3),
        polygonOffsets: view('uint32', 3),
        featureOffsets: view('uint32', 3)
      },
      output: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 8),
        weights: view('float32', 8),
        status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
      }
    });
    const sources = getGPUCommandNodeKernelSources(producer.getCommandNodes(graph));
    expect(
      sources.some(record =>
        record.source.includes('total > CAPACITY || pairTotal > PAIR_CAPACITY')
      )
    ).toBe(false);
    expect(
      sources.some(record =>
        record.source.includes(
          'candidateOverflow[candidateOverflowOffset] = select(0u, 1u, pairTotal > PAIR_CAPACITY)'
        )
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});
