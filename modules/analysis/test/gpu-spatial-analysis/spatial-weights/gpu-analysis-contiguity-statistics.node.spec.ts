// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUAnalysisContiguityStatistics,
  GPU_ANALYSIS_CONTIGUITY_STATISTICS_MAXIMUM_FEATURE_COUNT,
  decodeGPUAnalysisContiguityStatistics,
  type GPUAnalysisContiguityStatisticsProps
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {getGPUAnalysisOperator} from '../../../src/gpu-spatial-analysis/ontology/index';
import {createNullWebGPUDevice, createVectorView} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  featureCount = 4
): GPUAnalysisContiguityStatisticsProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `contiguity-test-${serial++}`, format, length);
  return {
    id: 'profile-contiguity',
    criterion: 'queen',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', Math.max(4, featureCount * 4)),
      ringOffsets: view('uint32', featureCount + 1),
      polygonOffsets: view('uint32', featureCount + 1),
      featureOffsets: view('uint32', featureCount + 1)
    },
    output: {
      weights: {
        offsets: view('uint32', featureCount + 1),
        neighbors: view('uint32', 16),
        weights: view('float32', 16),
        status: {
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1),
          candidateOverflow: view('uint32', 1)
        }
      },
      cardinality: view('uint32', featureCount),
      counts: view('uint32', 6),
      median: view('float32', 1),
      validity: view('uint32', 1)
    }
  };
}

it('validates profile capacities, required status and snap precision at construction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    const construct = (overrides: Partial<GPUAnalysisContiguityStatisticsProps>) =>
      new GPUAnalysisContiguityStatistics({...props, ...overrides});
    expect(() => construct({})).not.toThrow();
    expect(() => construct({criterion: 'bishop' as never})).toThrow(/profile-contiguity criterion/);
    expect(() => construct({pairCapacity: 0})).toThrow(/profile-contiguity pairCapacity/);
    expect(() => construct({snapTolerance: -1})).toThrow(/profile-contiguity snapTolerance/);
    expect(() => construct({snapTolerance: 0.25})).toThrow(/coordinateExtent is required/);
    expect(() => construct({snapTolerance: 0.25, coordinateExtent: 2 ** 24})).toThrow(
      /snapTolerance/
    );
    expect(() => construct({snapTolerance: 0.25, coordinateExtent: 4})).not.toThrow();
    expect(() => construct({snapTolerance: 0.25, coordinateExtent: NaN})).toThrow(
      /coordinateExtent/
    );
    expect(() => construct({output: {...props.output, cardinality: props.output.counts}})).toThrow(
      /cardinality must have 4 rows/
    );
    expect(() => construct({output: {...props.output, counts: props.output.validity}})).toThrow(
      /counts.length/
    );
    expect(() =>
      construct({
        output: {
          ...props.output,
          validity: createTransientView(graph, 'empty-validity', 'uint32', 0)
        }
      })
    ).toThrow(/validity.length/);
    expect(() =>
      construct({output: {...props.output, median: props.output.validity as never}})
    ).toThrow(/output.median/);
    expect(() =>
      construct({
        output: {
          ...props.output,
          weights: {...props.output.weights, status: {overflow: props.output.validity} as never}
        }
      })
    ).toThrow(/requiredCount and candidateOverflow/);
    expect(() => new GPUAnalysisContiguityStatistics(createProps(graph, 0))).toThrow();
    expect(GPU_ANALYSIS_CONTIGUITY_STATISTICS_MAXIMUM_FEATURE_COUNT).toBe(2 ** 24);
    expect(() => new GPUAnalysisContiguityStatistics(createProps(graph, 2 ** 24 + 1))).toThrow(
      /featureCount/
    );
    expect(() =>
      construct({
        polygons: {
          ...props.polygons,
          positions: createVectorView('chunked', 'float32x2', [
            props.polygons.positions as never
          ]) as never
        }
      })
    ).toThrow(/one packed view/);
  } finally {
    device.destroy();
  }
});

it('rejects output aliasing, foreign graphs and excessive storage before declaring nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    expect(
      () =>
        new GPUAnalysisContiguityStatistics({
          ...props,
          output: {...props.output, validity: props.output.weights.status.overflow}
        })
    ).toThrow(/share buffers/);
    expect(
      () =>
        new GPUAnalysisContiguityStatistics({
          ...props,
          output: {
            ...props.output,
            cardinality: graph.createDataView(props.polygons.featureOffsets.buffer, {
              format: 'uint32',
              length: 4
            })
          }
        })
    ).toThrow(/share buffers/);
    const otherGraph = new GPUCommandGraph(device);
    expect(() => new GPUAnalysisContiguityStatistics(props).getCommandNodes(otherGraph)).toThrow(
      /target graph/
    );
    const foreign = createProps(otherGraph);
    expect(() =>
      new GPUAnalysisContiguityStatistics({
        ...props,
        output: {...props.output, median: foreign.output.median}
      }).getCommandNodes(graph)
    ).toThrow(/target graph/);
    const excessive = createTransientView(
      graph,
      'excessive-counts',
      'uint32',
      Math.floor(device.limits.maxStorageBufferBindingSize / 4) + 1
    );
    expect(() =>
      new GPUAnalysisContiguityStatistics({
        ...props,
        output: {...props.output, counts: excessive}
      }).getCommandNodes(graph)
    ).toThrow(/input\/output binding/);
  } finally {
    device.destroy();
  }
});

it('delegates topology grow plans without claiming the truncated required count is exact', () => {
  const device = createNullWebGPUDevice();
  try {
    const contributor = new GPUAnalysisContiguityStatistics(
      createProps(new GPUCommandGraph(device))
    );
    const current = contributor.getWeightsCapacityPlan();
    expect(current.slots.capacity).toBeGreaterThanOrEqual(16);
    expect(current.pairs.capacity).toBe(64);
    const grown = contributor.getWeightsCapacityPlan({requiredCount: 100, candidateOverflow: true});
    expect(grown.slots.capacity).toBeGreaterThanOrEqual(100);
    expect(grown.pairs.capacity).toBeGreaterThanOrEqual(128);
    const explicit = new GPUAnalysisContiguityStatistics({...contributor.props, pairCapacity: 7});
    expect(explicit.getWeightsCapacityPlan().pairs.capacity).toBe(7);
    expect(explicit.getWeightsCapacityPlan({candidateOverflow: true}).pairs.capacity).toBe(14);
  } finally {
    device.destroy();
  }
});

it('rejects oversized pair scratch without mutating the graph and permits a same-ID retry', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const criterion of ['queen', 'rook'] as const) {
      const graph = new GPUCommandGraph(device);
      const props = {...createProps(graph), id: undefined, criterion};
      const maximumPairCapacity = Math.floor(device.limits.maxStorageBufferBindingSize / 4);
      const declareScratch = vi.spyOn(graph, 'createTransientBuffer');
      try {
        const rejected = new GPUAnalysisContiguityStatistics({
          ...props,
          pairCapacity: maximumPairCapacity + 1
        });
        expect(() => rejected.getCommandNodes(graph)).toThrow(
          /analysis-contiguity-statistics pairCapacity scratch/
        );
        expect(declareScratch).not.toHaveBeenCalled();
        expect(() =>
          graph.add(new GPUAnalysisContiguityStatistics({...props, pairCapacity: 64}))
        ).not.toThrow();
        expect(declareScratch).toHaveBeenCalled();
      } finally {
        declareScratch.mockRestore();
      }
    }
  } finally {
    device.destroy();
  }
});

it('refuses an exact summary for candidate-only truncation even with a stale successful validity', () => {
  expect(
    decodeGPUAnalysisContiguityStatistics({
      counts: [1, 1, 3, 0, 1, 3],
      median: [0],
      validity: [1],
      overflow: [0],
      candidateOverflow: [1]
    })
  ).toEqual({exact: false, overflow: false, candidateOverflow: true});
});

it('decodes hand-counted chain and island profiles and refuses every overflow combination', () => {
  // A--B--C and an island: cardinalities [1,2,1,0], sorted [0,1,1,2].
  const values = {
    counts: [4, 0, 1, 0, 2, 1],
    median: [1],
    validity: [1],
    overflow: [0],
    candidateOverflow: [0]
  };
  expect(decodeGPUAnalysisContiguityStatistics(values)).toEqual({
    exact: true,
    overflow: false,
    candidateOverflow: false,
    slots: 4,
    islands: 1,
    minimum: 0,
    median: 1,
    maximum: 2
  });
  // An even profile's median interpolates the middle two cardinalities, not a rounded rank.
  expect(
    decodeGPUAnalysisContiguityStatistics({...values, counts: [6, 0, 0, 1, 2, 0], median: [1.5]})
  ).toMatchObject({exact: true, median: 1.5});
  expect(
    decodeGPUAnalysisContiguityStatistics({...values, counts: [0, 0, 4, 0, 0, 4], median: [0]})
  ).toMatchObject({exact: true, islands: 4, minimum: 0, median: 0, maximum: 0});
  for (const overflow of [0, 1]) {
    for (const candidateOverflow of [0, 1]) {
      for (const validity of [0, 1]) {
        if (!overflow && !candidateOverflow && validity) continue;
        expect(
          decodeGPUAnalysisContiguityStatistics({
            ...values,
            overflow: [overflow],
            candidateOverflow: [candidateOverflow],
            validity: [validity]
          })
        ).toEqual({
          exact: false,
          overflow: Boolean(overflow),
          candidateOverflow: Boolean(candidateOverflow)
        });
      }
    }
  }
  for (const name of ['counts', 'median', 'validity', 'overflow', 'candidateOverflow'] as const) {
    expect(() => decodeGPUAnalysisContiguityStatistics({...values, [name]: []})).toThrow(
      new RegExp(`${name}.length`)
    );
  }
});

it('catalogues the profile as topological with an explicit limit and pending GPU evidence', () => {
  expect(getGPUAnalysisOperator('GPUAnalysisContiguityStatistics')).toMatchObject({
    inputs: ['polygons'],
    outputs: ['spatial-weights', 'attributes', 'summary'],
    coordinateModels: ['topological'],
    tier: 'scale',
    evidence: {status: 'partial'},
    limits: [{name: 'featureCount', maximum: 2 ** 24}]
  });
});
