// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {addLocalVariabilityRecipe} from '../../../src/gpu-spatial-analysis/recipes/local-variability-recipe';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {CPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-contiguity-weights-operation';
import {GPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {CPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {createPairStarvedContiguityFixture} from './contiguity-loss.test-utils';

function vector(format: 'float32x2' | 'float32' | 'uint32', values: Float32Array | Uint32Array) {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

it('final3 local variability retains every declared bounded-status view in both returned ports', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = (format: 'uint32' | 'float32', length: number) =>
    createTransientView(graph, `h3-${serial++}`, format, length);
  try {
    const status = Object.fromEntries(
      [
        'overflow',
        'candidateOverflow',
        'count',
        'requiredCount',
        'invalidCount',
        'approximation',
        'uncertainCount',
        'iterationCount',
        'converged',
        'iterationLimitReached'
      ].map(name => [name, view('uint32', 1)])
    );
    const result = addLocalVariabilityRecipe(graph, {
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 6),
        weights: view('float32', 6),
        status
      },
      values: view('float32', 4),
      parameters: view('float32', 8)
    } as never);
    for (const [name, value] of Object.entries(status)) {
      expect(result.intermediates.weights.status?.[name as 'overflow'] === value, name).toBe(true);
      expect(
        result.status.stages.find(stage => stage.stage === 'weights')?.status[
          name as 'overflow'
        ] === value,
        name
      ).toBe(true);
    }
  } finally {
    device.destroy();
  }
});

for (const Backend of [CPUContiguityWeightsOperation, GPUContiguityWeightsOperation])
  it(`final3 ${Backend.name} rejects duplicate final and candidate output names`, () => {
    expect(
      () =>
        new Backend({
          criterion: 'queen',
          featureCount: 2,
          capacity: 1,
          pairCapacity: 8,
          polygons: {positions: 'p', ringOffsets: 'r', polygonOffsets: 'o', featureOffsets: 'f'},
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            status: {overflow: 'final', candidateOverflow: 'final'}
          }
        })
    ).toThrow(/output.*candidateOverflow|already|same|unique/);
  });

it('final3 distinct named contiguity loss outputs retain final-only truncation', async () => {
  const fixture = createPairStarvedContiguityFixture();
  const operation = new CPUContiguityWeightsOperation({
    criterion: 'queen',
    featureCount: 2,
    capacity: 1,
    pairCapacity: 8,
    polygons: {positions: 'p', ringOffsets: 'r', polygonOffsets: 'o', featureOffsets: 'f'},
    output: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {overflow: 'final', candidateOverflow: 'candidate'}
    }
  });
  const inputs = Object.fromEntries(
    Object.entries(fixture.input).map(([name, values]) => [
      name,
      vector(name === 'positions' ? 'float32x2' : 'uint32', values)
    ])
  );
  const result = await operation.execute({inputs} as Parameters<typeof operation.execute>[0]);
  expect(result).toHaveProperty('final', Uint32Array.of(1));
  expect(result).toHaveProperty('candidate', Uint32Array.of(0));
});

it('final3 named GPU SKATER resolves and captures both loss ports during lowering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const status = {overflow: 'final', candidateOverflow: 'candidate'};
    const props = {
      rowCount: 2,
      edgeCount: 2,
      maximumRegionCount: 2,
      weights: {offsets: 'o', neighbors: 'n', weights: 'w', status},
      values: 'v',
      treeEdgeFlags: 't',
      componentLabels: 'c',
      parameters: 'p',
      output: {labels: 'l'}
    };
    const operation = new GPUSkaterRegionsOperation(props);
    status.overflow = 'mutated';
    const views = Object.fromEntries(
      [
        'offsets',
        'neighbors',
        'weights',
        'values',
        'treeEdgeFlags',
        'componentLabels',
        'parameters',
        'labels',
        'weightsOverflow',
        'weightsCandidateOverflow'
      ].map(name => [
        name,
        createTransientView(
          graph,
          name,
          name === 'values' || name === 'weights' ? 'float32' : 'uint32',
          name === 'offsets' ? 3 : name.startsWith('weights') && name !== 'weights' ? 1 : 2
        )
      ])
    );
    const contributor = operation.createContributor(
      name => views[name] as never
    ) as import('../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions').GPUSkaterRegions;
    expect(operation.inputs['weightsOverflow']).toBe('final');
    expect(contributor.props.weights.status?.overflow).toBe(views['weightsOverflow']);
    expect(contributor.props.weights.status?.candidateOverflow).toBe(
      views['weightsCandidateOverflow']
    );
    expect(
      contributor.getCommandNodes(graph).some(node => node.id.includes('-incomplete-labels'))
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const cause of ['overflow', 'candidateOverflow'] as const)
  it(`final3 named SKATER propagates ${cause} before traversing missing components and recovers`, async () => {
    const props = {
      rowCount: 2,
      edgeCount: 2,
      maximumRegionCount: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {[cause]: 'loss'}
      },
      values: 'values',
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      parameters: 'parameters',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const operation = new CPUSkaterRegionsOperation(props as never);
    expect(Object.values(new GPUSkaterRegionsOperation(props as never).inputs)).toContain('loss');
    for (const flag of [0, 1, 0]) {
      const inputs = Object.fromEntries(
        Object.entries({
          offsets: Uint32Array.of(0, 1, 2),
          neighbors: Uint32Array.of(1, 0),
          weights: Float32Array.of(1, 1),
          values: Float32Array.of(0, 2),
          treeEdgeFlags: Uint32Array.of(flag ? 0 : 1, 0),
          componentLabels: flag ? Uint32Array.of(0xffffffff, 0xffffffff) : Uint32Array.of(0, 0),
          parameters: Uint32Array.of(2, 1),
          loss: Uint32Array.of(flag)
        }).map(([name, values]) => [
          name,
          vector(values instanceof Float32Array ? 'float32' : 'uint32', values)
        ])
      );
      // CPU execute receives resolved port keys, as the mixed executor does.
      inputs[cause === 'overflow' ? 'weightsOverflow' : 'weightsCandidateOverflow'] =
        inputs['loss'];
      const result = await operation.execute({inputs} as Parameters<typeof operation.execute>[0]);
      expect(result).toHaveProperty(
        'labels',
        Uint32Array.from(flag ? [0xffffffff, 0xffffffff] : [0, 1])
      );
      expect(result).toHaveProperty('regionCount', Uint32Array.of(flag ? 0 : 2));
      expect(result).toHaveProperty('cutGains', Float32Array.of(flag ? NaN : 2));
    }
  });
