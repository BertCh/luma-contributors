// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {createLocalPermutationReplay} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createReplayTest} from './local-permutation-test-fixture';

function createReplay(
  overrides: {nullDistributionLength?: number; rowLength?: number} = {},
  configuration: Parameters<typeof createReplayTest>[1] = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const test = createReplayTest(graph, configuration);
  const testNodes = test.getCommandNodes(graph);
  const replay = createLocalPermutationReplay({
    test,
    row: createTransientView(graph, 'row', 'uint32', overrides.rowLength ?? 1),
    output: {
      nullDistribution: createTransientView(
        graph,
        'null-distribution',
        'float32',
        overrides.nullDistributionLength ?? 99
      ),
      replayInfo: createTransientView(graph, 'replay-info', 'float32', 3)
    }
  });
  return {graph, test, testNodes, replay};
}

describe('GPULocalPermutationReplay', () => {
  it('distinguishes unsupported two-chunk weights from a test that has not been lowered', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    try {
      const weights = {
        rowCount: 12,
        chunks: [0, 6].map(rowStart => ({
          rowStart,
          rowCount: 6,
          weights: {
            offsets: createTransientView(graph, `chunk-${rowStart}-offsets`, 'uint32', 7),
            neighbors: createTransientView(graph, `chunk-${rowStart}-neighbors`, 'uint32', 12),
            weights: createTransientView(graph, `chunk-${rowStart}-weights`, 'float32', 12)
          }
        }))
      };
      const test = createReplayTest(graph, {weights});
      expect(test.getCommandNodes(graph).length).toBeGreaterThan(0);
      expect(() =>
        createLocalPermutationReplay({
          test,
          row: createTransientView(graph, 'pin', 'uint32', 1),
          output: {
            nullDistribution: createTransientView(graph, 'draws', 'float32', 99),
            replayInfo: createTransientView(graph, 'info', 'float32', 3)
          }
        }).getCommandNodes(graph)
      ).toThrow(/test\.weights.*whole-CSR/);
    } finally {
      graph.device.destroy();
    }
  });
  it('does not advertise an unsupported keySpace prop', () => {
    const source = readFileSync(
      new URL(
        '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-replay.ts',
        import.meta.url
      ),
      'utf8'
    );
    expect(source).not.toMatch(/keySpace\?:\s*string/);
  });
  it('snapshots nested output and input references before validating', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const test = createReplayTest(graph);
    test.getCommandNodes(graph);
    const row = createTransientView(graph, 'original-row', 'uint32', 1);
    const nullDistribution = createTransientView(graph, 'original-distribution', 'float32', 99);
    const replayInfo = createTransientView(graph, 'original-info', 'float32', 3);
    const props = {test, row, output: {nullDistribution, replayInfo}};
    const replay = createLocalPermutationReplay(props);
    props.output.nullDistribution = createTransientView(
      graph,
      'replacement-distribution',
      'float32',
      1
    );
    props.output.replayInfo = createTransientView(graph, 'replacement-info', 'float32', 3);
    const nodes = replay.getCommandNodes(graph);
    const writes = nodes[1].resources!.filter(resource => resource.usage === 'storage-read-write');
    expect(writes.map(resource => 'buffer' in resource && resource.buffer)).toContain(
      nullDistribution
    );
    expect(nullDistribution.length).toBe(99);
    props.row = createTransientView(graph, 'replacement-row', 'uint32', 1);
    const otherGraph = new GPUCommandGraph(createNullWebGPUDevice());
    props.test = createReplayTest(otherGraph, {id: 'unlowered'});
    expect(() => replay.getCommandNodes(graph)).not.toThrow();
    const setupResources = replay.getCommandNodes(graph)[0].resources!;
    expect(setupResources.some(resource => 'buffer' in resource && resource.buffer === row)).toBe(
      true
    );
    expect(
      setupResources.some(resource => 'buffer' in resource && resource.buffer === replayInfo)
    ).toBe(true);
    otherGraph.device.destroy();
    graph.device.destroy();
  });
  it('shares the permutation kernel text with the test and writes the null distribution', () => {
    const {graph, testNodes, replay} = createReplay();
    const replayNodes = replay.getCommandNodes(graph);
    expect(replayNodes.map(node => node.id)).toEqual([
      'golden-replay-setup',
      'golden-replay-permute'
    ]);
    const [testSource] = getGPUCommandNodeKernelSources(
      testNodes.filter(node => node.id === 'golden-permute')
    );
    const [, replaySource] = getGPUCommandNodeKernelSources(replayNodes);
    for (const shared of [
      'createPhiloxStream(key, index, permutation, RANDOM_TAG)',
      'nextPhiloxBelow(&stream, others - drawn)',
      'fn readSwap(',
      'let simulated = getLocalTerm(fx, fy, lagA, lagB);'
    ]) {
      expect(testSource.source).toContain(shared);
      expect(replaySource.source).toContain(shared);
    }
    expect(replaySource.source).toContain(
      'nullDistribution[nullDistributionOffset + permutation] = simulated;'
    );
    expect(replaySource.source).not.toContain('exceedances');
    expect(testSource.source).not.toContain('nullDistribution');
    expect(getStorageBindingCount(replaySource.source)).toBeLessThanOrEqual(8);
    graph.device.destroy();
  });

  it('emits WGSL that parses for every statistic', () => {
    for (const statistic of [
      'localMoran',
      'localGeary',
      'localG',
      'localGStar',
      'localBivariateMoran',
      'localLeeL',
      'localJoinCount'
    ] as const) {
      const {graph, replay} = createReplay({}, {statistic});
      const sources = getParsedKernelSources(replay.getCommandNodes(graph));
      expect(sources.size).toBe(2);
      graph.device.destroy();
    }
  });

  it('binds the negative-value count in the setup of G and G* only', () => {
    for (const [statistic, expected] of [
      ['localG', true],
      ['localGStar', true],
      ['localMoran', false]
    ] as const) {
      const {graph, replay} = createReplay({}, {statistic});
      const [setup] = getGPUCommandNodeKernelSources(replay.getCommandNodes(graph));
      expect(setup.source.includes('negativeCount[negativeCountOffset] == 0u'), statistic).toBe(
        expected
      );
      expect(setup.source.includes('var<storage, read> negativeCount: array<u32>'), statistic).toBe(
        expected
      );
      graph.device.destroy();
    }
  });

  it('requires a lowered test', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const test = createReplayTest(graph);
    const replay = createLocalPermutationReplay({
      test,
      row: createTransientView(graph, 'row', 'uint32', 1),
      output: {
        nullDistribution: createTransientView(graph, 'null-distribution', 'float32', 99),
        replayInfo: createTransientView(graph, 'replay-info', 'float32', 3)
      }
    });
    expect(() => replay.getCommandNodes(graph)).toThrow(
      /^golden-replay replay requires its test to have been lowered first/
    );
    graph.device.destroy();
  });

  it('rejects a row view that is not one element', () => {
    expect(() => createReplay({rowLength: 2})).toThrow(/row length/);
  });

  it('rejects a null distribution shorter than the maximum permutation count', () => {
    expect(() => createReplay({nullDistributionLength: 98})).toThrow(/nullDistribution length/);
  });
});
