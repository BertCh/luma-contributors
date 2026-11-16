// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUGaussianPredictiveScores} from '../../../src/gpu-spatial-analysis/scoring-rules/gpu-gaussian-predictive-scores';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
let nextPropsId = 0;
function createProps(graph: GPUCommandGraph, lengths: number[], padded = false) {
  const propsId = nextPropsId++;
  const rows = (name: string) =>
    createVectorView(
      name,
      'float32',
      lengths.map((length, chunkIndex) => {
        const byteOffset = padded ? (chunkIndex % 63) * 4 : 0;
        const backing = createTransientView(
          graph,
          `${name}-${propsId}-${chunkIndex}`,
          'float32',
          Math.max(length, 1) + byteOffset / 4
        );
        return graph.createDataView(backing.buffer, {format: 'float32', length, byteOffset});
      })
    );
  return {
    id: 'chunk-scores',
    means: rows('means'),
    variances: rows('variances'),
    observations: rows('observations'),
    output: {crps: rows('crps')}
  };
}

it('validates row topology and writable ranges while preserving larger packed capacities', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  expect(new GPUGaussianPredictiveScores(props).estimatedCommandNodeCount).toBe(4);
  expect(
    () =>
      new GPUGaussianPredictiveScores({
        ...props,
        variances: createProps(graph, [2, 3, 2]).variances
      })
  ).toThrow(/^chunk-scores variances chunk 0.*topology/);
  expect(() => new GPUGaussianPredictiveScores({...props, output: {crps: props.means}})).toThrow(
    /share buffers/
  );
  const repeated = createProps(graph, [2, 2]);
  expect(
    () =>
      new GPUGaussianPredictiveScores({
        ...repeated,
        output: {
          crps: createVectorView('repeated', 'float32', [
            repeated.output.crps.data[0],
            repeated.output.crps.data[0]
          ])
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  const capacity = new GPUGaussianPredictiveScores({
    means: createTransientView(graph, 'packed-means', 'float32', 3),
    variances: createTransientView(graph, 'packed-variances', 'float32', 5),
    observations: createTransientView(graph, 'packed-observations', 'float32', 6),
    output: {crps: createTransientView(graph, 'packed-crps', 'float32', 7)}
  });
  expect(capacity.getCommandNodes(graph)).toHaveLength(1);
});

it('checks a later aligned binding before any contributor graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUGaussianPredictiveScores(createProps(graph, [1, 33_554_432], true));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^chunk-scores means chunk 1 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

it('supports 40 million scores beyond one packed binding and skips empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUGaussianPredictiveScores(
    createProps(graph, [0, 10_000_000, 10_000_000, 10_000_000, 10_000_000, 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(5);
});

it('keeps one source and pipeline as chunk count and unaligned prefixes grow', () => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUGaussianPredictiveScores(
      createProps(
        graph,
        Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
        true
      )
    );
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBe(1);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(Object.keys(constants)).toHaveLength(0);
      const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
    }
  }
  expect(keys.size).toBe(1);
});

it('prefixes chunk scratch resource IDs so two contributors can share one graph', () => {
  const graph = new GPUCommandGraph(device);
  const first = new GPUGaussianPredictiveScores({...createProps(graph, [1, 4, 2]), id: 'first'});
  const second = new GPUGaussianPredictiveScores({...createProps(graph, [1, 4, 2]), id: 'second'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const firstNodes = first.getCommandNodes(graph);
  const secondNodes = second.getCommandNodes(graph);
  expect(allocate.mock.calls.map(([props]) => props.id)).toEqual(['first-chunks', 'second-chunks']);
  const nodeIds = [...firstNodes, ...secondNodes].map(node => node.id);
  expect(new Set(nodeIds).size).toBe(nodeIds.length);
  expect(firstNodes.every(node => node.id.startsWith('first-'))).toBe(true);
  expect(secondNodes.every(node => node.id.startsWith('second-'))).toBe(true);
});
