// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_MAP_MATCHING_PARAMETER_LENGTH,
  GPUMapMatching,
  getGPUMapMatchingMaximumPointCount,
  type GPUMapMatchingProps
} from '../../../src/gpu-network/map-matching/index';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  createNullWebGPUDevice,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(graph: GPUCommandGraph, pointCount = 6, trackCount = 1): GPUMapMatchingProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `map-matching-robustness-view-${serial++}`, format, length);
  return {
    id: 'map-matching',
    positions: view('float32x2', pointCount),
    trackOffsets: view('uint32', trackCount + 1),
    topology: {
      nodePositions: view('float32x2', 4),
      offsets: view('uint32', 5),
      neighbors: view('uint32', 8)
    },
    parameters: view('float32', GPU_MAP_MATCHING_PARAMETER_LENGTH),
    cellSize: 50,
    bounds: {minimum: [0, 0], maximum: [200, 200]},
    output: {matchedEdges: view('uint32', pointCount)}
  };
}

it('GPUMapMatching checks its per-point scratch against the storage binding limit', () => {
  const device = createNullWebGPUDevice();
  // The routes table takes 8 * 9 * 4 = 288 bytes per point at 8 candidates: 128 MiB / 288.
  const maximum = getGPUMapMatchingMaximumPointCount(device);
  expect(maximum).toBe(466033);
  expect(getGPUMapMatchingMaximumPointCount(device, 4)).toBe(Math.floor(134217728 / 80));
  // One candidate: the 20-byte hop table binds tighter than the 12-byte candidate table.
  expect(getGPUMapMatchingMaximumPointCount(device, 1, true)).toBe(Math.floor(134217728 / 20));
  const getNodes = (pointCount: number, candidateCount?: number) => {
    const graph = new GPUCommandGraph(device, {id: `map-matching-scratch-${serial++}`});
    const props = createProps(graph, pointCount);
    return () => new GPUMapMatching({...props, candidateCount}).getCommandNodes(graph);
  };
  expect(getNodes(maximum)).not.toThrow();
  expect(getNodes(maximum + 1)).toThrow(/positions rows must be at most 466033/);
  expect(getNodes(maximum + 1, 4)).not.toThrow();
});

it('GPUMapMatching validates minimumFixSpacingSigmas and traversedEdges', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-validation'});
  const base = createProps(graph);
  const view = <Format extends 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `map-matching-validation-${serial++}`, format, length);
  expect(() => new GPUMapMatching({...base, minimumFixSpacingSigmas: -1})).toThrow(
    /minimumFixSpacingSigmas must be a non-negative number/
  );
  const traversedEdges = {edges: view('uint32', 8), counts: view('uint32', 1)};
  const withOutput = (overrides: Partial<GPUMapMatchingProps>) => () =>
    new GPUMapMatching({
      ...base,
      output: {...base.output, traversedEdges},
      ...overrides
    });
  expect(withOutput({})).toThrow(/traversedEdgeCapacity must be an integer/);
  expect(withOutput({traversedEdgeCapacity: 4})).toThrow(
    /output\.traversedEdges\.edges must contain trackCount \* traversedEdgeCapacity rows/
  );
  expect(() => withOutput({traversedEdgeCapacity: 8})()).not.toThrow();
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        traversedEdgeCapacity: 8,
        output: {
          ...base.output,
          traversedEdges: {...traversedEdges, counts: view('uint32', 2)}
        }
      })
  ).toThrow(/output\.traversedEdges\.counts must contain 1 rows/);
  expect(
    () =>
      new GPUMapMatching({
        ...base,
        traversedEdgeCapacity: 8,
        output: {
          ...base.output,
          traversedEdges: {...traversedEdges, status: {overflow: view('uint32', 2)}}
        }
      })
  ).toThrow(/output\.traversedEdges\.status\.overflow must contain 1 rows/);
});

it('GPUMapMatching without the new options keeps its node list', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'map-matching-plain-nodes'});
  const ids = new GPUMapMatching(createProps(graph)).getCommandNodes(graph).map(node => node.id);
  expect(ids.some(id => /fix-filter|predecessors|hops|route-edges/.test(id))).toBe(false);
  expect(ids).toContain('map-matching-clear-starts');
});

it('GPUMapMatching fix filter and traversed route kernels parse and stay within the limits', () => {
  for (const withTable of [false, true]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
      id: `map-matching-new-kernels-${withTable}`
    });
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `map-matching-new-kernels-${serial++}`, format, length);
    const base = createProps(graph);
    const nodes = new GPUMapMatching({
      ...base,
      minimumFixSpacingSigmas: 2,
      traversedEdgeCapacity: 16,
      routeNodeBudget: 128,
      ...(withTable
        ? {
            routeTable: {
              offsets: view('uint32', 5),
              targets: view('uint32', 16),
              costs: view('float32', 16),
              costLimit: view('float32', 1)
            }
          }
        : {}),
      output: {
        ...base.output,
        trackLogLikelihoods: view('float32', 1),
        breaks: view('uint32', 6),
        traversedEdges: {edges: view('uint32', 16), counts: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(ids).toContain('map-matching-fix-filter');
    expect(ids).not.toContain('map-matching-starts');
    expect(ids.indexOf('map-matching-hops')).toBeLessThan(ids.indexOf('map-matching-route-edges'));
    const sources = getParsedKernelSources(nodes);
    for (const [id, source] of sources) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
    expect(sources.get('map-matching-routes')).toContain(
      'predecessors[predecessorsOffset + point]'
    );
    expect(sources.get('map-matching-routes')).not.toContain('pointStarts');
    expect(sources.get('map-matching-forward')).toContain('DROPPED_FIX');
    expect(sources.get('map-matching-route-edges')).toContain('tableParent');
    for (const id of ['map-matching-route-edges', 'map-matching-routes']) {
      expect(getPrivateMemoryBytes(sources.get(id) ?? ''), id).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
    }
  }
});
