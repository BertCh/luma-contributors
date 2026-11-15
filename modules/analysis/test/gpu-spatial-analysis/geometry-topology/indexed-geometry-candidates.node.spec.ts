// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly/index';
import {GPUMakeValid} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-make-valid';
import {GPUOffsetCurve} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createTopologyOutput} from '../geometry-refinement/topology-output';
import {getInlinedCallSites} from '../segment-intersection/wgsl-call-sites';

function createHarness() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `geometry-index-${serial++}`, format, length);
  return {graph, view};
}

it('ring shells use a spatially sorted BVH and preserve ring-index ties', () => {
  const {graph, view} = createHarness();
  const nodes = new GPUSegmentRingAssembly({
    endpoints: view('float32x4', 128),
    groups: view('uint32', 128),
    output: {
      positions: view('float32x2', 256),
      ringOffsets: view('uint32', 18),
      ringShells: view('uint32', 17),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(nodes.some(node => node.id.includes('shell-index-sort'))).toBe(true);
  expect(nodes.some(node => node.id.includes('shell-index-bvh'))).toBe(true);
  // All coordinate bits must survive sorting: one distant shell must not collapse nearby
  // centers into an unresolved Morton-key tie.
  expect(nodes.some(node => node.id.includes('shell-index-sort-low'))).toBe(true);
  expect(nodes.some(node => node.id.includes('shell-index-sort-high'))).toBe(true);
  const shellKeys = sources.find(({id}) => id.endsWith('-shell-index-keys'))!.source;
  expect(shellKeys).toContain('getOrderedKey(');
  expect(shellKeys).not.toContain('65535.0');
  const traversal = sources.find(({id}) => id.endsWith('-ring-shells'))!.source;
  expect(traversal).toContain('findShell(node, holeBounds)');
  expect(traversal).toContain('boundsContain(vec4f(minimum, maximum), holeBounds)');
  expect(traversal).toContain('area == bestArea && shell > best');
  expect(traversal).not.toContain('shell < rings; shell++');
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
  // The successor kernel used to inline two separate copies of the large exact predicate.
  const match = sources.find(({id}) => id.endsWith('-match'))!.source;
  expect(match).toContain('exactOrientSigns(');
  for (const [name, sites] of getInlinedCallSites(match, 'orientSign')) {
    expect(sites, name).toBe(1);
  }
});

it('make-valid indexes source groups and inclusive heights without scanning segment capacity', () => {
  const {graph, view} = createHarness();
  const nodes = new GPUMakeValid({
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      ringOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      featureOffsets: view('uint32', 3)
    },
    intersectionCapacity: 16,
    precision: {vertexTolerance: 1e-5, storage: 'float32', predicates: 'exact-or-uncertain'},
    output: createTopologyOutput(view)
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(nodes.some(node => node.id.includes('parity-group-sort'))).toBe(true);
  expect(nodes.some(node => node.id.includes('parity-bvh'))).toBe(true);
  const parity = sources.find(({id}) => id.endsWith('-face-parity'))!.source;
  expect(parity).not.toContain('segment < SEGMENT_CAPACITY; segment++');
  expect(parity).toContain('lowerBoundSortedGroups(0u, SEGMENT_CAPACITY, group)');
  expect(parity).toContain('upperBoundSortedGroups(groupFirst, SEGMENT_CAPACITY, group)');
  expect(parity).toContain('nodeFirst < groupLast && nodeLast > groupFirst');
  expect(parity).not.toContain('f32(group)');
  expect(parity).toContain('minimum.y <= max(a.y, b.y) && maximum.y >= min(a.y, b.y)');
  expect(parity).toContain('5u * segment + 4u] == group');
  for (const {id, source} of sources) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
  }
});

it('offset-curve generation reaches orientation only through the shared exact helper', () => {
  const {graph, view} = createHarness();
  const nodes = new GPUOffsetCurve({
    positions: view('float32x2', 3),
    lineOffsets: view('uint32', 2),
    geometryType: 'lines',
    joinStyle: 'bevel',
    parameters: view('float32', 4),
    output: {positions: view('float32x2', 6)}
  }).getCommandNodes(graph);
  const source = getGPUCommandNodeKernelSources(nodes)[0].source;
  expect(source).toContain('fn exactOrientSigns(');
  expect(() => new WgslReflect(source)).not.toThrow();
  for (const [name, sites] of getInlinedCallSites(source, 'orientSign')) {
    expect(sites, name).toBe(1);
  }
});

it.each([
  {limit: 80, ringCapacity: 8, binding: 'shell records'},
  {limit: 112, ringCapacity: 8, binding: 'shell bounds'},
  {limit: 100, ringCapacity: 5, binding: 'shell BVH nodes'}
])('ring $binding rejects a storage-binding budget before creating index nodes', ({
  limit,
  ringCapacity,
  binding
}) => {
  const {graph, view} = createHarness();
  graph.device.limits.maxStorageBufferBindingSize = limit;
  const contributor = new GPUSegmentRingAssembly({
    id: 'bounded-shells',
    endpoints: view('float32x4', 16),
    output: {
      positions: view('float32x2', 32),
      ringOffsets: view('uint32', ringCapacity + 1),
      ringShells: view('uint32', ringCapacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    new RegExp(`^bounded-shells ${binding}.*limit ${limit}`)
  );
});

it('make-valid rejects parity scratch that exceeds the storage-binding budget', () => {
  const {graph, view} = createHarness();
  graph.device.limits.maxStorageBufferBindingSize = 512;
  const contributor = new GPUMakeValid({
    id: 'bounded-parity',
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 8),
      ringOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      featureOffsets: view('uint32', 3)
    },
    intersectionCapacity: 16,
    precision: {vertexTolerance: 1e-5, storage: 'float32', predicates: 'exact-or-uncertain'},
    output: createTopologyOutput(view)
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^bounded-parity parity segment records.*limit 512/
  );
});
