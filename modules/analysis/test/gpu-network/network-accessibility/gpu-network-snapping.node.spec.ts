// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {computeNetworkCostMatrixOnCPU} from '../../../src/gpu-network/network-accessibility/cpu-network-cost-matrix-operation';
import {GPUNetworkSnapping} from '../../../src/gpu-network/network-accessibility/gpu-network-snapping';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it.each([
  false,
  true
])('directed intersection-aware snapping uses endpoint fractions for zero-cost seeds (reverse=%s)', reverse => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `snap-${serial++}`, format, length);
  const contributor = new GPUNetworkSnapping({
    id: 'snap',
    positions: view('float32x2', 1),
    startDirection: reverse ? 'reverse' : 'forward',
    intersectionAware: true,
    topology: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 2),
      weights: view('float32', 2),
      nodePositions: view('float32x2', 3)
    },
    output: {
      snappedEdges: view('uint32', 1),
      startNodes: view('uint32', 2),
      startCosts: view('float32', 2)
    }
  });
  const source = getParsedKernelSources(contributor.getCommandNodes(graph)).get('snap-seeds');
  expect(source).toContain('fraction == 0.0');
  expect(source).toContain('fraction == 1.0');
  const seedNodes = new Uint32Array(2);
  const seedCosts = new Float32Array(2);
  const storage: Record<string, Uint32Array | Float32Array> = {
    fractions: Float32Array.of(reverse ? 0.5 : 0),
    snappedEdges: Uint32Array.of(0),
    edgeSources: Uint32Array.of(0, 0),
    edgeTargets: Uint32Array.of(1, 2),
    sourceCosts: Float32Array.of(reverse ? 5 : 0),
    targetCosts: Float32Array.of(reverse ? 5 : 10),
    seedNodes,
    seedCosts
  };
  const record = getWGSLKernelNodeRecord(graph, 'snap-seeds')!;
  const bindings = Object.fromEntries(
    record.bindings.map((binding, index) => [index, storage[binding.name]])
  );
  new WgslExec(
    new WgslParser().parse(record.source.replace(/^\s*_\s*=[^;]*;/gm, ''))
  ).dispatchWorkgroups('main', [1, 1, 1], {0: bindings}, {constants: record.constants});
  expect(Array.from(seedNodes)).toEqual(reverse ? [0, 0xffffffff] : [0, 1]);
  expect(Array.from(seedCosts)).toEqual(reverse ? [5, -1] : [0, 10]);
  const matrix = computeNetworkCostMatrixOnCPU(
    {
      topology: {
        offsets: Uint32Array.from(reverse ? [0, 0, 1, 2] : [0, 2, 2, 2]),
        neighbors: Uint32Array.from(reverse ? [0, 0] : [1, 2]),
        weights: Float32Array.of(10, 1)
      },
      startNodes: seedNodes,
      startCosts: seedCosts
    },
    {rowCount: 1, startsPerRow: 2}
  );
  // Hand routes: reverse midpoint can reach only source 0; forward intersection can take
  // either outgoing arc, including 0->2=1. The stored seed order is source then target.
  expect(Array.from(matrix.costs)).toEqual(reverse ? [5, Infinity, Infinity] : [0, 10, 1]);
});

it.each([
  false,
  true
])('preserves a reverse start at exact target coordinates under reciprocal evaluation (indexed=%s, B2)', indexed => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `endpoint-${serial++}`, format, length);
  const contributor = new GPUNetworkSnapping({
    id: 'endpoint',
    positions: view('float32x2', 1),
    startDirection: 'reverse',
    intersectionAware: true,
    topology: {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 1),
      weights: view('float32', 1),
      nodePositions: view('float32x2', 2)
    },
    candidateCapacity: indexed ? 1 : undefined,
    maximumSnapDistance: indexed ? view('float32', 1) : undefined,
    output: {
      snappedEdges: view('uint32', 1),
      snapFractions: view('float32', 1),
      startNodes: view('uint32', 2),
      startCosts: view('float32', 2),
      status: indexed ? {overflow: view('uint32', 1)} : undefined
    }
  });
  contributor.getCommandNodes(graph);
  const storage: Record<string, Uint32Array | Float32Array> = {
    points: Float32Array.of(11, 0),
    nodePositions: Float32Array.of(0, 0, 11, 0),
    edgeSources: Uint32Array.of(0),
    edgeTargets: Uint32Array.of(1),
    joinRows: Uint32Array.of(0),
    snappedEdges: Uint32Array.of(99),
    fractions: Float32Array.of(-1),
    distances: Float32Array.of(-1),
    edgeCosts: Float32Array.of(11),
    sourceCosts: Float32Array.of(-1),
    targetCosts: Float32Array.of(-1),
    seedNodes: new Uint32Array(2),
    seedCosts: new Float32Array(2)
  };
  const run = (suffix: string) => {
    const record = getWGSLKernelNodeRecord(graph, `endpoint-${suffix}`)!;
    // Exercise the permitted Metal lowering a/b -> a*f32(1/b). Here b=11^2=121.
    // The f32 product is 0.9999999403953552, although the input point is exactly end.
    const source = record.source
      .replace(
        'dot(point - start, segment) / denominator',
        `dot(point - start, segment) * ${Math.fround(1 / 121)}`
      )
      .replace(/^\s*_\s*=[^;]*;/gm, '');
    const bindings = Object.fromEntries(
      record.bindings.map((binding, index) => [index, storage[binding.name]])
    );
    new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
      'main',
      [1, 1, 1],
      {0: bindings},
      {constants: record.constants}
    );
  };
  run(indexed ? 'project' : 'scan');
  run('costs');
  run('seeds');
  expect(Array.from(storage['fractions'])).toEqual([1]);
  expect(Array.from(storage['seedNodes'])).toEqual([0, 1]);
  expect(Array.from(storage['seedCosts'])).toEqual([11, 0]);
  const result = computeNetworkCostMatrixOnCPU(
    {
      topology: {
        offsets: Uint32Array.of(0, 0, 1),
        neighbors: Uint32Array.of(0),
        weights: Float32Array.of(11)
      },
      startNodes: storage['seedNodes'] as Uint32Array,
      startCosts: storage['seedCosts'] as Float32Array
    },
    {rowCount: 1, startsPerRow: 2}
  );
  expect(Array.from(result.costs)).toEqual([11, 0]);
});
