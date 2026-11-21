// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from '../spatial-validation/spatial-design-audit.test-utils';
import {
  MERGE_GLOBAL_QUOTIENTS,
  MERGE_LOCAL_QUOTIENTS
} from './categorical-merge-fixture.test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it.each([
  'colocation',
  'dependence'
] as const)('AMG merge %s admits live flags and suppresses only logical prefixes', kind => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `merge-loss-${serial++}`, format, length);
  const overflow = view('uint32', 1);
  const candidateOverflow = view('uint32', 1);
  const requiredCount = view('uint32', 1);
  const complete = view('uint32', 1);
  const weights = {
    offsets: view('uint32', 4),
    neighbors: view('uint32', 3),
    weights: view('float32', 3),
    status: {overflow, candidateOverflow, requiredCount, complete}
  };
  const categories = view('uint32', 3);
  const categoryRowCounts = view('uint32', 4);
  // Three rows and two categories: 4 global, 6 local, or 18 statistics words.
  // Two extra words in every output must survive, including during loss.
  const results: [GraphDataView<'float32'>, number][] =
    kind === 'colocation'
      ? [
          [view('float32', 6), 4],
          [view('float32', 8), 6]
        ]
      : [[view('float32', 20), 18]];
  const construct = (
    status: Partial<typeof weights.status> & {
      converged?: GraphDataView<'uint32'>;
      iterationLimitReached?: GraphDataView<'uint32'>;
      unresolvedCount?: GraphDataView<'uint32'>;
    } = weights.status
  ) =>
    kind === 'colocation'
      ? new GPUColocationQuotient({
          weights: {...weights, status},
          categories,
          categoryCount: 2,
          output: {
            globalQuotients: results[0][0],
            localQuotients: results[1][0],
            categoryRowCounts
          }
        })
      : new GPULocalCategoryDependence({
          weights: {...weights, status},
          categories,
          categoryCount: 2,
          output: {
            statistics: results[0][0],
            categoryRowCounts
          }
        });
  try {
    for (const status of [{}, {overflow}, {candidateOverflow}, {overflow, requiredCount}]) {
      expect(() => construct(status)).not.toThrow();
    }
    expect(() => construct({requiredCount, candidateOverflow})).toThrow(
      /requiredCount requires weights.status.overflow/
    );
    expect(() => construct({converged: complete})).toThrow(/weights.status.*converged/);
    expect(() => construct({iterationLimitReached: overflow})).toThrow(
      /weights.status.*iterationLimitReached/
    );
    expect(() => construct({overflow, unresolvedCount: view('uint32', 1)})).toThrow(
      new RegExp(
        `^${kind === 'colocation' ? 'colocation-quotient' : 'local-category-dependence'} weights\\.status\\.unresolvedCount.*omit`
      )
    );
    const allNodes = construct().getCommandNodes(graph);
    const nodes = allNodes.filter(node => node.id.includes('-incomplete-'));
    const countNodes = allNodes.filter(
      node => /-(clear-rows|count-rows)$/.test(node.id) || node.id === 'colocation-quotient-shares'
    );
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2, 3));
    memory.write(weights.neighbors, Uint32Array.of(1, 2, 0));
    memory.write(weights.weights, Float32Array.of(1, 1, 1));
    memory.write(categoryRowCounts, Uint32Array.of(99, 99, 99, 99));
    for (const flags of [
      [0, 0, 1],
      [8, 0, 1],
      [0, 128, 1],
      [8, 128, 1],
      [0, 0, 0],
      [0, 0, 1]
    ]) {
      memory.write(overflow, Uint32Array.of(flags[0]));
      memory.write(candidateOverflow, Uint32Array.of(flags[1]));
      memory.write(complete, Uint32Array.of(flags[2]));
      const lost = Boolean(flags[0] || flags[1] || !flags[2]);
      memory.write(categories, lost ? Uint32Array.of(1, 0, 1) : Uint32Array.of(0, 0, 1));
      executeCommandNodesOnCPU(countNodes, memory);
      for (const [result] of results)
        memory.write(result, new Float32Array(result.length).fill(2.5));
      executeCommandNodesOnCPU(nodes, memory);
      for (const [result, length] of results) {
        const actual = Array.from(memory.readFloat32(result));
        expect(actual.slice(length)).toEqual([2.5, 2.5]);
        if (flags[0] || flags[1] || !flags[2])
          expect(actual.slice(0, length).every(Number.isNaN)).toBe(true);
        else expect(actual.slice(0, length)).toEqual(new Array(length).fill(2.5));
      }
      expect(Array.from(memory.readUint32(categoryRowCounts))).toEqual(
        lost ? [1, 2, 99, 99] : [2, 1, 99, 99]
      );
    }
    // Every status view is a live input: writing category counts over one must be refused.
    // Use a valid scalar overlapping the output, so status-shape validation cannot mask aliasing.
    const aliasedStatus = new GraphDataView(categoryRowCounts.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: categoryRowCounts.byteOffset,
      byteStride: 4,
      rowByteLength: 4
    });
    for (const name of ['overflow', 'candidateOverflow', 'requiredCount', 'complete'] as const) {
      weights.status[name] = aliasedStatus;
      expect(construct).toThrow(/must not share buffers with inputs/);
      weights.status[name] = view('uint32', 1);
    }
    for (const name of ['overflow', 'candidateOverflow', 'complete'] as const) {
      weights.status[name] = view('float32', 1) as never;
      expect(() => construct()).toThrow(/uint32/);
      weights.status[name] = view('uint32', 0);
      expect(() => construct()).toThrow(/uint32/);
      weights.status[name] = view('uint32', 1);
    }
  } finally {
    device.destroy();
  }
});

it('B2 hand-derived colocation fixture agrees with the replay expectations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `fixture-${serial++}`, format, length);
  try {
    const nodes = new GPUColocationQuotient({
      weights: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 3),
        weights: view('float32', 3)
      },
      categories: view('uint32', 3),
      categoryCount: 2,
      output: {globalQuotients: view('float32', 4), localQuotients: view('float32', 6)}
    }).getCommandNodes(graph);
    // Hand sums of neighbor shares: A->A=1, A->B=1, B->A=1, B->B=0.
    // Chance is 1/2,1/2,1,0 respectively. Two linked A rows, one linked B row.
    const values = {
      sums: Float32Array.of(1, 1, 1, 0),
      rowCounts: Uint32Array.of(2, 1),
      linkedCounts: Uint32Array.of(2, 1),
      categories: Uint32Array.of(0, 0, 1),
      globalQuotients: new Float32Array(4),
      localValues: Float32Array.of(1, 0, 0, 1, 1, 0)
    };
    for (const kernel of getGPUCommandNodeKernelSources(
      nodes.filter(node => /-(global|local)$/.test(node.id))
    )) {
      executeSpatialAuditKernel(kernel, values);
    }
    expect(Array.from(values.globalQuotients)).toEqual(MERGE_GLOBAL_QUOTIENTS);
    expect(Array.from(values.localValues)).toEqual(MERGE_LOCAL_QUOTIENTS);
  } finally {
    device.destroy();
  }
});
