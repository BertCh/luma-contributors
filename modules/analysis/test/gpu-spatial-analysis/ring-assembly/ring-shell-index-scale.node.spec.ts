// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createRingShellIndexProbe, createRingShellIndexScene} from './ring-shell-index-harness';
import {getRingShellIndexNodes} from '../../../src/analysis-core/ring-assembly/ring-shell-index';

useCPUExecutorTestHooks();

it('compact shell records preserve signed areas, uint32 groups and unused rows', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const bounds = createTransientView(graph, 'record-bounds', 'float32', 16);
  const areas = createTransientView(graph, 'record-areas', 'float32', 4);
  const isHole = createTransientView(graph, 'record-holes', 'uint32', 4);
  const count = createTransientView(graph, 'record-count', 'uint32', 1);
  const groups = createTransientView(graph, 'record-groups', 'uint32', 4);
  const index = getRingShellIndexNodes(graph, {
    id: 'compact-records',
    operation: 'test',
    bounds,
    areas,
    isHole,
    count,
    groups
  });
  const memory = createCPUGraphMemory();
  memory.write(areas, Float32Array.of(4, -1, 0, 0));
  memory.write(isHole, Uint32Array.of(0, 1, 0, 1));
  memory.write(count, Uint32Array.of(2));
  memory.write(groups, Uint32Array.of(0xfffffffe, 0xfffffffe, 0, 0));
  executeCommandNodesOnCPU([index.nodes[0]], memory);
  // Hand-derived IEEE-754 areas; unused rows must not be mistaken for shells or holes.
  expect(Array.from(memory.readUint32(index.records))).toEqual([
    0x40800000, 0, 0xfffffffe, 0xbf800000, 1, 0xfffffffe, 0, 2, 0, 0, 2, 0
  ]);
});

it('two-word sorting retains low coordinate bits, negative centers and stable identical centers', () => {
  const step = 2 ** -23;
  const centers = [
    [1 + step, 1],
    [1, 1],
    [1, 1 + step],
    [1 + 2 * step, 1],
    [-1, 1],
    [1, -1],
    [1e9 + 64, 64],
    [1, 1]
  ];
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const bounds = createTransientView(graph, 'key-bounds', 'float32', 4 * centers.length);
  const areas = createTransientView(graph, 'key-areas', 'float32', centers.length);
  const isHole = createTransientView(graph, 'key-holes', 'uint32', centers.length);
  const count = createTransientView(graph, 'key-count', 'uint32', 1);
  const index = getRingShellIndexNodes(graph, {
    id: 'key-order',
    operation: 'test',
    bounds,
    areas,
    isHole,
    count
  });
  const memory = createCPUGraphMemory();
  memory.write(bounds, Float32Array.from(centers.flatMap(center => [...center, ...center])));
  memory.write(areas, new Float32Array(centers.length).fill(1));
  memory.write(isHole, new Uint32Array(centers.length));
  memory.write(count, Uint32Array.of(centers.length));
  executeCommandNodesOnCPU(index.nodes, memory);
  // Independent BigInt bit interleave, without the shader's spreadBits masks or two-sort recipe.
  const orderedBits = (value: number) => {
    const bits = new Uint32Array(Float32Array.of(value).buffer)[0];
    return BigInt((bits & 0x80000000) !== 0 ? ~bits >>> 0 : (bits ^ 0x80000000) >>> 0);
  };
  const key = (center: number[]) => {
    let result = 0n;
    for (let bit = 0n; bit < 32n; bit++) {
      result |= ((orderedBits(center[0]) >> bit) & 1n) << (2n * bit);
      result |= ((orderedBits(center[1]) >> bit) & 1n) << (2n * bit + 1n);
    }
    return result;
  };
  const expected = centers
    .map((center, row) => ({row, key: key(center)}))
    .sort((first, second) =>
      first.key < second.key ? -1 : first.key > second.key ? 1 : first.row - second.row
    )
    .map(value => value.row);
  expect(Array.from(memory.readUint32(index.bvh.leafIds))).toEqual(expected);
});

it('the annulus scale fixture preserves bit reversal, holes, padding and the distant shell', () => {
  const scene = createRingShellIndexScene(12);
  expect(scene.annulusCount).toBe(4096);
  expect([0, 1, 2, 3].map(annulus => scene.bounds[8 * annulus])).toEqual([0, 6144, 3072, 9216]);
  expect(Array.from(scene.bounds.slice(4 * 8192, 4 * 8193))).toEqual([1e9, 0, 1e9 + 128, 128]);
  expect(scene.ringCount).toBe(8193);
  expect(Array.from(scene.isHole.slice(0, 4))).toEqual([0, 1, 0, 1]);
});

it('full-coordinate Morton ordering prunes disjoint annuli despite a distant outlier', () => {
  let previousVisits = 0;
  for (const annulusBits of [10, 11, 12]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const probe = createRingShellIndexProbe(graph, annulusBits);
    const memory = createCPUGraphMemory();
    for (const input of probe.inputs) memory.write(input.view, input.values);
    // Execute production index preparation, with the published sort/BVH primitives emulated.
    // An independent recursive box walk measures its pruning without interpreting exact winding
    // at every candidate. The headless spec counts iterations of the production traversal itself.
    executeCommandNodesOnCPU(
      probe.nodes.filter(node => node.id !== 'shell-probe-measured'),
      memory
    );
    const minima = memory.readFloat32(probe.index.minima);
    const maxima = memory.readFloat32(probe.index.maxima);
    const leaves = memory.readUint32(probe.index.leaves);
    const internalCount = leaves.length - 1;
    let visits = 0;
    const perHoleLimit = 2 * Math.ceil(Math.log2(probe.scene.ringCapacity)) + 1;
    for (let annulus = 0; annulus < probe.scene.annulusCount; annulus++) {
      const shell = 2 * annulus;
      const bounds = probe.scene.bounds.subarray(4 * (shell + 1), 4 * (shell + 2));
      let visited = 0;
      const candidates: number[] = [];
      const visit = (node: number) => {
        visited++;
        if (
          minima[2 * node] > bounds[0] ||
          minima[2 * node + 1] > bounds[1] ||
          maxima[2 * node] < bounds[2] ||
          maxima[2 * node + 1] < bounds[3]
        )
          return;
        if (node < internalCount) {
          visit(2 * node + 1);
          visit(2 * node + 2);
        } else {
          candidates.push(leaves[node - internalCount]);
        }
      };
      visit(0);
      expect(visited, `hole ${shell + 1} visits`).toBeLessThanOrEqual(perHoleLimit);
      expect(candidates, `hole ${shell + 1} candidates`).toEqual([shell]);
      visits += visited;
    }
    expect(visits).toBe(probe.scene.annulusCount * perHoleLimit);
    if (previousVisits > 0) expect(visits).toBeLessThan(2.3 * previousVisits);
    previousVisits = visits;
  }
}, 180_000);

it('full-coordinate Morton ordering also refines negative y centers', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const probe = createRingShellIndexProbe(graph, 6, true);
  const memory = createCPUGraphMemory();
  for (const input of probe.inputs) memory.write(input.view, input.values);
  executeCommandNodesOnCPU(
    probe.nodes.filter(node => node.id !== 'shell-probe-measured'),
    memory
  );
  const minima = memory.readFloat32(probe.index.minima);
  const maxima = memory.readFloat32(probe.index.maxima);
  const leaves = memory.readUint32(probe.index.leaves);
  const internalCount = leaves.length - 1;
  // A distant shell can split the coordinate order at one leaf in this rotated scene. Visiting
  // its ancestor path as well as the matching shell's path still takes at most two search paths.
  const perHoleLimit = 4 * Math.ceil(Math.log2(probe.scene.ringCapacity)) + 1;
  for (let annulus = 0; annulus < probe.scene.annulusCount; annulus++) {
    const bounds = probe.scene.bounds.subarray(4 * (2 * annulus + 1), 4 * (2 * annulus + 2));
    let visited = 0;
    const candidates: number[] = [];
    const visit = (node: number) => {
      visited++;
      if (
        minima[2 * node] > bounds[0] ||
        minima[2 * node + 1] > bounds[1] ||
        maxima[2 * node] < bounds[2] ||
        maxima[2 * node + 1] < bounds[3]
      )
        return;
      if (node < internalCount) {
        visit(2 * node + 1);
        visit(2 * node + 2);
      } else {
        candidates.push(leaves[node - internalCount]);
      }
    };
    visit(0);
    expect(visited).toBeLessThanOrEqual(perHoleLimit);
    expect(candidates).toEqual([2 * annulus]);
  }
});
