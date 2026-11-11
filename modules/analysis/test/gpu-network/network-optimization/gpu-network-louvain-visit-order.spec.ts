// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  PERMUTATION_RANDOM_WGSL,
  PhiloxStream
} from '../../../src/analysis-core/kernels/permutation-random';
import {PHILOX_PURPOSES} from '../../../src/analysis-core/kernels/philox-random';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUNetworkLouvainVisitOrderSource,
  GPUNetworkLouvain
} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createOptimizationHarness} from './optimization-test-harness';

/**
 * Runs Louvain on `pairCount` separate edges `(2 k, 2 k + 1)` of weight 1. Whichever vertex of a
 * pair is visited first joins the community of the other one (the move raises modularity), and
 * the other vertex then has no other neighboring community. So the label of a pair is the vertex
 * visited second, and the labels show the visit order inside every pair.
 */
async function runPairs(
  device: Device,
  pairCount: number,
  seed: number | undefined
): Promise<number[]> {
  const nodeCount = 2 * pairCount;
  const harness = createOptimizationHarness(device);
  const weights = new Float32Array(nodeCount * nodeCount);
  for (let pair = 0; pair < pairCount; pair++) {
    weights[2 * pair * nodeCount + 2 * pair + 1] = 1;
    weights[(2 * pair + 1) * nodeCount + 2 * pair] = 1;
  }
  const output = {
    communities: harness.output('uint32', nodeCount),
    objectiveTrace: harness.output('float32', 1 + 4 * 4),
    summary: harness.output('uint32', 4)
  };
  const result = await harness.run(
    new GPUNetworkLouvain({
      nodeCount,
      seed,
      maximumLevels: 4,
      maximumIterations: 4,
      weights: harness.input('float32', weights),
      parameters: harness.input('float32', Float32Array.of(1, 0)),
      output
    })
  );
  expect(result(output.summary).slice(0, 2), `${nodeCount} vertices`).toEqual([1, 1]);
  // Every pair merged: all the weight is internal, and each of the `pairCount` communities holds
  // the share 1 / pairCount of the strength, so the objective is 1 - 1 / pairCount.
  const trace = result(output.objectiveTrace).filter(Number.isFinite);
  expect(trace.at(-1), `${nodeCount} vertices`).toBeCloseTo(1 - 1 / pairCount, 6);
  return result(output.communities);
}

/** Labels of the pairs for a visit order: each pair takes the vertex that comes later in it. */
function getPairLabels(order: readonly number[]): number[] {
  const position = new Array<number>(order.length);
  order.forEach((vertex, index) => {
    position[vertex] = index;
  });
  return Array.from({length: order.length}, (_, vertex) => {
    const partner = vertex ^ 1;
    return position[vertex] > position[partner] ? vertex : partner;
  });
}

/** Fisher-Yates order from the registered Philox stream of the contributor, in TypeScript. */
function getPhiloxVisitOrder(vertexCount: number, seed: number): number[] {
  const order = Array.from({length: vertexCount}, (_, vertex) => vertex);
  const stream = new PhiloxStream([seed, PHILOX_PURPOSES.networkLouvainVisitOrder.id], 0, 0, 0);
  for (let remaining = vertexCount; remaining > 1; remaining--) {
    const selected = stream.nextBelow(remaining);
    [order[remaining - 1], order[selected]] = [order[selected], order[remaining - 1]];
  }
  return order;
}

const EVEN_VERTEX_COUNTS = [4, 8, 10, 12, 14, 16, 20, 24, 28, 32];

it('GPUNetworkLouvain without a seed visits vertices in ascending order at every size', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Ascending order: the even vertex of every pair moves first, so every pair takes its odd vertex.
  // Found while pinning network-paths-08: with 12 to 24 vertices the unseeded kernel visited
  // vertex 0 at every step (its identity order table read back as zeros on this GPU), so only the
  // first pair merged and the run still reported valid and converged.
  for (const vertexCount of EVEN_VERTEX_COUNTS) {
    expect(await runPairs(device, vertexCount / 2, undefined), `${vertexCount} vertices`).toEqual(
      Array.from({length: vertexCount}, (_, vertex) => vertex | 1)
    );
  }
});

it('GPUNetworkLouvain seeded visit order comes from the registered Philox stream (network-paths-08)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Review network-paths-08: the private xorshift32 started from `seed ^ 0x9e3779b9`, so this seed
  // gave state 0 for ever, every draw 0 and the fixed rotation 1, 2, ..., n - 1, 0 as the "random"
  // order. That rotation labels the pairs 0, 0, 3, 3, 5, 5, ...
  const pairCount = 8;
  const vertexCount = 2 * pairCount;
  const stuckSeed = 0x9e3779b9;
  const rotation = Array.from({length: vertexCount}, (_, index) => (index + 1) % vertexCount);
  const rotationLabels = getPairLabels(rotation);
  expect(rotationLabels.slice(0, 6)).toEqual([0, 0, 3, 3, 5, 5]);
  let evenFirstCount = 0;
  const seeds = [stuckSeed, 0, 1, 17, 0xffffffff, 123456789, 2026, 42, 7, 99, 31337, 271828];
  for (const seed of seeds) {
    const labels = await runPairs(device, pairCount, seed);
    expect(labels, `seed ${seed}`).toEqual(getPairLabels(getPhiloxVisitOrder(vertexCount, seed)));
    for (let pair = 0; pair < pairCount; pair++) {
      expect(labels[2 * pair]).toBe(labels[2 * pair + 1]);
      evenFirstCount += labels[2 * pair] === 2 * pair + 1 ? 1 : 0;
    }
    if (seed === stuckSeed) {
      expect(labels).not.toEqual(rotationLabels);
    }
  }
  // In a uniform shuffle either vertex of a pair comes first with probability 1/2: 96 fair coins,
  // mean 48, standard deviation 4.9. The bound is 5 standard deviations.
  const coinCount = seeds.length * pairCount;
  expect(Math.abs(evenFirstCount - coinCount / 2)).toBeLessThan(5 * Math.sqrt(coinCount / 4));
});

it('GPUNetworkLouvain seeded runs follow the Philox order at every size up to the 32-vertex limit (fr-fields-05)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const vertexCount of EVEN_VERTEX_COUNTS) {
    for (const seed of [0, 17, 0x9e3779b9]) {
      expect(
        await runPairs(device, vertexCount / 2, seed),
        `${vertexCount} vertices, seed ${seed}`
      ).toEqual(getPairLabels(getPhiloxVisitOrder(vertexCount, seed)));
    }
  }
});

/**
 * Reads the contributor's visit-order table back from the GPU: a kernel of one binding that runs
 * the contributor's own WGSL and then reads the table by a loop index inside a loop nest with a
 * membership test, the shape of the Louvain sweep (and the shape in which a function-scope
 * identity table read back as zeros).
 */
async function readVisitOrder(device: Device, vertexCount: number, seed: number) {
  const graph = new GPUCommandGraph(device, {id: `louvain-visit-order-${vertexCount}-${seed}`});
  const buffer = createOutputBuffer(device, vertexCount);
  const source = getGPUNetworkLouvainVisitOrderSource(vertexCount, seed);
  graph.add({
    getCommandNodes: <Parameters>(target: GPUCommandGraph<Parameters>) => [
      createWGSLKernelNode<Parameters>(target, {
        id: 'read-visit-order',
        operation: 'GPUNetworkLouvain',
        variant: `read-visit-order-${vertexCount}-${seed}`,
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {
            name: 'visited',
            view: importGraphBuffer(target, 'visited', buffer, 'uint32', vertexCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `${PERMUTATION_RANDOM_WGSL}
${source.declarations}
const VERTEX_COUNT: u32 = ${vertexCount}u;`,
        body: `${source.body}
var groups: array<u32, ${vertexCount}>;
for (var vertex = 0u; vertex < VERTEX_COUNT; vertex++) { groups[vertex] = vertex; }
for (var round = 0u; round < 2u; round++) {
  for (var visit = 0u; visit < VERTEX_COUNT; visit++) {
    let group = louvainVisitOrder[visit];
    var present = false;
    for (var vertex = 0u; vertex < VERTEX_COUNT; vertex++) { if (groups[vertex] == group) { present = true; } }
    if (!present) { continue; }
    visited[visitedOffset + visit] = group;
  }
}`
      })
    ]
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return await readUint32(buffer, vertexCount);
  } finally {
    compiled.destroy();
    buffer.destroy();
  }
}

it('the Louvain visit-order table reads back as the full Philox order from 2 to 512 entries (fr-fields-05)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fix review fr-fields-05: the whole order, not one bit per pair, at sizes on both sides of the
  // ranges in which a function-scope identity table read back as zeros (from 12 entries up
  // inside the old Louvain kernel, 40 to 96 in a kernel like this one), and well past the class
  // limit of 32. The table has a literal initializer since the closing pass of 2026-11-11, so no
  // loop fills it; the order must be what it was when an identity loop did.
  for (const vertexCount of [2, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128, 300, 512]) {
    for (const seed of [0, 17, 0x9e3779b9, 0xffffffff, 2026]) {
      const expected = getPhiloxVisitOrder(vertexCount, seed);
      const actual = await readVisitOrder(device, vertexCount, seed);
      expect(actual, `${vertexCount} entries, seed ${seed}`).toEqual(expected);
      // A permutation, and (past a handful of entries) not the identity.
      expect([...actual].sort((left, right) => left - right)).toEqual(
        Array.from({length: vertexCount}, (_, vertex) => vertex)
      );
      if (vertexCount >= 8) {
        expect(actual).not.toEqual(Array.from({length: vertexCount}, (_, vertex) => vertex));
      }
    }
  }
});
