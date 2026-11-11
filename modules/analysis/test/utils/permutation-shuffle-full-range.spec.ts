// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {PERMUTATION_RANDOM_WGSL} from '../../src/analysis-core/kernels/permutation-random';
import {
  getPermutationShuffleWGSL,
  getSwapOrNotRounds,
  PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS,
  type SwapOrNotRounds
} from '../../src/analysis-core/kernels/permutation-shuffle';
import {PHILOX_PURPOSES} from '../../src/analysis-core/kernels/philox-random';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createOutputBuffer, readUint32, submitGraph} from './gpu-contributor-test-utils';
import {getSwapOrNotReferenceIndex} from './permutation-shuffle-reference';
import {requireTestDevice} from './require-test-device';

// The WGSL swap-or-not shuffle on domains above 2^31 (math review finding core-numerics-02):
// `key + count - position` wrapped in u32 there, so a round was no longer an involution and the
// result was not a permutation. Expected values are the exact-integer definition of the shuffle
// (`permutation-shuffle-reference.ts`), not the kernel's 32-bit arithmetic.

const COUNTS = [0x80000000, 0x80000001, 3_000_000_000, 0xfffffffe, 0xffffffff];
const WORKGROUP = 64;
const POSITIONS_PER_LANE = 4;

/** Position of sample `slot` of lane `lane`: both ends of the domain, the middle and two thirds. */
function getSamplePosition(lane: number, slot: number, count: number): number {
  if (slot === 0) {
    return lane;
  }
  if (slot === 1) {
    return count - 1 - lane;
  }
  if (slot === 2) {
    return Math.floor(count / 2) + lane;
  }
  return Math.floor(count / 3) * 2 + lane * 65537;
}

it('the WGSL swap-or-not shuffle is a bijection that matches the definition above 2^31', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const stride = WORKGROUP * POSITIONS_PER_LANE;
  const words = COUNTS.length * stride;
  const output = createOutputBuffer(device, words);
  const graph = new GPUCommandGraph(device, {id: 'shuffle-full-range-test'});
  const outputView = importGraphBuffer(graph, 'shuffle', output, 'uint32', words);
  graph.add({
    getCommandNodes: target => [
      createWGSLKernelNode(target, {
        id: 'shuffle-full-range-kernel',
        operation: 'ShuffleFullRangeTest',
        bindings: [{name: 'shuffle', view: outputView, type: 'u32', access: 'read_write'}],
        workgroupSize: WORKGROUP,
        invocationCount: COUNTS.length * WORKGROUP,
        guardIndex: false,
        declarations: `${PERMUTATION_RANDOM_WGSL}
${getPermutationShuffleWGSL(PHILOX_PURPOSES.permutationShuffle.id, PERMUTATION_SHUFFLE_FULL_RANGE_ROUNDS)}
const SHUFFLE_COUNTS = array<u32, ${COUNTS.length}>(${COUNTS.map(count => `${count}u`).join(', ')});
fn getSamplePosition(lane: u32, slot: u32, count: u32) -> u32 {
  if (slot == 0u) { return lane; }
  if (slot == 1u) { return count - 1u - lane; }
  if (slot == 2u) { return count / 2u + lane; }
  return count / 3u * 2u + lane * 65537u;
}`,
        body: `let testCase = index / ${WORKGROUP}u;
  let count = SHUFFLE_COUNTS[testCase];
  loadSwapOrNotRounds(vec2<u32>(99u, 7u), 3u, count, localInvocationIndex, ${WORKGROUP}u);
  workgroupBarrier();
  for (var slot = 0u; slot < ${POSITIONS_PER_LANE}u; slot++) {
    shuffle[shuffleOffset + testCase * ${stride}u + localInvocationIndex * ${POSITIONS_PER_LANE}u + slot] =
      getSwapOrNotPermutationIndex(getSamplePosition(localInvocationIndex, slot, count), count);
  }`
      })
    ]
  });
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const gpu = await readUint32(output, words);
  for (const [countIndex, count] of COUNTS.entries()) {
    const rounds = getSwapOrNotRounds([99, 7], 3, count, PHILOX_PURPOSES.permutationShuffle.id);
    const reversed: SwapOrNotRounds = {
      keys: Uint32Array.from(rounds.keys).reverse(),
      salts: Uint32Array.from(rounds.salts).reverse()
    };
    const images = new Set<number>();
    const positions = new Set<number>();
    for (let lane = 0; lane < WORKGROUP; lane++) {
      for (let slot = 0; slot < POSITIONS_PER_LANE; slot++) {
        const position = getSamplePosition(lane, slot, count);
        const image = gpu[countIndex * stride + lane * POSITIONS_PER_LANE + slot];
        expect(image, `count ${count} position ${position}`).toBe(
          getSwapOrNotReferenceIndex(position, count, rounds)
        );
        // Every round is an involution, so the reversed rounds undo the shuffle.
        expect(getSwapOrNotReferenceIndex(image, count, reversed)).toBe(position);
        positions.add(position);
        images.add(image);
      }
    }
    // Distinct positions have distinct images.
    expect(images.size, `count ${count}`).toBe(positions.size);
  }
  compiled.destroy();
  output.destroy();
});
