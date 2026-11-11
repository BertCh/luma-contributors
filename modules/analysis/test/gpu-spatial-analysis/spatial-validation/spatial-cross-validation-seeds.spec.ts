// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review, interpolation-density-02. The block folds used to be
// `hash(block ^ seed) % foldCount`: a small seed only relabelled the blocks, so 8 blocks in 5
// folds left fold 4 empty for every seed 0..7. The properties below are the definition of a
// seeded block design, not a copy of the kernel: every fold holds a block when there are at
// least as many blocks as folds, fold sizes differ by at most one block, a block is never split,
// and the seed changes the design.

const INVALID_FOLD = 0xffffffff;

type FoldRun = {folds: number[]; counts: number[]; status: number[]};

async function runBlockFolds(props: {
  blocks: number[];
  foldCount: number;
  seed: number;
  mask?: number[];
  positions?: number[];
}): Promise<FoldRun> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = props.blocks.length;
  const positions = props.positions ?? props.blocks.flatMap((_, row) => [row * 10, 0]);
  const graph = new GPUCommandGraph(device, {id: `block-folds-${props.seed}`});
  const positionsBuffer = createInputBuffer(device, Float32Array.from(positions));
  const blocksBuffer = createInputBuffer(device, Uint32Array.from(props.blocks));
  const maskBuffer = props.mask && createInputBuffer(device, Uint32Array.from(props.mask));
  const foldsBuffer = createOutputBuffer(device, rows);
  const trainingBuffer = createOutputBuffer(device, rows * props.foldCount);
  const testingBuffer = createOutputBuffer(device, rows * props.foldCount);
  const countsBuffer = createOutputBuffer(device, 3 * props.foldCount);
  const statusBuffer = createOutputBuffer(device, props.foldCount);
  graph.add(
    new GPUSpatialCrossValidation({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
      blocks: importGraphBuffer(graph, 'blocks', blocksBuffer, 'uint32', rows),
      mask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
      foldCount: props.foldCount,
      seed: props.seed,
      output: {
        folds: importGraphBuffer(graph, 'folds', foldsBuffer, 'uint32', rows),
        training: importGraphBuffer(
          graph,
          'training',
          trainingBuffer,
          'uint32',
          rows * props.foldCount
        ),
        testing: importGraphBuffer(
          graph,
          'testing',
          testingBuffer,
          'uint32',
          rows * props.foldCount
        ),
        counts: importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', 3 * props.foldCount),
        status: importGraphBuffer(graph, 'status', statusBuffer, 'uint32', props.foldCount)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    folds: await readUint32(foldsBuffer, rows),
    counts: await readUint32(countsBuffer, 3 * props.foldCount),
    status: await readUint32(statusBuffer, props.foldCount)
  };
  compiled.destroy();
  for (const buffer of [
    positionsBuffer,
    blocksBuffer,
    foldsBuffer,
    trainingBuffer,
    testingBuffer,
    countsBuffer,
    statusBuffer
  ]) {
    buffer.destroy();
  }
  maskBuffer?.destroy();
  return result;
}

/** Number of distinct blocks assigned to each fold. */
function getBlocksPerFold(blocks: number[], folds: number[], foldCount: number): number[] {
  const blockSets = Array.from({length: foldCount}, () => new Set<number>());
  for (const [row, fold] of folds.entries()) {
    if (fold !== INVALID_FOLD) {
      blockSets[fold].add(blocks[row]);
    }
  }
  return blockSets.map(blockSet => blockSet.size);
}

for (const {blockCount, foldCount} of [
  {blockCount: 8, foldCount: 5},
  {blockCount: 16, foldCount: 10},
  {blockCount: 64, foldCount: 10}
]) {
  it(`GPUSpatialCrossValidation deals ${blockCount} blocks into ${foldCount} folds with no empty fold for every seed`, async () => {
    // Two rows per block, in an order that does not follow the block id.
    const blocks = Array.from({length: 2 * blockCount}, (_, row) => (row * 7 + 3) % blockCount);
    const designs: string[] = [];
    for (let seed = 0; seed < 8; seed++) {
      const {folds, counts, status} = await runBlockFolds({blocks, foldCount, seed});
      expect(folds.every(fold => fold < foldCount)).toBe(true);
      // A block is never split between folds.
      const foldOfBlock = new Map<number, number>();
      for (const [row, fold] of folds.entries()) {
        expect(foldOfBlock.get(blocks[row]) ?? fold).toBe(fold);
        foldOfBlock.set(blocks[row], fold);
      }
      // Fold sizes in blocks: floor(blocks / folds) or one more, the larger ones first.
      const quotient = Math.floor(blockCount / foldCount);
      const remainder = blockCount % foldCount;
      expect(getBlocksPerFold(blocks, folds, foldCount)).toEqual(
        Array.from({length: foldCount}, (_, fold) => quotient + (fold < remainder ? 1 : 0))
      );
      // Test count of every fold is two rows per block, and no fold reports an empty test set.
      for (let fold = 0; fold < foldCount; fold++) {
        expect(counts[3 * fold + 1]).toBe(2 * (quotient + (fold < remainder ? 1 : 0)));
        expect(status[fold] & 2).toBe(0);
      }
      designs.push(folds.join(','));
    }
    // Eight seeds give eight different designs. For the smallest case, 8 blocks, there are
    // 8! / (2! 2! 2!) = 5040 designs, so the chance of a repeat among the 28 pairs of seeds is
    // about 28 / 5040 = 0.0056; the seeds are fixed, so the outcome is deterministic.
    expect(new Set(designs).size).toBe(8);
  });
}

it('GPUSpatialCrossValidation ranks only blocks that hold a valid row, and block ids need not be dense', async () => {
  // Six blocks with sparse ids. Block 4000000000 has only a masked row and block 77 only a
  // non-finite position, so four blocks remain for four folds: one block per fold.
  const blocks = [900, 5, 4000000000, 5, 77, 123456789, 900, 31];
  const mask = [1, 1, 0, 1, 1, 1, 1, 1];
  const positions = [0, 0, 10, 0, 20, 0, 30, 0, Number.NaN, 0, 50, 0, 60, 0, 70, 0];
  for (let seed = 0; seed < 4; seed++) {
    const {folds, status} = await runBlockFolds({blocks, mask, positions, foldCount: 4, seed});
    expect(folds[2]).toBe(INVALID_FOLD);
    expect(folds[4]).toBe(INVALID_FOLD);
    expect(folds[0]).toBe(folds[6]);
    expect(folds[1]).toBe(folds[3]);
    expect([folds[0], folds[1], folds[5], folds[7]].sort()).toEqual([0, 1, 2, 3]);
    expect(status).toEqual([0, 0, 0, 0]);
  }
});

// Review follow-up. The fold of a block depends on which blocks hold a usable row, so a mask
// that removes whole blocks re-deals the rest. The alternative, ranking every block id present
// in the input whatever its mask, makes the assignment a function of the seed and the block set
// only, but it brings the empty folds back: with 8 blocks of which 4 are usable and 4 folds, the
// 4 usable blocks land in 4 different folds for only 16 of the 70 equally likely rank subsets.
// The two tests below pin the rule that was chosen and the property it keeps.

it('GPUSpatialCrossValidation leaves no fold empty when a mask removes whole blocks', async () => {
  // Eight blocks, three rows each; the mask removes blocks 1, 2, 5 and 6 entirely.
  const blocks = Array.from({length: 24}, (_, row) => row % 8);
  const maskedBlocks = new Set([1, 2, 5, 6]);
  const mask = blocks.map(block => (maskedBlocks.has(block) ? 0 : 1));
  for (let seed = 0; seed < 16; seed++) {
    const {folds, status} = await runBlockFolds({blocks, mask, foldCount: 4, seed});
    for (const [row, fold] of folds.entries()) {
      expect(fold === INVALID_FOLD).toBe(mask[row] === 0);
    }
    // Four usable blocks in four folds: one block each, for every seed.
    expect(getBlocksPerFold(blocks, folds, 4)).toEqual([1, 1, 1, 1]);
    expect(status).toEqual([0, 0, 0, 0]);
  }
});

it('GPUSpatialCrossValidation keeps the folds when a mask removes rows but no whole block', async () => {
  const blocks = Array.from({length: 24}, (_, row) => row % 8);
  // Rows 0 to 7 are masked: every block loses its first row and keeps two.
  const mask = blocks.map((_, row) => (row < 8 ? 0 : 1));
  for (let seed = 0; seed < 4; seed++) {
    const unmasked = await runBlockFolds({blocks, foldCount: 3, seed});
    const masked = await runBlockFolds({blocks, mask, foldCount: 3, seed});
    expect(masked.folds.slice(8)).toEqual(unmasked.folds.slice(8));
    expect(masked.folds.slice(0, 8)).toEqual(new Array<number>(8).fill(INVALID_FOLD));
  }
});
