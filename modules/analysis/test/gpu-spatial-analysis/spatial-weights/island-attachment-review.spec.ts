// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {GPUIslandAttachment} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createIslandAttachmentFallbackNodes,
  ISLAND_ATTACHMENT_NEAREST_DECLARATIONS
} from '../../../src/gpu-spatial-analysis/spatial-weights/island-attachment-fallback';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createIslandAttachmentHarness,
  expectIslandAttachmentCSR,
  ISLAND_ATTACHMENT_NONE,
  type IslandAttachmentScene
} from './island-attachment-test-utils';

/** Empty rows are exactly the supplied islands; every other row has distinctive binary-exact weights. */
function createScene(
  rows: number,
  islands: ReadonlySet<number>,
  wideRow = false
): IslandAttachmentScene {
  const positions = new Float32Array(rows * 2);
  const offsets = new Uint32Array(rows + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < rows; row++) {
    positions[row * 2] = 100 + (row % 1000);
    positions[row * 2 + 1] = 100 + Math.floor(row / 1000);
    if (!islands.has(row)) {
      for (const other of wideRow && row === 2 ? [3, 4, 5] : [row === rows - 1 ? 0 : row + 1]) {
        neighbors.push(other);
        weights.push(((row % 31) + 1) * 0.25 + (other % 4) * 0.0625);
      }
    }
    offsets[row + 1] = neighbors.length;
  }
  return {
    positions,
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

/**
 * Direct production-fallback harness: avoids a second exhaustive grid walk for the 65K-query
 * stress cases. Every representative coincides, so the exact answer is row 0 (row 1 for self 0).
 * Smaller internal caps exercise both cursors without a 16-million-query allocation/search.
 */
function createFallbackHarness(
  device: Device,
  rows: number,
  queryBatchSize?: number,
  representativeBatchSize?: number
) {
  const graph = new GPUCommandGraph(device);
  const positionBuffer = createOutputBuffer(device, rows * 2);
  positionBuffer.write(new Float32Array(rows * 2).fill(7));
  const unresolvedBuffer = createOutputBuffer(device, rows);
  const attachmentBuffer = createOutputBuffer(device, rows);
  const countBuffer = createOutputBuffer(device, 1);
  const positions = importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rows);
  const unresolved = importGraphBuffer(graph, 'unresolved', unresolvedBuffer, 'uint32', rows);
  const attachment = importGraphBuffer(graph, 'attachment', attachmentBuffer, 'uint32', rows);
  const uncertainCount = importGraphBuffer(graph, 'uncertain-count', countBuffer, 'uint32', 1);
  graph.add(
    createIslandAttachmentFallbackNodes(graph, {
      id: 'direct-fallback',
      positions,
      unresolved,
      attachment,
      uncertainCount,
      nearestDeclarations: ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
      queryBatchSize,
      representativeBatchSize
    })
  );
  const compiled = graph.compile();
  return {
    async run(selected: ReadonlySet<number>) {
      const flags = Uint32Array.from({length: rows}, (_, row) => Number(selected.has(row)));
      unresolvedBuffer.write(flags);
      attachmentBuffer.write(new Uint32Array(rows).fill(ISLAND_ATTACHMENT_NONE));
      submitGraph(device, compiled, undefined);
      const expected = Array.from({length: rows}, (_, row) =>
        selected.has(row) ? (row === 0 ? 1 : 0) : ISLAND_ATTACHMENT_NONE
      );
      expect(await readUint32(countBuffer, 1)).toEqual([selected.size]);
      expect(await readUint32(attachmentBuffer, rows)).toEqual(expected);
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [positionBuffer, unresolvedBuffer, attachmentBuffer, countBuffer])
        buffer.destroy();
    }
  };
}

describe('GPUIslandAttachment complete expanded-limit outputs', () => {
  for (const reciprocal of [false, true]) {
    for (const truncated of [false, true]) {
      it(`131,073 rows retain sorted CSR and distinctive weights; reciprocal=${reciprocal}, truncated=${truncated}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const rows = 131_073;
        const scene = createScene(rows, new Set([0, 1, 65536]), true);
        // Three islands surround row 2 at distance one and are farther from one another.
        scene.positions.set([-1, 0, 1, 0, 0, 0], 0);
        scene.positions.set([0, 1], 65536 * 2);
        const expected = new Array<number>(rows).fill(ISLAND_ATTACHMENT_NONE);
        expected[0] = expected[1] = expected[65536] = 2;
        // Both capacities cut row 2 partway through: two of three original entries without
        // reciprocity, or four of six merged entries with it (including original weights).
        const capacity = truncated ? (reciprocal ? 6 : 4) : scene.neighbors.length + 2 * rows;
        const harness = createIslandAttachmentHarness(device, scene, {reciprocal, capacity});
        try {
          const result = await harness.run();
          expect(result.attachment).toEqual(expected);
          expect(result.uncertainCount).toBe(0);
          expect(result.requiredCount).toBe(rows + (reciprocal ? 5 : 2));
          expectIslandAttachmentCSR(result, scene, expected, reciprocal, capacity);
          if (truncated) {
            expect(result.offsets.slice(0, 5)).toEqual([0, 1, 2, capacity, capacity]);
            expect(result.neighbors).toEqual(reciprocal ? [2, 2, 0, 1, 3, 4] : [2, 2, 3, 4]);
            expect(result.overflow).toBe(1);
          }
        } finally {
          harness.destroy();
        }
      }, 120_000);
    }
  }

  it('nonzero-offset packed subviews preserve guards through many -> zero -> one unresolved queries', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 300;
    const selections = [
      new Set(Array.from({length: 257}, (_, row) => row)),
      new Set<number>(),
      new Set([17])
    ];
    const makeScene = (islands: ReadonlySet<number>) => {
      const scene = createScene(rows, islands);
      scene.positions.fill(7);
      return scene;
    };
    const harness = createIslandAttachmentHarness(device, makeScene(selections[0]), {
      guarded: true
    });
    const capacity = rows - selections[0].size + 2 * rows;
    try {
      for (const islands of selections) {
        const scene = makeScene(islands);
        harness.writeScene(scene);
        const result = await harness.run();
        const expected = Array.from({length: rows}, (_, row) =>
          islands.has(row) ? (row === 0 ? 1 : 0) : ISLAND_ATTACHMENT_NONE
        );
        expect(result.attachment).toEqual(expected);
        expect(result.uncertainCount).toBe(islands.size);
        expectIslandAttachmentCSR(result, scene, expected, true, capacity);
      }
    } finally {
      harness.destroy();
    }
  }, 120_000);

  for (const unresolvedCount of [255, 256, 257, 65535, 65536]) {
    it(`${unresolvedCount} unresolved queries compact and cover every indirect workgroup`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // 65,536 workgroups exceed the portable 65,535-wide x dimension. These two stress
      // cases intentionally do exhaustive work, but allocate only about 1 MiB of test buffers.
      const rows = unresolvedCount >= 65535 ? 65536 : 300;
      const harness = createFallbackHarness(device, rows);
      try {
        await harness.run(new Set(Array.from({length: unresolvedCount}, (_, row) => row)));
      } finally {
        harness.destroy();
      }
    }, 300_000);
  }

  it('small query and representative batches roll over, retain earlier ties and reset on rerun', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createFallbackHarness(device, 33, 3, 7);
    try {
      // Sparse selection makes compaction row order different from the original row IDs.
      await harness.run(new Set(Array.from({length: 17}, (_, position) => position * 2)));
      await harness.run(new Set());
      await harness.run(new Set([32]));
    } finally {
      harness.destroy();
    }
  }, 120_000);
});

it('island attachment emits a closed-form CSR into disjoint ranges imported once', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  try {
    const packed = rig.output('uint32', 130);
    const offsets = rig.output('uint32', 3);
    const attachment = rig.output('uint32', 2);
    const overflow = rig.output('uint32', 1);
    const requiredCount = rig.output('uint32', 1);
    const uncertainCount = rig.output('uint32', 1);
    const weights = {
      offsets: rig.input(Uint32Array.of(0, 0, 1), 'uint32'),
      neighbors: rig.input(Uint32Array.of(0), 'uint32'),
      weights: rig.input(Float32Array.of(2), 'float32')
    };
    const producer = new GPUIslandAttachment({
      positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
      weights,
      nearestSearch: 'scan',
      reciprocal: false,
      output: {
        offsets: offsets.view,
        attachment: attachment.view,
        neighbors: rig.graph.createDataView(packed.view.buffer, {format: 'uint32', length: 2}),
        weights: rig.graph.createDataView(packed.view.buffer, {
          format: 'float32',
          byteOffset: 256,
          length: 2
        }),
        added: rig.graph.createDataView(packed.view.buffer, {
          format: 'uint32',
          byteOffset: 512,
          length: 2
        }),
        status: {
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          uncertainCount: uncertainCount.view
        }
      }
    });
    rig.run(producer);
    expect(await offsets.readUint32()).toEqual([0, 1, 2]);
    expect(await attachment.readUint32()).toEqual([1, 0xffffffff]);
    const words = await packed.readUint32();
    expect(words.slice(0, 2)).toEqual([1, 0]);
    expect((await packed.readFloat32()).slice(64, 66)).toEqual([1, 2]);
    expect(words.slice(128, 130)).toEqual([1, 0]);
    expect(await requiredCount.readUint32()).toEqual([2]);
    expect(await overflow.readUint32()).toEqual([0]);
    expect(await uncertainCount.readUint32()).toEqual([0]);
  } finally {
    rig.destroy();
  }
});
