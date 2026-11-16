// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {
  GPUIslandAttachment,
  type GPUIslandAttachmentNearestSearch
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

export const ISLAND_ATTACHMENT_NONE = 0xffffffff;
const GUARD_WORD = 0xa5a5a5a5;

export type IslandAttachmentScene = {
  positions: Float32Array;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights?: Float32Array;
};

export type IslandAttachmentResult = {
  attachment: number[];
  offsets: number[];
  neighbors: number[];
  weights: number[];
  added: number[];
  requiredCount: number;
  overflow: number;
  uncertainCount: number;
  milliseconds: number;
};

/** Independent map-and-sort CSR oracle. Attachments are a CPU oracle or a fixture's closed form. */
export function getIslandAttachmentCSRReference(
  scene: IslandAttachmentScene,
  attachment: readonly number[],
  reciprocal: boolean,
  capacity: number
): Pick<
  IslandAttachmentResult,
  'offsets' | 'neighbors' | 'weights' | 'added' | 'requiredCount' | 'overflow'
> {
  const inverse = new Map<number, number[]>();
  if (reciprocal) {
    for (let row = 0; row < attachment.length; row++) {
      const other = attachment[row];
      if (other === ISLAND_ATTACHMENT_NONE) continue;
      const islands = inverse.get(other) ?? [];
      islands.push(row);
      inverse.set(other, islands);
    }
  }
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  const added: number[] = [];
  let requiredCount = 0;
  for (let row = 0; row < attachment.length; row++) {
    const entries = new Map<number, {weight: number; added: number}>();
    for (let slot = scene.offsets[row]; slot < scene.offsets[row + 1]; slot++) {
      entries.set(scene.neighbors[slot], {weight: scene.weights?.[slot] ?? 1, added: 0});
    }
    const additions = [...(inverse.get(row) ?? [])];
    if (attachment[row] !== ISLAND_ATTACHMENT_NONE) additions.push(attachment[row]);
    for (const other of additions) {
      if (other !== row && !entries.has(other)) entries.set(other, {weight: 1, added: 1});
    }
    requiredCount += entries.size;
    for (const other of [...entries.keys()].sort((first, second) => first - second)) {
      if (neighbors.length === capacity) break;
      const entry = entries.get(other)!;
      neighbors.push(other);
      weights.push(entry.weight);
      added.push(entry.added);
    }
    offsets.push(Math.min(requiredCount, capacity));
  }
  return {
    offsets,
    neighbors,
    weights,
    added,
    requiredCount,
    overflow: Number(requiredCount > capacity)
  };
}

export function expectIslandAttachmentCSR(
  actual: IslandAttachmentResult,
  scene: IslandAttachmentScene,
  attachment: readonly number[],
  reciprocal = true,
  capacity = scene.neighbors.length + 2 * attachment.length
): void {
  const expected = getIslandAttachmentCSRReference(scene, attachment, reciprocal, capacity);
  for (const field of [
    'offsets',
    'neighbors',
    'weights',
    'added',
    'requiredCount',
    'overflow'
  ] as const) {
    expect(actual[field], field).toEqual(expected[field]);
  }
}

/** All inputs and outputs can be packed subviews with a 252-byte prefix and trailing guards. */
export function createIslandAttachmentHarness(
  device: Device,
  scene: IslandAttachmentScene,
  options: {
    nearestSearch?: GPUIslandAttachmentNearestSearch;
    reciprocal?: boolean;
    capacity?: number;
    guarded?: boolean;
  } = {}
) {
  const rows = scene.positions.length / 2;
  const capacity = options.capacity ?? scene.neighbors.length + 2 * rows;
  const graph = new GPUCommandGraph(device);
  const records: {buffer: Buffer; prefix: number; words: number; length: number}[] = [];
  const prefix = options.guarded ? 63 : 0;
  const suffix = options.guarded ? 17 : 0;
  const storage = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => {
    const words = length * (format === 'float32x2' ? 2 : 1);
    const buffer = createOutputBuffer(device, prefix + words + suffix);
    const initial = new Uint32Array(prefix + words + suffix).fill(GUARD_WORD);
    buffer.write(initial);
    const handle = graph.importBuffer(
      {id: `storage-${records.length}`, byteLength: buffer.byteLength, usage: buffer.usage},
      buffer
    );
    const view: GraphDataView<Format> = graph.createDataView(handle, {
      format,
      byteOffset: prefix * 4,
      length
    });
    const record = {buffer, prefix, words, length};
    records.push(record);
    return {...record, view};
  };
  const positions = storage('float32x2', rows);
  const inputOffsets = storage('uint32', rows + 1);
  // Reserve enough input capacity to change island membership on later encodings.
  const inputCapacity = Math.max(rows, scene.neighbors.length, 1);
  const inputNeighbors = storage('uint32', inputCapacity);
  const inputWeights = storage('float32', inputCapacity);
  const offsets = storage('uint32', rows + 1);
  const neighbors = storage('uint32', capacity);
  const weights = storage('float32', capacity);
  const attachment = storage('uint32', rows);
  const added = storage('uint32', capacity);
  const requiredCount = storage('uint32', 1);
  const overflow = storage('uint32', 1);
  const uncertainCount = storage('uint32', 1);
  const write = (record: (typeof records)[number], data: Float32Array | Uint32Array) => {
    record.buffer.write(data, record.prefix * 4);
  };
  const writeScene = (next: IslandAttachmentScene) => {
    if (
      next.positions.length !== rows * 2 ||
      next.offsets.length !== rows + 1 ||
      next.neighbors.length > inputCapacity
    ) {
      throw new Error('island test scene must retain its row shape and input capacity');
    }
    write(positions, next.positions);
    write(inputOffsets, next.offsets);
    const neighborValues = new Uint32Array(inputCapacity);
    neighborValues.set(next.neighbors);
    write(inputNeighbors, neighborValues);
    const weightValues = new Float32Array(inputCapacity);
    for (let slot = 0; slot < next.neighbors.length; slot++)
      weightValues[slot] = next.weights?.[slot] ?? 1;
    write(inputWeights, weightValues);
  };
  writeScene(scene);
  graph.add(
    new GPUIslandAttachment({
      positions: positions.view,
      weights: {
        offsets: inputOffsets.view,
        neighbors: inputNeighbors.view,
        weights: inputWeights.view
      },
      nearestSearch: options.nearestSearch ?? 'grid',
      reciprocal: options.reciprocal,
      output: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: weights.view,
        attachment: attachment.view,
        added: added.view,
        status: {
          overflow: overflow.view,
          requiredCount: requiredCount.view,
          uncertainCount: uncertainCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  const read = async (
    record: (typeof records)[number],
    floating = false,
    count = record.length
  ): Promise<number[]> => {
    const bytes = await record.buffer.readAsync();
    const byteOffset = bytes.byteOffset + record.prefix * 4;
    return Array.from(
      floating
        ? new Float32Array(bytes.buffer, byteOffset, count)
        : new Uint32Array(bytes.buffer, byteOffset, count)
    );
  };
  return {
    writeScene,
    async run(): Promise<IslandAttachmentResult> {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      const outputOffsets = await read(offsets);
      const written = outputOffsets[rows];
      const result: IslandAttachmentResult = {
        attachment: await read(attachment),
        offsets: outputOffsets,
        neighbors: await read(neighbors, false, written),
        weights: await read(weights, true, written),
        added: await read(added, false, written),
        requiredCount: (await read(requiredCount))[0],
        overflow: (await read(overflow))[0],
        uncertainCount: (await read(uncertainCount))[0],
        milliseconds: performance.now() - start
      };
      if (options.guarded) {
        for (const record of records) {
          const words = await readUint32(record.buffer);
          expect(words.slice(0, record.prefix)).toEqual(new Array(prefix).fill(GUARD_WORD));
          expect(words.slice(record.prefix + record.words)).toEqual(
            new Array(suffix).fill(GUARD_WORD)
          );
        }
      }
      return result;
    },
    destroy() {
      compiled.destroy();
      for (const {buffer} of records) buffer.destroy();
    }
  };
}
