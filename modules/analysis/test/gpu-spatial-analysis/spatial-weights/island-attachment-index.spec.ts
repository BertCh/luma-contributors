// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The grid path of GPUIslandAttachment must choose the attachment the scan chooses, including at
// ties (the lowest row wins), duplicates, non-finite representatives and ties wider than the
// candidate list. Rows are shuffled so neither the row order nor the Morton order helps.

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUIslandAttachment,
  GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT,
  type GPUIslandAttachmentNearestSearch
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from '../neighbor-search/neighbor-search-oracle';

const NONE = 0xffffffff;

type Scene = {positions: Float32Array; offsets: Uint32Array; neighbors: Uint32Array};

/** Fisher-Yates permutation of the rows. */
function shuffle(rows: number, random: () => number): number[] {
  const order = Array.from({length: rows}, (_, row) => row);
  for (let row = rows - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    [order[row], order[other]] = [order[other], order[row]];
  }
  return order;
}

/**
 * Builds a scene from points: every row of `islandRows` has no edges, every other row gets one edge
 * to the next non-island row (sorted and unique, so a valid CSR). Rows are shuffled.
 */
function createScene(
  points: [number, number][],
  isIsland: (row: number) => boolean,
  seed: number
): Scene {
  const random = createSeededRandom(seed);
  const order = shuffle(points.length, random);
  const shuffled = order.map(row => points[row]);
  const islands = order.map(row => isIsland(row));
  const connected = shuffled.map((_, row) => row).filter(row => !islands[row]);
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < shuffled.length; row++) {
    if (!islands[row] && connected.length > 1) {
      const position = connected.indexOf(row);
      neighbors.push(connected[(position + 1) % connected.length]);
    }
    offsets.push(neighbors.length);
  }
  // A row's only neighbor can be itself in a one-row graph; the scenes below never do that.
  const positions = new Float32Array(shuffled.length * 2);
  shuffled.forEach(([x, y], row) => {
    positions[row * 2] = x;
    positions[row * 2 + 1] = y;
  });
  return {positions, offsets: Uint32Array.from(offsets), neighbors: Uint32Array.from(neighbors)};
}

/** The documented rule on exact (integer) distances: nearest finite other row, ties to the lowest row. */
function getReference(scene: Scene): number[] {
  const rows = scene.positions.length / 2;
  const attachment: number[] = new Array(rows).fill(NONE);
  for (let row = 0; row < rows; row++) {
    const x = scene.positions[row * 2];
    const y = scene.positions[row * 2 + 1];
    if (scene.offsets[row] !== scene.offsets[row + 1] || !Number.isFinite(x + y)) continue;
    let best = Infinity;
    for (let other = 0; other < rows; other++) {
      const otherX = scene.positions[other * 2];
      const otherY = scene.positions[other * 2 + 1];
      if (other === row || !Number.isFinite(otherX + otherY)) continue;
      const distance = (x - otherX) ** 2 + (y - otherY) ** 2;
      if (distance < best) {
        best = distance;
        attachment[row] = other;
      }
    }
  }
  return attachment;
}

async function runAttachment(
  device: Device,
  scene: Scene,
  nearestSearch: GPUIslandAttachmentNearestSearch
): Promise<{attachment: number[]; uncertainCount: number; milliseconds: number}> {
  const rows = scene.positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: `island-index-${nearestSearch}`});
  const buffers: Buffer[] = [];
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    let serial = 0;
    const upload = <Format extends 'float32x2' | 'uint32'>(
      format: Format,
      data: Float32Array | Uint32Array,
      length: number
    ) => {
      const buffer = createInputBuffer(device, data);
      buffers.push(buffer);
      return importGraphBuffer(graph, `input-${serial++}`, buffer, format, length);
    };
    const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
    };
    // Room for every row to gain an edge to its nearest and from the islands attached to it.
    const capacity = scene.neighbors.length + 2 * rows;
    const offsets = output('uint32', rows + 1);
    const neighbors = output('uint32', capacity);
    const weights = output('float32', capacity);
    const attachment = output('uint32', rows);
    const added = output('uint32', capacity);
    const overflow = output('uint32', 1);
    const requiredCount = output('uint32', 1);
    const uncertainCount = output('uint32', 1);
    graph.add(
      new GPUIslandAttachment({
        weights: {
          offsets: upload('uint32', scene.offsets, rows + 1),
          neighbors: upload('uint32', scene.neighbors, Math.max(scene.neighbors.length, 1)),
          weights: importGraphBuffer(
            graph,
            'input-weights',
            createInputBuffer(
              device,
              new Float32Array(Math.max(scene.neighbors.length, 1)).fill(1)
            ),
            'float32',
            Math.max(scene.neighbors.length, 1)
          )
        },
        positions: upload('float32x2', scene.positions, rows),
        nearestSearch,
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
    compiled = graph.compile();
    const start = performance.now();
    submitGraph(device, compiled, undefined);
    const result = Array.from(await readUint32(attachment.buffer, rows));
    const [rescanned] = await readUint32(uncertainCount.buffer, 1);
    return {attachment: result, uncertainCount: rescanned, milliseconds: performance.now() - start};
  } finally {
    compiled?.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function countAttached(attachment: number[]): number {
  return attachment.filter(value => value !== NONE).length;
}

describe('GPUIslandAttachment grid path equals the scan', () => {
  const ROWS = 6000;

  it('integer lattice with many ties and duplicates', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(11);
    const points: [number, number][] = Array.from({length: ROWS}, () => [
      Math.floor(random() * 120),
      Math.floor(random() * 120)
    ]);
    const islandRandom = createSeededRandom(12);
    const islands = Array.from({length: ROWS}, () => islandRandom() < 0.3);
    const scene = createScene(points, row => islands[row], 13);
    const reference = getReference(scene);
    expect(countAttached(reference)).toBeGreaterThan(1500);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    const auto = await runAttachment(device, scene, 'auto');
    expect(scan.attachment, 'scan equals the documented rule').toEqual(reference);
    expect(grid.attachment, 'grid equals the scan').toEqual(scan.attachment);
    expect(auto.attachment, 'auto above the cut-over equals the scan').toEqual(scan.attachment);
    expect(scan.uncertainCount, 'the scan never uses the fallback').toBe(0);
    // The count is reported, bounded by the number of islands, and the same for 'grid' and 'auto'.
    // (Ties here are few: the sixteenth listed candidate is clearly farther, so none rescan.)
    expect(grid.uncertainCount).toBeLessThanOrEqual(islands.filter(Boolean).length);
    expect(auto.uncertainCount).toBe(grid.uncertainCount);
  }, 120_000);

  it('real-valued positions, where the two rankings round differently', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(21);
    const points: [number, number][] = Array.from({length: ROWS}, () => [
      Math.fround(random() * 1000),
      Math.fround(random() * 1000)
    ]);
    // Rows placed a tiny step apart: distances that differ by a few float32 ulps.
    for (let row = 0; row < 600; row += 3) {
      const [x, y] = points[row];
      points[row + 1] = [Math.fround(x + 0.0001), y];
      points[row + 2] = [x, Math.fround(y + 0.0001)];
    }
    const islandRandom = createSeededRandom(22);
    const islands = Array.from({length: ROWS}, () => islandRandom() < 0.4);
    const scene = createScene(points, row => islands[row], 23);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(countAttached(scan.attachment)).toBeGreaterThan(2000);
    expect(grid.attachment).toEqual(scan.attachment);
    // Distinct distances: the sixteen listed candidates prove every answer, so nothing rescans.
    expect(grid.uncertainCount).toBe(0);
  }, 120_000);

  it('resolves ties wider than the candidate list to the lowest row', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Forty coincident connected representatives at (10, 10) and forty coincident islands at
    // (11, 10). Each of those islands has 39 other islands at distance 0, so its sixteen listed
    // candidates are all tied and cannot prove the lowest-row answer: it rescans, and attaches to
    // the lowest-row coincident island (distance 0 beats the connected ones at distance 1).
    const random = createSeededRandom(31);
    const points: [number, number][] = [];
    const islandRows = new Set<number>();
    for (let row = 0; row < ROWS; row++) {
      if (row < 40) points.push([10, 10]);
      else {
        points.push([Math.floor(random() * 90) + 20, Math.floor(random() * 90) + 20]);
        if (row % 3 === 0) islandRows.add(row);
      }
    }
    for (let row = 40; row < 80; row++) {
      points[row] = [11, 10];
      islandRows.add(row);
    }
    const scene = createScene(points, row => islandRows.has(row), 32);
    const reference = getReference(scene);
    const grid = await runAttachment(device, scene, 'grid');
    expect(grid.attachment).toEqual(reference);
    expect(countAttached(reference)).toBeGreaterThan(1000);
    expect(grid.uncertainCount).toBeGreaterThanOrEqual(40);
  }, 120_000);

  it('skips non-finite representatives and leaves non-finite islands unattached', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(41);
    const points: [number, number][] = Array.from({length: ROWS}, () => [
      Math.floor(random() * 300),
      Math.floor(random() * 300)
    ]);
    for (let row = 0; row < ROWS; row += 97) points[row] = [Number.NaN, 5];
    for (let row = 5; row < ROWS; row += 131) points[row] = [7, Number.POSITIVE_INFINITY];
    const islandRandom = createSeededRandom(42);
    const islands = Array.from({length: ROWS}, () => islandRandom() < 0.5);
    const scene = createScene(points, row => islands[row], 43);
    const reference = getReference(scene);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(scan.attachment).toEqual(reference);
    expect(grid.attachment).toEqual(reference);
  }, 120_000);

  it('coordinates whose squared distances overflow still equal the scan', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(51);
    const points: [number, number][] = Array.from({length: ROWS}, () => [
      Math.fround((random() - 0.5) * 3e38),
      Math.fround((random() - 0.5) * 3e38)
    ]);
    const islandRandom = createSeededRandom(52);
    const islands = Array.from({length: ROWS}, () => islandRandom() < 0.3);
    const scene = createScene(points, row => islands[row], 53);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(countAttached(scan.attachment)).toBeGreaterThan(1000);
    expect(grid.attachment).toEqual(scan.attachment);
  }, 120_000);

  it('counts exactly the islands whose candidate list cannot prove the answer', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // 5,960 connected representatives spread over a wide integer square, and forty islands that
    // coincide at one point away from them: only those forty have sixteen tied nearest candidates.
    const random = createSeededRandom(81);
    const points: [number, number][] = [];
    for (let row = 0; row < 6000; row++) {
      points.push(
        row < 40
          ? [50_000, 50_000]
          : [Math.floor(random() * 90_000) + 100_000, Math.floor(random() * 90_000) + 100_000]
      );
    }
    const scene = createScene(points, row => row < 40, 82);
    const grid = await runAttachment(device, scene, 'grid');
    expect(grid.attachment).toEqual(getReference(scene));
    expect(grid.uncertainCount).toBe(40);
    expect((await runAttachment(device, scene, 'scan')).uncertainCount).toBe(0);
  }, 120_000);

  it('a graph without islands and a single island are handled', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const points: [number, number][] = Array.from({length: 300}, (_, row) => [row % 17, row % 13]);
    const none = createScene(points, () => false, 61);
    expect(countAttached((await runAttachment(device, none, 'grid')).attachment)).toBe(0);
    const single = createScene(points, row => row === 7, 62);
    const grid = await runAttachment(device, single, 'grid');
    expect(grid.attachment).toEqual(getReference(single));
    expect(countAttached(grid.attachment)).toBe(1);
  }, 120_000);

  it('60,000 shuffled rows: grid equals the scan (timing is logged, not asserted)', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 60_000;
    const random = createSeededRandom(71);
    const points: [number, number][] = Array.from({length: rows}, () => [
      Math.floor(random() * 700),
      Math.floor(random() * 700)
    ]);
    const islandRandom = createSeededRandom(72);
    const islands = Array.from({length: rows}, () => islandRandom() < 0.25);
    const scene = createScene(points, row => islands[row], 73);
    // The first run of each path compiles its pipelines; the timings are of the second run.
    await runAttachment(device, scene, 'grid');
    await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    const scan = await runAttachment(device, scene, 'scan');
    console.info(
      `island attachment ${rows} rows: grid ${grid.milliseconds.toFixed(0)} ms, scan ${scan.milliseconds.toFixed(0)} ms`
    );
    expect(countAttached(scan.attachment)).toBeGreaterThan(10_000);
    expect(grid.attachment).toEqual(scan.attachment);
  }, 300_000);

  it('the cut-over constant matches the other indexed contributors', () => {
    expect(GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT).toBe(4096);
  });
});
