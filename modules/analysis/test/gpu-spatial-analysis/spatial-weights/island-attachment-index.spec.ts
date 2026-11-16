// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The grid path of GPUIslandAttachment must choose the attachment the scan chooses, including at
// ties (the lowest row wins), duplicates, non-finite representatives and ties wider than the
// candidate list. Rows are shuffled so neither the row order nor the Morton order helps.

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT,
  type GPUIslandAttachmentNearestSearch
} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-island-attachment';
import {
  createIslandAttachmentHarness,
  expectIslandAttachmentCSR,
  type IslandAttachmentResult
} from './island-attachment-test-utils';
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
): Promise<IslandAttachmentResult> {
  const harness = createIslandAttachmentHarness(device, scene, {nearestSearch});
  try {
    return await harness.run();
  } finally {
    harness.destroy();
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

  for (const tiedCount of [16, 1024]) {
    it(`${tiedCount} exactly equidistant representatives match the scan and CPU reference`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      // Nonzero, exactly representable distance 5. The island is shuffled among connected rows;
      // its own zero-distance candidate must be skipped. The remaining candidates cannot certify
      // the lowest-row answer, including across different lanes and representative rounds.
      const directions: [number, number][] = [
        [3, 4],
        [-3, 4],
        [3, -4],
        [-3, -4]
      ];
      const points: [number, number][] = Array.from(
        {length: tiedCount},
        (_, row) => directions[row % directions.length]
      );
      points.push([0, 0]);
      const scene = createScene(points, row => row === tiedCount, tiedCount + 100);
      const reference = getReference(scene);
      const scan = await runAttachment(device, scene, 'scan');
      const grid = await runAttachment(device, scene, 'grid');
      expect(scan.attachment).toEqual(reference);
      expect(grid.attachment).toEqual(reference);
      expect(grid.attachment).toEqual(scan.attachment);
      expect(grid.uncertainCount).toBe(1);
      expect(scan.uncertainCount).toBe(0);
    }, 120_000);
  }

  it('1,025 coincident islands exclude themselves and reduce ties to the lowest other row', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // The last representative round has a single occupied lane; every query needs the fallback.
    // There are no original edges, exercising padded input slots and reciprocal merging as well.
    const rows = 1025;
    const scene: Scene = {
      positions: new Float32Array(2 * rows).fill(7),
      offsets: new Uint32Array(rows + 1),
      neighbors: new Uint32Array(0)
    };
    const reference = getReference(scene);
    expect(reference).toEqual([1, ...new Array(rows - 1).fill(0)]);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(scan.attachment).toEqual(reference);
    expect(grid.attachment).toEqual(scan.attachment);
    expect(grid.uncertainCount).toBe(rows);
  }, 120_000);

  it('the fallback preserves the former 65,535-row boundary', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 65535;
    // Overflowing squared distances force the fallback. Only the very last representative is
    // nearest, exercising round 256 and the unoccupied lane at the 65,535-row boundary.
    const positions = new Float32Array(2 * rows);
    for (let row = 1; row < rows; row++) positions[2 * row] = Math.fround(3e38);
    positions[2 * (rows - 1)] = Math.fround(1e38);
    const scene: Scene = {
      positions,
      offsets: Uint32Array.from({length: rows + 1}, (_, row) => Math.max(row - 1, 0)),
      neighbors: Uint32Array.from({length: rows - 1}, (_, slot) =>
        slot === rows - 2 ? 1 : slot + 2
      )
    };
    const reference = getReference(scene);
    expect(reference[0]).toBe(rows - 1);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(scan.attachment).toEqual(reference);
    expect(grid.attachment).toEqual(scan.attachment);
    expect(grid.uncertainCount).toBe(1);
  }, 120_000);

  it('one million lattice rows with sparse islands follow the closed-form nearest rule', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rows = 1_000_000;
    const columns = 1000;
    const positions = new Float32Array(rows * 2);
    const offsets = new Uint32Array(rows + 1);
    const neighbors: number[] = [];
    const expected = new Array<number>(rows).fill(NONE);
    for (let row = 0; row < rows; row++) {
      positions[row * 2] = row % columns;
      positions[row * 2 + 1] = Math.floor(row / columns);
      if (row % 100_000 === 0) {
        // On the first column, up, right and down tie at one. The smallest row is up,
        // except at the origin where right wins. Selection uses original representatives.
        expected[row] = row === 0 ? 1 : row - columns;
      } else {
        neighbors.push(row === rows - 1 ? 1 : row + 1);
      }
      offsets[row + 1] = neighbors.length;
    }
    const scene = {positions, offsets, neighbors: Uint32Array.from(neighbors)};
    const result = await runAttachment(device, scene, 'auto');
    expectIslandAttachmentCSR(result, scene, expected);
    expect(result.attachment).toEqual(expected);
    expect(result.uncertainCount).toBe(0);
  }, 120_000);

  for (const rows of [65_536, 65_537, 131_073, 1_000_000]) {
    it(`${rows} rows: unproven nearest search spans representative batches`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const positions = new Float32Array(rows * 2);
      for (let row = 1; row < rows; row++) positions[row * 2] = Math.fround(3e38);
      positions[(rows - 1) * 2] = Math.fround(1e38);
      // At 131,073 rows the lowest tied representative is in batch one and the other
      // is the final row of batch three. At one million only the final row is nearest.
      if (rows === 131_073) positions[65535 * 2] = Math.fround(1e38);
      const scene: Scene = {
        positions,
        offsets: Uint32Array.from({length: rows + 1}, (_, row) => Math.max(row - 1, 0)),
        neighbors: Uint32Array.from({length: rows - 1}, (_, slot) =>
          slot === rows - 2 ? 1 : slot + 2
        )
      };
      const expected = new Array<number>(rows).fill(NONE);
      expected[0] = rows === 131_073 ? 65535 : rows - 1;
      expect(getReference(scene)).toEqual(expected);
      const result = await runAttachment(device, scene, 'grid');
      expect(result.attachment).toEqual(expected);
      expectIslandAttachmentCSR(result, scene, expected);
      expect(result.uncertainCount).toBe(1);
    }, 120_000);
  }

  it('no finite other representative leaves the island unattached', async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene: Scene = {
      positions: new Float32Array([0, 0, Number.NaN, 0, 0, Number.POSITIVE_INFINITY]),
      offsets: new Uint32Array(4),
      neighbors: new Uint32Array(0)
    };
    const reference = getReference(scene);
    const scan = await runAttachment(device, scene, 'scan');
    const grid = await runAttachment(device, scene, 'grid');
    expect(reference).toEqual([NONE, NONE, NONE]);
    expect(scan.attachment).toEqual(reference);
    expect(grid.attachment).toEqual(scan.attachment);
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
