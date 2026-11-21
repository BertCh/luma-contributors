// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const TILE_SIZE = 16;
const HALO_SIZE = TILE_SIZE + 2;
const COLUMN_OFFSETS = [1, 1, 0, -1, -1, -1, 0, 1];
const ROW_OFFSETS = [0, 1, 1, 1, 0, -1, -1, -1];

/**
 * Bounded float64 raster relaxation for the CPU cost-distance twin. Each plain round reads
 * round-start values; each sweep line sees its own earlier tile writes. These are explicit,
 * deterministic visibility choices among the GPU's permitted schedules. Inner tile updates
 * are synchronous, at most 32 per tile visit, just like the GPU. An unchanged round certifies
 * convergence and consumes one round of the caller's budget.
 * @internal
 */
export function relaxCostDistanceRasterOnCPU(options: {
  width: number;
  height: number;
  values: Float64Array;
  maximumIterations: number;
  sweepAfterIteration: number;
  costLimit: number;
  participates: (cell: number) => boolean;
  getEdgeCost: (cell: number, neighbor: number, direction: number) => number;
  settledAt?: Uint32Array;
}): {converged: boolean; iterationCount: number} {
  const {width, height, values} = options;
  const tilesX = Math.ceil(width / TILE_SIZE);
  const tilesY = Math.ceil(height / TILE_SIZE);
  const edgeCosts = new Float64Array(values.length * 8).fill(Infinity);
  for (let cell = 0; cell < values.length; cell++) {
    if (!options.participates(cell)) continue;
    const column = cell % width;
    const row = Math.floor(cell / width);
    for (let direction = 0; direction < 8; direction++) {
      const neighborColumn = column + COLUMN_OFFSETS[direction];
      const neighborRow = row + ROW_OFFSETS[direction];
      if (neighborColumn < 0 || neighborColumn >= width || neighborRow < 0 || neighborRow >= height)
        continue;
      const neighbor = neighborRow * width + neighborColumn;
      if (options.participates(neighbor))
        edgeCosts[cell * 8 + direction] = options.getEdgeCost(cell, neighbor, direction);
    }
  }
  const halo = new Float64Array(HALO_SIZE * HALO_SIZE);
  const next = new Float64Array(TILE_SIZE * TILE_SIZE);
  const relaxTile = (tile: number, read: Float64Array, iteration: number): boolean => {
    const originColumn = (tile % tilesX) * TILE_SIZE;
    const originRow = Math.floor(tile / tilesX) * TILE_SIZE;
    halo.fill(Infinity);
    for (let haloRow = 0; haloRow < HALO_SIZE; haloRow++) {
      for (let haloColumn = 0; haloColumn < HALO_SIZE; haloColumn++) {
        const column = originColumn + haloColumn - 1;
        const row = originRow + haloRow - 1;
        if (column >= 0 && column < width && row >= 0 && row < height)
          halo[haloRow * HALO_SIZE + haloColumn] = read[row * width + column];
      }
    }
    for (let inner = 0; inner < 2 * TILE_SIZE; inner++) {
      let changed = false;
      for (let localRow = 0; localRow < TILE_SIZE; localRow++) {
        for (let localColumn = 0; localColumn < TILE_SIZE; localColumn++) {
          const column = originColumn + localColumn;
          const row = originRow + localRow;
          const cell = row * width + column;
          const center = (localRow + 1) * HALO_SIZE + localColumn + 1;
          let best = halo[center];
          if (column < width && row < height && options.participates(cell)) {
            for (let direction = 0; direction < 8; direction++) {
              const neighbor =
                center + ROW_OFFSETS[direction] * HALO_SIZE + COLUMN_OFFSETS[direction];
              const candidate = halo[neighbor] + edgeCosts[cell * 8 + direction];
              if (candidate <= options.costLimit && candidate < best) best = candidate;
            }
          }
          next[localRow * TILE_SIZE + localColumn] = best;
          if (best < halo[center]) changed = true;
        }
      }
      if (!changed) break;
      for (let localRow = 0; localRow < TILE_SIZE; localRow++) {
        for (let localColumn = 0; localColumn < TILE_SIZE; localColumn++) {
          halo[(localRow + 1) * HALO_SIZE + localColumn + 1] =
            next[localRow * TILE_SIZE + localColumn];
        }
      }
    }
    let changed = false;
    for (let localRow = 0; localRow < TILE_SIZE; localRow++) {
      for (let localColumn = 0; localColumn < TILE_SIZE; localColumn++) {
        const column = originColumn + localColumn;
        const row = originRow + localRow;
        if (column >= width || row >= height) continue;
        const cell = row * width + column;
        const value = halo[(localRow + 1) * HALO_SIZE + localColumn + 1];
        if (value < read[cell]) {
          // Sweep lines write to their own snapshot and publish the same decrease globally.
          if (read !== snapshot) read[cell] = value;
          values[cell] = value;
          if (options.settledAt) options.settledAt[cell] = iteration + 1;
          changed = true;
        }
      }
    }
    return changed;
  };
  let snapshot = values.slice();
  for (let iteration = 0; iteration < options.maximumIterations; iteration++) {
    snapshot = values.slice();
    const phase =
      iteration < options.sweepAfterIteration ? 0 : (iteration - options.sweepAfterIteration) % 5;
    let changed = false;
    if (phase === 0) {
      for (let tile = 0; tile < tilesX * tilesY; tile++) {
        changed = relaxTile(tile, snapshot, iteration) || changed;
      }
    } else {
      const vertical = phase >= 3;
      const reverse = phase === 2 || phase === 4;
      const lineCount = vertical ? tilesX : tilesY;
      const stepCount = vertical ? tilesY : tilesX;
      for (let line = 0; line < lineCount; line++) {
        const lineValues = snapshot.slice();
        for (let step = 0; step < stepCount; step++) {
          const position = reverse ? stepCount - 1 - step : step;
          const tile = vertical ? position * tilesX + line : line * tilesX + position;
          changed = relaxTile(tile, lineValues, iteration) || changed;
        }
      }
    }
    if (!changed) return {converged: true, iterationCount: iteration + 1};
  }
  return {converged: false, iterationCount: options.maximumIterations};
}
