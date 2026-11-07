// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerMultipleTestingMethod} from '@luma-contributors/kepler';

/** Source rows must already describe this explicitly chosen row-major lattice. */
export type HotspotSettings = {
  width: number;
  height: number;
  criterion: 'rook' | 'queen';
  radius: number;
  minimumNeighbors: number;
  selfWeight: number;
  multipleTesting: KeplerMultipleTestingMethod;
  /** Restored explicit capacity; omitted configuration computes the full unmasked bound. */
  neighborCapacity?: number;
};

export function normalizeHotspotSettings(
  settings: HotspotSettings,
  rowCount: number
): HotspotSettings & {neighborCapacity: number} {
  if (
    !Number.isSafeInteger(rowCount) ||
    rowCount < 1 ||
    rowCount > 2 ** 24 ||
    !Number.isSafeInteger(settings.width) ||
    settings.width < 1 ||
    !Number.isSafeInteger(settings.height) ||
    settings.height < 1 ||
    settings.width * settings.height !== rowCount
  )
    throw new Error('Hotspot lattice width × height must equal source rows (at most 16777216)');
  if (!Number.isInteger(settings.radius) || settings.radius < 1 || settings.radius > 32)
    throw new Error('Hotspot radius must be an integer from 1 to 32');
  if (settings.criterion !== 'rook' && settings.criterion !== 'queen')
    throw new Error('Choose rook or queen lattice neighbors');
  if (
    !Number.isSafeInteger(settings.minimumNeighbors) ||
    settings.minimumNeighbors < 0 ||
    settings.minimumNeighbors > 2 ** 24
  )
    throw new Error('Invalid hotspot minimum neighbor count');
  if (!Number.isFinite(Math.fround(settings.selfWeight)) || settings.selfWeight < 0)
    throw new Error('Hotspot self weight must be finite and nonnegative');
  if (
    !['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].includes(
      settings.multipleTesting
    )
  )
    throw new Error('Unsupported hotspot multiple-testing method');
  // Every offset contributes exactly the number of source/destination pairs inside the shape.
  // This is a conservative bound for any selection, computed in at most (2 * 32 + 1)^2 steps.
  let requiredCount = 0;
  const horizontalRadius = Math.min(settings.radius, settings.width - 1);
  const verticalRadius = Math.min(settings.radius, settings.height - 1);
  for (let verticalOffset = -verticalRadius; verticalOffset <= verticalRadius; verticalOffset++) {
    for (
      let horizontalOffset = -horizontalRadius;
      horizontalOffset <= horizontalRadius;
      horizontalOffset++
    ) {
      if (
        (!horizontalOffset && !verticalOffset) ||
        (settings.criterion === 'rook' &&
          Math.abs(horizontalOffset) + Math.abs(verticalOffset) > settings.radius)
      )
        continue;
      requiredCount +=
        (settings.width - Math.abs(horizontalOffset)) *
        (settings.height - Math.abs(verticalOffset));
    }
  }
  const neighborCapacity = settings.neighborCapacity ?? requiredCount;
  if (
    !Number.isSafeInteger(neighborCapacity) ||
    neighborCapacity < requiredCount ||
    neighborCapacity > 2 ** 31 - 1
  )
    throw new Error(
      `Hotspot neighbor capacity must cover ${requiredCount} slots and fit the 2147483647-slot limit`
    );
  const scratchBytes = rowCount * 192 + (rowCount + 1) * 4 + neighborCapacity * 8 + 4 * 1024 * 1024;
  if (!Number.isSafeInteger(scratchBytes))
    throw new Error('Hotspot memory estimate is not a safe integer');
  return {
    width: settings.width,
    height: settings.height,
    criterion: settings.criterion,
    radius: settings.radius,
    minimumNeighbors: settings.minimumNeighbors,
    selfWeight: settings.selfWeight,
    multipleTesting: settings.multipleTesting,
    neighborCapacity
  };
}

/** Additional CSR and inference scratch beyond the runtime's ordinary per-row analysis reserve. */
export function getHotspotScratchBytes(settings: HotspotSettings, rowCount: number): number {
  const normalized = normalizeHotspotSettings(settings, rowCount);
  return (rowCount + 1) * 4 + normalized.neighborCapacity * 8 + rowCount * 64;
}
