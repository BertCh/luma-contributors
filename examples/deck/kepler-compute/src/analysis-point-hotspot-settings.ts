// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerMultipleTestingMethod} from '@luma-contributors/kepler';
import {
  normalizePointTopologySettings,
  type PointTopologySettings
} from './analysis-point-weights-settings';

export type PointHotspotSettings = PointTopologySettings & {
  minimumNeighbors: number;
  selfWeight: number;
  multipleTesting: KeplerMultipleTestingMethod;
  islands: 'exclude' | 'keep';
};

export function normalizePointHotspotSettings(
  settings: PointHotspotSettings,
  rowCount: number
): PointHotspotSettings {
  const topology = normalizePointTopologySettings(settings, rowCount);
  if (rowCount >= 2 ** 24)
    throw new Error('Point hotspots require fewer than 16777216 source rows');
  if (
    !Number.isSafeInteger(settings.minimumNeighbors) ||
    settings.minimumNeighbors < 1 ||
    settings.minimumNeighbors > 2 ** 24
  )
    throw new Error('Point hotspots require a positive integer minimum neighbor count');
  if (!Number.isFinite(Math.fround(settings.selfWeight)) || settings.selfWeight < 0)
    throw new Error('Point hotspot self weight must be finite and nonnegative');
  if (
    !['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].includes(
      settings.multipleTesting
    )
  )
    throw new Error('Unsupported point hotspot multiple-testing method');
  if (settings.islands !== 'exclude' && settings.islands !== 'keep')
    throw new Error('Choose exclude or keep for point hotspot islands');
  return {
    ...topology,
    minimumNeighbors: settings.minimumNeighbors,
    selfWeight: settings.selfWeight,
    multipleTesting: settings.multipleTesting,
    islands: settings.islands
  };
}
