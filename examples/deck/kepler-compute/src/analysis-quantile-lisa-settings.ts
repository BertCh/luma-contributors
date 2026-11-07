// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS} from '@luma-contributors/kepler';
import {
  normalizePointTopologySettings,
  type PointTopologySettings
} from './analysis-point-weights-settings';

export type QuantileLisaSettings = PointTopologySettings & {
  quantileCount: number;
  selectedQuantile: number;
  seed: number;
  permutationCount: number;
  significanceLevel: number;
};

export function normalizeQuantileLisaSettings(
  settings: QuantileLisaSettings,
  rowCount: number
): QuantileLisaSettings {
  const topology = normalizePointTopologySettings(settings, rowCount);
  if (rowCount < 3 || rowCount >= 2 ** 24)
    throw new Error('Quantile LISA requires 3 to 16777215 source rows');
  if (
    !Number.isSafeInteger(settings.quantileCount) ||
    settings.quantileCount < 2 ||
    settings.quantileCount >= rowCount
  )
    throw new Error('Quantile count must be at least 2 and smaller than the source row count');
  if (
    !Number.isSafeInteger(settings.selectedQuantile) ||
    settings.selectedQuantile < 1 ||
    settings.selectedQuantile > settings.quantileCount
  )
    throw new Error('Choose a quantile from 1 through the quantile count');
  if (!Number.isSafeInteger(settings.seed) || settings.seed < 0)
    throw new Error('Permutation seed must be a nonnegative safe integer');
  if (
    !Number.isSafeInteger(settings.permutationCount) ||
    settings.permutationCount < 1 ||
    settings.permutationCount > MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS
  )
    throw new Error(`Choose 1 to ${MAXIMUM_KEPLER_QUANTILE_LISA_PERMUTATIONS} permutations`);
  const significanceLevel = Math.fround(settings.significanceLevel);
  if (!(significanceLevel > 0 && significanceLevel < 1))
    throw new Error('Quantile LISA significance must remain between 0 and 1 in Float32');
  return {
    ...topology,
    quantileCount: settings.quantileCount,
    selectedQuantile: settings.selectedQuantile,
    seed: settings.seed,
    permutationCount: settings.permutationCount,
    significanceLevel: settings.significanceLevel
  };
}
