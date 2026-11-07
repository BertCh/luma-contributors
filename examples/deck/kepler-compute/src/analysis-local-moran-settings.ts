// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerMultipleTestingMethod} from '@luma-contributors/kepler';
import {
  normalizePointTopologySettings,
  type PointTopologySettings
} from './analysis-point-weights-settings';

/** Explicit binary point weights and the selected conditional inference for local Moran's I. */
export type LocalMoranSettings = PointTopologySettings & {
  minimumNeighbors: number;
  islands: 'exclude' | 'keep';
  multipleTesting: KeplerMultipleTestingMethod;
  significanceLevel: number;
  inference?: 'analytic-conditional-normal-two-sided' | 'permutation-conditional-two-sided';
  permutationCount?: number;
  seed?: number;
  /** Bivariate mode: the weighted lag is taken over this column; requires permutations. */
  secondColumn?: string;
  /**
   * Multivariate Local Geary: 2 to 8 distinct columns (first is the primary column). Conditional
   * permutations only, up to 9999 draws, up to 2^24-1 rows (tiled above 256), correction none or bonferroni.
   */
  variableColumns?: string[];
};

export function normalizeLocalMoranSettings(
  settings: LocalMoranSettings,
  rowCount: number
): LocalMoranSettings {
  const topology = normalizePointTopologySettings(settings, rowCount);
  if (rowCount >= 2 ** 24) throw new Error('Local Moran requires fewer than 16777216 source rows');
  if (
    !Number.isSafeInteger(settings.minimumNeighbors) ||
    settings.minimumNeighbors < 1 ||
    settings.minimumNeighbors >= 2 ** 24
  )
    throw new Error('Local Moran requires a positive integer minimum neighbor count');
  if (settings.islands !== 'exclude' && settings.islands !== 'keep')
    throw new Error('Choose exclude or keep for Local Moran islands');
  if (
    !['none', 'bonferroni', 'holm', 'benjamini-hochberg', 'benjamini-yekutieli'].includes(
      settings.multipleTesting
    )
  )
    throw new Error('Unsupported Local Moran multiple-testing method');
  const significanceLevel = Math.fround(settings.significanceLevel);
  if (!Number.isFinite(significanceLevel) || significanceLevel <= 0 || significanceLevel >= 1)
    throw new Error('Local Moran significance must remain between 0 and 1 in Float32');
  const inference = settings.inference ?? 'analytic-conditional-normal-two-sided';
  if (
    inference !== 'analytic-conditional-normal-two-sided' &&
    inference !== 'permutation-conditional-two-sided'
  )
    throw new Error('Choose analytic or conditional permutation inference for Local Moran');
  if (inference === 'permutation-conditional-two-sided') {
    if (rowCount >= 2 ** 24 || settings.minimumNeighbors !== 1)
      throw new Error(
        'Permutation Local Moran requires fewer than 2^24 rows and minimum neighbors 1'
      );
    if (
      !Number.isInteger(settings.permutationCount) ||
      settings.permutationCount === undefined ||
      settings.permutationCount < 1 ||
      settings.permutationCount > 9999
    )
      throw new Error('Local Moran requires 1 to 9999 permutations');
    if (
      !Number.isInteger(settings.seed) ||
      settings.seed === undefined ||
      settings.seed < 0 ||
      settings.seed > 0xffffffff
    )
      throw new Error('Local Moran permutation seed must be an unsigned 32-bit integer');
  }
  if (settings.variableColumns !== undefined) {
    const variables = settings.variableColumns;
    if (
      settings.secondColumn !== undefined ||
      variables.length < 2 ||
      variables.length > 8 ||
      new Set(variables).size !== variables.length
    )
      throw new Error('Multivariate Local Geary requires 2 to 8 distinct variables');
    if (rowCount >= 2 ** 24 || rowCount < 2)
      throw new Error('Multivariate Local Geary supports 2 to 2^24-1 source rows');
    if (
      inference !== 'permutation-conditional-two-sided' ||
      (settings.permutationCount ?? 0) > 9999 ||
      (settings.multipleTesting !== 'none' && settings.multipleTesting !== 'bonferroni')
    )
      throw new Error(
        'Multivariate Local Geary uses up to 9999 permutations and none or Bonferroni correction'
      );
  }
  // Bivariate Local Moran has no analytic variance.
  if (settings.secondColumn !== undefined && inference !== 'permutation-conditional-two-sided')
    throw new Error('Bivariate Local Moran requires conditional permutation inference');
  return {
    ...topology,
    ...(settings.secondColumn === undefined ? {} : {secondColumn: settings.secondColumn}),
    ...(settings.variableColumns === undefined
      ? {}
      : {variableColumns: [...settings.variableColumns]}),
    minimumNeighbors: settings.minimumNeighbors,
    islands: settings.islands,
    multipleTesting: settings.multipleTesting,
    significanceLevel: settings.significanceLevel,
    inference,
    ...(inference === 'permutation-conditional-two-sided'
      ? {permutationCount: settings.permutationCount, seed: settings.seed}
      : {})
  };
}
