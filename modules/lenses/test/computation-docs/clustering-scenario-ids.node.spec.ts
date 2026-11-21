// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getLensElementId} from './lens-directives';
import {extractInventory} from '../../../../website/scripts/extract-inventory';

const SCENARIOS = [
  ['GPUDBSCAN', 'spatial-clustering'],
  ['GPUHDBSCAN', 'hierarchical-density-clustering']
] as const;

test.each(SCENARIOS)('keeps the docs lens ID for %s', (symbol, scenario) => {
  expect(getLensElementId(symbol)).toBe(scenario);
});

test('inventory preserves primary scenario IDs and archived secondary links', async () => {
  const {symbols} = await extractInventory();
  const scenariosBySymbol = {
    GPUDBSCAN: [
      'gpu-spatial-analysis.spatial-clustering',
      'gpu-spatial-analysis.cluster-and-outline-recipe',
      'gpu-spatial-analysis.group-convex-hull'
    ],
    GPUHDBSCAN: ['gpu-spatial-analysis.hierarchical-density-clustering'],
    GPUKeyLookup: [
      'gpu-dataframe.key-join',
      'gpu-spatial-analysis.population-support-alignment-recipe',
      'gpu-spatial-analysis.population-support-alignment'
    ],
    GPUPycnophylacticInterpolation: ['gpu-spatial-analysis.pycnophylactic']
  };
  for (const [symbol, [primary, ...secondary]] of Object.entries(scenariosBySymbol)) {
    const entry = symbols.find(candidate => candidate.symbol === symbol);
    expect(entry?.lensScenarios[0], symbol).toBe(primary);
    expect(entry?.lensScenarios.slice(1), symbol).toEqual(expect.arrayContaining(secondary));
  }
}, 60000);
