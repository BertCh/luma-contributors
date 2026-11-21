// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {WEIGHTS_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/weights';
import {expectFormulaBindingsExist, expectSpatialEntryWellFormed} from './formula-bindings';

const SOURCE_DIRECTORY = 'modules/analysis/src/gpu-spatial-analysis';

const kernelSources: Record<string, string[]> = {
  GPUAnalysisSpacing: [`${SOURCE_DIRECTORY}/analysis-spacing/analysis-spacing-statistics-nodes.ts`],
  GPUAnalysisContiguityStatistics: [
    `${SOURCE_DIRECTORY}/spatial-weights/gpu-analysis-contiguity-statistics.ts`
  ],
  GPUContiguityWeights: [`${SOURCE_DIRECTORY}/spatial-weights/gpu-contiguity-weights.ts`],
  GPULatticeWeights: [`${SOURCE_DIRECTORY}/spatial-weights/gpu-lattice-weights.ts`],
  GPUSpatialWeightsTransform: [
    `${SOURCE_DIRECTORY}/spatial-weights/gpu-spatial-weights-transform.ts`
  ],
  GPUChunkedSpatialWeightsTransform: [
    `${SOURCE_DIRECTORY}/spatial-weights/gpu-spatial-weights-transform.ts`
  ],
  GPUSpatialLag: [`${SOURCE_DIRECTORY}/spatial-weights/gpu-spatial-lag.ts`],
  GPUNeighborSearch: [
    `${SOURCE_DIRECTORY}/neighbor-search/gpu-neighbor-search.ts`,
    `${SOURCE_DIRECTORY}/neighbor-search/neighbor-search-kernels.ts`
  ],
  GPUSpatialWeightsSummary: [`${SOURCE_DIRECTORY}/spatial-weights/gpu-spatial-weights-summary.ts`]
};

const EXPECTED_ELEMENTS = [
  'GPUAnalysisSpacing',
  'GPUAnalysisContiguityStatistics',
  'GPUContiguityWeights',
  'GPULatticeWeights',
  'GPUSpatialWeightsTransform',
  'GPUSpatialLag',
  'GPUNeighborSearch',
  'GPUSpatialWeightsAlgebra',
  'GPUSpatialWeightsSummary',
  'GPUSpatialWeightsTranspose',
  'GPUSpatialWeightsValidation',
  'GPUMapColoring',
  'GPUChunkedSpatialWeightsValidation',
  'GPUChunkedSpatialWeightsTranspose',
  'GPUChunkedSpatialWeightsTransform'
];

test('the weights pack covers exactly its fifteen elements', () => {
  expect(Object.keys(WEIGHTS_OPERATORS).sort()).toEqual([...EXPECTED_ELEMENTS].sort());
});

test('chunked validation and transpose explicitly have no formula', () => {
  for (const name of ['GPUChunkedSpatialWeightsValidation', 'GPUChunkedSpatialWeightsTranspose'])
    expect(WEIGHTS_OPERATORS[name].formula).toMatchObject({formula: 'none'});
});

test('weights algebra advertises the canonical XOR spelling', () => {
  const choice = WEIGHTS_OPERATORS['GPUSpatialWeightsAlgebra'].choices.find(
    choice => typeof choice !== 'string' && choice.name === 'operation'
  );
  expect(choice).toMatchObject({
    alternatives: expect.arrayContaining(['symmetric-difference'])
  });
  expect(choice).not.toMatchObject({
    alternatives: expect.arrayContaining(['symmetricDifference'])
  });
});

for (const [name, entry] of Object.entries(WEIGHTS_OPERATORS)) {
  test(`${name}: weights pack and curated kernel bindings`, () => {
    expectSpatialEntryWellFormed(`weights/${name}`, entry);
    if ('formula' in entry.formula) {
      expect(kernelSources[name], `${name} has no formula, so no kernel sources`).toBeUndefined();
      return;
    }
    expect(kernelSources[name], `Add ${name}'s WGSL generator sources`).toBeDefined();
    expectFormulaBindingsExist(entry.formula, kernelSources[name]);
  });
}

test('formula binding checks reject stale identifiers for the weights formulas', () => {
  const formula = WEIGHTS_OPERATORS['GPUSpatialLag'].formula;
  if ('formula' in formula) throw new Error('Spatial lag must have a curated formula');
  expect(() =>
    expectFormulaBindingsExist(
      {
        ...formula,
        symbols: formula.symbols.map((symbol, index) =>
          index === 0 ? {...symbol, binding: 'wgsl:missingKernelIdentifier'} : symbol
        )
      },
      kernelSources['GPUSpatialLag']
    )
  ).toThrow();
});
