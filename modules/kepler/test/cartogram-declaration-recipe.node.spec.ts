// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  parseKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';

const dorling = {
  kind: 'dorling-cartogram',
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  crs: 'EPSG:32618',
  bounds: [0, 0, 1000, 1000],
  xColumn: 'east',
  yColumn: 'north',
  valueColumn: 'area',
  areaScale: 100,
  maximumIterations: 0,
  overlapTolerance: 1,
  attraction: 0
};
const base = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'recipe',
  source: {datasetId: 'data', revision: 1},
  population: 'all',
  backendPolicy: 'require-gpu',
  operation: dorling
};
const declaration = {metersPerUnit: 0.3048, planeOrigin: [1000, 2000], anchor: [-87.6, 41.9]};

test('a cartogram declaration round trips through recipe serialization', () => {
  const recipe = parseKeplerAnalysisRecipe({...base, display: {cartogramDeclaration: declaration}});
  expect(recipe.display?.cartogramDeclaration).toEqual(declaration);
  const restored = parseKeplerAnalysisRecipe(JSON.parse(serializeKeplerAnalysisRecipe(recipe)));
  expect(restored.display).toEqual({cartogramDeclaration: declaration});
  expect(parseKeplerAnalysisRecipe(base).display).toBeUndefined();
});

test('declarations are validated and limited to cartograms', () => {
  const parse = (cartogramDeclaration: unknown, operation: unknown = dorling) =>
    parseKeplerAnalysisRecipe({...base, operation, display: {cartogramDeclaration}});
  expect(() => parse({...declaration, metersPerUnit: 0})).toThrow(/declaration/);
  expect(() => parse({...declaration, anchor: [-87.6, 91]})).toThrow(/declaration/);
  expect(() => parse({...declaration, planeOrigin: [1, Number.NaN]})).toThrow(/declaration/);
  expect(() => parse({...declaration, extra: 1})).toThrow(/Unknown/);
  expect(() =>
    parse(declaration, {kind: 'standardize', column: 'a', method: 'z-score', variance: 'sample'})
  ).toThrow(/needs a cartogram/);
  expect(() => parseKeplerAnalysisRecipe({...base, display: {other: 1}})).toThrow(/Unknown/);
});
