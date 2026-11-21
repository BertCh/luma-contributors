// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, test} from 'vitest';
import {
  createDataFramePackData,
  createDataFrameElementPackData,
  getDataFrameOperatorEntries,
  DATAFRAME_FAMILY_OPERATORS
} from '../../../src/gpu-dataframe/computation-pack/pack';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';

const scopes = ['local', 'focal', 'block', 'zonal', 'global', 'frontier', 'pairwise', 'solver'];
const reductions = [
  'none',
  'count',
  'sum',
  'weighted-mean',
  'argmin',
  'histogram',
  'mode',
  'select',
  'scan-sort',
  'solve'
];

test('pack merges seven families without duplicates and scopes each element variable kinds', () => {
  const pack = createDataFramePackData();
  const entries = getDataFrameOperatorEntries();
  expect(pack.id).toBe('gpu-dataframe');
  expect(DATAFRAME_FAMILY_OPERATORS).toHaveLength(7);
  expect(
    DATAFRAME_FAMILY_OPERATORS.filter(family => 'GPUVariableStandardization' in family)
  ).toHaveLength(1);
  expect(entries['GPUVariableStandardization'].family).toBe('classification');
  expect(pack.operators?.['GPUVariableStandardization']?.reduction).toBe('sum');
  const keys = DATAFRAME_FAMILY_OPERATORS.flatMap(family => Object.keys(family));
  expect(new Set(keys).size).toBe(keys.length);
  expect(Object.keys(entries)).toHaveLength(keys.length);
  expect(pack.analysisStages).toEqual([
    'select',
    'filter',
    'derive',
    'join',
    'group',
    'aggregate',
    'order',
    'classify',
    'summarise',
    'infer'
  ]);
  for (const [name, entry] of Object.entries(entries)) {
    expect(scopes).toContain(entry.descriptor.scope);
    if (entry.descriptor.scopeSecondary) expect(scopes).toContain(entry.descriptor.scopeSecondary);
    expect(reductions).toContain(entry.descriptor.reduction);
    expect(entry.stages.every(stage => pack.analysisStages?.includes(stage))).toBe(true);
    expect(entry.datumKinds.length).toBeGreaterThan(0);
    expect(createDataFrameElementPackData(name).variableKinds).toEqual(entry.variables ?? {});
    expect(pack.operators?.[name]?.variableKinds).toEqual(entry.variables ?? {});
    for (const file of entry.sourceFiles)
      expect(readFileSync(resolve(file), 'utf8').length).toBeGreaterThan(0);
  }
  expect(createDataFrameElementPackData('missing').variableKinds).toEqual({});
  expect(
    createDataFrameElementPackData('GPUClassBreaks', {threshold: 1000}).metadata?.['threshold']
  ).toBe(1000);
});

test('every curated formula has exactly its tagged symbols and real source identifiers', () => {
  for (const [name, entry] of Object.entries(getDataFrameOperatorEntries())) {
    const formula = entry.formula;
    if ('formula' in formula) {
      expect(formula.formula).toBe('none');
      expect(formula.reason.length).toBeGreaterThan(10);
      continue;
    }
    expect(formula.derivation).toBe('curated');
    const formulaName = name === 'GPUKeyLookup' ? 'GPUKeyJoin' : name;
    expect(formula.id).toBe(`gpu-dataframe:${formulaName}`);
    const terms = new Set(listTexTerms(formula.tex).map(term => term.symbolId));
    expect(terms).toEqual(new Set(formula.symbols.map(symbol => symbol.id)));
    expect(terms.size).toBe(formula.symbols.length);
    const sources = entry.sourceFiles.map(file => readFileSync(resolve(file), 'utf8')).join('\n');
    for (const symbol of formula.symbols) {
      expect(symbol.binding).toMatch(/^wgsl:/);
      expect(sources, `${name}: ${symbol.binding}`).toMatch(
        new RegExp(`\\b${symbol.binding!.slice(5)}\\b`)
      );
    }
  }
});
