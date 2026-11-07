// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  applyLensUserInput,
  getLensUserInputDatasetId,
  listLensUserInputRoles,
  LensUserInputRoleError,
  runLensScenarioWithUserInput,
  type LensScenarioDescriptor
} from '@luma-contributors/lenses/computation-lenses';
import {mapCatalogLensUserData} from '../src/analysis/catalog-lens';

const embedded = {id: 'f-chicago-tracts', rowCount: 3, columns: {a: [1, 2, 3], b: [4, 5, 6]}};

function makeDescriptor(
  run: () => Promise<never>,
  overrides: Partial<LensScenarioDescriptor> = {}
): LensScenarioDescriptor {
  return {
    id: 'gpu-dataframe.test',
    entry: 'gpu-dataframe',
    element: 'GPUTest',
    title: 'Test',
    audience: 'engineer',
    rungs: ['result'],
    defaultRung: 'result',
    dataset: {id: 'f-chicago-tracts', licence: 'test', source: 'test'},
    load: async () => ({run}),
    ...overrides
  };
}

test('mapped columns stand in for the embedded dataset only while a run is active', async () => {
  expect(applyLensUserInput(embedded)).toBe(embedded);
  let seen: number[] = [];
  const descriptor = makeDescriptor(async () => {
    const dataset = applyLensUserInput(embedded);
    seen = [dataset.rowCount, ...dataset.columns.a];
    return {} as never;
  });
  const result = await runLensScenarioWithUserInput(descriptor, {} as Device, {
    datasetId: 'f-chicago-tracts',
    rowCount: 2,
    columns: {a: [10, null]}
  });
  expect(seen).toEqual([2, 10, null]);
  expect(result.accessedRoles).toEqual(['a']);
  expect(applyLensUserInput(embedded)).toBe(embedded);
});

test('an unmapped role fails with its name and the run state is released', async () => {
  const descriptor = makeDescriptor(async () => {
    void applyLensUserInput(embedded).columns.b;
    return {} as never;
  });
  const failure = await runLensScenarioWithUserInput(descriptor, {} as Device, {
    datasetId: 'f-chicago-tracts',
    rowCount: 1,
    columns: {a: [1]}
  }).catch(error => error);
  expect(failure).toBeInstanceOf(LensUserInputRoleError);
  expect(failure.role).toBe('b');
  expect(applyLensUserInput(embedded)).toBe(embedded);
});

test('only dataframe scenarios on embedded tables accept user data', async () => {
  const descriptor = makeDescriptor(async () => ({}) as never);
  expect(getLensUserInputDatasetId(descriptor)).toBe('f-chicago-tracts');
  const spatial = makeDescriptor(async () => ({}) as never, {entry: 'gpu-spatial-analysis'});
  expect(getLensUserInputDatasetId(spatial)).toBeUndefined();
  await expect(
    runLensScenarioWithUserInput(spatial, {} as Device, {
      datasetId: 'f-chicago-tracts',
      rowCount: 1,
      columns: {}
    })
  ).rejects.toThrow(/cannot run on user data/);
  const listing = await listLensUserInputRoles(descriptor);
  expect(listing?.roles.map(role => role.role)).toContain('poverty150Pct');
});

test('field mapping builds numeric columns and rejects missing fields', () => {
  const table = {x: [1, Number.NaN, 3], y: [4, 5, 6]};
  const mapped = mapCatalogLensUserData(
    {a: 'x', b: '', c: 'y'},
    3,
    name => table[name as 'x' | 'y']
  );
  expect(Object.keys(mapped.columns)).toEqual(['a', 'c']);
  expect([...mapped.columns.a]).toEqual([1, null, 3]);
  expect(() => mapCatalogLensUserData({a: 'z'}, 3, () => undefined)).toThrow(/not numeric/);
});
