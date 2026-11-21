// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import * as arrow from 'apache-arrow';
import type {Device} from '@luma.gl/core';
import {
  applyLensUserInput,
  getLensScenario,
  getLensUserInputDatasetId,
  listLensUserInputRoles,
  LensUserInputRoleError,
  runLensScenarioWithUserInput,
  type LensScenarioRunOptions,
  type LensScenarioDescriptor
} from '@luma-contributors/lenses/computation-lenses';
import {mapCatalogLensUserData} from '../src/analysis/catalog-lens';
import {readLensUserData} from '../../../examples/deck/kepler-compute/src/analysis-catalog-lens-mapping';
import {loadDataFrameLensData} from '../../lenses/src/computation-lenses/scenarios/gpu-dataframe/dataframe-lens-support';
import '../../lenses/src/computation-lenses/scenarios/gpu-dataframe/column-quantiles.lens';

test('quantile roles and required-role preflight use the declaration without GPU loading', async () => {
  const descriptor = getLensScenario('gpu-dataframe.column-quantiles')!;
  const listing = await listLensUserInputRoles(descriptor);
  expect(listing?.roles).toEqual([
    {port: 'values', role: 'poverty150Pct', required: true, missing: 'exclude', example: 27.6}
  ]);
  await expect(
    runLensScenarioWithUserInput(
      {
        ...descriptor,
        load: async () => {
          throw new Error('must not load GPU');
        }
      },
      {} as Device,
      {datasetId: 'f-chicago-tracts', rowCount: 1, columns: {}}
    )
  ).rejects.toThrow(/poverty150Pct/);
});

for (const datasetId of ['f-chicago-nature', 'f-ghcn-stations'] as const) {
  test(`run resolver reaches the ${datasetId} loader branch`, async () => {
    const descriptor = makeDescriptor(
      async (_device, options) => {
        const dataset = await loadDataFrameLensData(datasetId, options?.datasetResolver);
        expect(dataset.rowCount).toBe(1);
        expect((dataset.columns as Record<string, number[]>)['mapped']).toEqual([99]);
        return {} as never;
      },
      {dataset: {id: datasetId, licence: 'test', source: 'test'}}
    );
    expect(
      (
        await runLensScenarioWithUserInput(descriptor, {} as Device, {
          datasetId,
          rowCount: 1,
          columns: {mapped: [99]}
        })
      ).accessedRoles
    ).toEqual(['mapped']);
  });
}

test('interleaved mapped loaders leave ordinary embedded loads isolated, including failure', async () => {
  let release!: () => void;
  let entered!: () => void;
  const paused = new Promise<void>(resolve => {
    release = resolve;
  });
  const started = new Promise<void>(resolve => {
    entered = resolve;
  });
  const first = runLensScenarioWithUserInput(
    makeDescriptor(async (_device, options) => {
      entered();
      await paused;
      const dataset = await loadDataFrameLensData('f-chicago-tracts', options?.datasetResolver);
      expect([dataset.rowCount, dataset.columns.poverty150Pct[0]]).toEqual([1, 99]);
      return {} as never;
    }),
    {} as Device,
    {datasetId: 'f-chicago-tracts', rowCount: 1, columns: {poverty150Pct: [99]}}
  );
  await started;
  const embedded = await loadDataFrameLensData('f-chicago-tracts');
  expect([embedded.rowCount, embedded.columns.poverty150Pct[0]]).toEqual([791, 27.6]);
  const failure = await runLensScenarioWithUserInput(
    makeDescriptor(async (_device, options) => {
      const dataset = await loadDataFrameLensData('f-chicago-tracts', options?.datasetResolver);
      expect(dataset.columns.poverty150Pct).toEqual([42]);
      void Reflect.get(dataset.columns, 'income');
      return {} as never;
    }),
    {} as Device,
    {datasetId: 'f-chicago-tracts', rowCount: 1, columns: {poverty150Pct: [42]}}
  ).catch(error => error);
  expect(failure).toBeInstanceOf(LensUserInputRoleError);
  expect(failure.accessedRoles).toEqual(['poverty150Pct', 'income']);
  release();
  expect((await first).accessedRoles).toEqual(['poverty150Pct']);
  expect((await loadDataFrameLensData('f-chicago-tracts')).rowCount).toBe(791);
});

test('panel mapping preserves missing measurements and genuine zero', () => {
  const values = [null, 0, NaN, undefined, 12];
  const mapped = readLensUserData(
    {
      id: 'native',
      fields: [{name: 'score', type: 'real'}],
      dataContainer: {numRows: () => 5, valueAt: row => values[row]}
    },
    {poverty150Pct: 'score'}
  );
  expect(Array.from(mapped.columns['poverty150Pct'])).toEqual([null, 0, null, null, 12]);
});

const embedded = {id: 'f-chicago-tracts', rowCount: 3, columns: {a: [1, 2, 3], b: [4, 5, 6]}};

function makeDescriptor(
  run: (device: Device, options?: LensScenarioRunOptions) => Promise<never>,
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
  const descriptor = makeDescriptor(async (_device, options) => {
    const dataset = applyLensUserInput(embedded, options?.datasetResolver);
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
  const descriptor = makeDescriptor(async (_device, options) => {
    void applyLensUserInput(embedded, options?.datasetResolver).columns.b;
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
  expect(Array.from(mapped.columns['a'])).toEqual([1, null, 3]);
  expect(() => mapCatalogLensUserData({a: 'z'}, 3, () => undefined)).toThrow(/not numeric/);
});

test('native Arrow Int64 measurements preserve safe integers and exclude unsafe integers', () => {
  const vector = arrow.vectorFromArray([-8n, null, 6n, 0n, 9007199254740993n], new arrow.Int64());
  const mapped = readLensUserData(
    {
      id: 'integers',
      fields: [{name: 'score', type: 'integer'}],
      dataContainer: {numRows: () => vector.length, valueAt: row => vector.get(row)}
    },
    {poverty150Pct: 'score'}
  );
  expect(Array.from(mapped.columns['poverty150Pct'])).toEqual([-8, null, 6, 0, null]);
});

test('legacy mapped loaders refuse a silently ignored mapping', async () => {
  const descriptor = makeDescriptor(async () => {
    const dataset = await loadDataFrameLensData('f-chicago-tracts');
    expect([dataset.rowCount, dataset.columns.poverty150Pct[0]]).toEqual([791, 27.6]);
    return {} as never;
  });
  await expect(
    runLensScenarioWithUserInput(descriptor, {} as Device, {
      datasetId: 'f-chicago-tracts',
      rowCount: 2,
      columns: {poverty150Pct: [-4, 2]}
    })
  ).rejects.toThrow(/datasetResolver/);
});

for (const length of [1, 41])
  test('mapped column length ' + length + ' must match rowCount before loading', async () => {
    let loaded = false;
    const descriptor = {
      ...getLensScenario('gpu-dataframe.column-quantiles')!,
      load: async () => {
        loaded = true;
        throw new Error('unexpected load');
      }
    };
    await expect(
      runLensScenarioWithUserInput(descriptor, {} as Device, {
        datasetId: 'f-chicago-tracts',
        rowCount: 40,
        columns: {poverty150Pct: Array(length).fill(2)}
      })
    ).rejects.toThrow(/poverty150Pct.*length.*40/);
    expect(loaded).toBe(false);
  });
