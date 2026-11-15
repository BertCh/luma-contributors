// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import {ResidentChartProvider} from '../../../examples/deck/kepler-compute/src/resident-chart-provider';
import type {NativeChartConfiguration} from '../../../examples/deck/kepler-compute/src/resident-chart-plan';

const state = vi.hoisted(() => {
  vi.resetModules();
  return {
    widened: 0,
    destroyedMeasures: 0,
    rejectWidening: false,
    runtimeProps: [] as {measure?: unknown}[]
  };
});
vi.mock('../../../examples/deck/kepler-compute/src/resident-chart-measure-widening', () => ({
  widenResidentChartMeasure: async () => {
    state.widened++;
    // Mirrors the real range check, which rejects before any copy.
    if (state.rejectWidening)
      throw new Error('Integer style values exceed the exact Float32 range of 2^24.');
    return {
      vector: {widened: true},
      destroy() {
        state.destroyedMeasures++;
      }
    };
  }
}));
vi.mock('../../../examples/deck/kepler-compute/src/resident-chart-runtime', () => ({
  ResidentScalarChartRuntime: class {
    constructor(props: {measure?: unknown}) {
      state.runtimeProps.push(props);
    }
    async read() {
      return {value: 1};
    }
    cancel() {}
    async destroy() {}
  }
}));

const makeChart = (id: string, field: string, aggregation: string): NativeChartConfiguration => ({
  id,
  type: 'bigNumber',
  title: id,
  dataId: 'source',
  applyFilters: true,
  display: {},
  chartDisplay: {},
  axis: {field: {name: field, type: 'integer'}, aggregation}
});

function makeProvider() {
  state.widened = 0;
  state.destroyedMeasures = 0;
  state.rejectWidening = false;
  state.runtimeProps = [];
  const explorer = {
    dataset: {
      id: 'source',
      revision: 1,
      value: {
        columns: {whole: {format: 'sint32'}, amount: {format: 'float32'}},
        timeColumns: {},
        prepared: {dictionaries: {}}
      }
    },
    currentRevision: 1,
    isCurrent: true
  } as unknown as KeplerSessionExplorer;
  return new ResidentChartProvider({
    session: {} as KeplerComputeSession,
    explorer,
    subscribe: () => () => undefined
  });
}

test('integer sum, median and deviation share one widened Float32 copy; native extrema and Float32 do not widen', async () => {
  const provider = makeProvider();
  for (const [id, field, aggregation] of [
    ['a', 'whole', 'sum'],
    ['b', 'whole', 'median'],
    ['c', 'whole', 'stdev']
  ]) {
    expect(await provider.read(makeChart(id, field, aggregation))).toMatchObject({status: 'ready'});
  }
  expect(state.widened).toBe(1);
  expect(state.runtimeProps.map(props => props.measure)).toEqual(
    Array.from({length: 3}, () => ({widened: true}))
  );
  await provider.read(makeChart('d', 'whole', 'minimum'));
  await provider.read(makeChart('e', 'amount', 'median'));
  expect(state.widened).toBe(1);
  expect(state.runtimeProps.slice(3).map(props => props.measure)).toEqual([undefined, undefined]);
  await provider.destroy();
  expect(state.destroyedMeasures).toBe(1);
});

test('a source outside the exact Float32 integer range rejects the read and may retry', async () => {
  const provider = makeProvider();
  state.rejectWidening = true;
  await expect(provider.read(makeChart('a', 'whole', 'sum'))).rejects.toThrow('2^24');
  state.rejectWidening = false;
  expect(await provider.read(makeChart('a', 'whole', 'sum'))).toMatchObject({status: 'ready'});
  expect(state.widened).toBe(2);
  await provider.destroy();
  expect(state.destroyedMeasures).toBe(1);
});
