// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {dirname, join} from 'node:path';
import {runInNewContext} from 'node:vm';
import {expect, test} from 'vitest';
import {
  usesNativeTopologyPredicate,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';
import {normalizeKeplerScalarRange} from '../src/selection/scalar-range';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {planKeplerConsumerFilterIds} from '../src/selection/selection-scope-plan';

type NativeFilter = KeplerComputeFilter & {dataId: string[]; gpu?: boolean};
// Execute the pinned pure functions without importing Kepler's unrelated WebGL renderer graph.
const require = createRequire(import.meta.url);
const gpuSource = readFileSync(
  join(dirname(require.resolve('@kepler.gl/table')), '../esm/gpu-filter-utils.js'),
  'utf8'
);
const filterSource = readFileSync(
  join(dirname(require.resolve('@kepler.gl/utils')), '../esm/filter-utils.js'),
  'utf8'
);
const plotSource = readFileSync(
  join(dirname(require.resolve('@kepler.gl/utils')), '../esm/plot.js'),
  'utf8'
);
const gpuFunctions = [
  'getFilterGpuChannelCount',
  'channelOccupiedByFilter',
  'setFilterGpuMode',
  'assignGpuChannels',
  'assignGpuChannel',
  'assignGpuEndChannel',
  'resetFilterGpuMode'
];
const filterFunctions = [
  'isTimeIntervalFilter',
  'isNumericIntervalFilter',
  'disableGpuIfInverted',
  'isValidFilterValue',
  'getFilterRecord'
];
const extract = (source: string, name: string): string => {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`Pinned native function missing: ${name}`);
  return source.slice(start, source.indexOf('\n}', start) + 2);
};
const native = runInNewContext(
  `${gpuFunctions.map(name => extract(gpuSource, name)).join('\n')}\n${filterFunctions.map(name => extract(filterSource, name)).join('\n')}\n({assignGpuChannel, assignGpuChannels, resetFilterGpuMode, setFilterGpuMode, disableGpuIfInverted, getFilterRecord})`,
  {
    MAX_GPU_FILTERS: 4,
    FILTER_TYPES: {
      range: 'range',
      timeRange: 'timeRange',
      multiSelect: 'multiSelect',
      select: 'select',
      polygon: 'polygon'
    },
    ANIMATION_WINDOW: {interval: 'interval'},
    toArray: (value: unknown) => (Array.isArray(value) ? value : value == null ? [] : [value]),
    _toConsumableArray: (value: unknown[]) => [...value],
    _excluded2: ['gpuChannel', 'gpuEndChannel'],
    _objectSpread: Object.assign,
    _objectWithoutProperties: (value: Record<string, unknown>, excluded: string[]) =>
      Object.fromEntries(Object.entries(value).filter(([key]) => !excluded.includes(key))),
    set: (path: [string | number], value: unknown, object: Record<string, unknown> | unknown[]) =>
      Object.assign(Array.isArray(object) ? [...object] : {...object}, {[path[0]]: value})
  }
) as {
  assignGpuChannel(filter: NativeFilter, filters: NativeFilter[]): NativeFilter;
  assignGpuChannels(filters: NativeFilter[]): NativeFilter[];
  resetFilterGpuMode(filters: NativeFilter[]): NativeFilter[];
  setFilterGpuMode(filter: NativeFilter, filters: NativeFilter[]): NativeFilter;
  disableGpuIfInverted(filter: NativeFilter): NativeFilter;
  getFilterRecord(datasetId: string, filters: NativeFilter[]): {cpu: NativeFilter[]};
};
const {
  assignGpuChannel,
  assignGpuChannels,
  resetFilterGpuMode,
  setFilterGpuMode,
  disableGpuIfInverted,
  getFilterRecord
} = native;

function range(id: string): NativeFilter {
  return {
    id,
    dataId: ['source'],
    name: ['value'],
    type: 'range',
    value: [1, 2],
    gpu: true
  };
}

function expectNativeClassification(filters: ReturnType<typeof range>[]) {
  const native = getFilterRecord('source', filters);
  expect(filters.filter(usesNativeTopologyPredicate).map(filter => filter.id)).toEqual(
    native.cpu.map(filter => filter.id)
  );
}

test('native restore allocation and historical capacity fallback remain authoritative after deletion', () => {
  const restored = assignGpuChannels(
    resetFilterGpuMode(Array.from({length: 5}, (_, index) => range(`range-${index}`)))
  );
  expect(restored.map(filter => filter.gpu)).toEqual([true, true, true, true, false]);
  expectNativeClassification(restored);
  const afterDelete = restored.slice(1);
  expectNativeClassification(afterDelete);
  expect(usesNativeTopologyPredicate(afterDelete[3])).toBe(true);
  const added = assignGpuChannel(setFilterGpuMode(range('new-range'), afterDelete), afterDelete);
  expect(added.gpu).toBe(true);
  expectNativeClassification([...afterDelete, added]);
});

test('inversion and numeric interval playback follow the pinned CPU filterRecord rule', () => {
  const original = assignGpuChannels([range('first'), range('second')]);
  const inverted = disableGpuIfInverted({...original[0], inverted: true});
  expect(inverted.gpu).toBe(false);
  expectNativeClassification([inverted, original[1]]);
  const restored = assignGpuChannel(
    setFilterGpuMode({...inverted, inverted: false, gpu: true}, [inverted, original[1]]),
    [inverted, original[1]]
  );
  expect(restored.gpu).toBe(true);
  expectNativeClassification([restored, original[1]]);
  const playback = {...restored, animationWindow: 'interval' as const};
  expectNativeClassification([playback, original[1]]);
  expect(usesNativeTopologyPredicate(playback)).toBe(true);
});

test('timestamp interval channel capacity and categorical predicates retain native allocation metadata', () => {
  const time: NativeFilter = {
    id: 'time',
    dataId: ['source'],
    name: ['start'],
    endName: ['end'],
    type: 'timeRange',
    value: [1, 2],
    gpu: true
  };
  const category: NativeFilter = {
    id: 'category',
    dataId: ['source'],
    name: ['category'],
    type: 'multiSelect',
    value: ['a'],
    gpu: false
  };
  const filters = assignGpuChannels(
    resetFilterGpuMode([range('first'), time, range('last'), range('overflow'), category])
  );
  expect(filters.map(filter => filter.gpu)).toEqual([true, true, true, false, false]);
  expect(filters[1].gpuEndChannel).toHaveLength(1);
  expectNativeClassification(filters);
});

test('native numeric playback predecessor boundaries retain half-open Float32 and integer membership', () => {
  const plot = runInNewContext(
    `var float64View = new DataView(new ArrayBuffer(8));\n${['adjacentFloat', 'nextDown', 'intervalBinFromMarks'].map(name => extract(plotSource, name)).join('\n')}\n({intervalBinFromMarks})`,
    {
      bisectCenter: (marks: number[], value: number) => marks.indexOf(value)
    }
  ) as {intervalBinFromMarks(marks: number[], value: number, offset: number): [number, number]};
  const first = plot.intervalBinFromMarks([0, 1, 2], 0, 0);
  const last = plot.intervalBinFromMarks([0, 1, 2], 1, 0);
  expect(first[1]).toBeLessThan(1);
  expect(last).toEqual([1, 2]);
  for (const format of ['float32', 'uint32', 'sint32'] as const) {
    const firstBounds = normalizeKeplerScalarRange(format, first[0], first[1]);
    const lastBounds = normalizeKeplerScalarRange(format, last[0], last[1]);
    for (const value of [0, 1, 2]) {
      expect(value >= firstBounds.minimum && value <= firstBounds.maximum).toBe(
        value >= first[0] && value <= first[1]
      );
      expect(value >= lastBounds.minimum && value <= lastBounds.maximum).toBe(
        value >= last[0] && value <= last[1]
      );
    }
  }
});

test('consumer exclusions match pinned Charts CPU-id and GPU-field functions with paired datasets', () => {
  const chartSource = readFileSync(
    join(dirname(require.resolve('@kepler.gl/components')), '../esm/map/charts/chart-panel.js'),
    'utf8'
  );
  type SelectedFilter = NativeFilter & {selected: number[]};
  const filters: SelectedFilter[] = [
    {
      id: 'owned-cpu',
      dataId: ['other-source', 'source'],
      name: ['unrelated', 'axis'],
      gpu: false,
      selected: [0, 2]
    },
    {id: 'same-field-cpu', dataId: ['source'], name: ['axis'], gpu: false, selected: [0, 1]},
    {id: 'axis-gpu', dataId: ['source'], name: ['axis'], gpu: true, selected: [1, 2, 3]},
    {id: 'other-gpu', dataId: ['source'], name: ['other'], gpu: true, selected: [0, 2, 3]}
  ];
  const dataset = {
    allIndexes: [0, 1, 2, 3],
    dataContainer: {},
    fields: [{name: 'axis'}, {name: 'other'}],
    filterRecord: {cpu: filters.filter(filter => !filter.gpu)},
    gpuFilter: {
      filterValueUpdateTriggers: {gpuFilter_0: {name: 'axis'}, gpuFilter_1: {name: 'other'}}
    }
  };
  const oracle = runInNewContext(
    `${['fieldNameForFilter', 'cpuIndexExcludingFilterIds'].map(name => extract(chartSource, name)).join('\n')}\n${extract(plotSource, 'getSkipIndexes')}\n({cpuIndexExcludingFilterIds,getSkipIndexes})`,
    {
      getFilterFunction: (
        field: {name: string} | null,
        _dataId: string,
        filter: SelectedFilter
      ) => {
        expect(field?.name).toBe('axis');
        return ({index}: {index: number}) => filter.selected.includes(index);
      }
    }
  ) as {
    cpuIndexExcludingFilterIds(
      dataset: unknown,
      dataId: string,
      excludedIds: string[],
      layers: unknown[]
    ): number[];
    getSkipIndexes(dataset: unknown, filter: undefined, names: string[]): number[];
  };
  const mask = new GPUVector({name: 'test-topology', type: 'data', format: 'uint32', data: []});
  for (const excludedCpuFilterIds of [['owned-cpu'], ['owned-cpu', 'same-field-cpu']]) {
    const excludedGpuFieldNames = ['axis'];
    const nativeCpuRows = oracle.cpuIndexExcludingFilterIds(
      dataset,
      'source',
      excludedCpuFilterIds,
      []
    );
    const skippedChannels = oracle.getSkipIndexes(dataset, undefined, excludedGpuFieldNames);
    const nativeRows = nativeCpuRows.filter(row =>
      filters
        .filter(filter => filter.gpu)
        .every(
          (filter, channel) => skippedChannels.includes(channel) || filter.selected.includes(row)
        )
    );
    const selectedIds = planKeplerConsumerFilterIds(
      filters.map(filter => ({
        filterId: filter.id,
        mask,
        dynamicDomain: false,
        nativeTopology: !filter.gpu,
        nativeFieldName: filter.name![filter.dataId.indexOf('source')]
      })),
      {id: 'chart', excludedCpuFilterIds, excludedGpuFieldNames}
    );
    const residentRows = dataset.allIndexes.filter(row =>
      filters
        .filter(filter => selectedIds.includes(filter.id))
        .every(filter => filter.selected.includes(row))
    );
    expect(residentRows).toEqual(nativeRows);
  }
});
