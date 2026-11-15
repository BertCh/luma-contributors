// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {
  assertSamplePresentation,
  assertSampleRenderedLayers,
  assertSampleRenderedViews,
  assertSampleRenderedLayersWhenReady
} from '../../../examples/deck/kepler-compute/scripts/sample-map-presentation.mjs';

function fixture() {
  const saved = {
    config: {
      visState: {
        layers: [
          {
            id: 'parcel',
            type: 'point',
            config: {
              dataId: 'parcels',
              isVisible: true,
              columns: {lat: 'lat', lng: 'lng'},
              visConfig: {radiusRange: [0, 100], opacity: 0.8}
            },
            visualChannels: {
              colorField: {name: 'value', type: 'integer'},
              colorScale: 'quantile',
              sizeField: {name: 'value', type: 'integer'},
              sizeScale: 'sqrt'
            }
          }
        ],
        filters: [
          {
            id: 'filter',
            name: ['value'],
            value: [10, 40],
            dataId: ['parcels'],
            type: 'range',
            layerId: ['parcel']
          }
        ],
        splitMaps: [{layers: {parcel: true}}, {layers: {parcel: false}}]
      },
      mapState: {latitude: 34, longitude: -118, zoom: 7, pitch: 20, bearing: 12, isSplit: true}
    }
  };
  const actual = structuredClone(saved);
  actual.config.visState.layers[0]!.config.isVisible = false;
  return {
    saved,
    actual,
    limitations: [{layerId: 'parcel', fallbackLayerId: 'parcel-sample-locations'}]
  };
}

test('complete normalized presentation preserves original limited encodings and each split view', () => {
  const {saved, actual, limitations} = fixture();
  const comparable = actual as unknown as {
    config: {visState: {splitMaps: {layers: Record<string, unknown>}[]}};
  };
  comparable.config.visState.splitMaps[0]!.layers['parcel-sample-locations'] = {isVisible: true};
  comparable.config.visState.splitMaps[1]!.layers['parcel-sample-locations'] = {isVisible: false};
  actual.config.visState.layers.push({
    id: 'parcel-sample-locations',
    type: 'point',
    config: {}
  } as never);
  expect(() => assertSamplePresentation(actual, saved, limitations)).not.toThrow();
  assertSampleRenderedLayers([{id: 'parcel-sample-locations', visible: true}], saved, limitations);
  expect(() => assertSampleRenderedLayers([], saved, limitations)).toThrow(
    'expected visible rendered layer'
  );
});

test.each([
  'channels',
  'settings',
  'view',
  'filter-type',
  'filter-source',
  'camera',
  'extra-channel',
  'extra-setting',
  'filter-value',
  'dataset',
  'layer-type',
  'extra-layer',
  'extra-filter',
  'extra-view-layer'
] as const)('rejects lost or changed %s even with unchanged layer/filter identities', mutation => {
  const {saved, actual} = fixture();
  actual.config.visState.layers[0]!.config.isVisible = true;
  if (mutation === 'channels') actual.config.visState.layers[0]!.visualChannels = {} as never;
  if (mutation === 'settings') actual.config.visState.layers[0]!.config.visConfig.opacity = 0.1;
  if (mutation === 'view') actual.config.visState.splitMaps.reverse();
  if (mutation === 'filter-type') actual.config.visState.filters[0]!.type = 'select';
  if (mutation === 'filter-source') actual.config.visState.filters[0]!.dataId = ['other'];
  if (mutation === 'camera') actual.config.mapState.latitude = 0;
  if (mutation === 'extra-channel')
    Object.assign(actual.config.visState.layers[0]!.visualChannels, {
      heightField: {name: 'value', type: 'integer'}
    });
  if (mutation === 'extra-setting')
    Object.assign(actual.config.visState.layers[0]!.config.visConfig, {extruded: true});
  if (mutation === 'filter-value') actual.config.visState.filters[0]!.value = [0, 100];
  if (mutation === 'dataset') actual.config.visState.layers[0]!.config.dataId = 'other';
  if (mutation === 'layer-type') actual.config.visState.layers[0]!.type = 'grid';
  if (mutation === 'extra-layer')
    actual.config.visState.layers.push({...actual.config.visState.layers[0]!, id: 'extra'});
  if (mutation === 'extra-filter')
    actual.config.visState.filters.push({...actual.config.visState.filters[0]!, id: 'extra'});
  if (mutation === 'extra-view-layer')
    Object.assign(actual.config.visState.splitMaps[0]!.layers, {extra: true});
  expect(() => assertSamplePresentation(actual, saved)).toThrow();
});

test('normalized settings are compared exactly, including complete palettes and label defaults', () => {
  const {saved, actual} = fixture();
  actual.config.visState.layers[0]!.config.isVisible = true;
  const normalizedSettings = {
    colorRange: {name: 'Ocean Green', colors: ['#111111', '#222222']},
    textLabel: [{field: null, collisionEnabled: false, weight: 400}]
  };
  Object.assign(saved.config.visState.layers[0]!.config, normalizedSettings);
  Object.assign(actual.config.visState.layers[0]!.config, structuredClone(normalizedSettings));
  assertSamplePresentation(actual, saved);
  const actualConfig = actual.config.visState.layers[0]!
    .config as unknown as typeof normalizedSettings;
  actualConfig.colorRange.colors.reverse();
  expect(() => assertSamplePresentation(actual, saved)).toThrow('parcel.config');
  actualConfig.colorRange.colors.reverse();
  delete (actualConfig.textLabel[0] as Partial<(typeof actualConfig.textLabel)[0]>).weight;
  expect(() => assertSamplePresentation(actual, saved)).toThrow('parcel.config');
});

test('world-flights must render both its animated and static paths', () => {
  const {saved} = fixture();
  saved.config.visState.splitMaps = [];
  saved.config.visState.layers = ['lluahhf', 'jrj4wei'].map(id => ({
    ...saved.config.visState.layers[0]!,
    id,
    type: id === 'jrj4wei' ? 'geojson' : 'trip'
  }));
  const rendered = [{id: 'lluahhf', renderId: 'lluahhf-trips', visible: true}];
  expect(() => assertSampleRenderedLayers(rendered, saved)).toThrow('jrj4wei');
  assertSampleRenderedLayers(
    [...rendered, {id: 'jrj4wei', renderId: 'jrj4wei-paths', visible: true}],
    saved
  );
});

test('a layer omitted from every split view is hidden according to Kepler split-map semantics', () => {
  const {saved} = fixture();
  saved.config.visState.splitMaps = [{layers: {}}, {layers: {}}] as never;
  assertSampleRenderedLayers([], saved);
  expect(() => assertSampleRenderedLayers([{id: 'parcel', visible: true}], saved)).toThrow(
    'hidden rendered layer parcel'
  );
});

test('split render assertions require visibility in the correct viewport, including limited fallbacks', () => {
  const {saved, limitations} = fixture();
  const fallback = 'parcel-sample-locations';
  const views = [
    {id: 'view-0', layers: [{id: fallback, visible: true}]},
    {id: 'view-1', layers: [{id: fallback, visible: false}]}
  ];
  assertSampleRenderedViews(views, saved, limitations, 'parcels');
  const flipped = structuredClone(views);
  flipped[0]!.layers[0]!.visible = false;
  flipped[1]!.layers[0]!.visible = true;
  expect(() => assertSampleRenderedViews(flipped, saved, limitations, 'parcels')).toThrow(
    'parcels view 0: expected visible rendered layer'
  );
  const leaked = structuredClone(views);
  leaked[1]!.layers[0]!.visible = true;
  expect(() => assertSampleRenderedViews(leaked, saved, limitations, 'parcels')).toThrow(
    'parcels view 1: hidden rendered layer'
  );
  expect(() => assertSampleRenderedViews(views.slice(0, 1), saved, limitations)).toThrow(
    'rendered viewport count'
  );
});

test('Pittsburgh layers draw in opposite split views and must not leak into the other view', () => {
  const {saved} = fixture();
  const first = saved.config.visState.layers[0]!;
  saved.config.visState.layers = [
    {...first, id: 'j0guklb'},
    {...first, id: 'vtwrd0n'}
  ];
  saved.config.visState.splitMaps = [
    {layers: {j0guklb: true, vtwrd0n: false}},
    {layers: {j0guklb: false, vtwrd0n: true}}
  ] as never;
  const views = [
    {
      id: 'view-0',
      layers: [
        {id: 'j0guklb', visible: true},
        {id: 'vtwrd0n', visible: false}
      ]
    },
    {
      id: 'view-1',
      layers: [
        {id: 'j0guklb', visible: false},
        {id: 'vtwrd0n', visible: true}
      ]
    }
  ];
  assertSampleRenderedViews(views, saved, [], 'movement_pittsburgh');
  views[1]!.layers[0]!.visible = true;
  expect(() => assertSampleRenderedViews(views, saved, [], 'movement_pittsburgh')).toThrow(
    'movement_pittsburgh view 1: hidden rendered layer j0guklb'
  );
});

test('render readiness waits for asynchronous styles and still rejects a permanently missing layer', async () => {
  vi.useFakeTimers();
  try {
    const {saved} = fixture();
    saved.config.visState.splitMaps = [];
    const rendered = [{id: 'parcel', visible: true}];
    const read = vi.fn().mockResolvedValueOnce([]).mockResolvedValue(rendered);
    const ready = assertSampleRenderedLayersWhenReady(read, saved, [], 'nyc_earnings', {
      timeout: 1000,
      interval: 250
    });
    await vi.advanceTimersByTimeAsync(250);
    await expect(ready).resolves.toEqual(rendered);
    expect(read).toHaveBeenCalledTimes(2);
    const missing = assertSampleRenderedLayersWhenReady(async () => [], saved, [], 'nyc_earnings', {
      timeout: 1000,
      interval: 250
    });
    const assertion = expect(missing).rejects.toThrow(
      'nyc_earnings: expected visible rendered layer parcel'
    );
    await vi.advanceTimersByTimeAsync(1000);
    await assertion;
  } finally {
    vi.useRealTimers();
  }
});
