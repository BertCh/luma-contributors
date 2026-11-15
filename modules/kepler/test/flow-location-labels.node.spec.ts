// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {afterAll, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());
import {
  getFlowLocationLabelSignature,
  selectFlowLocationLabels
} from '../../../examples/deck/kepler-compute/src/flow-location-labels';

vi.mock('@flowmap.gl/data', () => ({
  makeLocationWeightGetter: () => () => 1,
  clusterLocations: (locations: unknown[]) => [{zoom: 0, clusters: locations.length}]
}));
// The label module imports GPU text modules that need a device; only its pure helpers run here.
vi.mock('../../../examples/deck/kepler-compute/src/label-runtime', () => ({
  makePointLabelRuntime: () => undefined
}));
vi.mock('../../../examples/deck/kepler-compute/gpu-layers/index', () => ({
  GPUPointLabelLayer: class {}
}));

function makeLayersData(names: string[], radii: number[], positions: number[][]) {
  const position = new Float64Array(positions.flatMap(([lon, lat]) => [lon, lat, 0]));
  return {
    locationLabels: names,
    circleAttributes: {
      length: names.length,
      attributes: {
        getPosition: {value: position, size: 3},
        getInRadius: {value: Float32Array.from(radii), size: 1},
        getOutRadius: {value: Float32Array.from(radii), size: 1}
      }
    }
  } as never;
}

const viewport = {
  width: 100,
  height: 100,
  project: ([lon, lat]: [number, number]): [number, number] => [lon, lat]
};

test('location labels keep in-view named locations and keep the largest circles over the limit', () => {
  const layersData = makeLayersData(
    ['A', '', 'C', 'Off', 'E'],
    [5, 5, 20, 5, 10],
    [
      [10, 10],
      [20, 20],
      [30, 30],
      [500, 500],
      [40, 40]
    ]
  );
  const all = selectFlowLocationLabels(layersData, viewport);
  expect(all.names).toEqual(['A', 'C', 'E']);
  expect(Array.from(all.positions)).toEqual([10, 10, 30, 30, 40, 40]);
  const limited = selectFlowLocationLabels(layersData, viewport, 2);
  // Largest radii C (20) and E (10) win; original order is preserved.
  expect(limited.names).toEqual(['C', 'E']);
  expect(getFlowLocationLabelSignature(limited)).not.toEqual(getFlowLocationLabelSignature(all));
});

test('location labels flatten control characters and truncate long names', () => {
  const layersData = makeLayersData(
    ['a\tb\nc', 'x'.repeat(100)],
    [1, 1],
    [
      [1, 1],
      [2, 2]
    ]
  );
  const rows = selectFlowLocationLabels(layersData, viewport);
  expect(rows.names[0]).toBe('a b c');
  expect(rows.names[1]).toHaveLength(48);
});

test('a time-filter mask update re-aggregates flow totals while the mask identity is unchanged', async () => {
  const {FlowRuntime} = await import('../../../examples/deck/kepler-compute/src/flow-runtime');
  const {FlowReferenceAggregator} = await import(
    '../../../examples/deck/kepler-compute/src/flow-gpu-aggregation'
  );
  const table = new arrow.Table({
    lon0: arrow.makeVector(new Float64Array([0, 0, 1, 0])),
    lat0: arrow.makeVector(new Float64Array([0, 0, 1, 0])),
    lon1: arrow.makeVector(new Float64Array([1, 1, 0, 1])),
    lat1: arrow.makeVector(new Float64Array([1, 1, 0, 1])),
    count: arrow.makeVector(new Float64Array([2, 3, 5, 7]))
  });
  let revision = 0;
  const requestUpdate = vi.fn();
  // Word i of the mask selects row i; a Kepler time filter rewrites these words in place.
  const words = Uint32Array.of(1, 1, 1, 1);
  const mask = {words} as never;
  const runtime = await FlowRuntime.create(table, {
    coordinates: {longitude: 'lon0', latitude: 'lat0'},
    roles: {targetLongitude: 'lon1', targetLatitude: 'lat1', count: 'count'},
    requestUpdate,
    getRevision: () => revision,
    createAggregator: async () =>
      new FlowReferenceAggregator(
        {
          sourceLongitude: table.getChild('lon0'),
          sourceLatitude: table.getChild('lat0'),
          targetLongitude: table.getChild('lon1'),
          targetLatitude: table.getChild('lat1'),
          count: table.getChild('count')
        },
        table.numRows,
        selection => (selection as {words: Uint32Array}).words
      )
  });
  runtime.binding.update(mask);
  await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(1));
  const total = () =>
    runtime.binding.getLayerData().flows.reduce((sum, flow) => sum + flow.count, 0);
  expect(total()).toBe(2 + 3 + 5 + 7);
  // The time window narrows to rows 1 and 2.
  words.set([0, 1, 1, 0]);
  revision++;
  runtime.binding.update(mask);
  await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(2));
  expect(total()).toBe(3 + 5);
  runtime.destroy();
});

test('collision avoidance keeps the larger of two overlapping labels and every clear label', () => {
  const layersData = makeLayersData(
    ['Big city', 'Neighbor', 'Far away'],
    [20, 5, 5],
    [
      [50, 50],
      [52, 50],
      [10, 10]
    ]
  );
  // Without collision all in-view labels are kept.
  expect(selectFlowLocationLabels(layersData, viewport).names).toEqual([
    'Big city',
    'Neighbor',
    'Far away'
  ]);
  const placed = selectFlowLocationLabels(layersData, viewport, undefined, {labelSize: 14});
  // The smaller circle's label overlaps the larger one's and is dropped; original order is kept.
  expect(placed.names).toEqual(['Big city', 'Far away']);
  expect(Array.from(placed.radii)).toEqual([20, 5]);
});

test('collision avoidance honors the label limit after placement', () => {
  const positions = Array.from({length: 6}, (_, index) => [10 + index * 15, 50]);
  const layersData = makeLayersData(
    positions.map((_, index) => `L${index}`),
    [1, 2, 3, 4, 5, 6],
    positions
  );
  const placed = selectFlowLocationLabels(layersData, viewport, 2, {labelSize: 4});
  expect(placed.names).toEqual(['L4', 'L5']);
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('@flowmap.gl/data');
  vi.doUnmock('../../../examples/deck/kepler-compute/src/label-runtime');
  vi.doUnmock('../../../examples/deck/kepler-compute/gpu-layers/index');
  vi.resetModules();
});
