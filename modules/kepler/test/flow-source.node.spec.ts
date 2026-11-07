// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {
  makeFlowData,
  prepareFlowSource
} from '../../../examples/deck/kepler-compute/src/flow-source';

// The pinned package uses extensionless ESM imports that Node cannot resolve; browser checks
// exercise the real clustering. These stand-ins only record that clustering is invoked.
vi.mock('@flowmap.gl/data', () => ({
  makeLocationWeightGetter: () => () => 1,
  clusterLocations: (locations: unknown[]) => [{zoom: 0, clusters: locations.length}]
}));

const table = new arrow.Table({
  lon0: arrow.makeVector(new Float64Array([0, 0, 1, 0, Number.NaN])),
  lat0: arrow.makeVector(new Float64Array([0, 0, 1, 0, 0])),
  lon1: arrow.makeVector(new Float64Array([1, 1, 0, 1, 1])),
  lat1: arrow.makeVector(new Float64Array([1, 1, 0, 1, 1])),
  count: arrow.makeVector(new Float64Array([2, 3, 5, 7, 11])),
  origin: arrow.vectorFromArray(['A', 'B', 'C', 'D', 'E']),
  destination: arrow.vectorFromArray(['X', 'Y', 'Z', 'W', 'V'])
});
const roles = {
  targetLongitude: 'lon1',
  targetLatitude: 'lat1',
  count: 'count',
  sourceName: 'origin',
  targetName: 'destination'
};

test('flow locations follow native first-seen ids, names and non-finite row exclusion', async () => {
  const source = await prepareFlowSource(table, {longitude: 'lon0', latitude: 'lat0'}, roles);
  // Native keys are "lat:lon"; the first row names its source A and target X.
  expect(source.locations.map(location => [location.id, location.name])).toEqual([
    [1, 'A'],
    [2, 'X']
  ]);
  expect(Array.from(source.sourceIds)).toEqual([1, 1, 2, 1, 0]);
  expect(Array.from(source.targetIds)).toEqual([2, 2, 1, 2, 0]);
  expect(source.clusterLevels.length).toBeGreaterThan(0);
  expect(source.bounds).toEqual([0, 0, 1, 1]);
});

test('selected rows become flows without repacking and missing counts count once', async () => {
  const source = await prepareFlowSource(table, {longitude: 'lon0', latitude: 'lat0'}, roles);
  expect(makeFlowData(source, undefined).flows.map(flow => flow.index)).toEqual([0, 1, 2, 3]);
  expect(makeFlowData(source, Uint32Array.of(0, 1, 0, 1, 1)).flows).toEqual([
    {index: 1, sourceId: 1, targetId: 2, count: 3},
    {index: 3, sourceId: 1, targetId: 2, count: 7}
  ]);
  const unweighted = await prepareFlowSource(
    table,
    {longitude: 'lon0', latitude: 'lat0'},
    {targetLongitude: 'lon1', targetLatitude: 'lat1'}
  );
  expect(Array.from(unweighted.counts)).toEqual([1, 1, 1, 1, 1]);
  expect(unweighted.locations.map(location => location.name)).toEqual(['Location#1', 'Location#2']);
});

test('flow runtime aggregates each selection revision once and reads only unique pairs', async () => {
  const {FlowRuntime} = await import('../../../examples/deck/kepler-compute/src/flow-runtime');
  const {FlowReferenceAggregator} = await import(
    '../../../examples/deck/kepler-compute/src/flow-gpu-aggregation'
  );
  let revision = 0;
  const requestUpdate = vi.fn();
  const words = Uint32Array.of(1, 0, 0, 1, 1);
  const mask = {words} as never;
  const runtime = await FlowRuntime.create(table, {
    coordinates: {longitude: 'lon0', latitude: 'lat0'},
    roles,
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
  // Creation aggregates all rows once for the clustering weights: 2 unique pairs.
  expect(runtime.source.locations.map(location => location.name)).toEqual(['A', 'X']);
  expect(runtime.readbackCount).toBe(1);
  expect(runtime.binding.getLayerData().flows).toEqual([]);
  runtime.binding.update(mask);
  runtime.binding.update(mask);
  await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(1));
  // Rows 0 and 3 share (1, 2): one datum with the summed count and first row.
  expect(runtime.binding.getLayerData().flows).toEqual([
    {index: 0, sourceId: 1, targetId: 2, count: 9}
  ]);
  expect(runtime.readbackCount).toBe(2);
  const bytesAfterFirst = runtime.readbackBytes;
  revision++;
  runtime.binding.update(mask);
  await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(2));
  expect(runtime.readbackCount).toBe(3);
  expect(runtime.readbackBytes).toBeGreaterThan(bytesAfterFirst);
  runtime.destroy();
});
