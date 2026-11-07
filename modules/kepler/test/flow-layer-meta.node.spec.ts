// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getFlowLayerMeta,
  hasFlowLayerMetaChanged
} from '../../../examples/deck/kepler-compute/src/flow-layer-meta';

const locations = [{id: 1, name: 'A', lon: -74, lat: 40.7}];
const flowSource = (bounds: [number, number, number, number] | null) => ({
  bounds,
  getLayerData: () => ({locations, clusterLevels: []})
});

test('flow metadata comes from the flow runtime, falling back only without finite endpoints', () => {
  const meta = getFlowLayerMeta(flowSource([-74, 40, -73, 41]), [0, 0, 1, 1]);
  expect(meta.bounds).toEqual([-74, 40, -73, 41]);
  expect(meta.locations).toBe(locations);
  expect(getFlowLayerMeta(flowSource(null), [0, 0, 1, 1]).bounds).toEqual([0, 0, 1, 1]);
  expect(getFlowLayerMeta(flowSource(null)).bounds).toBeNull();
});

test('metadata change detection is identity based for data and value based for bounds', () => {
  const meta = getFlowLayerMeta(flowSource([1, 2, 3, 4]));
  expect(hasFlowLayerMetaChanged(undefined, meta)).toBe(true);
  expect(hasFlowLayerMetaChanged({...meta, bounds: [1, 2, 3, 4]}, meta)).toBe(false);
  expect(hasFlowLayerMetaChanged({...meta, bounds: [1, 2, 3, 5]}, meta)).toBe(true);
  expect(hasFlowLayerMetaChanged({...meta, locations: []}, meta)).toBe(true);
});
