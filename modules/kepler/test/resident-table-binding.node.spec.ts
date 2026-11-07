// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  prepareResidentPolygonFilter,
  prepareNativePolygonFilter
} from '../../../examples/deck/kepler-compute/src/resident-table-binding';
import type {KeplerComputeFilter} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

const layers = [{id: 'points', type: 'point', config: {dataId: 'observations'}}];
const shell = [
  [-2, -2],
  [2, -2],
  [2, 2],
  [-2, 2],
  [-2, -2]
];
const hole = [
  [-1, -1],
  [-1, 1],
  [1, 1],
  [1, -1],
  [-1, -1]
];
function filter(geometry: unknown): KeplerComputeFilter {
  return {
    id: 'native-polygon',
    type: 'polygon',
    dataId: ['observations'],
    layerId: ['points'],
    value: {
      type: 'Feature',
      id: 'drawn-feature',
      geometry,
      properties: {filterId: 'native-polygon', isVisible: true}
    }
  };
}

describe('native Kepler polygon provider', () => {
  test('reads the pinned FeatureValue shape with closed shell and holes without changing source coordinates', () => {
    const source = filter({type: 'Polygon', coordinates: [shell, hole]});
    const prepared = prepareResidentPolygonFilter(source, 'observations', 'points', layers);
    expect(prepared.selection).toEqual({kind: 'polygon', polygons: [[shell, hole]]});
    expect(prepared.layerIds).toEqual(['points']);
    expect(prepared.topologyKey).toBe('0,2:0,4,8');
    expect(shell).toHaveLength(5);
    expect(prepared.inverted).toBe(false);
  });

  test('preserves union semantics inside MultiPolygon and native inversion as a separate operation', () => {
    const source = {
      ...filter({type: 'MultiPolygon', coordinates: [[shell, hole], [shell]]}),
      inverted: true
    };
    const prepared = prepareResidentPolygonFilter(source, 'observations', 'points', layers);
    expect(prepared.selection).toEqual({kind: 'polygon', polygons: [[shell, hole], [shell]]});
    expect(prepared.inverted).toBe(true);
    expect(prepared.topologyKey).toBe('0,2,3:0,4,8,12');
  });

  test('does not apply a target layer filter to another layer or the full dataset', () => {
    const source = filter({type: 'Polygon', coordinates: [shell]});
    expect(() =>
      prepareResidentPolygonFilter(
        {...source, layerId: ['another-layer']},
        'observations',
        'points',
        layers
      )
    ).toThrow('declared resident point layer');
    expect(() =>
      prepareResidentPolygonFilter(source, 'observations', 'points', [
        layers[0],
        {id: 'second', type: 'point', config: {dataId: 'observations'}}
      ])
    ).toThrow('separate per-layer');
    expect(() =>
      prepareResidentPolygonFilter(source, 'observations', 'points', [
        {...layers[0], type: 'polygon'}
      ])
    ).toThrow('declared point layer');
  });

  test('accepts native linear antimeridian geometry and rejects malformed geometries before allocation', () => {
    // Native booleanWithin has no antimeridian handling: edges are linear in raw degrees.
    expect(
      prepareResidentPolygonFilter(
        filter({
          type: 'Polygon',
          coordinates: [
            [
              [179, 0],
              [-179, 0],
              [-179, 1],
              [179, 1]
            ]
          ]
        }),
        'observations',
        'points',
        layers
      ).selection.kind
    ).toBe('polygon');
    expect(() =>
      prepareResidentPolygonFilter(
        filter({type: 'LineString', coordinates: shell}),
        'observations',
        'points',
        layers
      )
    ).toThrow('Polygon and MultiPolygon');
    expect(() =>
      prepareResidentPolygonFilter(
        filter({
          type: 'Polygon',
          coordinates: [
            [
              ['0', 0],
              [1, 0],
              [1, 1]
            ]
          ]
        }),
        'observations',
        'points',
        layers
      )
    ).toThrow('numeric longitude');
  });
});

test('native polygon conversion supports multiple registered layer semantics and explicit rectangle metadata', () => {
  const nativeLayers = [
    ...layers,
    {id: 'arc', type: 'arc', config: {dataId: 'observations', isVisible: false}},
    {id: 'other-dataset', type: 'point', config: {dataId: 'other'}}
  ];
  const native: KeplerComputeFilter = {
    ...filter({type: 'Polygon', coordinates: [shell]}),
    layerId: ['points', 'arc', 'other-dataset'],
    value: {
      type: 'Feature',
      geometry: {type: 'Polygon', coordinates: [shell]},
      properties: {shape: 'Rectangle', bbox: [-2, -2, 2, 2]}
    }
  };
  expect(
    prepareNativePolygonFilter(native, 'observations', nativeLayers, ['points', 'arc'])
  ).toMatchObject({layerIds: ['points', 'arc'], rectangleBounds: [-2, -2, 2, 2]});
  expect(() =>
    prepareNativePolygonFilter(native, 'observations', nativeLayers, ['points'])
  ).toThrow('registered resident geometry');
  expect(
    prepareNativePolygonFilter(
      {...native, enabled: false, value: 'invalid'},
      'observations',
      nativeLayers,
      []
    )
  ).toBeUndefined();
  expect(
    prepareNativePolygonFilter({...native, layerId: ['deleted']}, 'observations', nativeLayers, [])
  ).toBeUndefined();
});
