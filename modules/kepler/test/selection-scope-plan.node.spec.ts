// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUVector} from '@luma.gl/graph/gpu-data';
import {describe, expect, test} from 'vitest';
import {
  planKeplerSelectionScopes,
  planKeplerConsumerFilterIds,
  type KeplerSelectionLayer,
  type KeplerPolygonMaskFilter
} from '../src/selection/selection-scope-plan';

const positions = new GPUVector({name: 'points', type: 'data', format: 'float32x2', data: []});
const mask = new GPUVector({name: 'mask', type: 'data', format: 'uint32', data: []});
const layers: KeplerSelectionLayer[] = [
  {id: 'points', geometryRevision: 1, geometry: {kind: 'point', positions}},
  {id: 'hidden', geometryRevision: 1, visible: false, geometry: {kind: 'point', positions}},
  {id: 'untargeted', geometryRevision: 1, geometry: {kind: 'point', positions}}
];
const polygon: KeplerPolygonMaskFilter = {
  id: 'polygon',
  layerIds: ['points', 'hidden'],
  selection: {kind: 'rectangle', bounds: [0, 0, 1, 1]}
};

describe('native selection scope planning', () => {
  test('Charts exclude owned CPU ids and all matching GPU axis fields independently', () => {
    const filters = [
      {
        filterId: 'owned-category',
        mask,
        dynamicDomain: true,
        nativeTopology: true,
        nativeFieldName: 'axis'
      },
      {
        filterId: 'other-category',
        mask,
        dynamicDomain: true,
        nativeTopology: true,
        nativeFieldName: 'axis'
      },
      {
        filterId: 'axis-range-1',
        mask,
        dynamicDomain: true,
        nativeTopology: false,
        nativeFieldName: 'axis'
      },
      {
        filterId: 'axis-range-2',
        mask,
        dynamicDomain: true,
        nativeTopology: false,
        nativeFieldName: 'axis'
      },
      {
        filterId: 'other-range',
        mask,
        dynamicDomain: true,
        nativeTopology: false,
        nativeFieldName: 'other'
      },
      {filterId: 'disabled', mask, dynamicDomain: true, enabled: false}
    ];
    expect(planKeplerConsumerFilterIds(filters, {id: 'all-source'})).toEqual([
      'owned-category',
      'other-category',
      'axis-range-1',
      'axis-range-2',
      'other-range'
    ]);
    expect(
      planKeplerConsumerFilterIds(filters, {
        id: 'chart',
        excludedCpuFilterIds: ['owned-category', 'other-range'],
        excludedGpuFieldNames: ['axis']
      })
    ).toEqual(['other-category', 'other-range']);
    expect(
      planKeplerConsumerFilterIds(filters, {
        id: 'big-number-count',
        excludedCpuFilterIds: ['owned-category']
      })
    ).toEqual(['other-category', 'axis-range-1', 'axis-range-2', 'other-range']);
    expect(() =>
      planKeplerConsumerFilterIds(
        [{filterId: 'missing', mask, nativeTopology: false, dynamicDomain: false}],
        {id: 'chart', excludedGpuFieldNames: ['axis']}
      )
    ).toThrow('field metadata');
  });
  test('targets only named layers, includes hidden targets and separates dynamic domain filters', () => {
    const plan = planKeplerSelectionScopes(
      layers,
      [
        {filterId: 'first-on-value', mask, dynamicDomain: true, nativeTopology: false},
        {filterId: 'second-on-value', mask, dynamicDomain: false},
        {filterId: 'disabled', mask, dynamicDomain: true, enabled: false}
      ],
      [polygon, {...polygon, id: 'disabled-polygon', enabled: false, layerIds: ['untargeted']}]
    );
    expect(plan).toEqual({
      scalarFilterIds: ['first-on-value', 'second-on-value'],
      domainFilterIds: ['first-on-value'],
      nativeTopologyFilterIds: ['second-on-value'],
      layerFilterIds: {points: ['polygon'], hidden: ['polygon'], untargeted: []},
      targetedLayerIds: ['points', 'hidden']
    });
  });
  test('disable/delete/unknown target leaves native population at the scalar base', () => {
    expect(
      planKeplerSelectionScopes(layers, [], [{...polygon, enabled: false}]).targetedLayerIds
    ).toEqual([]);
    expect(
      planKeplerSelectionScopes(layers, [], [{...polygon, layerIds: ['deleted']}]).targetedLayerIds
    ).toEqual([]);
    expect(planKeplerSelectionScopes(layers, [], []).targetedLayerIds).toEqual([]);
  });
  test('rejects ambiguous identities and accepts arbitrary string layer ids safely', () => {
    expect(() => planKeplerSelectionScopes([layers[0], layers[0]], [], [])).toThrow(
      'Duplicate selection layer'
    );
    expect(() =>
      planKeplerSelectionScopes(
        layers,
        [{filterId: 'polygon', mask, dynamicDomain: false}],
        [polygon]
      )
    ).toThrow('Duplicate native filter');
    const unusual = {...layers[0], id: '__proto__'};
    expect(
      planKeplerSelectionScopes([unusual], [], [{...polygon, layerIds: ['__proto__']}])
        .targetedLayerIds
    ).toEqual(['__proto__']);
  });
});
