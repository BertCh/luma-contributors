// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {LAYER_TYPES} from '@kepler.gl/constants';
import {
  ActionTypes,
  applyMapState,
  receiveMapConfig,
  runGroupBy,
  runJoin,
  runSpatialJoin,
  runSuitability,
  toggleSplitMap,
  updateMap,
  wrapTo
} from '@kepler.gl/actions';
import {
  assertSingleViewConfiguration,
  getResidentActionRejection,
  type ResidentOperationState
} from '../../../examples/deck/kepler-compute/src/resident-action-guards';

const state: ResidentOperationState = {
  visState: {
    groupBys: [
      {id: 'group', dataId: 'resident'},
      {id: 'ordinary-group', dataId: 'ordinary'}
    ],
    joins: [{id: 'join', leftDataId: 'ordinary', rightDataId: 'resident'}],
    suitabilities: [{id: 'score', dataId: 'resident'}],
    layers: [
      {id: 'resident-layer', config: {dataId: 'resident'}},
      {id: 'ordinary-layer', config: {dataId: 'ordinary'}}
    ]
  },
  mapState: {isSplit: false}
};
const isResidentDataset = (datasetId: string) => datasetId === 'resident';

describe('resident native action boundary', () => {
  it('routes resident editor extraction before native CPU filtering while preserving ordinary extraction', () => {
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.EXTRACT_DATA_FROM_FEATURE, layerId: 'resident-layer'},
        isResidentDataset
      )?.reason
    ).toMatch(/Export Data/);
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.EXTRACT_DATA_FROM_FEATURE, layerId: 'ordinary-layer'},
        isResidentDataset
      )
    ).toBeUndefined();
    for (const type of [LAYER_TYPES.vectorTile, LAYER_TYPES.rasterTile])
      expect(
        getResidentActionRejection(
          {visState: {layers: [{id: 'tiles', type, config: {dataId: 'resident'}}]}},
          {type: ActionTypes.EXTRACT_DATA_FROM_FEATURE, layerId: 'tiles'},
          isResidentDataset
        )
      ).toBeUndefined();
  });
  it.each([
    [runGroupBy('group'), 'groupBys', 'sql'],
    [runJoin('join'), 'joins', 'sql'],
    [runSpatialJoin('join'), 'joins', 'analysis'],
    [runSuitability('score'), 'suitabilities', 'analysis']
  ])('rejects resident native operation before a retained-row executor runs: %j', (action, list, route) => {
    let nativeExecutions = 0;
    const rejection = getResidentActionRejection(
      state,
      wrapTo('compute', action),
      isResidentDataset
    );
    if (!rejection) nativeExecutions++;
    expect(nativeExecutions).toBe(0);
    expect(rejection?.operation?.list).toBe(list);
    expect(rejection?.route).toBe(route);
    expect(rejection?.reason).toMatch(/resident/);
  });

  it('preserves nonresident operations, unrelated actions, and other map instances', () => {
    expect(
      getResidentActionRejection(state, runGroupBy('ordinary-group'), isResidentDataset)
    ).toBeUndefined();
    expect(
      getResidentActionRejection(state, {type: 'unrelated'}, isResidentDataset)
    ).toBeUndefined();
    expect(
      getResidentActionRejection(state, wrapTo('another-map', runJoin('join')), isResidentDataset)
    ).toBeUndefined();
  });

  it('blocks native split entry but allows returning an existing split state to one view', () => {
    expect(getResidentActionRejection(state, toggleSplitMap(), isResidentDataset)?.reason).toMatch(
      /one map view/
    );
    expect(
      getResidentActionRejection({mapState: {isSplit: true}}, toggleSplitMap(), isResidentDataset)
    ).toBeUndefined();
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.SET_MAP_SPLIT_MODE, payload: {mapSplitMode: 'SWIPE_COMPARE'}},
        isResidentDataset
      )
    ).toBeDefined();
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.SET_MAP_SPLIT_MODE, payload: {mapSplitMode: 'SWIPE_COMPARE'}},
        isResidentDataset,
        undefined,
        2
      )
    ).toBeUndefined();
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.SET_MAP_SPLIT_MODE, payload: {mapSplitMode: 'SINGLE_MAP'}},
        isResidentDataset
      )
    ).toBeUndefined();
  });

  it('rejects split saved configurations before source replacement or native reducer work', () => {
    const config = {version: 'v1', config: {visState: {splitMaps: [{layers: {}}, {layers: {}}]}}};
    expect(() => assertSingleViewConfiguration(config)).toThrow(/one map view/);
    expect(
      getResidentActionRejection(state, receiveMapConfig(config), isResidentDataset)
    ).toBeDefined();
    expect(
      getResidentActionRejection(
        state,
        {type: ActionTypes.ADD_DATA_TO_MAP, payload: {config}},
        isResidentDataset
      )
    ).toBeDefined();
    expect(() =>
      assertSingleViewConfiguration({
        version: 'v1',
        config: {visState: {splitMaps: []}, mapState: {isSplit: false, mapSplitMode: 'SINGLE_MAP'}}
      })
    ).not.toThrow();
  });

  it('admits at most two split views only when shared-device view support is declared', () => {
    const config = {
      version: 'v1',
      config: {
        visState: {splitMaps: [{layers: {a: true}}, {layers: {b: true}}]},
        mapState: {mapSplitMode: 'DUAL_MAP'}
      }
    };
    expect(() => assertSingleViewConfiguration(config, 2)).not.toThrow();
    expect(() =>
      assertSingleViewConfiguration({config: {mapState: {mapSplitMode: 'SWIPE_COMPARE'}}}, 2)
    ).not.toThrow();
    expect(() =>
      assertSingleViewConfiguration({config: {mapState: {mapSplitMode: 'SWIPE_COMPARE'}}})
    ).toThrow(/one map view/);
    expect(() => assertSingleViewConfiguration(config)).toThrow(/one map view/);
    expect(() =>
      assertSingleViewConfiguration(
        {version: 'v1', config: {visState: {splitMaps: [{}, {}, {}]}}},
        2
      )
    ).toThrow(/one map view/);
    expect(() =>
      assertSingleViewConfiguration({config: {mapState: {mapSplitMode: 'SPLIT_3'}}}, 2)
    ).toThrow(/one map view/);
  });

  it('blocks viewport JSON and direct map-state paths that allocate another map', () => {
    expect(
      getResidentActionRejection(state, updateMap({isSplit: true}), isResidentDataset)
    ).toBeDefined();
    expect(
      getResidentActionRejection(state, applyMapState({isSplit: true}), isResidentDataset)
    ).toBeDefined();
    expect(
      getResidentActionRejection(state, updateMap({longitude: -74, zoom: 10}), isResidentDataset)
    ).toBeUndefined();
  });
});
