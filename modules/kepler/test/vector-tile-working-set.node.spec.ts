// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  VectorTileFeatureRegistry,
  resolveVectorTileFeatureIdentity
} from '../../../examples/deck/kepler-compute/src/vector-tile-feature-ids';
import {VectorTileWorkingSet} from '../../../examples/deck/kepler-compute/src/vector-tile-working-set';
import {
  decideVectorTileAnalysis,
  describeVectorTileAnalysisScope
} from '../../../examples/deck/kepler-compute/src/vector-tile-analysis-scope';
import {
  buildDecodedTileBatches,
  computeDecodedTileFilterMask,
  computeDecodedTileStyleDomain
} from '../../../examples/deck/kepler-compute/src/vector-tile-decoded-batch';

const options = {sourceLayer: 'roads'};

describe('vector tile feature ids', () => {
  it('is independent of tile and geometry, and deterministic', () => {
    const fragmentA = {properties: {name: 'A1', lanes: 2}};
    const fragmentB = {properties: {lanes: 2, name: 'A1'}};
    const first = resolveVectorTileFeatureIdentity(fragmentA, options);
    expect(resolveVectorTileFeatureIdentity(fragmentB, options).featureId).toBe(first.featureId);
    expect(first.stability).toBe('property-hash');
    expect(first.collisionRisk).toBe(true);
  });

  it('prefers source ids then unique properties', () => {
    expect(resolveVectorTileFeatureIdentity({id: 7, properties: {}}, options)).toMatchObject({
      featureId: 'roads:i:7',
      stability: 'source-id'
    });
    expect(resolveVectorTileFeatureIdentity({properties: {OBJECTID: 9}}, options)).toMatchObject({
      featureId: 'roads:p:OBJECTID:9',
      stability: 'unique-property'
    });
    expect(
      resolveVectorTileFeatureIdentity(
        {properties: {code: 'x'}},
        {...options, uniqueIdProperty: 'code'}
      ).featureId
    ).toBe('roads:p:code:x');
  });

  it('keeps one id for a split feature through evict and reload', () => {
    const registry = new VectorTileFeatureRegistry(options);
    const [idInFirstTile] = registry.registerTile('3/1/1', [{id: 5}, {id: 6}]);
    const [idInSecondTile] = registry.registerTile('3/2/1', [{id: 5}]);
    expect(idInFirstTile).toBe(idInSecondTile);
    expect(registry.summarize()).toMatchObject({
      uniqueFeatureCount: 2,
      fragmentCount: 3,
      splitFeatureCount: 1
    });
    registry.evictTile('3/1/1');
    expect(registry.summarize()).toMatchObject({uniqueFeatureCount: 1, splitFeatureCount: 0});
    const [reloaded] = registry.registerTile('3/1/1', [{id: 5}]);
    expect(reloaded).toBe(idInFirstTile);
  });
});

describe('vector tile working set', () => {
  it('evicts least recently used non-visible tiles within the budget', () => {
    const evicted: string[] = [];
    const set = new VectorTileWorkingSet({budgetBytes: 100, onEvictTile: key => evicted.push(key)});
    set.admitTile('a', 40);
    set.admitTile('b', 40);
    set.touchTile('a');
    const admission = set.admitTile('c', 40);
    expect(admission).toEqual({admitted: true, evictedTileKeys: ['b']});
    expect(evicted).toEqual(['b']);
    expect(set.getStatus()).toMatchObject({usedBytes: 80, tileCount: 2, evictionCount: 1});
  });

  it('never evicts visible tiles and reports rejection', () => {
    const set = new VectorTileWorkingSet({budgetBytes: 100});
    set.admitTile('a', 60);
    set.setVisibleTileKeys(['a']);
    expect(set.admitTile('b', 60)).toMatchObject({
      admitted: false,
      rejectedReason: 'visible-tiles-fill-budget'
    });
    expect(set.admitTile('huge', 101)).toMatchObject({rejectedReason: 'larger-than-budget'});
    expect(set.getStatus().rejectedTileKeys).toEqual(['b', 'huge']);
    expect(set.has('a')).toBe(true);
  });

  it('reserves and releases memory and retries after registry refusal', () => {
    let reserved = 0;
    const capacity = 50;
    const set = new VectorTileWorkingSet({
      budgetBytes: 1000,
      reserveMemory: byteLength => {
        if (reserved + byteLength > capacity) throw new Error('capacity');
        reserved += byteLength;
        return {
          resize() {},
          release: () => {
            reserved -= byteLength;
          }
        };
      }
    });
    set.admitTile('a', 30);
    expect(set.admitTile('b', 30)).toEqual({admitted: true, evictedTileKeys: ['a']});
    expect(reserved).toBe(30);
    set.destroy();
    expect(reserved).toBe(0);
  });

  it('drops feature registry entries when a tile is evicted', () => {
    const set = new VectorTileWorkingSet({budgetBytes: 10, featureIdOptions: options});
    set.admitTile('a', 10, [{id: 1}]);
    set.admitTile('b', 10, [{id: 2}]);
    expect(set.featureRegistry?.summarize().uniqueFeatureCount).toBe(1);
  });
});

describe('vector tile analysis scope', () => {
  const inputs = {
    datasetId: 'tiles',
    zoom: 4.256,
    visibleTileKeys: ['a', 'b', 'c'],
    loadedTileKeys: ['a', 'b']
  };

  it('labels viewport populations and refuses global analysis', () => {
    const scope = describeVectorTileAnalysisScope(inputs);
    expect(scope).toMatchObject({
      kind: 'viewport-working-set',
      canInferGlobally: false,
      isTileCoverageComplete: false
    });
    expect(scope.label).toBe('Viewport working set (2 of 3 visible tiles, zoom 4.26, incomplete)');
    expect(decideVectorTileAnalysis(scope, {requiresCompleteDataset: true}).allowed).toBe(false);
    expect(decideVectorTileAnalysis(scope, {requiresCompleteDataset: false})).toEqual({
      allowed: true,
      resultLabel: scope.label
    });
  });

  it('does not treat fully loaded visible tiles as the complete dataset', () => {
    const scope = describeVectorTileAnalysisScope({...inputs, loadedTileKeys: ['a', 'b', 'c']});
    expect(scope.isTileCoverageComplete).toBe(true);
    expect(scope.canInferGlobally).toBe(false);
    expect(decideVectorTileAnalysis(scope, {requiresCompleteDataset: true}).allowed).toBe(false);
  });

  it('refuses counting without stable ids and allows a materialized source', () => {
    const scope = describeVectorTileAnalysisScope(inputs);
    expect(
      decideVectorTileAnalysis(scope, {
        requiresCompleteDataset: false,
        requiresStableFeatureIds: true
      }).allowed
    ).toBe(false);
    const complete = describeVectorTileAnalysisScope({...inputs, isSourceFullyMaterialized: true});
    expect(decideVectorTileAnalysis(complete, {requiresCompleteDataset: true}).allowed).toBe(true);
  });
});

describe('decoded tile batches', () => {
  const features = [
    {id: 1, properties: {value: 5}, geometry: {type: 'Point', coordinates: [1, 2]}},
    {id: 2, properties: {value: 15}, geometry: {type: 'Point', coordinates: [3, 4]}},
    {
      id: 3,
      properties: {value: 'x'},
      geometry: {
        type: 'LineString',
        coordinates: [
          [0, 0],
          [1, 1],
          [2, 2]
        ]
      }
    },
    {id: 4, properties: {}, geometry: null}
  ];

  it('groups by geometry family and filters', () => {
    const batches = buildDecodedTileBatches(features, {...options, numericFieldNames: ['value']});
    expect(batches.map(batch => batch.family)).toEqual(['point', 'line']);
    const [points, lines] = batches;
    expect(Array.from(points.positions)).toEqual([1, 2, 3, 4]);
    expect(Array.from(lines.vertexCounts)).toEqual([3]);
    const mask = computeDecodedTileFilterMask(points, [
      {kind: 'range', field: 'value', min: 10, max: 20}
    ]);
    expect(Array.from(mask)).toEqual([0, 1]);
    expect(computeDecodedTileStyleDomain(points, 'value')).toEqual([5, 15]);
    expect(computeDecodedTileStyleDomain(points, 'value', mask)).toEqual([15, 15]);
    expect(
      Array.from(
        computeDecodedTileFilterMask(lines, [{kind: 'range', field: 'value', min: 0, max: 9}])
      )
    ).toEqual([0]);
  });
});
