// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, test} from 'vitest';
import {PointPositionVariantRuntime} from '../../../examples/deck/kepler-compute/src/point-position-variants';
import {
  makeConnectionVariantBinding,
  resolveAggregationVariant,
  resolveVariantPositions
} from '../../../examples/deck/kepler-compute/src/point-variant-extended-sources';
import type {PointSelectorLayer} from '../../../examples/deck/kepler-compute/src/resident-point-layer-selectors';

function makeFakeSession() {
  const state = {reserved: 0, buffersAlive: 0, leasesAlive: 0};
  const session = {
    deviceLost: false,
    device: {
      lost: new Promise<never>(() => undefined),
      createBuffer: (props: {data?: ArrayBufferView; byteLength?: number}) => {
        state.buffersAlive++;
        return {
          byteLength: props.data?.byteLength ?? props.byteLength ?? 0,
          destroy() {
            state.buffersAlive--;
          }
        };
      }
    },
    registry: {
      reserveMemory(byteLength: number) {
        state.reserved += byteLength;
        let released = false;
        return {
          byteLength,
          release() {
            if (!released) state.reserved -= byteLength;
            released = true;
          },
          resize() {}
        };
      },
      acquireDataset() {
        state.leasesAlive++;
        let valid = true;
        return {
          isValid: () => valid,
          release: async () => {
            if (valid) state.leasesAlive--;
            valid = false;
          }
        };
      }
    }
  };
  return {state, session: session as never};
}

const prepared = {
  table: arrow.tableFromArrays({
    lng: new Float32Array([1, 2, 3]),
    lat: new Float32Array([4, 5, 6]),
    dropLng: new Float32Array([10, 20, 30]),
    dropLat: new Float32Array([40, 50, 60]),
    alt: new Float32Array([1, 2, 3])
  }),
  dictionaries: {},
  sourceBatches: [{sourceRowIndexOffset: 0, sourceRowCount: 3}]
} as never;
const dataset = {id: 'observations', revision: 1, prepared};
const primary = {longitude: 'lng', latitude: 'lat'};

function makeLayer(
  id: string,
  longitude: string,
  latitude: string,
  type = 'heatmap'
): PointSelectorLayer {
  return {
    id,
    type,
    config: {
      dataId: 'observations',
      isVisible: true,
      columns: {lng: {value: longitude}, lat: {value: latitude}, altitude: {value: null}}
    }
  };
}

describe('extended point position variants', () => {
  test('heatmap and cluster read variant positions only for a non-primary pair', () => {
    const {session} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    expect(
      resolveVariantPositions(variants, makeLayer('a', 'lng', 'lat'), dataset, primary).status
    ).toBe('primary');
    const result = resolveVariantPositions(
      variants,
      makeLayer('b', 'dropLng', 'dropLat', 'cluster'),
      dataset,
      primary
    );
    expect(result.status).toBe('variant');
    if (result.status === 'variant')
      expect(result.columns).toEqual({lng: 'dropLng', lat: 'dropLat'});
    expect(
      resolveVariantPositions(variants, makeLayer('c', 'missing', 'lat'), dataset, primary).status
    ).toBe('unavailable');
  });

  test('grid aggregation builds one runtime per pair and destroys it with the variant', async () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    let created = 0;
    let destroyed = 0;
    const context = {
      device: {} as never,
      registry: {} as never,
      requestUpdate: () => undefined,
      createRuntime: (props: {coordinateColumns?: unknown}) => {
        created++;
        expect(props.coordinateColumns).toEqual({lat: 'dropLat', lng: 'dropLng'});
        return {
          resolve: () => ({binding: {} as never}),
          destroy: async () => {
            destroyed++;
          }
        } as never;
      }
    };
    const aggregationDataset = {...dataset, rowCount: 3, batchRowCounts: [3], columns: {}};
    const resolve = (id: string, lng: string, lat: string) =>
      resolveAggregationVariant(
        variants,
        context,
        makeLayer(id, lng, lat, 'grid') as never,
        aggregationDataset,
        primary,
        {} as never,
        1
      );
    expect(resolve('a', 'lng', 'lat')).toBeUndefined();
    expect(resolve('a', 'dropLng', 'dropLat')?.binding).toBeDefined();
    expect(resolve('b', 'dropLng', 'dropLat')?.binding).toBeDefined();
    expect(created).toBe(1);
    await variants.retainLayers([], Promise.resolve());
    expect(destroyed).toBe(1);
    expect(state.reserved).toBe(0);
  });

  test('line and arc endpoints map to source and target variants', async () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    const layer: PointSelectorLayer = {
      id: 'arc',
      type: 'arc',
      config: {
        columns: {
          lng0: {value: 'dropLng'},
          lat0: {value: 'dropLat'},
          lng1: {value: 'lng'},
          lat1: {value: 'lat'}
        }
      }
    };
    const bound = {lng0: 'lng', lat0: 'lat', lng1: 'dropLng', lat1: 'dropLat'};
    const binding = makeConnectionVariantBinding(variants, layer, dataset, bound);
    expect(binding && 'columns' in binding ? binding.columns : undefined).toEqual({
      lng0: 'dropLng',
      lat0: 'dropLat',
      lng1: 'lng',
      lat1: 'lat'
    });
    expect(variants.variantCount).toBe(2);
    // Sub-keys survive retention for an existing layer and are freed with the layer.
    await variants.retainLayers(['arc'], Promise.resolve());
    expect(variants.variantCount).toBe(2);
    expect(
      makeConnectionVariantBinding(variants, layer, dataset, {
        lng0: 'dropLng',
        lat0: 'dropLat',
        lng1: 'lng',
        lat1: 'lat'
      })
    ).toBeUndefined();
    await variants.retainLayers(['arc'], Promise.resolve());
    expect(variants.variantCount).toBe(0);
    expect(state.reserved).toBe(0);
  });
});

test('line altitude changes create independent XYZ endpoint variants with stable cached storage', async () => {
  const {session, state} = makeFakeSession();
  const variants = new PointPositionVariantRuntime(session, () => undefined);
  const layer = makeLayer('line-altitude', 'lng', 'lat', 'line');
  layer.config.columns = {
    lng0: {value: 'lng'},
    lat0: {value: 'lat'},
    lng1: {value: 'dropLng'},
    lat1: {value: 'dropLat'},
    alt0: {value: 'alt'},
    alt1: {value: 'alt'}
  };
  const bound = {lng0: 'lng', lat0: 'lat', lng1: 'dropLng', lat1: 'dropLat'};
  const first = makeConnectionVariantBinding(variants, layer, dataset, bound)!;
  expect('unsupportedReason' in first).toBe(false);
  if ('unsupportedReason' in first) throw new Error(first.unsupportedReason);
  expect(first.sourcePositions.format).toBe('float32x3');
  expect(first.targetPositions.format).toBe('float32x3');
  expect(first.sourceAltitudeValidity?.length).toBe(3);
  const next = makeConnectionVariantBinding(variants, layer, dataset, bound)!;
  if ('unsupportedReason' in next) throw new Error(next.unsupportedReason);
  expect(next.sourcePositions).toBe(first.sourcePositions);
  expect(next.targetPositions).toBe(first.targetPositions);
  await variants.destroy(Promise.resolve());
  expect(state.reserved).toBe(0);
  expect(state.buffersAlive).toBe(0);
  expect(state.leasesAlive).toBe(0);
});
