// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, test} from 'vitest';
import {PointPositionVariantRuntime} from '../../../examples/deck/kepler-compute/src/point-position-variants';
import {
  makePointVariantBindingOverride,
  planPointLayerSelectorWithVariants
} from '../../../examples/deck/kepler-compute/src/point-variant-layer-source';
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
const source = {...primary, positions: {} as never, revision: 1};

function makeLayer(id: string, longitude: string, latitude: string): PointSelectorLayer {
  return {
    id,
    type: 'point',
    config: {
      dataId: 'observations',
      isVisible: true,
      columns: {lng: {value: longitude}, lat: {value: latitude}, altitude: {value: null}}
    }
  };
}

describe('point position variants', () => {
  test('different lng/lat pair prepares one shared variant, selectors per layer id', () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    const first = planPointLayerSelectorWithVariants(
      variants,
      makeLayer('a', 'dropLng', 'dropLat'),
      dataset,
      source
    );
    const second = planPointLayerSelectorWithVariants(
      variants,
      makeLayer('b', 'dropLng', 'dropLat'),
      dataset,
      source
    );
    expect(first.status).toBe('selector');
    expect(second.status).toBe('selector');
    if (first.status === 'selector' && second.status === 'selector') {
      expect(first.selector.id).toBe('a');
      expect(second.selector.id).toBe('b');
      expect(first.selector.geometryRevision).toBe(second.selector.geometryRevision);
      expect(first.selector.geometry).toEqual(second.selector.geometry);
    }
    expect(variants.variantCount).toBe(1);
    expect(variants.gpuByteLength).toBe(24);
    expect(state.reserved).toBe(24);
    expect(state.buffersAlive).toBe(1);
  });

  test('primary pair keeps shared positions and releases a held variant', async () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    planPointLayerSelectorWithVariants(
      variants,
      makeLayer('a', 'dropLng', 'dropLat'),
      dataset,
      source
    );
    const plan = planPointLayerSelectorWithVariants(
      variants,
      makeLayer('a', 'lng', 'lat'),
      dataset,
      source
    );
    expect(plan.status).toBe('selector');
    expect(variants.hasUnreferencedVariants).toBe(true);
    await variants.retainLayers(['a'], Promise.resolve());
    expect(variants.variantCount).toBe(0);
    expect(state.reserved).toBe(0);
    expect(state.buffersAlive).toBe(0);
    expect(state.leasesAlive).toBe(0);
  });

  test('removed layers and retired datasets free variant memory', async () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    planPointLayerSelectorWithVariants(
      variants,
      makeLayer('a', 'dropLng', 'dropLat'),
      dataset,
      source
    );
    await variants.retainLayers([], Promise.resolve());
    expect(state.reserved).toBe(0);
    planPointLayerSelectorWithVariants(
      variants,
      makeLayer('b', 'dropLat', 'dropLng'),
      dataset,
      source
    );
    expect(state.reserved).toBe(24);
    await variants.retireDataset('observations', Promise.resolve());
    expect(state.reserved).toBe(0);
    expect(state.buffersAlive).toBe(0);
  });

  test('unknown or invalid columns report a reason and allocate nothing', () => {
    const {session, state} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    const plan = planPointLayerSelectorWithVariants(
      variants,
      makeLayer('a', 'missing', 'dropLat'),
      dataset,
      source
    );
    expect(plan.status).toBe('mismatch');
    expect(state.reserved).toBe(0);
    expect(state.leasesAlive).toBe(0);
  });

  test('render override swaps positions and honors altitude validity', () => {
    const {session} = makeFakeSession();
    const variants = new PointPositionVariantRuntime(session, () => undefined);
    const layer = makeLayer('a', 'dropLng', 'dropLat');
    layer.config.columns['altitude'] = {value: 'alt'};
    let validity: unknown;
    const override = makePointVariantBindingOverride(variants, layer, dataset, primary, value => {
      validity = value;
      return {} as never;
    });
    expect(override?.point?.positions.format).toBe('float32x3');
    expect(override?.point?.columns).toEqual({lng: 'dropLng', lat: 'dropLat', altitude: 'alt'});
    expect(validity).toBeDefined();
    expect(
      makePointVariantBindingOverride(
        variants,
        makeLayer('b', 'lng', 'lat'),
        dataset,
        primary,
        () => ({}) as never
      )
    ).toBeUndefined();
  });
});
