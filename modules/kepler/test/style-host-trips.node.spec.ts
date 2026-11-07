// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import type {CommandEncoder} from '@luma.gl/core';
import type {GeoJSONTripGPUVectors} from '@luma.gl/engine/arrow';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerComputeSession} from '../src/session/compute-session';
import type {KeplerResidentStyle} from '../src/styles/resident-style';
import type {NativeStyleLayer} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import {TripRuntime} from '../../../examples/deck/kepler-compute/src/trip-runtime';
import {GeoJSONPropertyRuntime} from '../../../examples/deck/kepler-compute/src/geojson-property-runtime';

import {NativeStyleRuntime} from '../../../examples/deck/kepler-compute/src/style-host-runtime';
import {KeplerResidentStyleCache} from '../src/styles/style-cache';
const state = vi.hoisted(() => {
  // Resolve the host, cache and test doubles together before static imports in a shared worker.
  // Graph loading belongs to module bootstrap, not a timed beforeAll test hook.
  vi.resetModules();
  return {order: [] as string[], released: [] as string[]};
});
vi.mock('../../../examples/deck/kepler-compute/src/trip-runtime', () => ({
  TripRuntime: vi.fn(function (props) {
    return {
      binding: {
        column: props.column,
        getColor: {primitiveColors: props.colors},
        getWidth: {primitiveWidths: props.widths}
      },
      prepare: vi.fn(),
      encode: vi.fn(() => state.order.push('trip')),
      destroy: vi.fn(completion =>
        Promise.resolve(completion).then(() => {
          state.released.push(props.column);
        })
      ),
      accountedResidentByteLength: 12
    };
  })
}));
vi.mock('../../../examples/deck/kepler-compute/src/geojson-property-runtime', () => ({
  GeoJSONPropertyRuntime: vi.fn(function (_device, source) {
    const output = {...source.values};
    return {
      source,
      output,
      colors: output.format === 'unorm8x4' ? output : undefined,
      values: output.format === 'float32' ? output : undefined,
      setFallback: vi.fn(() => false),
      encode: vi.fn(() => state.order.push('property')),
      discardEncoding: vi.fn(),
      destroy: vi.fn(),
      accountedResidentByteLength: 8
    };
  })
}));
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  state.order = [];
  state.released = [];
});

function setup() {
  const source = new Proxy(
    {},
    {
      get() {
        throw new Error('Source rows must not be scanned');
      }
    }
  );
  const mask = {format: 'uint32', length: 3} as GPUVector<'uint32'>;
  const colors = {format: 'unorm8x4', length: 3} as GPUVector<'unorm8x4'>;
  const widths = {format: 'float32', length: 3} as GPUVector<'float32'>;
  const session = {
    device: {createFence: () => ({signaled: Promise.resolve(), destroy: vi.fn()})},
    registry: {
      reserveMemory: () => ({release: vi.fn()}),
      getDataset: () => ({
        id: 'trips',
        revision: 1,
        rowCount: 3,
        value: {
          source,
          columns: {value: {format: 'float32'}},
          analytics: {dictionaries: {}},
          prepared: {dictionaries: {}}
        }
      })
    }
  } as unknown as KeplerComputeSession;
  const set = vi.spyOn(KeplerResidentStyleCache.prototype, 'set').mockImplementation(
    plan =>
      ({
        currentPlan: plan,
        currentRevision: 1,
        isCurrent: true,
        canUpdate: () => true,
        colors: plan.kind === 'color' ? colors : undefined,
        values: plan.kind === 'scalar' ? widths : undefined,
        encode: () => state.order.push('style'),
        discardEncoding: vi.fn(),
        readLegend: async () => null
      }) as unknown as KeplerResidentStyle
  );
  vi.spyOn(KeplerResidentStyleCache.prototype, 'remove').mockResolvedValue();
  vi.spyOn(KeplerResidentStyleCache.prototype, 'destroy').mockResolvedValue();
  const runtime = new NativeStyleRuntime({session, requestUpdate: vi.fn()});
  const layer: NativeStyleLayer = {
    id: 'routes',
    type: 'trip',
    config: {
      dataId: 'trips',
      color: [1, 2, 3],
      colorField: {name: 'value'},
      colorScale: 'quantile',
      colorDomain: [0, 10],
      sizeField: {name: 'value'},
      sizeScale: 'sqrt',
      sizeDomain: [0, 10],
      visConfig: {
        colorRange: {colors: ['#000000', '#ffffff']},
        sizeRange: [1, 10],
        thickness: 2,
        stroked: true
      }
    }
  };
  const trips = {
    getPath: {},
    getTimestamps: {},
    sourceRowIds: {},
    source: {sourceStyleProperties: [], invalidStyleProperties: []}
  } as unknown as GeoJSONTripGPUVectors;
  return {runtime, layer, mask, colors, widths, trips, set, session};
}

test('trip styles reuse stable mappings, encode styles first, and gather analysis overrides before clearing raw colors', async () => {
  const context = setup();
  const options = {trips: context.trips, tripColumn: 'route'};
  const first = context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  expect(first.unsupportedReason).toBeUndefined();
  expect(TripRuntime).toHaveBeenCalledOnce();
  expect(vi.mocked(TripRuntime).mock.calls[0][0]).toMatchObject({
    colors: context.colors,
    widths: context.widths,
    selectionMask: context.mask,
    memoryRegistry: context.session.registry
  });
  context.runtime.resolve(context.layer, {selectionMask: context.mask}, {...options, zoom: 10});
  expect(TripRuntime).toHaveBeenCalledOnce();
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['style', 'style', 'trip']);
  const analysis = {format: 'unorm8x4', length: 3} as GPUVector<'unorm8x4'>;
  const overridden = context.runtime.resolve(
    context.layer,
    {selectionMask: context.mask, colorOverride: analysis},
    options
  );
  expect(overridden.colorOverride).toBeUndefined();
  expect(overridden.trip).not.toBe(first.trip);
  expect(vi.mocked(TripRuntime).mock.calls.at(-1)?.[0].colors).toBe(analysis);
  expect(context.set).toHaveBeenCalledTimes(2);
  await context.runtime.destroy(Promise.resolve());
});

test('trip topology retires behind the source fence and missing geometry remains explicitly unsupported', async () => {
  const context = setup();
  const missing = context.runtime.resolve(context.layer, {selectionMask: context.mask});
  expect(missing.enabled).toBe(false);
  expect(missing.unsupportedReason).toContain('Prepared trip geometry');
  const supported = context.runtime.resolve(
    context.layer,
    {selectionMask: context.mask},
    {trips: context.trips, tripColumn: 'route'}
  );
  expect(supported.unsupportedReason).toBeUndefined();
  let complete!: () => void;
  const retirement = context.runtime.retireDataset(
    'trips',
    new Promise<void>(resolve => {
      complete = resolve;
    })
  );
  expect(state.released).toEqual([]);
  complete();
  await retirement;
  expect(state.released).toEqual(['route']);
  await context.runtime.destroy(Promise.resolve());
});

test('embedded trip properties use GPU defaults before primitive gathering and native fields bypass malformed properties', async () => {
  const context = setup();
  const propertyColors = {
    format: 'unorm8x4',
    length: 3,
    data: [{length: 3}]
  } as GPUVector<'unorm8x4'>;
  const propertyWidths = {
    format: 'float32',
    length: 3,
    data: [{length: 3}]
  } as GPUVector<'float32'>;
  context.trips.properties = {
    lineColor: {values: propertyColors, validity: context.mask},
    lineWidth: {values: propertyWidths, validity: context.mask}
  };
  context.trips.source.sourceStyleProperties = ['lineColor', 'lineWidth'];
  context.trips.source.invalidStyleProperties = [
    {property: 'lineColor', sourceRowIndex: 0, reason: 'Malformed source color'}
  ];
  const options = {trips: context.trips, tripColumn: 'route'};
  const fieldBound = context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  expect(fieldBound.unsupportedReason).toBeUndefined();
  expect(GeoJSONPropertyRuntime).not.toHaveBeenCalled();
  context.layer.config.colorField = undefined;
  const invalid = context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  expect(invalid.unsupportedReason).toContain('Malformed source color');
  context.trips.source.invalidStyleProperties = [];
  context.layer.config.sizeField = undefined;
  const properties = context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  expect(properties.unsupportedReason).toBeUndefined();
  expect(GeoJSONPropertyRuntime).toHaveBeenCalledTimes(2);
  const args = vi.mocked(TripRuntime).mock.calls.at(-1)![0];
  expect(args.colors).not.toBe(propertyColors);
  expect(args.widths).not.toBe(propertyWidths);
  state.order = [];
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['property', 'property', 'trip']);
  await context.runtime.destroy(Promise.resolve());
});

test('observed trip properties without uploaded buffers are unsupported until a native field replaces them', async () => {
  const context = setup();
  context.layer.config.colorField = undefined;
  context.trips.source.sourceStyleProperties = ['lineColor'];
  const options = {trips: context.trips, tripColumn: 'route'};
  const missing = context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  expect(missing.enabled).toBe(false);
  expect(missing.unsupportedReason).toContain('property presence requires upload metadata');
  expect(TripRuntime).not.toHaveBeenCalled();
  context.layer.config.colorField = {name: 'value'};
  expect(
    context.runtime.resolve(context.layer, {selectionMask: context.mask}, options).unsupportedReason
  ).toBeUndefined();
  await context.runtime.destroy(Promise.resolve());
});

test('deleting a Trip publication waits for earlier replaced gather fences', async () => {
  const context = setup();
  const options = {trips: context.trips, tripColumn: 'route'};
  context.runtime.resolve(context.layer, {selectionMask: context.mask}, options);
  let release!: () => void;
  const fence = new Promise<void>(resolve => {
    release = resolve;
  });
  vi.spyOn(context.session.device, 'createFence').mockReturnValue({
    signaled: fence,
    destroy: vi.fn()
  } as never);
  context.runtime.resolve(
    context.layer,
    {selectionMask: {...context.mask} as GPUVector<'uint32'>},
    options
  );
  let completed = false;
  const pending = context.runtime.retireDataset('trips', Promise.resolve()).then(() => {
    completed = true;
  });
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  expect(completed).toBe(false);
  expect(state.released).toHaveLength(1);
  release();
  await pending;
  expect(state.released).toEqual(['route', 'route']);
  await context.runtime.destroy(Promise.resolve());
});
