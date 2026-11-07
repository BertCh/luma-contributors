// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, expect, test, vi} from 'vitest';
import type {CommandEncoder} from '@luma.gl/core';
import type {PreparedGeometryGPUVectors} from '@luma.gl/engine/arrow';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {KeplerComputeSession} from '../src/session/compute-session';
import type {NativeStyleLayer} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import {NativeStyleRuntime} from '../../../examples/deck/kepler-compute/src/style-host-runtime';
import {GeoJSONPropertyRuntime} from '../../../examples/deck/kepler-compute/src/geojson-property-runtime';
import {GeometryRuntime} from '../../../examples/deck/kepler-compute/src/geometry-runtime';

const state = vi.hoisted(() => {
  // Reset before static imports so the shared worker binds this host to this file's test doubles.
  // Graph loading belongs to module bootstrap, not a timed beforeAll test hook.
  vi.resetModules();
  return {order: [] as string[]};
});
vi.mock('../../../examples/deck/kepler-compute/src/geojson-property-runtime', () => ({
  GeoJSONPropertyRuntime: vi.fn(function (_device, source, fallback) {
    return {
      source,
      output: {...source.values},
      colors: source.values.format === 'unorm8x4' ? {...source.values} : undefined,
      values: source.values.format === 'float32' ? {...source.values} : undefined,
      fallback,
      setFallback: vi.fn(value => {
        const changed = JSON.stringify(fallback) !== JSON.stringify(value);
        fallback = value;
        return changed;
      }),
      encode: vi.fn(() => state.order.push('properties')),
      discardEncoding: vi.fn(),
      destroy: vi.fn(),
      accountedResidentByteLength: 8
    };
  })
}));
vi.mock('../../../examples/deck/kepler-compute/src/geometry-runtime', () => ({
  GeometryRuntime: vi.fn(function (props) {
    return {
      binding: {column: props.column, points: {colors: props.colors}},
      prepare: vi.fn(),
      setVerticalDefaults: vi.fn(),
      encode: vi.fn(() => state.order.push('geometry')),
      destroy: vi.fn(),
      accountedResidentByteLength: 8
    };
  })
}));
afterEach(() => {
  vi.clearAllMocks();
  state.order = [];
});

function setup() {
  const source = new Proxy(
    {},
    {
      get() {
        throw new Error('Source rows must not be read');
      }
    }
  );
  const values = {format: 'unorm8x4', length: 1, data: [{length: 1}]} as GPUVector<'unorm8x4'>;
  const validity = {format: 'uint32', length: 1, data: [{length: 1}]} as GPUVector<'uint32'>;
  const geometry = {
    metadata: {
      rowCount: 1,
      sourceStyleProperties: ['fillColor'],
      fixedRadius: true,
      invalidStyleProperties: [],
      featureTypes: {point: true, line: false, polygon: false}
    },
    properties: {fillColor: {values, validity}},
    buffers: []
  } as unknown as PreparedGeometryGPUVectors;
  const reservations: {release: ReturnType<typeof vi.fn>}[] = [];
  let fence: Promise<void> = Promise.resolve();
  const session = {
    device: {createFence: () => ({signaled: fence, destroy: vi.fn()})},
    registry: {
      getDataset: () => ({
        id: 'geo',
        revision: 1,
        rowCount: 1,
        value: {source, columns: {}, analytics: {dictionaries: {}}, prepared: {dictionaries: {}}}
      }),
      reserveMemory: () => {
        const reservation = {release: vi.fn()};
        reservations.push(reservation);
        return reservation;
      }
    }
  } as unknown as KeplerComputeSession;
  const requestUpdate = vi.fn();
  const runtime = new NativeStyleRuntime({session, requestUpdate});
  const layer: NativeStyleLayer = {
    id: 'geometry',
    type: 'geojson',
    config: {dataId: 'geo', color: [10, 20, 30], visConfig: {filled: true, stroked: false}}
  };
  const binding = {selectionMask: validity};
  return {
    runtime,
    layer,
    geometry,
    binding,
    reservations,
    requestUpdate,
    setFence: (next: Promise<void>) => {
      fence = next;
    }
  };
}

test('embedded properties update native defaults without reallocating source outputs or geometry and retire behind a fence', async () => {
  const context = setup();
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  const first = context.runtime.resolve(context.layer, context.binding, options);
  expect(first.unsupportedReason).toBeUndefined();
  expect(first.geojson?.fixedRadius).toBe(true);
  await Promise.resolve();
  context.requestUpdate.mockClear();
  context.layer.config.color = [40, 50, 60];
  const second = context.runtime.resolve(context.layer, context.binding, options);
  await Promise.resolve();
  expect(context.requestUpdate).toHaveBeenCalledOnce();
  expect(second.channels?.colors).toBe(first.channels?.colors);
  expect(GeoJSONPropertyRuntime).toHaveBeenCalledTimes(1);
  expect(GeometryRuntime).toHaveBeenCalledTimes(1);
  const property = vi.mocked(GeoJSONPropertyRuntime).mock.results[0].value;
  expect(property.setFallback).toHaveBeenLastCalledWith([40, 50, 60]);
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['properties', 'geometry']);
  context.runtime.discardEncoding();
  expect(property.discardEncoding).toHaveBeenCalledOnce();
  let complete!: () => void;
  const retirement = context.runtime.retireDataset(
    'geo',
    new Promise<void>(resolve => {
      complete = resolve;
    })
  );
  expect(property.destroy).not.toHaveBeenCalled();
  complete();
  await retirement;
  expect(property.destroy).toHaveBeenCalledOnce();
  for (const reservation of context.reservations)
    expect(reservation.release).toHaveBeenCalledOnce();
  await context.runtime.destroy(Promise.resolve());
});

test('active malformed embedded properties disable rendering but analysis overrides and disabled channels ignore them', async () => {
  const context = setup();
  context.geometry.metadata.invalidStyleProperties = [
    {sourceRowIndex: 0, property: 'fillColor', reason: 'Unsupported source color'}
  ];
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  const invalid = context.runtime.resolve(context.layer, context.binding, options);
  expect(invalid.enabled).toBe(false);
  expect(invalid.unsupportedReason).toContain('Unsupported source color');
  const override = context.geometry.properties!.fillColor!.values;
  const analysis = context.runtime.resolve(
    context.layer,
    {...context.binding, colorOverride: override},
    options
  );
  expect(analysis.unsupportedReason).toBeUndefined();
  expect(analysis.colorOverride).toBeUndefined();
  expect(vi.mocked(GeometryRuntime).mock.calls.at(-1)?.[0].colors).toBe(override);
  expect(GeoJSONPropertyRuntime).not.toHaveBeenCalled();
  context.layer.config.visConfig.filled = false;
  expect(context.runtime.resolve(context.layer, context.binding, options).enabled).not.toBe(false);
  expect(GeoJSONPropertyRuntime).not.toHaveBeenCalled();
  await context.runtime.destroy(Promise.resolve());
});

test('deleting a GeoJSON publication waits for older geometry and property generations', async () => {
  const context = setup();
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  context.runtime.resolve(context.layer, context.binding, options);
  const oldGeometry = vi.mocked(GeometryRuntime).mock.results[0].value;
  const oldProperty = vi.mocked(GeoJSONPropertyRuntime).mock.results[0].value;
  let release!: () => void;
  context.setFence(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  const replacement = {
    ...context.geometry,
    properties: {
      ...context.geometry.properties,
      fillColor: {
        ...context.geometry.properties!.fillColor!,
        values: {...context.geometry.properties!.fillColor!.values} as GPUVector<'unorm8x4'>
      }
    }
  };
  expect(
    context.runtime.resolve(context.layer, context.binding, {...options, geometry: replacement})
      .unsupportedReason
  ).toBeUndefined();
  let completed = false;
  const pending = context.runtime.retireDataset('geo', Promise.resolve()).then(() => {
    completed = true;
  });
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  expect(completed).toBe(false);
  expect(oldGeometry.destroy).not.toHaveBeenCalled();
  expect(oldProperty.destroy).not.toHaveBeenCalled();
  release();
  await pending;
  expect(oldGeometry.destroy).toHaveBeenCalledOnce();
  expect(oldProperty.destroy).toHaveBeenCalledOnce();
  for (const reservation of context.reservations)
    expect(reservation.release).toHaveBeenCalledOnce();
  await context.runtime.destroy(Promise.resolve());
});

test('polygon height and offset properties use distinct native fallback rules and retain stable gather outputs', async () => {
  const context = setup();
  context.geometry.metadata.featureTypes.polygon = true;
  const values = {format: 'float32', length: 1, data: [{length: 1}]} as GPUVector<'float32'>;
  const validity = context.binding.selectionMask;
  context.geometry.properties!.elevation = {values, validity};
  context.geometry.properties!.elevationOffset = {
    values: {...values} as GPUVector<'float32'>,
    validity
  };
  context.geometry.metadata.sourceStyleProperties = [
    ...context.geometry.metadata.sourceStyleProperties,
    'elevation',
    'elevationOffset'
  ];
  context.layer.config.visConfig.enable3d = true;
  context.layer.config.visConfig.elevationOffset = 50;
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  expect(
    context.runtime.resolve(context.layer, context.binding, options).unsupportedReason
  ).toBeUndefined();
  const calls = vi.mocked(GeoJSONPropertyRuntime).mock.calls;
  expect(calls.find(call => call[1].values === values)?.[2]).toBe(500);
  expect(
    calls.find(call => call[1].values === context.geometry.properties!.elevationOffset!.values)?.[2]
  ).toBe(50);
  const first = vi.mocked(GeometryRuntime).mock.calls.at(-1)![0];
  expect(first.elevations).toBeDefined();
  expect(first.elevationOffsets).toBeDefined();
  expect(first.elevations).not.toBe(first.elevationOffsets);
  context.layer.config.visConfig.elevationOffset = -20;
  context.runtime.resolve(context.layer, context.binding, options);
  expect(GeometryRuntime).toHaveBeenCalledTimes(1);
  expect(GeoJSONPropertyRuntime).toHaveBeenCalledTimes(3);
  context.layer.config.visConfig.enable3d = false;
  context.runtime.resolve(context.layer, context.binding, options);
  expect(vi.mocked(GeometryRuntime).mock.calls.at(-1)![0].elevations).toBe(0);
  expect(vi.mocked(GeometryRuntime).mock.calls.at(-1)![0].elevationOffsets).toBe(
    first.elevationOffsets
  );
  await context.runtime.destroy(Promise.resolve());
});

test('uniform polygon offset controls update without allocating a replacement geometry runtime', async () => {
  const context = setup();
  context.geometry.metadata.featureTypes.polygon = true;
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  context.layer.config.visConfig.elevationOffset = 5;
  context.runtime.resolve(context.layer, context.binding, options);
  context.layer.config.visConfig.elevationOffset = 20;
  context.runtime.resolve(context.layer, context.binding, options);
  expect(GeometryRuntime).toHaveBeenCalledTimes(1);
  const geometry = vi.mocked(GeometryRuntime).mock.results[0].value;
  expect(geometry.setVerticalDefaults).toHaveBeenLastCalledWith({
    elevation: 0,
    elevationOffset: 20
  });
  await context.runtime.destroy(Promise.resolve());
});

test('vertical source diagnostics follow actual polygon channel use', async () => {
  const context = setup();
  context.geometry.metadata.featureTypes.polygon = true;
  const values = {format: 'float32', length: 1, data: [{length: 1}]} as GPUVector<'float32'>;
  context.geometry.properties!.elevation = {values, validity: context.binding.selectionMask};
  context.geometry.metadata.sourceStyleProperties = ['fillColor', 'elevation'];
  context.geometry.metadata.invalidStyleProperties = [
    {sourceRowIndex: 0, property: 'elevation', reason: 'Invalid embedded height'}
  ];
  const options = {geometry: context.geometry, geometryColumn: 'shape'};
  expect(
    context.runtime.resolve(context.layer, context.binding, options).unsupportedReason
  ).toBeUndefined();
  context.layer.config.visConfig.enable3d = true;
  expect(
    context.runtime.resolve(context.layer, context.binding, options).unsupportedReason
  ).toContain('Invalid embedded height');
  context.layer.config.visConfig.filled = false;
  expect(
    context.runtime.resolve(context.layer, context.binding, options).unsupportedReason
  ).toBeUndefined();
  context.layer.config.visConfig.filled = true;
  context.layer.config.visConfig.enable3d = false;
  context.geometry.properties!.elevationOffset = {values, validity: context.binding.selectionMask};
  context.geometry.metadata.sourceStyleProperties = ['fillColor', 'elevationOffset'];
  context.geometry.metadata.invalidStyleProperties = [
    {sourceRowIndex: 0, property: 'elevationOffset', reason: 'Offset exceeds Float32'}
  ];
  // Even a flat polygon invokes the native offset accessor.
  expect(
    context.runtime.resolve(context.layer, context.binding, options).unsupportedReason
  ).toContain('Offset exceeds Float32');
  await context.runtime.destroy(Promise.resolve());
});
