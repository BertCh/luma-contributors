// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {afterEach, beforeAll, expect, test, vi} from 'vitest';
import type {CommandEncoder} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerComputeSession} from '../src/session/compute-session';
import type {NativeStyleLayer} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import type {ResidentDatasetBinding} from '../../../examples/deck/kepler-compute/src/resident-layers';
import {
  IconRuntime,
  makeGPUVectorsFromNativeIcons
} from '../../../examples/deck/kepler-compute/src/icon-runtime';

let NativeStyleRuntime: typeof import('../../../examples/deck/kepler-compute/src/style-host-runtime').NativeStyleRuntime;
let KeplerResidentStyleCache: typeof import('../src/styles/style-cache').KeplerResidentStyleCache;
const state = vi.hoisted(() => ({order: [] as string[], released: [] as string[], sequence: 0}));
vi.mock('../../../examples/deck/kepler-compute/src/icon-runtime', () => ({
  makeGPUVectorsFromNativeIcons: vi.fn((_device, source) => {
    const id = ++state.sequence;
    return {
      source,
      groups: source.groups,
      isCurrent: () => true,
      destroy: vi.fn(completion =>
        Promise.resolve(completion).finally(() => state.released.push(`source-${id}`))
      )
    };
  }),
  IconRuntime: vi.fn(function (props) {
    const id = ++state.sequence;
    return {
      binding: {
        columns: props.columns,
        groups: props.icons.source.groups.map(group => ({
          iconId: group.iconId,
          resolveSourceRow: group.resolveSourceRow,
          positions: props.positions,
          selectionMask: props.selectionMask,
          colors: props.colors,
          radii: props.radii
        }))
      },
      prepare: vi.fn(),
      encode: vi.fn(() => state.order.push('icon')),
      destroy: vi.fn(completion =>
        Promise.resolve(completion).finally(() => state.released.push(`gather-${id}`))
      )
    };
  })
}));
beforeAll(async () => {
  vi.resetModules();
  ({NativeStyleRuntime} = await import(
    '../../../examples/deck/kepler-compute/src/style-host-runtime'
  ));
  ({KeplerResidentStyleCache} = await import('../src/styles/style-cache'));
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.clearAllMocks();
  state.order = [];
  state.released = [];
  state.sequence = 0;
});

function setup() {
  const source = new arrow.Table({
    latitude: arrow.vectorFromArray([10, 11, null], new arrow.Float32()),
    longitude: arrow.vectorFromArray([20, 21, 22], new arrow.Float32()),
    marker: arrow.vectorFromArray(['triangle', null, 'triangle'], new arrow.Utf8())
  }).concat(
    new arrow.Table({
      latitude: arrow.vectorFromArray([12, 13], new arrow.Float32()),
      longitude: arrow.vectorFromArray([22, 23], new arrow.Float32()),
      marker: arrow.vectorFromArray(['diamond', 'triangle'], new arrow.Utf8())
    })
  );
  const reads = source.batches.map(batch => vi.spyOn(batch, 'getChild'));
  const mask = {format: 'uint32', length: 5} as GPUVector<'uint32'>;
  const positions = {format: 'float32x2', length: 5} as GPUVector<'float32x2'>;
  const colors = {format: 'unorm8x4', length: 5} as GPUVector<'unorm8x4'>;
  const radii = {format: 'float32', length: 5} as GPUVector<'float32'>;
  const dataset = {
    id: 'icons',
    revision: 1,
    rowCount: 5,
    value: {
      source,
      columns: {value: {format: 'float32'}},
      analytics: {dictionaries: {}},
      prepared: {dictionaries: {}}
    }
  };
  let fence: Promise<void> = Promise.resolve();
  const session = {
    device: {createFence: () => ({signaled: fence, destroy: vi.fn()})},
    registry: {getDataset: () => dataset}
  } as unknown as KeplerComputeSession;
  vi.spyOn(KeplerResidentStyleCache.prototype, 'set').mockImplementation(
    plan =>
      ({
        currentPlan: plan,
        currentRevision: 1,
        isCurrent: true,
        canUpdate: () => true,
        colors: plan.kind === 'color' ? colors : undefined,
        values: plan.kind === 'scalar' ? radii : undefined,
        encode: () => state.order.push('style'),
        discardEncoding: vi.fn(),
        readLegend: async () => null
      }) as never
  );
  vi.spyOn(KeplerResidentStyleCache.prototype, 'remove').mockResolvedValue();
  vi.spyOn(KeplerResidentStyleCache.prototype, 'destroy').mockResolvedValue();
  const runtime = new NativeStyleRuntime({session, requestUpdate: vi.fn()});
  const layer: NativeStyleLayer = {
    id: 'markers',
    type: 'icon',
    iconGeometryVersion: 1,
    iconGeometry: {triangle: [0, 0, 0, 1, 0, 0, 0, 1, 0], diamond: [0, 0, 0, 1, 1, 0, 0, 2, 0]},
    config: {
      dataId: 'icons',
      columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}, icon: {value: 'marker'}},
      color: [20, 30, 40],
      colorField: {name: 'value'},
      colorScale: 'linear',
      colorDomain: [0, 1],
      sizeField: {name: 'value'},
      sizeScale: 'linear',
      sizeDomain: [0, 1],
      visConfig: {colorRange: {colors: ['#000000', '#ffffff']}, radius: 3, radiusRange: [0, 10]}
    }
  };
  const binding: ResidentDatasetBinding = {
    selectionMask: mask,
    point: {positions, columns: {lat: 'latitude', lng: 'longitude'}}
  };
  return {
    runtime,
    layer,
    binding,
    dataset,
    reads,
    colors,
    radii,
    setFence: (next: Promise<void>) => {
      fence = next;
    }
  };
}

test('Icon host prepares immutable Arrow groups once and reuses them across native styles and masks', async () => {
  const context = setup();
  const first = context.runtime.resolve(context.layer, context.binding);
  expect(first.unsupportedReason).toBeUndefined();
  expect(first.icon?.groups.map(group => group.iconId)).toEqual(['triangle', 'diamond']);
  expect(first.icon?.groups[0].resolveSourceRow(0)).toBe(0);
  expect(first.icon?.groups[0].resolveSourceRow(1)).toBe(4);
  expect(first.icon?.groups[1].resolveSourceRow(0)).toBe(3);
  expect(first.icon?.groups[0].colors).toBe(context.colors);
  expect(first.icon?.groups[0].radii).toBe(context.radii);
  const reads = context.reads.map(read => read.mock.calls.length);
  for (let zoom = 0; zoom < 4; zoom++)
    context.runtime.resolve(context.layer, context.binding, {zoom});
  expect(makeGPUVectorsFromNativeIcons).toHaveBeenCalledTimes(1);
  expect(IconRuntime).toHaveBeenCalledTimes(1);
  context.runtime.resolve(context.layer, {
    ...context.binding,
    selectionMask: {...context.binding.selectionMask} as GPUVector<'uint32'>
  });
  expect(makeGPUVectorsFromNativeIcons).toHaveBeenCalledTimes(1);
  expect(IconRuntime).toHaveBeenCalledTimes(2);
  expect(context.reads.map(read => read.mock.calls.length)).toEqual(reads);
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['style', 'style', 'icon']);
  await context.runtime.destroy(Promise.resolve());
  expect(state.released.filter(value => value.startsWith('source'))).toHaveLength(1);
});

test('Icon host retries pending catalogs by version and caches unknown marker failures without rescanning', async () => {
  const context = setup();
  const pending = {...context.layer, iconGeometry: null};
  expect(context.runtime.resolve(pending, context.binding).unsupportedReason).toMatch(/loading/);
  expect(context.reads.every(read => read.mock.calls.length === 0)).toBe(true);
  expect(context.runtime.resolve(context.layer, context.binding).unsupportedReason).toBeUndefined();
  const missing = {
    ...context.layer,
    iconGeometry: {triangle: [0, 0, 0, 1, 0, 0, 0, 1, 0]},
    iconGeometryVersion: 2
  };
  expect(context.runtime.resolve(missing, context.binding).unsupportedReason).toMatch(/diamond/);
  const reads = context.reads.map(read => read.mock.calls.length);
  expect(context.runtime.resolve(missing, context.binding).unsupportedReason).toMatch(/diamond/);
  expect(context.reads.map(read => read.mock.calls.length)).toEqual(reads);
  expect(
    context.runtime.resolve({...context.layer, iconGeometryVersion: 3}, context.binding)
      .unsupportedReason
  ).toBeUndefined();
  expect(makeGPUVectorsFromNativeIcons).toHaveBeenCalledTimes(2);
  await context.runtime.destroy(Promise.resolve());
});

test('Icon source retirement waits for earlier replaced gathers even when the final caller fence is already settled', async () => {
  const context = setup();
  context.runtime.resolve(context.layer, context.binding);
  let release!: () => void;
  context.setFence(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  context.runtime.resolve(context.layer, {
    ...context.binding,
    selectionMask: {...context.binding.selectionMask} as GPUVector<'uint32'>
  });
  const retired = context.runtime.retireDataset('icons', Promise.resolve());
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  expect(state.released.some(value => value.startsWith('source'))).toBe(false);
  release();
  await retired;
  expect(state.released.at(-1)).toBe('source-1');
  await context.runtime.destroy(Promise.resolve());
});

test('Icon altitude and changed coordinate columns require matching explicit position uploads', async () => {
  const context = setup();
  const altitude = {
    ...context.layer,
    config: {
      ...context.layer.config,
      columns: {...context.layer.config.columns, altitude: {value: 'elevation'}}
    }
  };
  expect(context.runtime.resolve(altitude, context.binding).unsupportedReason).toMatch(
    /XYZ upload/
  );
  expect(context.reads.every(read => read.mock.calls.length === 0)).toBe(true);
  const changed = {
    ...context.layer,
    config: {
      ...context.layer.config,
      columns: {...context.layer.config.columns, lat: {value: 'other-latitude'}}
    }
  };
  expect(context.runtime.resolve(changed, context.binding).unsupportedReason).toMatch(
    /matching source-aligned/
  );
  expect(makeGPUVectorsFromNativeIcons).not.toHaveBeenCalled();
  await context.runtime.destroy(Promise.resolve());
});

test('dataset retirement also awaits older Icon catalogs that still borrow publication positions', async () => {
  const context = setup();
  context.runtime.resolve(context.layer, context.binding);
  let release!: () => void;
  context.setFence(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  context.runtime.resolve({...context.layer, iconGeometryVersion: 2}, context.binding);
  let completed = false;
  const retirement = context.runtime.retireDataset('icons', Promise.resolve()).then(() => {
    completed = true;
  });
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  expect(completed).toBe(false);
  expect(state.released).not.toContain('source-1');
  release();
  await retirement;
  expect(state.released).toContain('source-1');
  await context.runtime.destroy(Promise.resolve());
});
