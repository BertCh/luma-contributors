// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, beforeAll, expect, test, vi} from 'vitest';
import type {CommandEncoder} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerComputeSession} from '../src/session/compute-session';
import type {NativeStyleLayer} from '../../../examples/deck/kepler-compute/src/native-style-plan';
import type {H3GPUVectors} from '../../../examples/deck/kepler-compute/src/h3-runtime';
import {H3Runtime, H3MetadataRuntime} from '../../../examples/deck/kepler-compute/src/h3-runtime';

let NativeStyleRuntime: typeof import('../../../examples/deck/kepler-compute/src/style-host-runtime').NativeStyleRuntime;
const state = vi.hoisted(() => ({
  order: [] as string[],
  summary: {count: 2, minimumResolution: 7, maximumResolution: 7, hasPentagon: false},
  released: 0
}));
vi.mock('../../../examples/deck/kepler-compute/src/h3-runtime', () => ({
  H3MetadataRuntime: vi.fn(function (props) {
    let revision = props.revision;
    let snapshot;
    return {
      get snapshot() {
        return snapshot;
      },
      setRevision: vi.fn(next => {
        if (revision !== next) snapshot = undefined;
        revision = next;
      }),
      encode: vi.fn(() => state.order.push('metadata')),
      discardEncoding: vi.fn(),
      readMetadata: vi.fn(async () => {
        snapshot = {...state.summary, revision};
        return snapshot;
      }),
      destroy: vi.fn(completion => Promise.resolve(completion))
    };
  }),
  H3Runtime: vi.fn(function (props) {
    return {
      binding: {column: props.column, branch: props.branch},
      setCoverage: vi.fn(),
      setViewportMesh: vi.fn(),
      encode: vi.fn(() => state.order.push('h3')),
      discardEncoding: vi.fn(),
      destroy: vi.fn(completion =>
        Promise.resolve(completion).finally(() => {
          state.released++;
        })
      )
    };
  })
}));
beforeAll(async () => {
  vi.resetModules();
  ({NativeStyleRuntime} = await import(
    '../../../examples/deck/kepler-compute/src/style-host-runtime'
  ));
});
afterEach(() => {
  vi.clearAllMocks();
  state.order = [];
  state.summary = {count: 2, minimumResolution: 7, maximumResolution: 7, hasPentagon: false};
  state.released = 0;
});
function setup() {
  const display = {format: 'uint32', length: 3} as GPUVector<'uint32'>;
  const topology = {format: 'uint32', length: 3} as GPUVector<'uint32'>;
  const h3 = {source: {rowCount: 3}, validity: {data: [{length: 3}]}} as H3GPUVectors;
  let fence: Promise<void> = Promise.resolve();
  const session = {
    device: {createFence: () => ({signaled: fence, destroy: vi.fn()})},
    registry: {
      getDataset: () => ({
        id: 'cells',
        revision: 1,
        rowCount: 3,
        value: {columns: {}, analytics: {dictionaries: {}}, prepared: {dictionaries: {}}}
      })
    }
  } as unknown as KeplerComputeSession;
  const requestUpdate = vi.fn();
  const onUnsupported = vi.fn();
  const runtime = new NativeStyleRuntime({session, requestUpdate, onUnsupported});
  const layer: NativeStyleLayer = {
    id: 'h3-layer',
    type: 'hexagonId',
    config: {
      dataId: 'cells',
      color: [20, 30, 40],
      columns: {hex_id: {value: 'cell'}},
      visConfig: {filled: true, outline: true, coverage: 0.8, enable3d: false}
    }
  };
  const options = {
    h3,
    h3Column: 'cell',
    h3MetadataMask: topology,
    h3MetadataRevision: 1,
    h3Viewport: {
      latitude: 37.75,
      longitude: -122.4,
      projectFlat: (position: number[]) => position,
      distanceScales: {unitsPerMeter: [1, 1]}
    }
  };
  const binding = {selectionMask: display};
  return {
    runtime,
    layer,
    options,
    binding,
    display,
    topology,
    requestUpdate,
    onUnsupported,
    setFence: (next: Promise<void>) => {
      fence = next;
    }
  };
}

test('H3 host waits for current topology metadata, independently of display scope, and reuses viewport geometry', async () => {
  const context = setup();
  const pending = context.runtime.resolve(context.layer, context.binding, context.options);
  expect(pending.enabled).toBe(false);
  expect(pending.unsupportedReason).toBeUndefined();
  expect(H3Runtime).not.toHaveBeenCalled();
  expect(vi.mocked(H3MetadataRuntime).mock.calls[0][0].mask).toBe(context.topology);
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['metadata']);
  await context.runtime.readLegends();
  const result = context.runtime.resolve(context.layer, context.binding, context.options);
  expect(result.unsupportedReason).toBeUndefined();
  expect(result.h3?.branch).toBe('columns');
  expect(vi.mocked(H3Runtime).mock.calls[0][0].selectionMask).toBe(context.display);
  const runtime = vi.mocked(H3Runtime).mock.results[0].value;
  const mesh = vi.mocked(H3Runtime).mock.calls[0][0].viewportMesh;
  context.runtime.resolve(context.layer, context.binding, {...context.options, zoom: 10});
  expect(H3Runtime).toHaveBeenCalledOnce();
  expect(runtime.setViewportMesh).toHaveBeenLastCalledWith(mesh);
  state.order = [];
  context.runtime.encode({} as CommandEncoder);
  expect(state.order).toEqual(['metadata', 'h3']);
  await context.runtime.destroy(Promise.resolve());
});

test('H3 branch follows selected resolution/pentagon metadata and high precision ignores unsupported coverage fields', async () => {
  const context = setup();
  context.layer.config.coverageField = {name: 'not-uploaded'};
  context.layer.config.coverageScale = 'unsupported-coverage-scale';
  state.summary = {count: 1, minimumResolution: 7, maximumResolution: 7, hasPentagon: true};
  context.runtime.resolve(context.layer, context.binding, context.options);
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  const polygon = context.runtime.resolve(context.layer, context.binding, context.options);
  expect(polygon.unsupportedReason).toBeUndefined();
  expect(polygon.h3?.branch).toBe('polygons');
  expect(vi.mocked(H3Runtime).mock.calls[0][0].coverage).toBe(1);
  expect(vi.mocked(H3Runtime).mock.calls[0][0].coverages).toBeUndefined();
  context.layer.config.coverageField = undefined;
  const summaries = [
    {count: 2, minimumResolution: 6, maximumResolution: 8, hasPentagon: false},
    {count: 1, minimumResolution: 5, maximumResolution: 5, hasPentagon: false},
    {count: 1, minimumResolution: 8, maximumResolution: 8, hasPentagon: false}
  ];
  for (const [index, summary] of summaries.entries()) {
    state.summary = summary;
    const options = {...context.options, h3MetadataRevision: index + 2};
    expect(context.runtime.resolve(context.layer, context.binding, options).enabled).toBe(false);
    context.runtime.encode({} as CommandEncoder);
    await context.runtime.readLegends();
    expect(context.runtime.resolve(context.layer, context.binding, options).h3?.branch).toBe(
      index < 2 ? 'polygons' : 'columns'
    );
  }
  await context.runtime.destroy(Promise.resolve());
});

test('empty H3 topology clears geometry and deleting a publication waits for older branch resources', async () => {
  const context = setup();
  context.runtime.resolve(context.layer, context.binding, context.options);
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  context.runtime.resolve(context.layer, context.binding, context.options);
  let release!: () => void;
  context.setFence(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  state.summary = {
    count: 0,
    minimumResolution: 0xffffffff,
    maximumResolution: 0,
    hasPentagon: false
  };
  const options = {...context.options, h3MetadataRevision: 2};
  context.runtime.resolve(context.layer, context.binding, options);
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  const empty = context.runtime.resolve(context.layer, context.binding, options);
  expect(empty.unsupportedReason).toBeUndefined();
  expect(empty.h3?.branch).toBe('empty');
  expect(H3Runtime).toHaveBeenCalledOnce();
  let completed = false;
  const retirement = context.runtime.retireDataset('cells', Promise.resolve()).then(() => {
    completed = true;
  });
  await new Promise<void>(resolve => {
    setTimeout(resolve, 0);
  });
  expect(completed).toBe(false);
  expect(state.released).toBe(0);
  release();
  await retirement;
  expect(state.released).toBe(1);
  await context.runtime.destroy(Promise.resolve());
});

test('a superseded H3 metadata read schedules a frame to retry the current summary', async () => {
  const context = setup();
  context.runtime.resolve(context.layer, context.binding, context.options);
  context.runtime.encode({} as CommandEncoder);
  const metadata = vi.mocked(H3MetadataRuntime).mock.results[0].value;
  metadata.readMetadata.mockResolvedValueOnce(null);
  await Promise.resolve();
  context.requestUpdate.mockClear();
  await context.runtime.readLegends();
  expect(context.requestUpdate).toHaveBeenCalledOnce();
  expect(context.runtime.resolve(context.layer, context.binding, context.options).enabled).toBe(
    false
  );
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  expect(context.runtime.resolve(context.layer, context.binding, context.options).h3?.branch).toBe(
    'columns'
  );
  await context.runtime.destroy(Promise.resolve());
});

test('pending topology clears prior diagnostics even when a hidden layer is not resolved again after replay', async () => {
  const context = setup();
  context.runtime.resolve(context.layer, context.binding, context.options);
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  context.layer.config.coverageField = {name: 'missing'};
  expect(
    context.runtime.resolve(context.layer, context.binding, context.options).unsupportedReason
  ).toBeTruthy();
  await Promise.resolve();
  expect(context.onUnsupported.mock.calls.at(-1)?.[1]).toBeTruthy();
  const pending = context.runtime.resolve(context.layer, context.binding, {
    ...context.options,
    h3MetadataRevision: 2
  });
  expect(pending.enabled).toBe(false);
  expect(pending.unsupportedReason).toBeUndefined();
  await Promise.resolve();
  expect(context.onUnsupported).toHaveBeenLastCalledWith(context.layer.id, null);
  // No subsequent resolve is required to clear the global diagnostic for a hidden native layer.
  context.runtime.encode({} as CommandEncoder);
  await context.runtime.readLegends();
  expect(context.onUnsupported).toHaveBeenLastCalledWith(context.layer.id, null);
  await context.runtime.destroy(Promise.resolve());
});
