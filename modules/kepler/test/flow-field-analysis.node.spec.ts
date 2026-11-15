// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';

// The native GeoJSON parser is unused for these numeric-column sources.
vi.mock('@kepler.gl/layers', () => ({parseGeoJsonRawFeature: () => null}));
// Only the particle renderer is stubbed: binding, selection-cache readback, and grid builder are real.
vi.mock('../../../examples/deck/kepler-compute/src/flow-field-runtime', () => ({
  getFlowFieldSeed: () => 1,
  FlowFieldRuntime: class {
    static isSupported() {
      return true;
    }
    constructor(
      readonly device: unknown,
      readonly id: string,
      readonly grid: unknown,
      readonly plan: unknown
    ) {}
    onViewportAvailabilityChange: (() => void) | null = null;
    readonly stats = {viewportFailure: undefined};
    destroy() {}
  }
}));

import {
  FlowFieldBindings,
  type FlowFieldLayerConfig
} from '../../../examples/deck/kepler-compute/src/flow-field-bindings';
import type {FlowFieldSelectionContext} from '../../../examples/deck/kepler-compute/src/flow-field-filter';
import {prepareFlowFieldLagrangian} from '../../../examples/deck/kepler-compute/src/flow-field-lagrangian';
import {createRasterEmbeddingsHost} from '../../../examples/deck/kepler-compute/src/raster-embeddings-host';

function makeSource() {
  const longitude = Float32Array.from({length: 16}, (_, index) => index % 4);
  return arrow.tableFromArrays({
    longitude,
    latitude: Float32Array.from({length: 16}, (_, index) => Math.floor(index / 4)),
    east: Float32Array.from(longitude, value => (value < 2 ? 1 : 20)),
    north: new Float32Array(16),
    elevation: Float32Array.from(longitude, value => value * 111320)
  });
}
const config: FlowFieldLayerConfig = {
  columnMode: 'UV',
  columns: {
    lng: {value: 'longitude'},
    lat: {value: 'latitude'},
    u: {value: 'east'},
    v: {value: 'north'},
    altitude: {value: 'elevation'}
  },
  visConfig: {gridResolution: 4, smoothing: 0},
  color: [255, 255, 255]
};

function makeSelection() {
  let revision = 1;
  let read: () => Promise<Uint8Array> = async () =>
    new Uint8Array(new Uint32Array(16).fill(1).buffer);
  const context: FlowFieldSelectionContext = {
    mask: {
      length: 16,
      data: [{length: 16, byteOffset: 0, buffer: {readAsync: () => read()}}]
    } as never,
    getRevision: () => revision,
    requestUpdate: vi.fn()
  };
  return {
    context,
    advance: (nextRead: () => Promise<Uint8Array>) => {
      revision++;
      read = nextRead;
    }
  };
}

async function createFilteredBinding() {
  const bindings = new FlowFieldBindings();
  bindings.gridMode = 'cpu';
  const selection = makeSelection();
  const context = {
    device: {type: 'webgpu'} as never,
    table: makeSource(),
    selection: selection.context
  };
  bindings.resolve('flow', config, context);
  await vi.waitFor(() => expect(selection.context.requestUpdate).toHaveBeenCalled());
  bindings.resolve('flow', config, context);
  return {bindings, selection, context};
}

test('CPU analysis waits for the current mask and rebuilds without needing another render', async () => {
  const {bindings, selection, context} = await createFilteredBinding();
  let release!: (bytes: Uint8Array) => void;
  const pendingMask = new Promise<Uint8Array>(resolve => {
    release = resolve;
  });
  selection.advance(() => pendingMask);
  // Rendering intentionally retains revision 1 while the new selection readback is pending.
  expect(bindings.resolve('flow', config, context).bounds).toEqual([0, 0, 3, 3]);
  let settled = false;
  const snapshot = bindings.readGridForAnalysis('flow').then(grid => {
    settled = true;
    return grid;
  });
  await Promise.resolve();
  await Promise.resolve();
  const publishedBeforeMaskSettled = settled;
  release(
    new Uint8Array(Uint32Array.from({length: 16}, (_, index) => (index % 4 < 2 ? 1 : 0)).buffer)
  );
  const grid = await snapshot;
  expect(publishedBeforeMaskSettled).toBe(false);
  expect(grid.bounds).toEqual([0, 0, 1, 3]);
  expect(Math.max(...Array.from(grid.texels).filter((_, index) => index % 4 === 0))).toBe(1);
  bindings.destroy();
});

test('CPU analysis applies a replacement mask even when its revision number is unchanged', async () => {
  const {bindings, selection, context} = await createFilteredBinding();
  const words = Uint32Array.from({length: 16}, (_, index) => (index % 4 < 2 ? 1 : 0));
  selection.context.mask = {
    length: 16,
    data: [
      {length: 16, byteOffset: 0, buffer: {readAsync: async () => new Uint8Array(words.buffer)}}
    ]
  } as never;
  bindings.resolve('flow', config, context);
  const grid = await bindings.readGridForAnalysis('flow');
  expect(grid.bounds).toEqual([0, 0, 1, 3]);
  expect(grid.maximumSpeed).toBe(1);
  bindings.destroy();
});

test('CPU analysis follows a newer mask load that supersedes its pending revision', async () => {
  const {bindings, selection, context} = await createFilteredBinding();
  let release!: (bytes: Uint8Array) => void;
  const pendingMask = new Promise<Uint8Array>(resolve => {
    release = resolve;
  });
  selection.advance(() => pendingMask);
  bindings.resolve('flow', config, context);
  const snapshot = bindings.readGridForAnalysis('flow');
  const words = Uint32Array.from({length: 16}, (_, index) => (index % 4 < 2 ? 1 : 0));
  selection.advance(async () => new Uint8Array(words.buffer));
  bindings.resolve('flow', config, context);
  release(new Uint8Array(new Uint32Array(16).fill(1).buffer));
  const grid = await snapshot;
  expect(grid.bounds).toEqual([0, 0, 1, 3]);
  expect(grid.maximumSpeed).toBe(1);
  bindings.destroy();
});

test('CPU analysis rejects a failed mask load instead of returning the previous grid', async () => {
  const {bindings, selection, context} = await createFilteredBinding();
  selection.advance(async () => {
    throw new Error('mask readback failed');
  });
  bindings.resolve('flow', config, context);
  await expect(bindings.readGridForAnalysis('flow')).rejects.toThrow('mask readback failed');
  await expect(bindings.readGridForAnalysis('flow')).rejects.toThrow('mask readback failed');
  bindings.destroy();
});

test('flow snapshots preserve elevation provenance and physical preparation rejects its slopes', async () => {
  const bindings = new FlowFieldBindings();
  bindings.gridMode = 'cpu';
  bindings.resolve(
    'slope',
    {...config, columnMode: 'ELEVATION'},
    {device: {type: 'webgpu'} as never, table: makeSource()}
  );
  const snapshot = await bindings.readGridForAnalysis('slope');
  expect(Math.abs(snapshot.texels[0])).toBeCloseTo(1, 2);
  expect(snapshot).toHaveProperty('mode', 'ELEVATION');
  expect(() => prepareFlowFieldLagrangian(snapshot)).toThrow(/elevation.*slopes/i);
  bindings.destroy();
});

test.each([
  'start',
  'end'
])('physical diagnostics reject an elevation %s slice before GPU allocation', async which => {
  const bindings = new FlowFieldBindings();
  bindings.gridMode = 'cpu';
  for (const [id, mode] of [
    ['velocity', 'UV'],
    ['slope', 'ELEVATION']
  ] as const)
    bindings.resolve(
      id,
      {...config, columnMode: mode},
      {device: {type: 'webgpu'} as never, table: makeSource()}
    );
  const publishTable = vi.fn();
  const createBuffer = vi.fn(() => {
    throw new Error('GPU allocation reached');
  });
  const device = {type: 'webgpu', createBuffer} as never;
  const host = createRasterEmbeddingsHost({
    getDevice: () => device,
    getSource: () => undefined,
    readFlowGrid: id => bindings.readGridForAnalysis(id),
    publishTable
  });
  await expect(
    host.runFlow({
      layerId: which === 'start' ? 'slope' : 'velocity',
      endLayerId: which === 'end' ? 'slope' : undefined,
      timeStep: 1,
      stepCount: 4
    })
  ).rejects.toThrow(/elevation.*slopes/i);
  expect(createBuffer).not.toHaveBeenCalled();
  expect(publishTable).not.toHaveBeenCalled();
  host.destroy();
  bindings.destroy();
});
