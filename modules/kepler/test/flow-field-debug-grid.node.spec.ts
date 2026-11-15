// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterAll, describe, expect, test, vi} from 'vitest';

// Node uses isolate:false: discard previously evaluated consumers before applying this file's mocks.
vi.hoisted(() => vi.resetModules());

vi.mock('@deck.gl/layers', () => ({
  PathLayer: class {
    constructor(readonly props: Record<string, unknown>) {}
  },
  PolygonLayer: class {
    constructor(readonly props: Record<string, unknown>) {}
  }
}));
vi.mock('@luma.gl/graph/experimental/simulation', () => ({}));

const {
  FlowFieldDebugGridCache,
  buildFlowFieldDebugGridGeometry,
  buildFlowFieldDebugGridLayers,
  readFlowFieldDebugGrid
} = await import('../../../examples/deck/kepler-compute/src/flow-field-debug-grid');

const snapshot = {
  columns: 3,
  rows: 2,
  bounds: [0, 10, 4, 12] as [number, number, number, number],
  filled: Uint8Array.from([1, 0, 0, 0, 0, 1]),
  altitude: Float32Array.from([100, NaN, NaN, NaN, NaN, 50])
};

function makeRuntime(options: {revision?: number | null; texels?: Float32Array} = {}) {
  const readTexelsForDebug = vi.fn(async () => options.texels ?? new Float32Array(3 * 2 * 4));
  return {
    grid: {columns: 3, rows: 2, bounds: [0, 10, 4, 12] as const},
    stats: {selectionRevision: options.revision ?? null, gridReadbackCount: 0},
    gpu: {grid: {readTexelsForDebug}},
    readTexelsForDebug
  };
}

describe('flow field debug grid', () => {
  test('builds native lines and filled cells', () => {
    const {gridLines, filledCells} = buildFlowFieldDebugGridGeometry(snapshot, 2);
    expect(gridLines).toHaveLength(3 + 2);
    expect(gridLines[0].path).toEqual([
      [0, 10, 200],
      [0, 12, 0]
    ]);
    expect(gridLines[3].path[2]).toEqual([4, 10, 0]);
    expect(filledCells).toHaveLength(2);
    expect(filledCells[0].polygon[0]).toEqual([0, 10, 200]);
    expect(filledCells[0].polygon[2]).toEqual([1, 11, 200]);
    expect(filledCells[1].polygon[2]).toEqual([4, 12, 100]);
  });

  test('creates a path layer and a polygon layer', () => {
    const layers = buildFlowFieldDebugGridLayers(snapshot, {layerId: 'ff', elevationMultiplier: 1});
    const props = layers.map(layer => (layer as unknown as {props: Record<string, unknown>}).props);
    expect(props.map(entry => entry['id'])).toEqual(['ff-debug-grid', 'ff-debug-filled']);
    expect(props[0]['parameters']).toEqual({depthTest: false});
  });

  test('reads a GPU grid with validity as filled', async () => {
    const texels = new Float32Array(3 * 2 * 4);
    texels[2] = 1;
    texels[5 * 4 + 2] = 1;
    const runtime = makeRuntime({texels});
    const result = await readFlowFieldDebugGrid(runtime as never);
    expect(Array.from(result?.filled ?? [])).toEqual([1, 0, 0, 0, 0, 1]);
    expect(result?.altitude).toBeUndefined();
  });

  test('cache refreshes only on runtime, revision or bounds changes and never counts as a grid readback', async () => {
    const cache = new FlowFieldDebugGridCache();
    const requestUpdate = vi.fn();
    const options = {requestUpdate, elevationMultiplier: 1};
    const runtime = makeRuntime();
    expect(cache.getLayers('ff', runtime as never, options)).toEqual([]);
    await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(1));
    const first = cache.getLayers('ff', runtime as never, options);
    expect(first).toHaveLength(2);
    expect(cache.getLayers('ff', runtime as never, options)).toBe(first);
    expect(cache.debugReadCount).toBe(1);
    runtime.stats.selectionRevision = 4;
    cache.getLayers('ff', runtime as never, options);
    await vi.waitFor(() => expect(requestUpdate).toHaveBeenCalledTimes(2));
    expect(cache.debugReadCount).toBe(2);
    (runtime.grid as {bounds: readonly number[]}).bounds = [0, 10, 5, 12];
    cache.getLayers('ff', runtime as never, options);
    await vi.waitFor(() => expect(cache.debugReadCount).toBe(3));
    expect(runtime.stats.gridReadbackCount).toBe(0);
  });
});

// Remove mocked dependencies and their consumers before the next shared-worker spec.
afterAll(() => {
  vi.doUnmock('@deck.gl/layers');
  vi.doUnmock('@luma.gl/graph/experimental/simulation');
  vi.resetModules();
});
