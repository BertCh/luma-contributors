// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {NullDevice} from '@luma.gl/core/test-utils';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {KeplerComputeSession, type KeplerSessionExplorer} from '../src/session/compute-session';
import {GPUKeplerTableInspector} from '../src/exploration/gpu-table-inspector';
import {ResidentScopeBinding} from '../../../examples/deck/kepler-compute/src/resident-scope-binding';
import {
  installKeplerComputeTableBridge,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

test('source-only restore ignores foreign polygons without creating an empty scalar owner', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const source = new arrow.Table([
    ...new arrow.Table({
      h3: arrow.vectorFromArray(['8928308280fffff', '8928308280bffff'], new arrow.Utf8())
    }).batches,
    ...new arrow.Table({h3: arrow.vectorFromArray(['89283082807ffff'], new arrow.Utf8())}).batches
  ]);
  const dataset = session.importArrowDataset({
    id: 'h3-only',
    table: source,
    normalization: {columns: []}
  });
  const explorer = {
    dataset,
    setExternalActive: vi.fn(),
    invalidate: vi.fn(),
    encode: vi.fn(),
    close: vi.fn(async () => {})
  } as unknown as KeplerSessionExplorer;
  vi.spyOn(session, 'openExplorer').mockReturnValue(explorer);
  const positions = new GPUVector({
    name: 'h3-centers',
    type: 'data',
    format: 'float32x2',
    ownsData: true,
    data: [new Float32Array([0, 0, 1, 1]), new Float32Array([2, 2])].map(
      data =>
        new GPUData({
          format: 'float32x2',
          length: data.length / 2,
          buffer: device.createBuffer({data, usage: Buffer.STORAGE}),
          ownsBuffer: true
        })
    )
  });
  const failure = vi.fn();
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {},
    layers: [{id: 'h3-layer', geometryRevision: 1, geometry: {kind: 'point', positions}}],
    onUpdate() {},
    onUnsupported: failure
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'h3'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers?: readonly unknown[]): void {}
    },
    binding.bind(explorer)
  );
  const nativeLayers = [
    {id: 'geojson-layer', type: 'geojson', config: {dataId: 'first-source'}},
    {id: 'h3-layer', type: 'hexagonId', config: {dataId: dataset.id}}
  ];
  const foreign: KeplerComputeFilter = {
    id: 'selected-geometry',
    dataId: ['first-source'],
    type: 'polygon',
    layerId: ['geojson-layer'],
    value: {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-1, -1],
            [3, -1],
            [3, 3],
            [-1, 3],
            [-1, -1]
          ]
        ]
      }
    }
  };
  try {
    expect(Object.keys(dataset.value.columns)).toEqual([]);
    expect(binding.nativeFilteredMask.data.map(chunk => chunk.length)).toEqual([2, 1]);
    binding.beginLayerRegistration();
    table.filterTable([foreign], nativeLayers);
    expect(() => binding.endLayerRegistration()).not.toThrow();
    expect(failure).not.toHaveBeenCalled();
    expect(binding.scopes.currentPlan.scalarFilterIds).toEqual([]);
    expect(binding.scopes.currentPlan.targetedLayerIds).toEqual([]);
    table.filterTable(
      [{...foreign, id: 'own-polygon', dataId: [dataset.id], layerId: ['h3-layer']}],
      nativeLayers
    );
    expect(failure).not.toHaveBeenCalled();
    expect(binding.scopes.currentPlan.layerFilterIds['h3-layer']).toEqual(['own-polygon']);
    table.filterTable([], nativeLayers);
    expect(failure).not.toHaveBeenCalled();
    expect(binding.scopes.currentPlan.targetedLayerIds).toEqual([]);
    expect(binding.nativeFilteredMask.data.map(chunk => chunk.length)).toEqual([2, 1]);
  } finally {
    await binding.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    positions.destroy();
    device.destroy();
  }
});

test('consumer eligibility rejects unencoded, failed registration, invalidated and retired dataset masks', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const table = new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))});
  const dataset = session.importArrowDataset({id: 'consumer-source', table});
  // This Node case exercises the real provider lifecycle and owned NullDevice buffers. GPU graph
  // execution itself is covered by native-scope-binding.spec; no kernel is emulated here.
  const explorer = {
    dataset,
    setExternalActive: vi.fn(),
    invalidate: vi.fn(),
    encode: vi.fn(),
    close: vi.fn(async () => {})
  } as unknown as KeplerSessionExplorer;
  vi.spyOn(session, 'openExplorer').mockReturnValue(explorer);
  const failure = vi.fn();
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {},
    onUpdate() {},
    onUnsupported: failure
  });
  const encodeScopes = vi.spyOn(binding.scopes, 'encode').mockReturnValue(1);
  const encoder = device.createCommandEncoder();
  try {
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    expect(binding.canConsume).toBe(true);
    const previousRevision = binding.currentRevision;
    binding.beginLayerRegistration();
    expect(binding.currentRevision).toBeGreaterThan(previousRevision);
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    expect(encodeScopes).toHaveBeenCalledTimes(1);
    vi.spyOn(binding.scopes, 'setLayers').mockImplementationOnce(() => {
      throw new Error('Missing resident selector');
    });
    expect(() => binding.endLayerRegistration()).toThrow('Missing resident selector');
    expect(binding.isLayerRegistrationPending).toBe(false);
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    expect(encodeScopes).toHaveBeenCalledTimes(1);
    expect(failure).toHaveBeenCalledWith('Missing resident selector');
    binding.beginLayerRegistration();
    binding.endLayerRegistration();
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    expect(binding.canConsume).toBe(true);
    const encodedRevision = binding.currentRevision;
    binding.getConsumerMask({id: 'chart'});
    expect(binding.currentRevision).toBeGreaterThan(encodedRevision);
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    expect(binding.canConsume).toBe(true);
    session.importArrowDataset({id: dataset.id, table});
    expect(binding.canConsume).toBe(false);
    expect(() => binding.encode(encoder)).toThrow('stale');
  } finally {
    await binding.destroy(Promise.resolve());
    expect(binding.canConsume).toBe(false);
    await session.destroy(Promise.resolve());
    encoder.destroy();
    device.destroy();
  }
});

test('explicit prefix preview permission reaches native domain, plot and table-inspector leases', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const finite = session.importArrowDataset({
    id: 'prefix',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([1, 2]))})
  });
  const dataset = session.registry.registerDataset({
    ...finite,
    population: {kind: 'stream', streamId: 'stream', state: 'open'}
  });
  const openExplorer = vi.spyOn(session, 'openExplorer').mockImplementation((id, options) => {
    const lease = session.registry.acquireDataset(id, {
      allowIncomplete: options.allowIncompletePopulation
    });
    return {
      dataset: lease.dataset,
      allowIncompletePopulation: Boolean(options.allowIncompletePopulation),
      maximumInspectionPageSize: 1,
      setExternalActive: vi.fn(),
      invalidate: vi.fn(),
      encode: vi.fn(),
      close: (completion: PromiseLike<unknown>) => lease.release(completion)
    } as unknown as KeplerSessionExplorer;
  });
  const properties = {
    session,
    datasetId: dataset.id,
    fields: {value: {kind: 'range' as const, dimension: 'value', domain: [1, 2]}},
    onUpdate() {},
    onUnsupported: vi.fn()
  };
  expect(() => new ResidentScopeBinding(properties)).toThrow('incomplete');
  const binding = new ResidentScopeBinding({...properties, allowIncompletePopulation: true});
  const primary = session.openExplorer(dataset.id, {
    allowIncompletePopulation: true,
    workingMemoryBudgetBytes: 1024
  });
  const inspector = new GPUKeplerTableInspector({
    session,
    explorer: primary,
    sortMemoryBudgetBytes: 0
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'value'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[]): void {}
    },
    binding.bind(primary)
  );
  const acquireDataset = vi.spyOn(session.registry, 'acquireDataset');
  try {
    // A disabled filter keeps its bounded histogram population but needs no scalar kernel.
    table.filterTable([
      {
        id: 'plot',
        dataId: [dataset.id],
        name: ['value'],
        type: 'range',
        value: [1, 2],
        enabled: false
      }
    ]);
    expect(properties.onUnsupported).not.toHaveBeenCalled();
    // The primary and domain explorers are session explorers; the native-edge plot acquires its own
    // dataset lease, which must carry the same explicit prefix permission.
    expect(openExplorer).toHaveBeenCalledTimes(2);
    expect(
      openExplorer.mock.calls.every(([, options]) => options.allowIncompletePopulation === true)
    ).toBe(true);
    expect(acquireDataset).toHaveBeenCalledTimes(1);
    expect(acquireDataset).toHaveBeenCalledWith(dataset.id, {allowIncomplete: true});
    expect(inspector.columns).toEqual(['value']);
    expect(() => session.registry.beginRequest('analysis', [dataset.id])).toThrow('incomplete');
  } finally {
    await inspector.destroy();
    await binding.destroy(Promise.resolve());
    await primary.close(Promise.resolve());
    await session.destroy(Promise.resolve());
    expect(session.registry.memoryUsage.totalBytes).toBe(0);
    device.destroy();
  }
});

test('trusted chart time ownership changes invalidate current masks and reject stale or non-time registrations', async () => {
  const device = new NullDevice({});
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const source = new arrow.Table({
    timestamp: arrow.makeVector(
      arrow.makeData({
        type: new arrow.TimestampMillisecond(),
        data: new BigInt64Array([0n, 1n]),
        length: 2
      })
    )
  });
  const dataset = session.importArrowDataset({id: 'chart-time', table: source});
  const explorer = {
    dataset,
    setExternalActive: vi.fn(),
    invalidate: vi.fn(),
    encode: vi.fn(),
    close: vi.fn(async () => {})
  } as unknown as KeplerSessionExplorer;
  vi.spyOn(session, 'openExplorer').mockReturnValue(explorer);
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {timestamp: {kind: 'time', dimension: 'timestamp', domain: [0, 1]}},
    onUpdate() {},
    onUnsupported: vi.fn()
  });
  vi.spyOn(binding.scopes, 'encode').mockReturnValue(1);
  const encoder = device.createCommandEncoder();
  try {
    binding.encode(encoder);
    expect(binding.canConsume).toBe(true);
    const previous = binding.currentRevision;
    binding.setChartTimeNullEpochFilters([
      {filterId: 'arbitrary-persisted-id', field: 'timestamp'}
    ]);
    expect(binding.currentRevision).toBeGreaterThan(previous);
    expect(binding.canConsume).toBe(false);
    binding.encode(encoder);
    const current = binding.currentRevision;
    binding.setChartTimeNullEpochFilters([
      {filterId: 'arbitrary-persisted-id', field: 'timestamp'}
    ]);
    expect(binding.currentRevision).toBe(current);
    expect(binding.canConsume).toBe(true);
    expect(() =>
      binding.setChartTimeNullEpochFilters([{filterId: 'other', field: 'missing'}])
    ).toThrow('exact time field');
    expect(binding.currentRevision).toBe(current);
    binding.setChartTimeNullEpochFilters([]);
    expect(binding.canConsume).toBe(false);
    session.importArrowDataset({id: dataset.id, table: source});
    expect(() =>
      binding.setChartTimeNullEpochFilters([
        {filterId: 'arbitrary-persisted-id', field: 'timestamp'}
      ])
    ).toThrow('stale');
  } finally {
    await binding.destroy(Promise.resolve());
    await session.destroy(Promise.resolve());
    encoder.destroy();
    device.destroy();
  }
});
