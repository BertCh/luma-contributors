// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {ResidentScopeBinding} from '../../../examples/deck/kepler-compute/src/resident-scope-binding';
import {
  installKeplerComputeTableBridge,
  type KeplerComputeFilter
} from '../../../examples/deck/kepler-compute/src/kepler-table-bridge';

it('native scope binding publishes distinct same-field plots, dynamic domains and hidden-target layer unions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'scoped-source',
    table: new arrow.Table({
      value: arrow.makeVector(new Float32Array([0, 1, 2, 3, 4, 5])),
      category: arrow.makeVector(new Uint32Array([0, 1, 0, 1, 0, 1])),
      timestamp: arrow.makeVector(
        arrow.makeData({
          type: new arrow.TimestampMillisecond(),
          data: new BigInt64Array([0n, 1000n, 2000n, 3000n, 4000n, 5000n]),
          length: 6
        })
      )
    })
  });
  const positions = makePositions(device, [0, 1, 2, 3, 4, 5]);
  const reversed = makePositions(device, [5, 4, 3, 2, 1, 0]);
  const failures: string[] = [];
  const plotNotices = new Map<string, string>();
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {
      value: {kind: 'range', dimension: 'value', domain: [0, 5]},
      category: {kind: 'category', dimension: 'category', domain: [0, 1], categories: [0, 1]},
      timestamp: {kind: 'time', dimension: 'timestamp', domain: [0, 5000]}
    },
    plots: {value: {column: 'value', domain: [0, 6], binCount: 6}},
    layers: [
      {id: 'first', geometryRevision: 1, geometry: {kind: 'point', positions}},
      {
        id: 'hidden',
        geometryRevision: 1,
        geometry: {kind: 'point', positions: reversed},
        visible: false
      },
      {id: 'untargeted', geometryRevision: 1, geometry: {kind: 'point', positions}}
    ],
    onUpdate() {},
    onUnsupported: reason => {
      failures.push(reason);
    },
    onPlotUnsupported: (filterId, reason) => {
      if (reason) plotNotices.set(filterId, reason);
      else plotNotices.delete(filterId);
    }
  });
  const primary = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 1024 * 1024,
    maximumConcurrentReads: 1,
    maximumReadbackBytes: 4096,
    dimensions: [binding.externalDimension]
  });
  const bridgeBinding = binding.bind(primary);
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'value'}, {name: 'category'}, {name: 'timestamp'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers?: readonly unknown[]): void {}
    },
    bridgeBinding
  );
  const nativeLayers = ['first', 'hidden', 'untargeted'].map(id => ({
    id,
    type: 'point',
    config: {dataId: dataset.id}
  }));
  const lower: KeplerComputeFilter = {
    id: 'lower',
    dataId: dataset.id,
    name: ['value'],
    type: 'range',
    value: [1.1, 5],
    fixedDomain: true
  };
  const upper: KeplerComputeFilter = {
    id: 'upper',
    dataId: dataset.id,
    name: ['value'],
    type: 'range',
    value: [0, 4],
    fixedDomain: false
  };
  const category: KeplerComputeFilter = {
    id: 'category',
    dataId: dataset.id,
    name: ['category'],
    type: 'multiSelect',
    value: [1]
  };
  async function update(filters: readonly KeplerComputeFilter[]): Promise<void> {
    table.filterTable(filters, nativeLayers);
    expect(failures).toEqual([]);
    const encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    binding.publishHistograms(table.lumaCompute);
  }
  try {
    await update([lower, upper, category]);
    expect((await primary.readPage(0, 6, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([3])
    );
    expect(binding.getFilterHistogram('lower')?.map(bin => bin.count)).toEqual([0, 1, 0, 1, 0, 0]);
    expect(binding.getFilterHistogram('upper')?.map(bin => bin.count)).toEqual([0, 0, 0, 1, 0, 1]);
    expect(table.lumaCompute.updateFilterPlot(lower).bins).toEqual({
      [dataset.id]: binding.getFilterHistogram('lower')
    });
    expect(table.lumaCompute.updateFilterPlot(upper).bins).toEqual({
      [dataset.id]: binding.getFilterHistogram('upper')
    });
    expect(
      bridgeBinding.getDynamicLayerDomain?.('value', 'linear', table.lumaCompute.revision)
    ).toEqual([1, 3]);
    await update([{...lower, value: [4.1, 5]}, upper]);
    expect((await primary.readSelectedCount())?.selectedCount).toBe(0);
    expect(binding.getFilterHistogram('lower')?.map(bin => bin.count)).toEqual([1, 1, 1, 1, 1, 0]);
    expect(binding.getFilterHistogram('upper')?.map(bin => bin.count)).toEqual([0, 0, 0, 0, 0, 1]);
    await update([lower, {...upper, enabled: false}]);
    expect((await primary.readSelectedCount())?.selectedCount).toBe(4);
    expect(binding.getFilterHistogram('lower')?.map(bin => bin.count)).toEqual([1, 1, 1, 1, 1, 1]);
    expect(binding.getFilterHistogram('upper')?.map(bin => bin.count)).toEqual([0, 0, 1, 1, 1, 1]);
    await update([
      lower,
      upper,
      polygon(dataset.id, 'first-filter', 'first'),
      polygon(dataset.id, 'hidden-filter', 'hidden')
    ]);
    expect(await readMask(binding.getLayerMask('first'))).toEqual([0, 0, 0, 1, 0, 0]);
    expect(await readMask(binding.getLayerMask('hidden'))).toEqual([0, 0, 1, 0, 0, 0]);
    expect(await readMask(binding.getLayerMask('untargeted'))).toEqual([0, 0, 1, 1, 1, 0]);
    expect((await primary.readPage(0, 6, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([2, 3])
    );
    expect(
      bridgeBinding.getDynamicLayerDomain?.('value', 'linear', table.lumaCompute.revision)
    ).toEqual([0, 4]);
    binding.setLayers(
      [
        {
          id: 'hidden',
          geometryRevision: 1,
          geometry: {kind: 'point', positions: reversed},
          visible: false
        },
        {id: 'untargeted', geometryRevision: 1, geometry: {kind: 'point', positions}}
      ],
      nativeLayers.filter(layer => layer.id !== 'first')
    );
    expect(table.lumaCompute.status).toBe('pending');
    expect(table.lumaCompute.updateFilterPlot(lower).bins).toEqual({[dataset.id]: undefined});
    let encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    expect((await primary.readPage(0, 6, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([2])
    );
    binding.setLayers(
      [
        {id: 'first', geometryRevision: 1, geometry: {kind: 'point', positions}},
        {
          id: 'hidden',
          geometryRevision: 1,
          geometry: {kind: 'point', positions: reversed},
          visible: false
        },
        {id: 'untargeted', geometryRevision: 1, geometry: {kind: 'point', positions}}
      ],
      nativeLayers
    );
    encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    expect((await primary.readPage(0, 6, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([2, 3])
    );
    await update([lower]);
    expect(binding.getFilterHistogram('upper')).toBeUndefined();
    expect((await primary.readSelectedCount())?.selectedCount).toBe(4);
    const time: KeplerComputeFilter = {
      id: 'time',
      dataId: dataset.id,
      type: 'timeRange',
      name: ['timestamp'],
      value: [0, 5000]
    };
    await update([lower, time]);
    expect(plotNotices.has('time')).toBe(false);
    expect(binding.getFilterHistogram('time')).toEqual(
      [2000, 3000, 4000, 5000].map(x0 => ({x0, x1: x0 + 1000, count: 1, indexes: []}))
    );
    // 20,001 one-millisecond edges exceed the 16,384-bin native histogram bound.
    const excessiveTimeBins = {
      ...time,
      domain: [0, 20000] as [number, number],
      plotType: {type: 'histogram', interval: '1-millisecond'}
    };
    await update([lower, excessiveTimeBins]);
    expect(plotNotices.get('time')).toContain('16384');
    expect((await primary.readSelectedCount())?.selectedCount).toBe(4);
    await update([lower, {...excessiveTimeBins, enabled: false}]);
    expect(plotNotices.has('time')).toBe(false);
    await update([lower, time]);
    expect(plotNotices.has('time')).toBe(false);
    expect(binding.getFilterHistogram('time')?.reduce((sum, bin) => sum + bin.count, 0)).toBe(4);
    await update([lower]);
    expect(plotNotices.has('time')).toBe(false);
    table.filterTable([lower], nativeLayers);
    encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    const staleSummary = binding.readSummary();
    table.filterTable([lower, upper], nativeLayers);
    const retainedBeforeStaleCompletion = session.registry.memoryUsage.reservedBytes.gpu;
    expect(await staleSummary).toBe(false);
    await expect
      .poll(() => session.registry.memoryUsage.reservedBytes.gpu)
      .toBeLessThan(retainedBeforeStaleCompletion);
    table.filterTable([lower], nativeLayers);
    encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    const coalescedSummary = binding.readSummary();
    table.filterTable([lower, upper], nativeLayers);
    expect(binding.hasPlot('lower')).toBe(true);
    expect(binding.isSummaryCurrent).toBe(false);
    expect(binding.getFilterHistogram('lower')).toBeUndefined();
    encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await coalescedSummary).toBe(true);
    expect(binding.isSummaryCurrent).toBe(true);
    expect(binding.getFilterHistogram('lower')?.map(bin => bin.count)).toEqual([1, 1, 1, 1, 1, 0]);
    expect(binding.getFilterHistogram('upper')?.map(bin => bin.count)).toEqual([0, 0, 1, 1, 1, 1]);
    await update([]);
    expect(table.lumaCompute.status).toBe('ready');
    expect(failures).toEqual([]);
  } finally {
    const fence = device.createFence();
    try {
      await Promise.all([binding.destroy(fence.signaled), primary.close(fence.signaled)]);
      expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
      positions.destroy();
      reversed.destroy();
    }
  }
});

it('native registration transactions defer selector readiness and commit replacement layer geometry atomically', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'registration-source',
    table: new arrow.Table({
      value: arrow.makeVector(new Float32Array([0, 1, 2, 3]))
    })
  });
  const positions = makePositions(device, [0, 1, 2, 3]);
  const targets = makePositions(device, [0, 5, 2, 3]);
  const failures: string[] = [];
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {value: {kind: 'range', dimension: 'value', domain: [0, 3]}},
    layers: [{id: 'native-layer', geometryRevision: 1, geometry: {kind: 'point', positions}}],
    onUpdate() {},
    onUnsupported: reason => {
      failures.push(reason);
    }
  });
  const primary = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 1024 * 1024,
    maximumConcurrentReads: 1,
    maximumReadbackBytes: 4096,
    dimensions: [binding.externalDimension]
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'value'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers?: readonly unknown[]): void {}
    },
    binding.bind(primary)
  );
  const nativePoints = [{id: 'native-layer', type: 'point', config: {dataId: dataset.id}}];
  const nativeLines = [{id: 'native-layer', type: 'line', config: {dataId: dataset.id}}];
  const spatial: KeplerComputeFilter = {
    id: 'polygon',
    dataId: dataset.id,
    type: 'polygon',
    layerId: ['native-layer'],
    value: {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [-0.5, -0.5],
            [2.5, -0.5],
            [2.5, 0.5],
            [-0.5, 0.5],
            [-0.5, -0.5]
          ]
        ]
      }
    }
  };
  async function selectedCount(): Promise<number | undefined> {
    const encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    binding.publishHistograms(table.lumaCompute);
    return (await primary.readSelectedCount())?.selectedCount;
  }
  try {
    table.filterTable([spatial], nativePoints);
    expect(await selectedCount()).toBe(3);
    binding.beginLayerRegistration();
    binding.beginLayerRegistration();
    binding.setLayers([], []);
    // Kepler creates the replacement native layer before the subscriber registers its GPU selector.
    table.filterTable([spatial], nativeLines);
    expect(failures).toEqual([]);
    expect(table.lumaCompute.status).toBe('pending');
    expect(binding.isLayerRegistrationPending).toBe(true);
    expect(await binding.readSummary()).toBe(false);
    binding.setLayers(
      [
        {
          id: 'native-layer',
          geometryRevision: 2,
          geometry: {kind: 'endpoint-pair', sourcePositions: positions, targetPositions: targets}
        }
      ],
      nativeLines
    );
    binding.endLayerRegistration();
    expect(binding.isLayerRegistrationPending).toBe(true);
    binding.endLayerRegistration();
    expect(binding.isLayerRegistrationPending).toBe(false);
    expect(failures).toEqual([]);
    expect(await selectedCount()).toBe(2);
    expect((await primary.readPage(0, 4, ['value']))?.sourceRowIndices).toEqual(
      new Uint32Array([0, 2])
    );
    binding.beginLayerRegistration();
    binding.setLayers([], nativeLines);
    table.filterTable([spatial], nativeLines);
    expect(failures).toEqual([]);
    expect(() => binding.endLayerRegistration()).toThrow('registered resident geometry selector');
    expect(binding.isLayerRegistrationPending).toBe(false);
    expect(await binding.readSummary()).toBe(false);
    expect(failures).toHaveLength(1);
    binding.beginLayerRegistration();
    binding.setLayers(
      [{id: 'native-layer', geometryRevision: 3, geometry: {kind: 'point', positions}}],
      nativePoints
    );
    table.filterTable([spatial], nativePoints);
    binding.endLayerRegistration();
    expect(await selectedCount()).toBe(3);
    binding.beginLayerRegistration();
    table.filterTable(
      [{...spatial, value: {type: 'Feature', geometry: {type: 'Point', coordinates: [0, 0]}}}],
      nativePoints
    );
    expect(failures.at(-1)).toContain('Polygon and MultiPolygon');
    expect(() => binding.endLayerRegistration()).toThrow('Polygon and MultiPolygon');
    expect(binding.isLayerRegistrationPending).toBe(false);
    expect(table.lumaCompute.status).toBe('unsupported');
    expect(await binding.readSummary()).toBe(false);
    binding.setLayers(
      [{id: 'native-layer', geometryRevision: 4, geometry: {kind: 'point', positions}}],
      nativePoints
    );
    expect(await binding.readSummary()).toBe(false);
    expect(table.lumaCompute.status).toBe('unsupported');
    table.filterTable([spatial], nativePoints);
    expect(await selectedCount()).toBe(3);
    expect(table.lumaCompute.status).toBe('ready');
  } finally {
    const fence = device.createFence();
    try {
      await Promise.all([binding.destroy(fence.signaled), primary.close(fence.signaled)]);
      expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
      positions.destroy();
      targets.destroy();
    }
  }
});

it('native topology omits ordinary GPU ranges, includes CPU predicates and inverts completed nullable masks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 64 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'topology-source',
    table: new arrow.Table({
      category: arrow.makeVector(
        arrow.makeData({
          type: new arrow.Uint32(),
          data: new Uint32Array([0, 1, 0, 0, 1]),
          nullBitmap: new Uint8Array([0b10111]),
          nullCount: 1
        })
      ),
      value: arrow.makeVector(
        arrow.makeData({
          type: new arrow.Float32(),
          data: new Float32Array([0, 1, 2, 0, 4]),
          nullBitmap: new Uint8Array([0b10111]),
          nullCount: 1
        })
      ),
      timestamp: arrow.makeVector(
        arrow.makeData({
          type: new arrow.TimestampMillisecond(),
          data: new BigInt64Array([0n, 1000n, 2000n, 0n, 4000n]),
          length: 5,
          nullBitmap: new Uint8Array([0b10111]),
          nullCount: 1
        })
      )
    })
  });
  const positions = makePositions(device, [0, 1, 2, 3, 4]);
  const failures: string[] = [];
  const binding = new ResidentScopeBinding({
    session,
    datasetId: dataset.id,
    fields: {
      value: {kind: 'range', dimension: 'value', domain: [0, 4]},
      category: {kind: 'category', dimension: 'category', domain: [0, 1], categories: [0, 1]},
      timestamp: {kind: 'time', dimension: 'timestamp', domain: [0, 4000]}
    },
    layers: [{id: 'layer', geometryRevision: 1, geometry: {kind: 'point', positions}}],
    onUpdate() {},
    onUnsupported: reason => failures.push(reason)
  });
  const primary = session.openExplorer(dataset.id, {
    workingMemoryBudgetBytes: 1024 * 1024,
    dimensions: [binding.externalDimension],
    maximumReadbackBytes: 4096,
    maximumConcurrentReads: 1
  });
  const table = installKeplerComputeTableBridge(
    {
      id: dataset.id,
      fields: [{name: 'value'}, {name: 'timestamp'}, {name: 'category'}],
      allIndexes: [],
      filteredIndex: [],
      filteredIndexForDomain: [],
      filteredIndexByLayer: {},
      filterTable(_filters: readonly KeplerComputeFilter[], _layers?: readonly unknown[]): void {}
    },
    binding.bind(primary)
  );
  const nativeLayers = [{id: 'layer', type: 'point', config: {dataId: dataset.id}}];
  const range: KeplerComputeFilter = {
    id: 'range',
    dataId: [dataset.id],
    name: ['value'],
    type: 'range',
    value: [1, 2],
    gpu: true
  };
  async function update(filters: KeplerComputeFilter[]) {
    table.filterTable(filters, nativeLayers);
    expect(failures).toEqual([]);
    const encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    return {
      display: await readMask(binding.getLayerMask('layer')),
      topology: await readMask(binding.getLayerTopologyMask('layer'))
    };
  }
  try {
    expect(await update([range])).toEqual({display: [0, 1, 1, 0, 0], topology: [1, 1, 1, 1, 1]});
    expect(await update([{...range, gpu: false}])).toEqual({
      display: [0, 1, 1, 0, 0],
      topology: [0, 1, 1, 0, 0]
    });
    expect(await update([{...range, gpu: false, inverted: true}])).toEqual({
      display: [1, 0, 0, 1, 1],
      topology: [1, 0, 0, 1, 1]
    });
    expect(
      await update([
        {
          ...range,
          name: ['timestamp'],
          type: 'timeRange',
          value: [1000, 2000],
          gpu: false,
          inverted: true
        }
      ])
    ).toEqual({display: [1, 0, 0, 1, 1], topology: [1, 0, 0, 1, 1]});
    expect(
      await update([
        {...range, name: ['category'], type: 'multiSelect', value: [1], gpu: false, inverted: true}
      ])
    ).toEqual({display: [1, 0, 1, 1, 0], topology: [1, 0, 1, 1, 0]});
    expect(
      await update([{...range, animationWindow: 'interval', value: [0, 1 - Number.EPSILON / 2]}])
    ).toEqual({display: [1, 0, 0, 0, 0], topology: [1, 0, 0, 0, 0]});
    expect(await update([{...range, animationWindow: 'interval', value: [1, 4]}])).toEqual({
      display: [0, 1, 1, 0, 1],
      topology: [0, 1, 1, 0, 1]
    });
    expect(await update([{...range, inverted: true, enabled: false}])).toEqual({
      display: [1, 1, 1, 1, 1],
      topology: [1, 1, 1, 1, 1]
    });
    const chartTime: KeplerComputeFilter = {
      ...range,
      id: 'arbitrary-persisted-time-filter',
      name: ['timestamp'],
      type: 'timeRange',
      value: [0, 0]
    };
    expect((await update([chartTime])).display).toEqual([1, 0, 0, 0, 0]);
    expect(binding.getFilterHistogram(chartTime.id)?.map(({x0, count}) => ({x0, count}))).toEqual([
      {x0: 0, count: 2},
      {x0: 1000, count: 1},
      {x0: 2000, count: 1},
      {x0: 4000, count: 1}
    ]);
    binding.publishHistograms(table.lumaCompute);
    expect(table.lumaCompute.updateFilterPlot(chartTime).lumaComputePlot?.status).toBe('ready');
    const previousRevision = binding.currentRevision;
    binding.setChartTimeNullEpochFilters([{filterId: chartTime.id, field: 'timestamp'}]);
    expect(binding.currentRevision).toBeGreaterThan(previousRevision);
    expect(binding.canConsume).toBe(false);
    const encoder = device.createCommandEncoder();
    primary.encode(encoder);
    device.submit(encoder.finish());
    expect(await binding.readSummary()).toBe(true);
    expect(await readMask(binding.getLayerMask('layer'))).toEqual([1, 0, 0, 1, 0]);
    expect((await update([chartTime, {...chartTime, id: 'ordinary-time-slider'}])).display).toEqual(
      [1, 0, 0, 0, 0]
    );
    expect(binding.getFilterHistogram(chartTime.id)?.reduce((sum, bin) => sum + bin.count, 0)).toBe(
      1
    );
    expect(
      binding.getFilterHistogram('ordinary-time-slider')?.reduce((sum, bin) => sum + bin.count, 0)
    ).toBe(2);
    expect((await update([{...chartTime, value: [1, 2000]}])).display).toEqual([0, 1, 1, 0, 0]);
    expect((await update([chartTime])).display).toEqual([1, 0, 0, 1, 0]);
    binding.setChartTimeNullEpochFilters([{filterId: 'different-filter-id', field: 'timestamp'}]);
    expect((await update([chartTime])).display).toEqual([1, 0, 0, 0, 0]);
    const beforeInvalidPolicy = binding.currentRevision;
    expect(() =>
      binding.setChartTimeNullEpochFilters([{filterId: chartTime.id, field: 'value'}])
    ).toThrow('exact time field');
    expect(binding.currentRevision).toBe(beforeInvalidPolicy);
    binding.setChartTimeNullEpochFilters([]);
    await update([chartTime]);
    binding.publishHistograms(table.lumaCompute);
    const minuteFilter = {...chartTime, plotType: {type: 'histogram', interval: '1-minute'}};
    expect(table.lumaCompute.updateFilterPlot(minuteFilter).lumaComputePlot?.status).toBe(
      'pending'
    );
    expect(binding.canConsume).toBe(false);
    const minuteEncoder = device.createCommandEncoder();
    primary.encode(minuteEncoder);
    device.submit(minuteEncoder.finish());
    expect(await binding.readSummary()).toBe(true);
    binding.publishHistograms(table.lumaCompute);
    expect(table.lumaCompute.updateFilterPlot(minuteFilter).lumaComputePlot?.status).toBe('ready');
    expect(binding.getFilterHistogram(chartTime.id)).toEqual([
      {x0: 0, x1: 60000, count: 5, indexes: []}
    ]);
  } finally {
    const fence = device.createFence();
    try {
      await Promise.all([binding.destroy(fence.signaled), primary.close(fence.signaled)]);
      expect(session.registry.memoryUsage.reservedBytes.gpu).toBe(0);
      await session.destroy(fence.signaled);
    } finally {
      fence.destroy();
      positions.destroy();
    }
  }
});

function makePositions(device: Device, horizontal: number[]): GPUVector<'float32x2'> {
  return new GPUVector({
    type: 'buffer',
    name: 'positions',
    format: 'float32x2',
    length: horizontal.length,
    buffer: device.createBuffer({
      data: new Float32Array(horizontal.flatMap(value => [value, 0])),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    }),
    ownsBuffer: true
  });
}

function polygon(datasetId: string, id: string, layerId: string): KeplerComputeFilter {
  return {
    id,
    dataId: datasetId,
    type: 'polygon',
    layerId: [layerId],
    value: {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [2.5, -0.5],
            [3.5, -0.5],
            [3.5, 0.5],
            [2.5, 0.5],
            [2.5, -0.5]
          ]
        ]
      }
    }
  };
}

async function readMask(mask: GPUVector<'uint32'>): Promise<number[]> {
  return (
    await Promise.all(
      mask.data.map(async chunk => {
        if (!chunk.length) return [];
        const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
        return Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
      })
    )
  ).flat();
}
