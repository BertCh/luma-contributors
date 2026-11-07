// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {GPUVector} from '@luma-contributors/graph/gpu-data';
import {KeplerComputeSession} from '../src/session/compute-session';
import {createNullWebGPUDevice} from '../../graph/test/utils/gpu-contributor-test-utils';
import {HeatmapBindingRuntime} from '../../../examples/deck/kepler-compute/src/heatmap-binding-runtime';

async function fixture() {
  const device = createNullWebGPUDevice();
  vi.spyOn(device, 'getTextureFormatCapabilities').mockReturnValue({blend: true} as ReturnType<
    typeof device.getTextureFormatCapabilities
  >);
  const session = new KeplerComputeSession({device, memoryBudgetBytes: 32 * 1024 * 1024});
  const dataset = session.importArrowDataset({
    id: 'points',
    table: new arrow.Table({value: arrow.makeVector(new Float32Array([2, 8]))})
  });
  const positions = new GPUVector({name: 'points', type: 'data', format: 'float32x2', data: []});
  const selectionMask = new GPUVector({name: 'mask', type: 'data', format: 'uint32', data: []});
  const runtime = new HeatmapBindingRuntime(session, vi.fn());
  const options = {
    layerId: 'heat',
    dataset,
    positions,
    selectionMask,
    columns: {lat: 'north', lng: 'east'},
    revision: 1,
    domainRevision: 1
  };
  return {device, session, runtime, options};
}

test('heatmap admission preserves native constant-one weights and retires texture budget only after the fence', async () => {
  const context = await fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  let complete!: () => void;
  const completion = new Promise<void>(resolve => {
    complete = resolve;
  });
  try {
    const first = context.runtime.resolve(context.options).binding!;
    expect(first.weights).toBeUndefined();
    expect(first.columns).toEqual({lat: 'north', lng: 'east'});
    expect(context.runtime.resolve(context.options).binding?.revision).toBe(first.revision);
    expect(
      context.runtime.resolve({...context.options, revision: 2}).binding?.revision
    ).toBeGreaterThan(first.revision);
    expect(context.session.registry.memoryUsage.totalBytes).toBeGreaterThan(baseline);
    const retired = context.runtime.retainLayers([], completion);
    await Promise.resolve();
    expect(context.session.registry.memoryUsage.totalBytes).toBeGreaterThan(baseline);
    complete();
    await retired;
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    expect(context.options.dataset.value.columns['value'].data[0].buffer.destroyed).toBe(false);
  } finally {
    complete();
    await context.runtime.destroy(Promise.resolve());
    await context.session.destroy(Promise.resolve());
    context.device.destroy();
    vi.restoreAllMocks();
  }
});

test('heatmap rejected completion still releases source lease and texture reservation', async () => {
  const context = await fixture();
  const baseline = context.session.registry.memoryUsage.totalBytes;
  try {
    expect(
      context.runtime.resolve({...context.options, weightColumn: 'value', weightScale: 'quantile'})
        .reason
    ).toContain('linear');
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    context.runtime.resolve(context.options);
    await expect(
      context.runtime.retainLayers([], Promise.reject(new Error('lost fence')))
    ).rejects.toThrow('retirement');
    expect(context.session.registry.memoryUsage.totalBytes).toBe(baseline);
    await expect(context.runtime.destroy(Promise.resolve())).rejects.toThrow('retirement');
  } finally {
    await context.session.destroy(Promise.resolve());
    context.device.destroy();
    vi.restoreAllMocks();
  }
});
