// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {buildBitmapFontAtlas, type FontAtlas} from '@luma.gl/engine/text';
import {createNullWebGPUDevice} from '../../graph/test/utils/gpu-contributor-test-utils';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {preparePointLabelSource} from '../../../examples/deck/kepler-compute/src/label-source';
import {makePointLabelRuntime} from '../../../examples/deck/kepler-compute/src/label-runtime';

vi.mock('@luma.gl/engine/text', async importOriginal => {
  const actual = await importOriginal<typeof import('@luma.gl/engine/text')>();
  return {
    ...actual,
    buildBitmapFontAtlas: vi.fn(() => makeAtlas()),
    GPUTextResources: class {
      readonly atlasTexture = {ready: Promise.resolve(), isReady: true};
      readonly fontAtlas: FontAtlas;
      constructor(
        readonly device: Device,
        props: {fontAtlas: FontAtlas}
      ) {
        this.fontAtlas = props.fontAtlas;
      }
      destroy = vi.fn();
    }
  };
});
function makeAtlas(): FontAtlas {
  return {
    baselineOffset: 5,
    lineHeight: 64,
    width: 128,
    height: 64,
    pages: [{} as HTMLCanvasElement],
    xOffset: 0,
    yOffsetMin: 0,
    yOffsetMax: 0,
    renderSettings: {mode: 'bitmap', threshold: 0, smoothing: 0},
    mapping: {
      A: {
        x: 0,
        y: 0,
        width: 10,
        height: 12,
        advance: 10.5,
        anchorX: 5,
        anchorY: 9,
        layoutOffsetX: 0,
        layoutOffsetY: -9
      }
    }
  };
}
function fixture(device: Device) {
  const source = preparePointLabelSource({
    fieldName: 'name',
    chunks: [
      {id: 'first', values: ['A', null]},
      {id: 'empty', values: []},
      {id: 'last', values: ['A']}
    ],
    maxCharactersPerLabel: 10,
    maxGlyphs: 10,
    maxSourceBytes: 1024
  });
  const vector = <Format extends 'float32x3' | 'uint32'>(format: Format) =>
    new GPUVector({
      type: 'data',
      format,
      ownsData: true,
      data: source.chunks.map(
        chunk =>
          new GPUData({
            format,
            length: chunk.rowCount,
            ownsBuffer: true,
            buffer: device.createBuffer({
              byteLength: Math.max(4, chunk.rowCount * (format === 'float32x3' ? 12 : 4)),
              usage: Buffer.STORAGE | Buffer.COPY_DST
            })
          })
      )
    });
  return {source, positions: vector('float32x3'), mask: vector('uint32')};
}
const mapState = {longitude: 0, latitude: 45, zoom: 8};

test('point label style/filter/zoom resolution never rebuilds source glyphs and preserves native non-picking semantics', async () => {
  const device = createNullWebGPUDevice();
  const inputs = fixture(device);
  const read = inputs.positions.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'));
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const before = vi.mocked(buildBitmapFontAtlas).mock.calls.length;
  const runtime = makePointLabelRuntime(device, {
    source: inputs.source,
    positions: inputs.positions,
    workingMemoryBudgetBytes: 1 << 20,
    memoryRegistry: registry
  });
  await runtime.ready;
  // 128x64 RGBA8 full mip chain, two glyphs, one empty batch, and renderer controls.
  const atlasGPUBytes = (8192 + 2048 + 512 + 128 + 32 + 8 + 2 + 1) * 4;
  expect(runtime.gpuByteLength).toBe(atlasGPUBytes + 80 + 24 + 4 + 2 * 148);
  const first = runtime.resolve({
    label: {field: {name: 'name'}, size: 18},
    selectionMask: inputs.mask,
    radius: 3,
    radiusScale: 2,
    mapState
  });
  const second = runtime.resolve({
    label: {
      field: {name: 'name'},
      size: 24,
      color: [255, 0, 0],
      anchor: 'end',
      alignment: 'bottom'
    },
    selectionMask: inputs.mask,
    radius: 5,
    radiusScale: 4,
    mapState: {...mapState, zoom: 9}
  });
  if (!('layerProps' in first) || !('layerProps' in second))
    throw new Error('Expected supported label plan');
  expect(second.layerProps.batches).toBe(first.layerProps.batches);
  expect(second.layerProps.resources).toBe(first.layerProps.resources);
  expect(second.layerProps.getPosition).toBe(inputs.positions);
  expect(second.layerProps.selectionMask).toBe(inputs.mask);
  expect(second.layerProps.pickable).toBe(false);
  expect(second.layerProps.opacity).toBe(1);
  expect(second.layerProps.size).toBe(16);
  expect(second.layerProps.verticalPadding).toBe(24);
  expect(second.layerProps.radiusPixelScale).toBeCloseTo(first.layerProps.radiusPixelScale! * 4);
  expect(
    first.layerProps.batches.map(batch => [
      batch.rowCount,
      batch.rowIndexBase,
      batch.glyphRows.length
    ])
  ).toEqual([
    [2, 0, 1],
    [0, 2, 0],
    [1, 2, 1]
  ]);
  expect(vi.mocked(buildBitmapFontAtlas).mock.calls.length - before).toBe(1);
  read.forEach(spy => expect(spy).not.toHaveBeenCalled());
  const resources = first.layerProps.batches.flatMap(batch => [
    batch.glyphLayouts.buffer,
    batch.glyphRows.buffer
  ]);
  expect(registry.memoryUsage.reservedBytes).toEqual({
    cpu: runtime.cpuByteLength,
    gpu: runtime.gpuByteLength
  });
  const fence = Promise.withResolvers<void>();
  const pending = runtime.destroy(fence.promise);
  expect(runtime.isCurrent()).toBe(false);
  expect(resources.every(buffer => !buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBeGreaterThan(0);
  fence.resolve();
  await pending;
  expect(resources.every(buffer => buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  expect(inputs.positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  inputs.positions.destroy();
  inputs.mask.destroy();
  device.destroy();
});

test('point labels reject unsupported native controls instead of silently dropping them', async () => {
  const device = createNullWebGPUDevice();
  const inputs = fixture(device);
  const runtime = makePointLabelRuntime(device, {
    source: inputs.source,
    positions: inputs.positions,
    workingMemoryBudgetBytes: 1 << 20
  });
  for (const label of [
    {field: {name: 'other'}},
    {field: {name: 'name', type: 'timestamp'}},
    {field: {name: 'name', displayFormat: '0.00'}},
    {field: {name: 'name'}, weight: 700},
    {field: {name: 'name'}, outlineWidth: 1},
    {field: {name: 'name'}, background: true},
    {field: {name: 'name'}, collisionEnabled: true}
  ]) {
    expect(
      runtime.resolve({label, selectionMask: inputs.mask, radiusScale: 1, mapState})
    ).toHaveProperty('diagnostic');
  }
  // The label shader horizon-culls on a globe, so the same label resolves there.
  expect(
    runtime.resolve({
      label: {field: {name: 'name'}},
      selectionMask: inputs.mask,
      radiusScale: 1,
      mapState: {...mapState, globe: {enabled: true}}
    })
  ).not.toHaveProperty('diagnostic');
  await runtime.destroy(Promise.resolve());
  inputs.positions.destroy();
  inputs.mask.destroy();
  device.destroy();
});

test('label allocations reject before GPU upload on budget failure and release on captured device loss', async () => {
  const device = createNullWebGPUDevice();
  const inputs = fixture(device);
  const create = vi.spyOn(device, 'createBuffer');
  expect(() =>
    makePointLabelRuntime(device, {
      source: inputs.source,
      positions: inputs.positions,
      workingMemoryBudgetBytes: 1
    })
  ).toThrow(/budget/);
  expect(create).not.toHaveBeenCalled();
  const loss = Promise.withResolvers<{reason: 'destroyed'; message: string}>();
  Object.defineProperty(device, 'lost', {value: loss.promise, configurable: true});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1 << 20});
  const runtime = makePointLabelRuntime(device, {
    source: inputs.source,
    positions: inputs.positions,
    workingMemoryBudgetBytes: 1 << 20,
    memoryRegistry: registry
  });
  const stranded = new Promise<void>(() => {});
  Object.defineProperty(device, 'lost', {value: stranded});
  const pending = runtime.destroy(stranded);
  loss.resolve({reason: 'destroyed', message: 'test'});
  await pending;
  expect(registry.memoryUsage.totalBytes).toBe(0);
  inputs.positions.destroy();
  inputs.mask.destroy();
  device.destroy();
});
