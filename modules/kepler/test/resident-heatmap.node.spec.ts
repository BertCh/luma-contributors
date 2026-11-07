// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {HeatmapLayer} from '@deck.gl/aggregation-layers';
import type {UpdateParameters} from '@deck.gl/core';
import {expect, test, vi} from 'vitest';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {createNullWebGPUDevice} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {
  makeResidentHeatmapLayer,
  renderResidentHeatmapPasses,
  getResidentHeatmapReadiness,
  getResidentHeatmapUnsupportedReason
} from '../../../examples/deck/kepler-compute/src/resident-heatmap-layer';
import {getResidentHeatmapWeightShader} from '../../../examples/deck/kepler-compute/src/heatmap-weight-runtime';

function fixture() {
  const device = createNullWebGPUDevice();
  const positions = new GPUVector({
    format: 'float32x2',
    type: 'data',
    name: 'position',
    data: [
      new GPUData({format: 'float32x2', length: 2, buffer: device.createBuffer({byteLength: 16})})
    ]
  });
  const selectionMask = new GPUVector({
    format: 'uint32',
    type: 'data',
    name: 'mask',
    data: [new GPUData({format: 'uint32', length: 2, buffer: device.createBuffer({byteLength: 8})})]
  });
  return {device, inputs: {positions, selectionMask, revision: 1}};
}

test('resident wrapper preserves native configuration and constructor framing without CPU rows', () => {
  const {device, inputs} = fixture();
  class NativeHeatmap extends HeatmapLayer {
    override _updateTextureRenderingBounds(): void {
      throw new Error('native framing retained');
    }
  }
  const accessor = vi.fn(() => {
    throw new Error('CPU accessor');
  });
  const native = new NativeHeatmap({
    id: 'native',
    data: [{opaque: true}],
    getPosition: accessor,
    radiusPixels: 31,
    intensity: 2.25,
    threshold: 0.12,
    aggregation: 'MEAN',
    colorRange: [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ],
    extensions: [],
    pickable: true
  });
  try {
    const resident = makeResidentHeatmapLayer(native, inputs);
    const replay = makeResidentHeatmapLayer(native, {...inputs, revision: 2});
    expect(resident.constructor).toBe(replay.constructor);
    expect(resident).toBeInstanceOf(NativeHeatmap);
    expect(resident.props).toMatchObject({
      radiusPixels: 31,
      intensity: 2.25,
      threshold: 0.12,
      aggregation: 'MEAN',
      data: [],
      extensions: [],
      pickable: false,
      weightsTextureSize: 512,
      debounceTimeout: 0
    });
    expect((resident as HeatmapLayer).props.colorRange).toBe(native.props.colorRange);
    expect(() => (resident as HeatmapLayer)._updateTextureRenderingBounds()).toThrow(
      'native framing retained'
    );
    (resident as HeatmapLayer)._setupAttributes();
    expect(accessor).not.toHaveBeenCalled();
    expect(inputs.positions.data[0].buffer.destroyed).toBe(false);
  } finally {
    device.destroy();
  }
});

test('mask revisions invalidate native density while unchanged controls do not', () => {
  const {device, inputs} = fixture();
  const native = new HeatmapLayer({id: 'revision', radiusPixels: 20});
  const resident = makeResidentHeatmapLayer(native, inputs) as HeatmapLayer;
  Object.defineProperty(resident, 'context', {value: {viewport: {zoom: 5}}});
  resident.state = {zoom: 5} as HeatmapLayer['state'];
  const flags = (revision: number, radiusPixels = 20) =>
    resident._getChangeFlags({
      props: {...resident.props, radiusPixels, residentInputs: {...inputs, revision}},
      oldProps: resident.props,
      changeFlags: {viewportChanged: false}
    } as unknown as UpdateParameters<HeatmapLayer>);
  try {
    expect(flags(1).dataChanged).toBe(false);
    expect(flags(2).dataChanged).toBeTruthy();
    expect(flags(1, 30).dataChanged).toBeTruthy();
    expect(flags(1).viewportZoomChanged).toBe(false);
  } finally {
    device.destroy();
  }
});

test('resident heatmap explicitly rejects unsupported texture and globe capabilities', () => {
  const {device} = fixture();
  try {
    vi.spyOn(device, 'getTextureFormatCapabilities').mockReturnValue({blend: true} as ReturnType<
      typeof device.getTextureFormatCapabilities
    >);
    expect(getResidentHeatmapUnsupportedReason(device)).toBeUndefined();
    vi.spyOn(device, 'getTextureFormatCapabilities').mockReturnValue({blend: false} as ReturnType<
      typeof device.getTextureFormatCapabilities
    >);
    expect(getResidentHeatmapUnsupportedReason(device)).toContain('rgba16float');
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

test('resident kernel matches pinned native WebGPU Gaussian and documents the distinct GLSL patch', () => {
  const require = createRequire(import.meta.url);
  const deckDirectory = dirname(require.resolve('@deck.gl/aggregation-layers'));
  const nativeShader = readFileSync(join(deckDirectory, 'heatmap-layer/weights.wgsl.js'), 'utf8');
  const nativeMaximum = readFileSync(join(deckDirectory, 'heatmap-layer/max.wgsl.js'), 'utf8');
  const keplerPatch = readFileSync(
    join(dirname(require.resolve('@kepler.gl/layers')), 'heatmap-layer/deck-heatmap-layer.js'),
    'utf8'
  );
  const residentShader = getResidentHeatmapWeightShader('float32x2');
  for (const expression of [
    'exp(-pow(2.0 * distanceFromCenter, 2.0) / 0.05555)',
    '(1.77245385 * 0.166666)',
    'if (distanceFromCenter > 0.5)',
    'project_position_vec3_f64'
  ]) {
    expect(nativeShader).toContain(expression);
    expect(residentShader).toContain(expression);
  }
  expect(nativeMaximum).toContain('weights.r / max(1.0, weights.a)');
  expect(keplerPatch).toContain('value - 0.00443');
  expect(keplerPatch).toContain('fragColor.r = 1.0;');
  expect(residentShader).not.toContain('0.00443');
  expect(residentShader).toContain('attributes.weightValidity != 0u');
  expect(residentShader).toContain('attributes.selectionMask == 0u');
});

test('cold weight pipelines stay dirty and schedule a current-layer retry without source access', async () => {
  const {device, inputs} = fixture();
  const resident = makeResidentHeatmapLayer(new HeatmapLayer({id: 'cold'}), inputs) as HeatmapLayer;
  const pipeline = {isPending: true, isErrored: false};
  const render = vi.fn(() => true);
  const maximum = vi.fn();
  Object.defineProperty(resident, 'context', {value: {device, viewport: {zoom: 5}}});
  resident.state = {
    worldBounds: [0, 0, 1, 1],
    textureSize: 512,
    residentWeightRuntime: {transform: {model: {pipeline}}, render},
    maxWeightTransform: {
      model: {pipeline: {isPending: false, isErrored: false}, needsRedraw: () => false}
    }
  } as unknown as HeatmapLayer['state'];
  const retry = vi.spyOn(resident, 'setNeedsRedraw').mockImplementation(() => {});
  vi.spyOn(resident, 'getCurrentLayer').mockReturnValue(resident);
  vi.spyOn(resident, '_worldToCommonBounds').mockReturnValue([0, 0, 1, 1]);
  vi.spyOn(resident, '_updateMaxWeightValue').mockImplementation(maximum);
  try {
    resident._updateWeightmap();
    resident._updateWeightmap();
    expect(render).not.toHaveBeenCalled();
    renderResidentHeatmapPasses([resident]);
    renderResidentHeatmapPasses([resident]);
    expect(resident.state.isWeightMapDirty).toBe(true);
    expect(render).not.toHaveBeenCalled();
    expect(maximum).not.toHaveBeenCalled();
    await Promise.resolve();
    expect(retry).toHaveBeenCalledTimes(1);
    pipeline.isPending = false;
    renderResidentHeatmapPasses([resident]);
    expect(render).toHaveBeenCalledTimes(1);
    expect(maximum).toHaveBeenCalledTimes(1);
    expect(resident.state.isWeightMapDirty).toBe(false);
    expect(getResidentHeatmapReadiness(resident)).toMatchObject({
      revision: 1,
      completedRevision: 1,
      densityPending: false,
      presentationReady: false
    });
    retry.mockClear();
    render.mockReturnValueOnce(false);
    resident._updateWeightmap();
    renderResidentHeatmapPasses([resident]);
    expect(resident.state.isWeightMapDirty).toBe(true);
    expect(maximum).toHaveBeenCalledTimes(1);
    await Promise.resolve();
    expect(retry).toHaveBeenCalledTimes(1);
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});

test('density waits for producer encoding and native loading waits for a later display draw', () => {
  const {device, inputs} = fixture();
  const resident = makeResidentHeatmapLayer(
    new HeatmapLayer({id: 'ordered'}),
    inputs
  ) as HeatmapLayer;
  const operations: string[] = [];
  const model = {pipeline: {isPending: false, isErrored: false}, _lastDrawTimestamp: -1};
  Object.defineProperty(resident, 'context', {value: {device, viewport: {zoom: 5}}});
  Object.defineProperty(resident, 'internalState', {value: {isAsyncPropLoading: () => false}});
  vi.spyOn(resident, 'getSubLayers').mockReturnValue([
    {isLoaded: true, getModels: () => [model]} as unknown as HeatmapLayer
  ]);
  vi.spyOn(resident, 'setNeedsRedraw').mockImplementation(() => {});
  vi.spyOn(resident, '_worldToCommonBounds').mockReturnValue([0, 0, 1, 1]);
  vi.spyOn(resident, '_updateMaxWeightValue').mockImplementation(() => {
    operations.push('maximum');
  });
  resident.state = {
    worldBounds: [0, 0, 1, 1],
    textureSize: 512,
    residentWeightRuntime: {
      transform: {model},
      render: () => {
        operations.push('density');
        return true;
      }
    },
    maxWeightTransform: {model: {...model, needsRedraw: () => false}}
  } as unknown as HeatmapLayer['state'];
  try {
    resident._updateWeightmap();
    expect(operations).toEqual([]);
    expect(resident.isLoaded).toBe(false);
    operations.push('selection', 'normalized weights');
    renderResidentHeatmapPasses([resident]);
    expect(operations).toEqual(['selection', 'normalized weights', 'density', 'maximum']);
    expect(resident.isLoaded).toBe(false);
    model._lastDrawTimestamp = device.timestamp;
    expect(resident.isLoaded).toBe(true);
    renderResidentHeatmapPasses([resident]);
    expect(operations).toHaveLength(4);
    resident._updateWeightmap();
    expect(resident.isLoaded).toBe(false);
    renderResidentHeatmapPasses([resident]);
    expect(resident.isLoaded).toBe(false);
    model._lastDrawTimestamp = device.timestamp;
    expect(resident.isLoaded).toBe(true);
  } finally {
    vi.restoreAllMocks();
    device.destroy();
  }
});
