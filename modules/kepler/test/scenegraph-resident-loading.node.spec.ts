// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {ScenegraphLayer} from '@deck.gl/mesh-layers';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {ResidentScenegraphPointLayer} from '../../../examples/deck/kepler-compute/src/scenegraph-resident-layer';

// Loading/drawing before a model exists needs no Kepler browser bootstrap or GPU device.
vi.mock('@kepler.gl/layers', () => ({toCorsSafeGcsUrl: (url: string) => url}));

test('resident scenegraphs draw safely while an asynchronous model response is pending', async () => {
  const encodeInstances = vi.fn();
  const resident = new ResidentScenegraphPointLayer({
    id: 'delayed-scenegraph',
    positions: {format: 'float32x2', data: []} as unknown as GPUVector<'float32x2'>,
    scenegraph: 'https://example.invalid/delayed.glb',
    sizeScale: 1,
    orientation: [0, 0, 0],
    color: [255, 255, 255, 255]
  });
  Object.assign(resident, {
    state: {chunks: [{chunkIndex: 0, rowIndexBase: 0, data: {length: 1, attributes: {}}}]},
    drawInstances: encodeInstances
  });
  const chunk = resident.renderLayers()[0] as ScenegraphLayer;
  // Deck has not installed scenegraph/models until its async prop response resolves.
  Object.assign(chunk, {state: {}});
  let resolveResponse!: () => void;
  const response = new Promise<void>(resolve => {
    resolveResponse = resolve;
  });
  const parameters = {} as Parameters<ScenegraphLayer['draw']>[0];
  const model = {
    parameters: {depthFormat: 'depth24plus', depthCompare: 'less-equal', depthWriteEnabled: true},
    setParameters: vi.fn()
  };
  const loaded = response.then(() => {
    Object.assign(chunk, {state: {scenegraph: {}, models: [model]}});
  });
  expect(() => chunk.draw(parameters)).not.toThrow();
  await Promise.resolve();
  expect(() => chunk.draw(parameters)).not.toThrow();
  expect(encodeInstances).not.toHaveBeenCalled();
  expect(model.setParameters).not.toHaveBeenCalled();

  resolveResponse();
  await loaded;
  // Rendering an actual glTF needs a GPU. Only replace the stock draw after exercising
  // its real unloaded early return; verify the resident adapter resumes after loading.
  const stockDraw = vi.spyOn(ScenegraphLayer.prototype, 'draw').mockImplementation(() => {});
  try {
    chunk.draw(parameters);
    expect(encodeInstances).toHaveBeenCalledOnce();
    expect(model.setParameters).toHaveBeenCalledExactlyOnceWith({
      depthCompare: 'less-equal',
      depthWriteEnabled: true
    });
    expect(stockDraw).toHaveBeenCalledExactlyOnceWith(parameters);
  } finally {
    stockDraw.mockRestore();
  }
});
