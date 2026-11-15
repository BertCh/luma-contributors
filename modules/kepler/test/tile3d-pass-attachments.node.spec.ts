// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import type {Model} from '@luma.gl/engine';
import {expect, test, vi} from 'vitest';
import {prepareScenegraphPassAttachments} from '../../../examples/deck/kepler-compute/gpu-layers/effects/wgsl-shadow-scenegraph';
import {
  patchTiles3DRemovedExtensionsSource,
  createKeplerComputeDeckCompatPlugin
} from '../../../examples/deck/kepler-compute/kepler-compute-transform';

test('scenegraph models retain render parameters and let Model infer each pass depth format', () => {
  const parameters = {
    depthFormat: 'depth24plus' as const,
    depthWriteEnabled: true,
    depthCompare: 'less' as const,
    cullMode: 'back' as const,
    blend: true
  };
  const model = {
    parameters,
    setParameters: vi.fn(next => {
      Object.assign(model, {parameters: next});
    })
  };
  prepareScenegraphPassAttachments([model as unknown as Model]);
  expect(model.parameters).toEqual({
    depthWriteEnabled: true,
    depthCompare: 'less',
    cullMode: 'back',
    blend: true
  });
  expect(parameters.depthFormat).toBe('depth24plus');
  prepareScenegraphPassAttachments([model as unknown as Model]);
  expect(model.setParameters).toHaveBeenCalledTimes(1);
});

test('pinned 3D tiles metadata callback accepts raw, postprocessed and missing glTF', () => {
  const source = readFileSync(
    new URL(
      '../../../node_modules/@loaders.gl/tiles-3d/dist/tileset-3d/format-3d-tiles/tiles-3d-source.js',
      import.meta.url
    ),
    'utf8'
  );
  expect(source).toContain('content?.gltf?.json.extensionsRemoved');
  const patched = patchTiles3DRemovedExtensionsSource(source);
  expect(patchTiles3DRemovedExtensionsSource(patched)).toBe(patched);
  const statement = patched.match(/const extensionsRemoved = contents.flatMap\([^\n]+/u)?.[0];
  expect(statement).toBeDefined();
  const getRemovedExtensions = new Function('contents', `${statement}\nreturn extensionsRemoved;`);
  expect(
    getRemovedExtensions([
      {gltf: {json: {extensionsRemoved: ['KHR_draco_mesh_compression']}}},
      {gltf: {extensionsRemoved: ['EXT_meshopt_compression']}},
      {gltf: {}},
      {},
      undefined
    ])
  ).toEqual(['KHR_draco_mesh_compression', 'EXT_meshopt_compression']);
  const transform = createKeplerComputeDeckCompatPlugin().transform;
  expect(typeof transform).toBe('function');
  const transformSource = transform as (source: string, id: string) => {code: string} | null;
  expect(
    transformSource(
      source,
      '/node_modules/@loaders.gl/tiles-3d/dist/tileset-3d/format-3d-tiles/tiles-3d-source.js'
    )?.code
  ).toBe(patched);
  expect(transformSource(source, '/node_modules/other/source.js')).toBeNull();
});
