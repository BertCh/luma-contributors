// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import type {Device} from '@luma.gl/core';
import {VectorTileResidentTile} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-tile';
import {VectorTileResidentRenderer} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-layers';

test('vector tile geometry switches once per globe transition and releases its previous reservation', () => {
  let globeResolutionDegrees: number | undefined;
  const destroyed: string[] = [];
  const released: number[] = [];
  const create = vi.spyOn(VectorTileResidentTile, 'create').mockImplementation(
    options =>
      ({
        tileKey: options.tileKey,
        globeResolutionDegrees: options.globeResolutionDegrees,
        binding: {},
        gpuByteLength: 128,
        destroy: () => destroyed.push(options.tileKey)
      }) as unknown as VectorTileResidentTile
  );
  const renderer = new VectorTileResidentRenderer({
    getDevice: () => device,
    getGlobeResolutionDegrees: () => globeResolutionDegrees,
    getReserveMemory: () => byteLength => ({release: () => released.push(byteLength)})
  });
  const device = {} as Device;
  try {
    renderer.admitTile(
      'basemap',
      '0/0/0',
      [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}],
      []
    );
    renderer.getLayers();
    renderer.getLayers();
    expect(create).toHaveBeenCalledTimes(1);
    expect(create.mock.calls[0][0].globeResolutionDegrees).toBeUndefined();
    globeResolutionDegrees = 5;
    renderer.getLayers();
    renderer.getLayers();
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[1][0].globeResolutionDegrees).toBe(5);
    expect(destroyed).toHaveLength(1);
    expect(released).toEqual([128]);
    globeResolutionDegrees = undefined;
    renderer.getLayers();
    expect(create).toHaveBeenCalledTimes(3);
    expect(create.mock.calls[2][0].globeResolutionDegrees).toBeUndefined();
    expect(released).toEqual([128, 128]);
  } finally {
    renderer.destroy();
    create.mockRestore();
  }
  expect(released).toEqual([128, 128, 128]);
});
