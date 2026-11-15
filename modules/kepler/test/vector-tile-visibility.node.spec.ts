// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {WebMercatorViewport, _GlobeViewport as GlobeViewport} from '@deck.gl/core';
import {Tileset2D} from '@deck.gl/geo-layers';
import {expect, test, vi} from 'vitest';
import {VectorTileLiveController} from '../../../examples/deck/kepler-compute/src/vector-tile-live-controller';
import {VectorTileResidentRenderer} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-layers';
import {VectorTileResidentTile} from '../../../examples/deck/kepler-compute/src/vector-tile-resident-tile';

class TestMVTLayer {
  static layerName = 'MVTLayer';
  readonly id = 'data-vector-tiles';
  constructor(readonly props: Record<string, unknown> = {}) {}
  clone(props: Record<string, unknown>) {
    return new TestMVTLayer({...this.props, ...props});
  }
}

test('data tile cache transitions draw only the current viewport, and rebuild only visible globe geometry', () => {
  let globeResolutionDegrees: number | undefined;
  const device = {} as Device;
  const destroyed: string[] = [];
  const create = vi.spyOn(VectorTileResidentTile, 'create').mockImplementation(
    options =>
      ({
        tileKey: options.tileKey,
        globeResolutionDegrees: options.globeResolutionDegrees,
        binding: {points: {positions: {}, selectionMask: {}, resolveSourceRow: () => 0}},
        gpuByteLength: 128,
        update() {},
        destroy: () => destroyed.push(options.tileKey)
      }) as unknown as VectorTileResidentTile
  );
  const renderer = new VectorTileResidentRenderer({
    getDevice: () => device,
    getGlobeResolutionDegrees: () => globeResolutionDegrees,
    getReserveMemory: () => null
  });
  const controller = new VectorTileLiveController({
    residentRenderer: renderer,
    getReserveMemory: () => null,
    getCreateBuffer: () => null
  });
  const props = controller.wrapLayer(new TestMVTLayer()).props as Record<
    string,
    (argument: unknown) => void
  >;
  const content = [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}];
  const detailed = {index: {x: 256, y: 256, z: 9}, content};
  const overview = {index: {x: 0, y: 0, z: 0}, content};
  try {
    props['onTileLoad'](detailed);
    props['onTileLoad'](overview);
    props['onViewportLoad']([detailed]);
    const planar = renderer.getLayers();
    expect(planar).toHaveLength(1);
    expect(planar[0].id).toContain('/9/256/256-points');
    expect(create).toHaveBeenCalledTimes(1);
    globeResolutionDegrees = 5;
    props['onViewportLoad']([overview]);
    expect(controller.getState()[0].visibleTileKeys).toEqual(['0/0/0']);
    const globe = renderer.getLayers();
    expect(globe).toHaveLength(1);
    expect(globe[0].id).toContain('/0/0/0-points');
    expect(create).toHaveBeenCalledTimes(2);
    expect(create.mock.calls[1][0].globeResolutionDegrees).toBe(5);
    expect(destroyed).toEqual([]); // Hidden cached geometry is neither rebuilt nor evicted.
    props['onViewportLoad']([]);
    expect(renderer.getLayers()).toEqual([]);
    props['onViewportLoad']([detailed]);
    expect(renderer.getLayers()).toHaveLength(1);
    expect(create).toHaveBeenCalledTimes(3);
    expect(destroyed).toEqual(['data-vector-tiles/9/256/256']);
  } finally {
    controller.destroy();
    renderer.destroy();
    create.mockRestore();
  }
});

test('native refinement visibility reaches data tiles while an overview is still loading', async () => {
  const notify = vi.fn();
  const renderer = new VectorTileResidentRenderer({
    getDevice: () => null,
    getReserveMemory: () => null,
    onChange: notify
  });
  const controller = new VectorTileLiveController({
    residentRenderer: renderer,
    getReserveMemory: () => null,
    getCreateBuffer: () => null
  });
  let updateCount = 0;
  class CustomTileset extends Tileset2D {
    override update(...updateArguments: Parameters<Tileset2D['update']>) {
      updateCount++;
      return super.update(...updateArguments);
    }
  }
  const wrapped = controller.wrapLayer(new TestMVTLayer({TilesetClass: CustomTileset}));
  const VisibilityTileset = wrapped.props['TilesetClass'] as typeof Tileset2D;
  expect(VisibilityTileset).not.toBe(CustomTileset);
  let resolveOverview!: (content: unknown[]) => void;
  const pendingOverview = new Promise<unknown[]>(resolve => {
    resolveOverview = resolve;
  });
  const content = [{geometry: {type: 'Point', coordinates: [0, 0]}, properties: {}}];
  const tileset = new VisibilityTileset({
    minZoom: 0,
    maxZoom: 9,
    maxCacheSize: 100,
    getTileData: ({index}) => (index.z === 0 ? pendingOverview : content),
    onTileLoad: wrapped.props['onTileLoad'] as never,
    onTileUnload: wrapped.props['onTileUnload'] as never
  });
  const getNativeVisibleKeys = () =>
    tileset.tiles
      .filter(tile => tile.isVisible)
      .map(tile => `${tile.index.z}/${tile.index.x}/${tile.index.y}`);
  try {
    expect(tileset).toBeInstanceOf(CustomTileset);
    const planar = new WebMercatorViewport({
      width: 128,
      height: 128,
      longitude: 0,
      latitude: 0,
      zoom: 9
    });
    tileset.update(planar);
    await vi.waitFor(() => expect(tileset.isLoaded).toBe(true));
    tileset.update(planar);
    expect(renderer.getState()[0].visibleTileKeys).toEqual(getNativeVisibleKeys());
    expect(getNativeVisibleKeys().every(key => key.startsWith('9/'))).toBe(true);
    const globe = new GlobeViewport({
      width: 128,
      height: 128,
      longitude: 0,
      latitude: 0,
      zoom: 0.2,
      resolution: 72
    });
    tileset.update(globe);
    expect(tileset.isLoaded).toBe(false);
    expect(tileset.selectedTiles?.map(tile => tile.index)).toMatchObject([{x: 0, y: 0, z: 0}]);
    const fallbackKeys = getNativeVisibleKeys();
    expect(fallbackKeys.some(key => key.startsWith('9/'))).toBe(true);
    expect(renderer.getState()[0].visibleTileKeys).toEqual(fallbackKeys);
    // Selected tiles alone must not overwrite the native fallback/refinement visibility.
    (wrapped.props['onViewportLoad'] as (tiles: unknown) => void)(tileset.selectedTiles);
    expect(renderer.getState()[0].visibleTileKeys).toEqual(fallbackKeys);
    resolveOverview(content);
    await vi.waitFor(() => expect(tileset.isLoaded).toBe(true));
    tileset.update(globe);
    expect(renderer.getState()[0].visibleTileKeys).toEqual(['0/0/0']);
    expect(tileset.tiles.some(tile => tile.index.z === 9 && !tile.isVisible)).toBe(true);
    const previousNotifications = notify.mock.calls.length;
    tileset.update(globe);
    expect(notify).toHaveBeenCalledTimes(previousNotifications);
    expect(updateCount).toBeGreaterThan(0);
  } finally {
    tileset.finalize();
    controller.destroy();
    renderer.destroy();
  }
});
