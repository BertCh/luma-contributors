// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LayerClassesType} from '@kepler.gl/layers';
import {DatasetType} from '@kepler.gl/constants';
import {expect, test, vi} from 'vitest';
import {
  createPreparedLayerClasses,
  PREPARE_DATA_MESSAGE
} from '../../../examples/deck/kepler-compute/src/prepared-layer-classes';

class NativeTileLayer {
  readonly id = 'native-tile';
  readonly config = {dataId: 'tile-dataset'};
  renderLayer() {
    return [{id: 'native-deck-layer'}];
  }
}

test.each([
  DatasetType.VECTOR_TILE,
  DatasetType.RASTER_TILE,
  DatasetType.WMS_TILE,
  DatasetType.TILE_3D,
  DatasetType.BITMAP
])('URL-backed %s reaches its renderer without tabular publication', type => {
  const isPrepared = vi.fn(() => false);
  const onPreparationRequired = vi.fn();
  const classes = createPreparedLayerClasses(
    {rasterTile: NativeTileLayer} as unknown as LayerClassesType,
    {isPrepared, onPreparationRequired}
  );
  const layer = new classes.rasterTile({id: 'native-tile'});
  expect(layer.renderLayer({dataset: {type}} as never)).toEqual([{id: 'native-deck-layer'}]);
  expect(isPrepared).not.toHaveBeenCalled();
  expect(onPreparationRequired).not.toHaveBeenCalled();
});

test.each([
  DatasetType.LOCAL,
  DatasetType.EXTERNALLY_HOSTED
])('unprepared tabular %s still requires publication', type => {
  const onPreparationRequired = vi.fn();
  const classes = createPreparedLayerClasses(
    {rasterTile: NativeTileLayer} as unknown as LayerClassesType,
    {isPrepared: () => false, onPreparationRequired}
  );
  const layer = new classes.rasterTile({id: 'native-tile'});
  expect(layer.renderLayer({dataset: {type}} as never)).toEqual([]);
  expect(onPreparationRequired).toHaveBeenCalledWith('native-tile', PREPARE_DATA_MESSAGE);
});
