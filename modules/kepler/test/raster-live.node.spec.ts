// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {describe, expect, it} from 'vitest';
import {RasterLiveController} from '../../../examples/deck/kepler-compute/src/raster-live-controller';
import {
  createNumericSourceFromKeplerTiles,
  getKeplerTileSignature
} from '../../../examples/deck/kepler-compute/src/raster-live-source';
import {
  computeZonalTable,
  decodeZonalPolygons
} from '../../../examples/deck/kepler-compute/src/raster-live-zonal';

function makeTile(x: number, y: number, z: number, fill: number, mask?: number[]) {
  return {
    index: {x, y, z},
    data: {
      images: {
        imageBands: [{data: new Float32Array(4).fill(fill), width: 2, height: 2}],
        imageMask: mask ? {data: Uint8Array.from(mask), width: 2, height: 2} : null
      }
    }
  };
}

describe('raster live source', () => {
  it('copies Kepler tiles into a numeric source with mask nodata', () => {
    const source = createNumericSourceFromKeplerTiles('layer', [
      makeTile(1, 1, 2, 5, [1, 0, 1, 1]),
      null
    ]);
    expect(source.tiles).toHaveLength(1);
    expect([...source.tiles[0].bands[0].validity]).toEqual([1, 0, 1, 1]);
    expect(source.tiles[0].zoom).toBe(2);
  });

  it('turns RGBA-only tiles into four uint8 bands with alpha nodata', () => {
    const rgba = new Uint8ClampedArray([1, 2, 3, 255, 4, 5, 6, 0, 7, 8, 9, 255, 1, 1, 1, 255]);
    const source = createNumericSourceFromKeplerTiles('rgba', [
      {index: {x: 0, y: 0, z: 1}, data: {images: {imageRgba: {data: rgba, width: 2, height: 2}}}}
    ]);
    expect(source.bandCount).toBe(4);
    expect([...source.tiles[0].bands[3].validity]).toEqual([1, 0, 1, 1]);
  });

  it('changes signature for replacement sample buffers, even at the same tile coordinates', () => {
    const tile = makeTile(0, 0, 1, 1);
    expect(getKeplerTileSignature([tile])).toBe(getKeplerTileSignature([tile]));
    expect(getKeplerTileSignature([tile])).not.toBe(getKeplerTileSignature([makeTile(0, 0, 1, 2)]));
    expect(getKeplerTileSignature([makeTile(0, 0, 1, 1)])).not.toBe(
      getKeplerTileSignature([makeTile(1, 0, 1, 1)])
    );
  });
});

describe('raster live controller and zonal table', () => {
  const layer = {
    id: 'raster-1',
    type: 'rasterTile',
    _loadedTiles: [makeTile(0, 0, 1, 10), makeTile(0, 0, 2, 20)]
  };
  const controller = new RasterLiveController({
    getLayers: () => [layer],
    getDevice: () => undefined
  });

  it('identifies the highest loaded zoom and reports outside', () => {
    const inside = controller.identify('raster-1', -170, 80);
    expect(inside?.status).toBe('value');
    expect(inside?.bandValues).toEqual([20]);
    expect(controller.identify('raster-1', 100, -50)?.status).toBe('outside');
    expect(controller.getSnapshot('raster-1')?.zooms).toEqual([1, 2]);
  });

  it('decodes polygons and appends zonal columns preserving batches', () => {
    expect(decodeZonalPolygons('{"type":"Point","coordinates":[0,0]}')).toBeUndefined();
    const polygon = {
      type: 'Polygon',
      coordinates: [
        [
          [-175, 85],
          [-100, 85],
          [-100, 60],
          [-175, 60],
          [-175, 85]
        ]
      ]
    };
    const table = arrow.tableFromJSON([
      {id: 1, geom: JSON.stringify(polygon)},
      {id: 2, geom: 'not json'}
    ]);
    const result = computeZonalTable({
      controller,
      rasterLayerId: 'raster-1',
      table,
      geometryColumn: 'geom'
    });
    expect(result.skippedRowCount).toBe(1);
    expect(result.table.numRows).toBe(2);
    expect(result.table.getChild('zonal_count')?.get(0)).toBeGreaterThan(0);
    expect(result.table.getChild('zonal_count')?.get(1)).toBeNull();
  });
});
