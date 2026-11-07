// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../graph/test/utils/require-test-device';
import {KeplerComputeSession} from '../src/session/compute-session';
import {createTerrainRasterSourceAsync} from '../../../examples/deck/kepler-compute/src/analysis3-raster-terrain-gpu';
import {executeTerrainResidentPublication} from '../../../examples/deck/kepler-compute/src/terrain-resident-publication';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';

// Oracle: the CPU terrain reference (createTerrainRasterSourceAsync engine 'cpu') on the same
// tiles. The GPU-resident cells must equal it within float32 tolerance, with no cell readback.

const SIZE = 24;
const OPERATIONS = ['slope', 'aspect', 'hillshade', 'curvature'] as const;

function getElevation(x: number, row: number): number {
  return 30 * Math.sin(x / 3) * Math.cos(row / 4) + 2 * x + (row % 5);
}

/** The same surface as one tile: the oracle for the two-tile mosaic with neighbor halo. */
function makeMergedSource() {
  const values = new Float32Array(SIZE * SIZE);
  for (let row = 0; row < SIZE; row++)
    for (let x = 0; x < SIZE; x++) values[row * SIZE + x] = getElevation(x, row);
  return createNumericRasterSource({
    id: 'merged',
    kind: 'xyz-tiles',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 'merged-tile',
        width: SIZE,
        height: SIZE,
        extent: [0, 0, 0.024, 0.024],
        zoom: 3,
        bands: [{values, noDataValue: -9999}]
      })
    ]
  });
}

function makeSource() {
  const tiles = [0, 1].map(column => {
    const width = SIZE / 2;
    const values = new Float32Array(width * SIZE);
    for (let row = 0; row < SIZE; row++)
      for (let local = 0; local < width; local++) {
        const x = column * width + local;
        values[row * width + local] = 30 * Math.sin(x / 3) * Math.cos(row / 4) + 2 * x + (row % 5);
      }
    return createNumericRasterTile({
      id: `tile-${column}`,
      width,
      height: SIZE,
      extent: [column * 0.012, 0, (column + 1) * 0.012, 0.024],
      zoom: 3,
      bands: [{values, noDataValue: -9999}]
    });
  });
  return createNumericRasterSource({id: 'bumps', kind: 'xyz-tiles', crs: 'EPSG:4326', tiles});
}

for (const operation of OPERATIONS) {
  test(`resident terrain ${operation} publishes GPU cells that match the CPU reference`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const session = new KeplerComputeSession({device, memoryBudgetBytes: 256 * 1024 * 1024});
    const source = makeSource();
    const result = await executeTerrainResidentPublication({
      session,
      source,
      operation,
      resultId: `terrain-${operation}`,
      isCurrent: () => true
    });
    try {
      expect(result.rowCount).toBe(2 * (SIZE / 2) * SIZE);
      expect(result.chunkCount).toBe(2);
      expect(result.contributors).toEqual([
        operation === 'curvature' ? 'GPUTerrainCurvature' : 'GPUTerrainDerivatives'
      ]);
      // Only the min/max summary of the six columns came back, never the cells.
      expect(result.readbackBytes).toBe(6 * 3 * 4);
      const resident = result.dataset.value.resident!;
      await resident.ensureRows(Array.from({length: result.rowCount}, (_, row) => row));
      const table = result.dataset.value.source;
      const gpuValues = table.getChild(operation)!;
      const cpu = await createTerrainRasterSourceAsync(
        makeMergedSource(),
        {operation},
        {engine: 'cpu'}
      );
      const band = cpu.source.tiles[0].bands[0];
      const tileWidth = SIZE / 2;
      let compared = 0;
      let nodata = 0;
      for (let tileIndex = 0; tileIndex < 2; tileIndex++)
        for (let row = 0; row < SIZE; row++)
          for (let local = 0; local < tileWidth; local++) {
            const actual = gpuValues.get(tileIndex * tileWidth * SIZE + row * tileWidth + local) as
              | number
              | null;
            const cell = row * SIZE + tileIndex * tileWidth + local;
            const label = `${operation} tile ${tileIndex} row ${row} column ${local}`;
            if (!band.validity[cell]) {
              nodata++;
              expect(actual, `${label} nodata`).toBeNull();
              continue;
            }
            expect(actual, label).not.toBeNull();
            const expected = band.values[cell];
            const scale = Math.max(1, Math.abs(expected));
            expect(Math.abs((actual as number) - expected) / scale, label).toBeLessThan(5e-3);
            compared++;
          }
      expect(compared).toBeGreaterThan(300);
      // The outer ring of the two-tile mosaic has no neighbors, so it must be null, not dropped.
      expect(nodata).toBeGreaterThan(0);
      // Row 1, column 1 of tile 0 is valid; its center is 1.5 pixels from the west edge.
      const centerIndex = tileWidth + 1;
      expect(table.getChild('longitude')!.get(centerIndex) as number).toBeCloseTo(0.0015, 6);
      expect(table.getChild('latitude')!.get(centerIndex) as number).toBeCloseTo(0.024 - 0.0015, 6);
      expect(table.getChild('tile_index')!.get(centerIndex)).toBe(0);
    } finally {
      await result.destroy();
    }
  });
}
