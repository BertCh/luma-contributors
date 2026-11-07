// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRegionalRasterChangeRecipe: two slices of a raster stack differenced and summarized per zone.
// The raster is the per-cell change (red grew, blue shrank); zone fills show the mean change and
// the dots the area whose change passes the threshold.

import {addRegionalRasterChangeRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUChangeDetectionParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {addReadout} from './lib/readout';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './regional-raster-change-recipe.parameters';

const SLICE_COUNT = 4;

export default defineOperationRun({
  parameters,
  caption: 'Change between two slices per cell, and mean change per zone.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 6, 4);
    return {
      world,
      grid,
      zones,
      zoneRaster: rasterizeZones(zones.positions, zones.count, 4, grid),
      noise: Float32Array.from({length: grid.width * grid.height}, random)
    };
  },
  build(builder, {world, grid, zones, zoneRaster, noise}) {
    const cellCount = grid.width * grid.height;
    const mesh = uploadQuadZones(builder, zones);
    const slices = builder.upload(
      'slices',
      'float32',
      Float32Array.from({length: cellCount * SLICE_COUNT}, (_, row) => {
        const cell = Math.floor(row / SLICE_COUNT);
        const slice = row % SLICE_COUNT;
        const [x, y] = getCellCenter(grid, cell);
        const growth = Math.exp(-(((x - 0.7 * world.width) / 0.25) ** 2 + ((y - 0.6) / 0.25) ** 2));
        const decline = Math.exp(-(((x - 0.25 * world.width) / 0.2) ** 2 + ((y - 0.4) / 0.3) ** 2));
        return 20 + slice * (6 * growth - 5 * decline) + noise[cell] * 1.5;
      })
    );
    const parameterBuffer = builder.allocate('change-parameters', 'float32', 8);
    const threshold = builder.allocate('threshold', 'float32', 1);
    const recipe = addRegionalRasterChangeRecipe(builder.graph, {
      grid: grid.grid,
      change: {slices: slices.view, sliceCount: SLICE_COUNT, parameters: parameterBuffer.view},
      zones: builder.upload('zone-raster', 'uint32', zoneRaster).view,
      zoneCount: zones.count,
      cellAreas: builder.upload(
        'cell-areas',
        'float32',
        new Float32Array(cellCount).fill(grid.cellSize ** 2)
      ).view,
      threshold: threshold.view
    });
    return {
      outputs: {
        difference: addReadout(builder, 'difference', recipe.outputs.difference),
        meanChanges: addReadout(builder, 'mean-changes', recipe.outputs.meanChanges),
        changedArea: addReadout(builder, 'changed-area', recipe.outputs.changedArea),
        mesh
      },
      update: current => {
        threshold.write(Float32Array.of(current.threshold));
        parameterBuffer.write(
          getGPUChangeDetectionParameterValues({
            beforeSlice: current.beforeSlice,
            afterSlice: current.afterSlice
          })
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.difference,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      ramp: 'diverging',
      range: [-18, 18]
    },
    getQuadFill(outputs.mesh, inputs.zones, outputs.meanChanges, {
      ramp: 'diverging',
      range: [-18, 18],
      alpha: 0.35
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    },
    // Area beyond the threshold, per zone (brighter = more area changed).
    {
      layer: 'dots',
      positions: outputs.mesh.centers,
      values: outputs.changedArea,
      ramp: 'sequential',
      radius: 12,
      range: [0, 0.12]
    }
  ]
});
