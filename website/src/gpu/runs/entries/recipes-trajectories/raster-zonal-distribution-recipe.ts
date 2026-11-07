// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRasterZonalDistributionRecipe: the distribution of raster cell values inside each zone. The
// raster shows through muted; zones are filled by the chosen quantile of their cells.

import {addRasterZonalDistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './raster-zonal-distribution-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Quantile of the raster values inside each zone.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 7, 4);
    const field = createSmoothField(random, 4, 3);
    return {
      world,
      grid,
      zones,
      zoneRaster: rasterizeZones(zones.positions, zones.count, 4, grid),
      field,
      noise: Float32Array.from({length: grid.width * grid.height}, random)
    };
  },
  build(builder, {world, grid, zones, zoneRaster, field, noise}) {
    const cellCount = grid.width * grid.height;
    const mesh = uploadQuadZones(builder, zones);
    const band = builder.allocate('band', 'float32', cellCount);
    const probabilities = builder.allocate('probabilities', 'float32', 1);
    const quantiles = builder.allocate('quantiles', 'float32', zones.count);
    addRasterZonalDistributionRecipe(builder.graph, {
      grid: grid.grid,
      input: {id: 'band', format: 'float32', storage: {kind: 'buffer', values: band.view}},
      zones: builder.upload('zone-raster', 'uint32', zoneRaster).view,
      zoneCount: zones.count,
      probabilities: probabilities.view,
      outputs: {quantiles: quantiles.view}
    });
    return {
      outputs: {band, quantiles, mesh},
      update: current => {
        probabilities.write(Float32Array.of(current.probability));
        band.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            return (
              current.relief * 100 * field(x / world.width, y / world.height) +
              current.roughness * 40 * noise[index]
            );
          })
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.band,
      columns: inputs.grid.width,
      rows: inputs.grid.height
    },
    getQuadFill(outputs.mesh, inputs.zones, outputs.quantiles, {ramp: 'sequential', alpha: 0.5}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});
