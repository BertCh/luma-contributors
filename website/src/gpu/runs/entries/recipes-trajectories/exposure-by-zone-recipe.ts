// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addExposureByZoneRecipe: share of each zone's population living where the hazard reaches the
// threshold. The hazard raster is muted underneath; zones are filled by their exposed share.

import {addExposureByZoneRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {addReadout} from './lib/readout';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './exposure-by-zone-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Share of population exposed to a hazard above the threshold, per zone.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 8, 5);
    return {
      world,
      grid,
      zones,
      zoneRaster: rasterizeZones(zones.positions, zones.count, 4, grid),
      populationField: createSmoothField(random, 4, 3),
      centerX: 0.3 + 0.4 * random()
    };
  },
  build(builder, {world, grid, zones, zoneRaster, populationField, centerX}, values) {
    const cellCount = grid.width * grid.height;
    const mesh = uploadQuadZones(builder, zones);
    const hazard = builder.allocate('hazard', 'float32', cellCount);
    const population = builder.allocate('population', 'float32', cellCount);
    const threshold = builder.allocate('threshold', 'float32', 1);
    const weightMeaning = values.weightMeaning === 1 ? 'density' : 'count';
    const recipe = addExposureByZoneRecipe(builder.graph, {
      grid: grid.grid,
      hazard: {id: 'hazard', format: 'float32', storage: {kind: 'buffer', values: hazard.view}},
      population: {
        id: 'population',
        format: 'float32',
        storage: {kind: 'buffer', values: population.view}
      },
      weightMeaning,
      cellAreas: builder.upload(
        'cell-areas',
        'float32',
        Float32Array.from(
          {length: cellCount},
          (_, index) =>
            grid.cellSize ** 2 *
            (0.1 + 6 * Math.exp(-(((getCellCenter(grid, index)[1] - 0.5) / 0.15) ** 2)))
        )
      ).view,
      zones: builder.upload('zone-raster', 'uint32', zoneRaster).view,
      zoneCount: zones.count,
      threshold: threshold.view
    });
    const exposedShare = addReadout(builder, 'exposed-share', recipe.outputs.exposedShare);
    return {
      outputs: {hazard, exposedShare, mesh},
      update: current => {
        threshold.write(Float32Array.of(current.threshold));
        hazard.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            // A hazard that decays from a river-like valley across the map.
            const distance = Math.abs(y - (0.5 + 0.25 * Math.sin((x / world.width - centerX) * 6)));
            return Math.exp(-((distance / current.spread) ** 2) * 6);
          })
        );
        population.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            return 100 * populationField(x / world.width, y / world.height) ** 2;
          })
        );
      }
    };
  },
  rebuildOn: ['weightMeaning'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.hazard,
      columns: inputs.grid.width,
      rows: inputs.grid.height
    },
    getQuadFill(outputs.mesh, inputs.zones, outputs.exposedShare, {
      ramp: 'sequential',
      alpha: 0.6,
      range: [0, 1]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});
