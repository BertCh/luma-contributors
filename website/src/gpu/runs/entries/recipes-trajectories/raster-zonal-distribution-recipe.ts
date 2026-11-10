// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRasterZonalDistributionRecipe: the distribution of raster cell values inside each zone. The
// raster shows through muted; zones are filled by the chosen quantile of their cells.

import {addRasterZonalDistributionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {getBuiltFraction} from './lib/land-cover';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {findZoneAt, getZoneRing} from './lib/zone-lookup';
import {parameters} from './raster-zonal-distribution-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Quantiles of WorldCover built-up fraction with adjustable noise in each zone.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 7, 4);
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
              current.relief * 100 * getBuiltFraction(x / world.width, y / world.height) +
              current.roughness * 40 * noise[index]
            );
          })
        );
      }
    };
  },
  // x is the probability of the quantile; the zone under the pointer reports its own value.
  interaction: {
    kind: 'level',
    hint: 'slide the quantile',
    rest: ({toUnit, inputs}) => [
      getFractionOf(findParameter(parameters, 'probability'), parameters[0].default),
      toUnit([inputs.zones.centers[0], inputs.zones.centers[1]])[1]
    ],
    resolve(pointer, {inputs, outputs, peek}) {
      const probability = getParameterAt(findParameter(parameters, 'probability'), pointer.unitX);
      const zone = findZoneAt(inputs.zones, pointer.x, pointer.y);
      const quantiles = peek(outputs.quantiles);
      return {
        values: {probability},
        glyphs: [{shape: 'polyline', points: getZoneRing(inputs.zones, zone), closed: true}],
        readout: `probabilities ${probability.toFixed(2)} · zone ${zone} q = ${quantiles ? quantiles[zone].toFixed(0) : '...'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.band,
      columns: inputs.grid.width,
      rows: inputs.grid.height
    },
    getQuadFill(outputs.mesh, inputs.zones, outputs.quantiles, {
      ramp: 'sequential',
      alpha: 0.6,
      range: [0, 120]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});
