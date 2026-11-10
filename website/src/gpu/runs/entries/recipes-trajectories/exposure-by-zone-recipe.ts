// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addExposureByZoneRecipe: share of each zone's population living where the hazard reaches the
// threshold. The hazard raster is muted underneath; zones are filled by their exposed share.

import {addExposureByZoneRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {getBuiltFraction} from './lib/land-cover';
import {addReadout} from './lib/readout';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './exposure-by-zone-recipe.parameters';

/** Hazard at a world position: it decays from a river-like valley across the map. */
function getHazardAt(
  x: number,
  y: number,
  worldWidth: number,
  centerX: number,
  spread: number
): number {
  const distance = Math.abs(y - (0.5 + 0.25 * Math.sin((x / worldWidth - centerX) * 6)));
  return Math.exp(-((distance / spread) ** 2) * 6);
}

export default defineOperationRun({
  parameters,
  caption: 'Modeled flood exposure weighted by the built-up fraction in Hyde Park.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 8, 5);
    return {
      world,
      grid,
      zones,
      zoneRaster: rasterizeZones(zones.positions, zones.count, 4, grid),
      centerX: 0.3 + 0.4 * random()
    };
  },
  build(builder, {world, grid, zones, zoneRaster, centerX}, values) {
    const cellCount = grid.width * grid.height;
    const mesh = uploadQuadZones(builder, zones);
    const hazard = builder.allocate('hazard', 'float32', cellCount);
    const population = builder.allocate('population', 'float32', cellCount);
    const exposed = builder.allocate('exposed', 'float32', cellCount);
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
      outputs: {hazard, exposed, exposedShare, mesh},
      update: current => {
        threshold.write(Float32Array.of(current.threshold));
        const hazardValues = Float32Array.from({length: cellCount}, (_, index) => {
          const [x, y] = getCellCenter(grid, index);
          return getHazardAt(x, y, world.width, centerX, current.spread);
        });
        hazard.write(hazardValues);
        // The cells at or above the threshold, drawn crisp over the zone fill.
        exposed.write(hazardValues.map(value => (value >= current.threshold ? 1 : 0)));
        population.write(
          Float32Array.from({length: cellCount}, (_, index) => {
            const [x, y] = getCellCenter(grid, index);
            return 100 * getBuiltFraction(x / world.width, y / world.height);
          })
        );
      }
    };
  },
  rebuildOn: ['weightMeaning'],
  // The hazard under the pointer is the threshold: cells at or above it are the exposed ones.
  interaction: {
    kind: 'level',
    hint: 'pick a hazard level',
    rest: ({toUnit, inputs, world}) => {
      // Half the peak hazard sits spread * sqrt(ln 2 / 6) above the valley line.
      const x = world.width / 2;
      const valley = 0.5 + 0.25 * Math.sin((x / world.width - inputs.centerX) * 6);
      return toUnit([x, valley + parameters[2].default * Math.sqrt(Math.LN2 / 6)]);
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const hazard = getHazardAt(pointer.x, pointer.y, world.width, inputs.centerX, values.spread);
      const threshold = snapToParameter(findParameter(parameters, 'threshold'), hazard);
      const shares = peek(outputs.exposedShare);
      const largest = shares ? Math.max(0, ...Array.from(shares).filter(Number.isFinite)) : 0;
      return {
        values: {threshold},
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `threshold ${threshold.toFixed(2)} · exposed share up to ${Math.round(largest * 100)}%`
      };
    }
  },
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
      alpha: 0.4,
      range: [0, 1]
    }),
    {
      layer: 'raster',
      values: outputs.exposed,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      alpha: 0.7
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});
