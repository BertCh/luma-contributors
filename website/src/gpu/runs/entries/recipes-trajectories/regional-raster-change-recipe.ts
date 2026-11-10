// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addRegionalRasterChangeRecipe: two slices of a raster stack differenced and summarized per zone.
// The raster is the per-cell change (red grew, blue shrank); zone fills show the mean change and
// the dots the area whose change passes the threshold.

import {addRegionalRasterChangeRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUChangeDetectionParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {getVegetationIndex} from './lib/ndvi';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {getRasterCell} from '../../pick';
import {addReadout} from './lib/readout';
import {createQuadZones, uploadQuadZones} from './lib/quads';
import {createWorldGrid, getCellCenter, rasterizeZones} from './lib/rasterize';
import {parameters} from './regional-raster-change-recipe.parameters';

const SLICE_COUNT = 4;

export default defineOperationRun({
  parameters,
  caption: 'Change in Sentinel-2 NDVI over the Dixie Fire area, averaged per zone.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const zones = createQuadZones(random, world, 6, 4);
    const cellCount = grid.width * grid.height;
    const sliceValues = Float32Array.from({length: cellCount * SLICE_COUNT}, (_, row) => {
      const cell = Math.floor(row / SLICE_COUNT);
      const slice = row % SLICE_COUNT;
      const [x, y] = getCellCenter(grid, cell);
      return getVegetationIndex(x / world.width, y / world.height, slice);
    });
    // The cell whose change snaps to the default threshold: where the pointer rests.
    const thresholdParameter = parameters[2];
    let restCell = 0;
    let restError = Infinity;
    for (let cell = 0; cell < cellCount; cell++) {
      const change = Math.abs(
        sliceValues[cell * SLICE_COUNT + parameters[1].default] -
          sliceValues[cell * SLICE_COUNT + parameters[0].default]
      );
      const error = Math.abs(
        snapToParameter(thresholdParameter, change) - thresholdParameter.default
      );
      if (error < restError) {
        restError = error;
        restCell = cell;
      }
    }
    return {
      world,
      grid,
      zones,
      zoneRaster: rasterizeZones(zones.positions, zones.count, 4, grid),
      sliceValues,
      restPoint: getCellCenter(grid, restCell)
    };
  },
  build(builder, {grid, zones, zoneRaster, sliceValues}) {
    const cellCount = grid.width * grid.height;
    const mesh = uploadQuadZones(builder, zones);
    const changed = builder.allocate('changed', 'float32', cellCount);
    const slices = builder.upload('slices', 'float32', sliceValues);
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
        changed,
        difference: addReadout(builder, 'difference', recipe.outputs.difference),
        meanChanges: addReadout(builder, 'mean-changes', recipe.outputs.meanChanges),
        changedArea: addReadout(builder, 'changed-area', recipe.outputs.changedArea),
        mesh
      },
      update: current => {
        threshold.write(Float32Array.of(current.threshold));
        // Cells whose change passes the threshold, outlined crisp over the difference.
        changed.write(
          Float32Array.from({length: cellCount}, (_, cell) =>
            Math.abs(
              sliceValues[cell * SLICE_COUNT + current.afterSlice] -
                sliceValues[cell * SLICE_COUNT + current.beforeSlice]
            ) >= current.threshold
              ? 1
              : 0
          )
        );
        parameterBuffer.write(
          getGPUChangeDetectionParameterValues({
            beforeSlice: current.beforeSlice,
            afterSlice: current.afterSlice
          })
        );
      }
    };
  },
  // The change under the pointer is the threshold: the cells beyond it are the changed area.
  interaction: {
    kind: 'level',
    hint: 'pick a change level',
    rest: ({toUnit, inputs}) => toUnit(inputs.restPoint),
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const cell = getRasterCell(
        inputs.grid.width,
        inputs.grid.height,
        [0, 0, world.width, world.height],
        pointer.x,
        pointer.y
      );
      const change = Math.abs(
        inputs.sliceValues[cell.index * SLICE_COUNT + values.afterSlice] -
          inputs.sliceValues[cell.index * SLICE_COUNT + values.beforeSlice]
      );
      const threshold = snapToParameter(findParameter(parameters, 'threshold'), change);
      const areas = peek(outputs.changedArea);
      const total = areas ? Array.from(areas).reduce((sum, area) => sum + (area || 0), 0) : 0;
      return {
        values: {threshold},
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `threshold ${threshold.toFixed(2)} NDVI · changed area ${total.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.difference,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      ramp: 'diverging',
      range: [-0.55, 0.55]
    },
    {
      layer: 'raster',
      values: outputs.changed,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      alpha: 0.4
    },
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
      radius: 8,
      range: [0, 0.12]
    }
  ]
});
