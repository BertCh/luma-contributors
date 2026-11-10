// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addStraightLineCatchmentsRecipe: every cell allocated to its nearest facility by straight-line
// distance. Colors are catchments, white dots the facilities; cells farther than maximumDistance
// stay empty.

import {addStraightLineCatchmentsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUDistanceFieldParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {getRainStations} from '../../../data/point-observations';
import {defineOperationRun} from '../../define-run';
import {createWorldGrid} from './lib/rasterize';
import {parameters} from './straight-line-catchments-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Catchments of selected NOAA climate stations in the Carolinas.',
  createInputs: ({world, random}) => {
    const homes = getRainStations(random, world, 14, true).positions;
    return {grid: createWorldGrid(world), homes, facilities: Float32Array.from(homes)};
  },
  build(builder, inputs) {
    const {grid, facilities} = inputs;
    const cellCount = grid.width * grid.height;
    const parameterBuffer = builder.allocate('distance-parameters', 'float32', 8);
    const facilityCount = builder.allocate('facility-count', 'uint32', 1);
    const allocation = builder.allocate('allocation', 'uint32', cellCount);
    const distances = builder.allocate('distances', 'float32', cellCount);
    const facilityPositions = builder.upload('facilities', 'float32x2', facilities);
    const catchmentValues = builder.upload(
      'values',
      'float32',
      new Float32Array(cellCount).fill(1)
    );
    addStraightLineCatchmentsRecipe(builder.graph, {
      grid: grid.grid,
      parameters: parameterBuffer.view,
      facilities: {kind: 'points', positions: facilityPositions.view},
      facilityCount: facilityCount.view,
      values: catchmentValues.view,
      outputs: {allocation: allocation.view, distances: distances.view}
    });
    return {
      outputs: {allocation, facilityPositions},
      update: current => {
        facilityCount.write(Uint32Array.of(current.facilityCount));
        facilityPositions.write(inputs.facilities);
        parameterBuffer.write(
          getGPUDistanceFieldParameterValues({maximumDistance: current.maximumDistance}, grid.grid)
        );
      }
    };
  },
  // The pointer carries the nearest station: the catchments re-form around it.
  interaction: {
    kind: 'drag',
    hint: 'move a station',
    rest: ({toUnit, inputs}) => toUnit([inputs.homes[0], inputs.homes[1]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      let station = 0;
      for (let candidate = 1; candidate < values.facilityCount; candidate++) {
        if (
          Math.hypot(
            pointer.x - inputs.homes[candidate * 2],
            pointer.y - inputs.homes[candidate * 2 + 1]
          ) <
          Math.hypot(
            pointer.x - inputs.homes[station * 2],
            pointer.y - inputs.homes[station * 2 + 1]
          )
        ) {
          station = candidate;
        }
      }
      inputs.facilities = Float32Array.from(inputs.homes);
      inputs.facilities[station * 2] = pointer.x;
      inputs.facilities[station * 2 + 1] = pointer.y;
      const allocation = peek(outputs.allocation);
      let covered = 0;
      if (allocation) for (const owner of allocation) if (owner === station) covered++;
      return {
        rerun: true,
        glyphs: [
          {shape: 'ring', at: [pointer.x, pointer.y], radius: values.maximumDistance, dashed: true},
          {shape: 'crosshair', at: [pointer.x, pointer.y]}
        ],
        readout: `station ${station} · catchment ${allocation ? Math.round((covered / allocation.length) * 100) : '...'}% of cells`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.allocation,
      columns: inputs.grid.width,
      rows: inputs.grid.height
    },
    {
      layer: 'dots',
      positions: outputs.facilityPositions,
      count: values.facilityCount,
      radius: 6,
      color: [1, 1, 1, 1]
    }
  ]
});
