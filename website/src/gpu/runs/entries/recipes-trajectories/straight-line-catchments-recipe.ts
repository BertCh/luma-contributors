// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addStraightLineCatchmentsRecipe: every cell allocated to its nearest facility by straight-line
// distance. Colors are catchments, white dots the facilities; cells farther than maximumDistance
// stay empty.

import {addStraightLineCatchmentsRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getGPUDistanceFieldParameterValues} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createWorldGrid} from './lib/rasterize';
import {parameters} from './straight-line-catchments-recipe.parameters';

const FACILITY_CAPACITY = 14;

export default defineOperationRun({
  parameters,
  caption: 'Nearest-facility catchments on a grid.',
  createInputs: ({random, world}) => ({
    grid: createWorldGrid(world),
    facilities: Float32Array.from({length: FACILITY_CAPACITY * 2}, (_, index) =>
      index % 2 === 0 ? (0.06 + 0.88 * random()) * world.width : 0.06 + 0.88 * random()
    )
  }),
  build(builder, {grid, facilities}) {
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
        parameterBuffer.write(
          getGPUDistanceFieldParameterValues({maximumDistance: current.maximumDistance}, grid.grid)
        );
      }
    };
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
