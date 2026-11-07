// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSurfacePreparationRecipe: gaps in a noisy surface filled from nearby cells, then a rank filter
// smooths what is left. The muted raster is the input with its holes; the output is the prepared
// surface.

import {addSurfacePreparationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {createWorldGrid, getCellCenter} from './lib/rasterize';
import {parameters} from './surface-preparation-recipe.parameters';

const NO_DATA = -9999;

export default defineOperationRun({
  parameters,
  caption: 'Holes filled and speckle filtered: the prepared surface.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const cellCount = grid.width * grid.height;
    return {
      world,
      grid,
      surface: createSmoothField(random, 4, 3),
      gapField: createSmoothField(random, 4, 5),
      speckle: Float32Array.from({length: cellCount}, () => random() - 0.5)
    };
  },
  build(builder, {world, grid, surface, gapField, speckle}, values) {
    const cellCount = grid.width * grid.height;
    const input = builder.allocate('input', 'float32', cellCount);
    const shown = builder.allocate('shown', 'float32', cellCount);
    const result = builder.allocate('values', 'float32', cellCount);
    const validity = builder.allocate('validity', 'uint32', cellCount);
    const originalValidity = builder.allocate('original-validity', 'uint32', cellCount);
    const filled = builder.allocate('filled', 'uint32', cellCount);
    const filtered = builder.allocate('filtered', 'uint32', cellCount);
    const unresolvedCount = builder.allocate('unresolved', 'uint32', 1);
    addSurfacePreparationRecipe(builder.graph, {
      grid: grid.grid,
      input: {
        id: 'input',
        format: 'float32',
        storage: {kind: 'buffer', values: input.view},
        noDataValue: NO_DATA
      },
      maximumDistance: values.maximumDistance,
      method: 'inverse-distance',
      filter:
        values.radius > 0
          ? {
              radius: values.radius,
              quantile: values.quantile,
              borderMode: 'nodata',
              noDataPolicy: 'ignore'
            }
          : undefined,
      outputs: {
        values: result.view,
        validity: validity.view,
        originalValidity: originalValidity.view,
        filled: filled.view,
        filtered: filtered.view,
        unresolvedCount: unresolvedCount.view
      }
    });
    return {
      outputs: {shown, result},
      update: current => {
        const holed = Float32Array.from({length: cellCount}, (_, index) => {
          const [x, y] = getCellCenter(grid, index);
          if (gapField(x / world.width, y / world.height) > current.gaps) return NO_DATA;
          return surface(x / world.width, y / world.height) + speckle[index] * 0.25;
        });
        input.write(holed);
        shown.write(holed.map(value => (value === NO_DATA ? Number.NaN : value)));
      }
    };
  },
  rebuildOn: ['maximumDistance', 'radius', 'quantile'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.shown,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      range: [0, 1]
    },
    {
      layer: 'raster',
      values: outputs.result,
      columns: inputs.grid.width,
      rows: inputs.grid.height,
      range: [0, 1],
      smoothing: true
    }
  ]
});
