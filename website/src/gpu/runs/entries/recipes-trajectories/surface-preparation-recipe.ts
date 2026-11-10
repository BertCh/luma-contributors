// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSurfacePreparationRecipe: gaps in a noisy surface filled from nearby cells, then a rank filter
// smooths what is left. The muted raster is the input with its holes; the output is the prepared
// surface.

import {addSurfacePreparationRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getVegetationIndex} from './lib/ndvi';
import {defineOperationRun} from '../../define-run';
import {createSmoothField} from '../../inputs';
import {createWorldGrid, getCellCenter} from './lib/rasterize';
import {parameters} from './surface-preparation-recipe.parameters';

const NO_DATA = -9999;

export default defineOperationRun({
  parameters,
  caption: 'Gap filling and rank filtering on a July 2021 Sentinel-2 NDVI scene.',
  createInputs: ({random, world}) => {
    const grid = createWorldGrid(world);
    const cellCount = grid.width * grid.height;
    return {
      world,
      grid,
      gapField: createSmoothField(random, 4, 5),
      speckle: Float32Array.from({length: cellCount}, () => random() - 0.5)
    };
  },
  build(builder, {world, grid, gapField, speckle}, values) {
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
          const value = getVegetationIndex(x / world.width, y / world.height, 0);
          return Number.isFinite(value) ? value + speckle[index] * 0.1 : NO_DATA;
        });
        input.write(holed);
        shown.write(holed.map(value => (value === NO_DATA ? Number.NaN : value)));
      }
    };
  },
  rebuildOn: ['maximumDistance', 'radius', 'quantile'],
  // A curtain between the holed input (left) and the prepared surface (right).
  interaction: {
    kind: 'swipe',
    hint: 'swipe input and prepared',
    rest: [0.5, 0.5],
    resolve: (pointer, {values}) => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: `input | prepared · fill ${values.maximumDistance} cells, rank r ${values.radius} q ${values.quantile}`
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      {
        layer: 'raster',
        values: outputs.shown,
        columns: inputs.grid.width,
        rows: inputs.grid.height,
        range: [0, 1],
        clip: {side: 'left', x: curtain}
      },
      {
        layer: 'raster',
        values: outputs.result,
        columns: inputs.grid.width,
        rows: inputs.grid.height,
        range: [0, 1],
        smoothing: true,
        clip: {side: 'right', x: curtain}
      }
    ];
  }
});
