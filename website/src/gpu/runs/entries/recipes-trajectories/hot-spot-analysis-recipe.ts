// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addHotSpotAnalysisRecipe on a raster lattice: noisy values with a high and a low cluster become
// Getis-Ord Gi* z-scores (red hot, blue cold); cells under minimumNeighbors stay empty.

import {
  addHotSpotAnalysisRecipe,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './hot-spot-analysis-recipe.parameters';

const COLUMNS = 32;
const ROWS = 18;

export default defineOperationRun({
  parameters,
  caption: 'Gi* z-scores of a noisy lattice: hot cluster red, cold cluster blue.',
  createInputs: ({random}) => ({
    noise: Float32Array.from({length: COLUMNS * ROWS}, () => random())
  }),
  build(builder, {noise}, values) {
    const cellValues = builder.allocate('values', 'float32', COLUMNS * ROWS);
    const zScores = builder.allocate('z-scores', 'float32', COLUMNS * ROWS);
    const parameterBuffer = new GPUParameterBuffer(builder.device, {
      id: 'autocorrelation-parameters',
      format: 'float32',
      length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    });
    const neighborCount = (2 * values.radius + 1) ** 2;
    addHotSpotAnalysisRecipe(builder.graph, {
      source: {
        kind: 'lattice',
        values: cellValues.view,
        width: COLUMNS,
        height: ROWS,
        radius: values.radius,
        neighborCapacity: COLUMNS * ROWS * neighborCount
      },
      parameters: parameterBuffer,
      minimumNeighbors: values.minimumNeighbors,
      outputs: {zScores: zScores.view}
    });
    return {
      outputs: {zScores},
      update: current =>
        cellValues.write(
          Float32Array.from(noise, (value, index) => {
            const x = index % COLUMNS;
            const y = Math.floor(index / COLUMNS);
            const hot = Math.exp(-(((x - 8) / 4) ** 2 + ((y - 6) / 3) ** 2));
            const cold = Math.exp(-(((x - 23) / 4) ** 2 + ((y - 12) / 3) ** 2));
            return 10 + value * 2 + current.strength * 3 * (hot - cold);
          })
        )
    };
  },
  rebuildOn: ['radius', 'minimumNeighbors'],
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.zScores,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'diverging',
      range: [-4, 4]
    }
  ]
});
