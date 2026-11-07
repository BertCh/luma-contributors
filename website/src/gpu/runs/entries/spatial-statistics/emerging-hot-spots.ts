// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUEmergingHotSpots: Gi* hot spot z-scores in every time slice of a space-time cube, then a
// Mann-Kendall trend per cell. The picture is the trend z-score per cell: red where the hot spot is
// intensifying (the blob that grows over the 12 slices), blue where it fades. The muted layer is
// the last slice of the input cube.

import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './emerging-hot-spots.parameters';

const WIDTH = 48;
const HEIGHT = 27;
const SLICES = 12;

export default defineOperationRun({
  parameters,
  caption: 'Trend of the hot spot z-score per cell over twelve time slices.',
  createInputs: ({random}) => ({
    noise: Float32Array.from({length: WIDTH * HEIGHT * SLICES}, () => random())
  }),
  build(builder, inputs) {
    const cellCount = WIDTH * HEIGHT;
    const cube = builder.allocate('cube', 'float32', cellCount * SLICES);
    const lastSlice = builder.allocate('last-slice', 'float32', cellCount);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
    );
    const giZScores = builder.allocate('gi-z-scores', 'float32', cellCount * SLICES);
    const trendZ = builder.allocate('trend-z', 'float32', cellCount);
    const trendP = builder.allocate('trend-p', 'float32', cellCount);
    const trendS = builder.allocate('trend-s', 'sint32', cellCount);
    const category = builder.allocate('category', 'uint32', cellCount);
    const hotSliceCount = builder.allocate('hot-slice-count', 'uint32', cellCount);
    const coldSliceCount = builder.allocate('cold-slice-count', 'uint32', cellCount);
    builder.graph.add(
      new GPUEmergingHotSpots({
        values: cube.view,
        gridWidth: WIDTH,
        gridHeight: HEIGHT,
        sliceCount: SLICES,
        maximumRadius: 3,
        parameters: parameterBuffer.view,
        output: {
          giZScores: giZScores.view,
          trendZ: trendZ.view,
          trendP: trendP.view,
          trendS: trendS.view,
          category: category.view,
          hotSliceCount: hotSliceCount.view,
          coldSliceCount: coldSliceCount.view
        }
      })
    );
    return {
      outputs: {trendZ, lastSlice},
      update: current => {
        // Counts around 10; a hot spot at the left grows over time, one at the right fades.
        const data = new Float32Array(cellCount * SLICES);
        const last = new Float32Array(cellCount);
        for (let cell = 0; cell < cellCount; cell++) {
          const column = cell % WIDTH;
          const row = Math.floor(cell / WIDTH);
          for (let slice = 0; slice < SLICES; slice++) {
            const progress = slice / (SLICES - 1);
            const growing = Math.exp(-(((column - 14) / 5) ** 2 + ((row - 12) / 4) ** 2));
            const fading = Math.exp(-(((column - 34) / 5) ** 2 + ((row - 14) / 4) ** 2));
            const value =
              8 +
              inputs.noise[cell * SLICES + slice] * 4 +
              current.trend * 14 * (growing * progress + fading * (1 - progress)) +
              (1 - current.trend) * 7 * (growing + fading) * 0.5;
            data[cell * SLICES + slice] = value;
            if (slice === SLICES - 1) last[cell] = value;
          }
        }
        cube.write(data);
        lastSlice.write(last);
        parameterBuffer.write(
          getGPUEmergingHotSpotsParameterValues({
            radius: current.radius,
            temporalWindow: current.temporalWindow
          })
        );
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'raster', context: true, values: outputs.lastSlice, columns: WIDTH, rows: HEIGHT},
    {
      layer: 'raster',
      values: outputs.trendZ,
      columns: WIDTH,
      rows: HEIGHT,
      ramp: 'diverging',
      range: [-4, 4],
      alpha: 0.9
    }
  ]
});
