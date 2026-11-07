// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterSeasonalBreak fits a seasonal curve (one harmonic) with a trend to every cell's
// 36-month series and looks for one level shift. Where a region of the map drops in the second
// half the break is found; the picture is the BIC improvement of the best break (zero means no
// break). Minimum gain is a construction prop; shift and noise rewrite the data.

import {GPURasterSeasonalBreak} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './seasonal-break-detection.parameters';
import {createStackInput} from './support/stacks';

const SLICE_COUNT = 36;

export default defineOperationRun({
  parameters,
  rebuildOn: ['minimumBICGain'],
  caption: 'A seasonal series per cell in; the strength of its level shift out.',
  createInputs: ({random, aspect}) => createStackInput(random, aspect, 24, SLICE_COUNT),
  build(builder, inputs, values) {
    const {cellCount, sliceCount} = inputs;
    const series = builder.allocate('series', 'float32', cellCount * sliceCount);
    const output = {
      coefficients: builder.allocate('coefficients', 'float32', cellCount * 5),
      components: builder.allocate('components', 'float32', cellCount * sliceCount * 3),
      breakIndex: builder.allocate('break-index', 'uint32', cellCount),
      improvement: builder.allocate('improvement', 'float32', cellCount),
      status: builder.allocate('status', 'uint32', cellCount)
    };
    builder.graph.add(
      new GPURasterSeasonalBreak({
        cellCount,
        sliceCount,
        values: series.view,
        period: 12,
        minimumSegmentLength: 8,
        minimumBICGain: values.minimumBICGain,
        output: {
          coefficients: output.coefficients.view,
          components: output.components.view,
          breakIndex: output.breakIndex.view,
          improvement: output.improvement.view,
          status: output.status.view
        }
      })
    );
    return {
      outputs: {improvement: output.improvement},
      update: current => {
        const data = new Float32Array(cellCount * sliceCount);
        for (let cell = 0; cell < cellCount; cell++) {
          // Cells in the high part of the region field lose vegetation at month 20.
          const breaks = inputs.region[cell] > 0.55;
          for (let slice = 0; slice < sliceCount; slice++) {
            data[cell * sliceCount + slice] =
              0.5 +
              0.2 * Math.sin((slice / 12) * Math.PI * 2) -
              (breaks && slice >= 20 ? current.shift : 0) +
              current.noise * inputs.noise[cell * sliceCount + slice];
          }
        }
        series.write(data);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      values: outputs.improvement,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      smoothing: false
    }
  ]
});
