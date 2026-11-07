// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTemporal composites twelve dates of a vegetation index per cell. Clouds make the
// per-date values invalid (QA validity 0), and the composite skips them: mean and median average
// what remains, maximum is the classic max-value composite that suppresses clouds. The operation
// is a construction prop; cloud cover rewrites the validity mask.

import {GPURasterTemporal} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './temporal-composite.parameters';
import {createStackInput} from './support/stacks';

const OPERATIONS = ['mean', 'minimum', 'maximum', 'median'] as const;
const SLICE_COUNT = 12;

export default defineOperationRun({
  parameters,
  rebuildOn: ['operation'],
  caption: 'Twelve cloudy dates in; one composite value per cell out.',
  createInputs: ({random, aspect}) => createStackInput(random, aspect, 48, SLICE_COUNT),
  build(builder, inputs, values) {
    const {cellCount, sliceCount} = inputs;
    const stackValues = new Float32Array(cellCount * sliceCount);
    for (let cell = 0; cell < cellCount; cell++) {
      for (let slice = 0; slice < sliceCount; slice++) {
        // Seasonal greenness curve per cell, scaled by its baseline level, plus noise.
        const season = 0.5 + 0.4 * Math.sin(((slice - 3) / sliceCount) * Math.PI * 2);
        stackValues[cell * sliceCount + slice] =
          0.15 + 0.7 * inputs.level[cell] * season + 0.06 * inputs.noise[cell * sliceCount + slice];
      }
    }
    const stack = builder.upload('stack', 'float32', stackValues);
    const validity = builder.allocate('validity', 'uint32', cellCount * sliceCount);
    const composite = builder.allocate('composite', 'float32', cellCount);
    const counts = builder.allocate('counts', 'uint32', cellCount);
    const chosenSlice = builder.allocate('chosen-slice', 'uint32', cellCount);
    const status = builder.allocate('status', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterTemporal({
        cellCount,
        sliceCount,
        values: stack.view,
        validity: validity.view,
        operation: OPERATIONS[Math.round(values.operation)],
        output: {
          values: composite.view,
          counts: counts.view,
          chosenSlice: chosenSlice.view,
          status: status.view
        }
      })
    );
    const firstDate = builder.upload(
      'first-date',
      'float32',
      Float32Array.from({length: cellCount}, (_, cell) => stackValues[cell * sliceCount])
    );
    return {
      outputs: {firstDate, composite},
      update: current =>
        validity.write(
          Uint32Array.from(inputs.clouds, cloud => (cloud < current.cloudCover ? 0 : 1))
        )
    };
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.firstDate,
      columns: inputs.columns,
      rows: inputs.rows,
      range: [0, 0.9]
    },
    {
      layer: 'raster',
      values: outputs.composite,
      columns: inputs.columns,
      rows: inputs.rows,
      range: [0, 0.9],
      smoothing: true
    }
  ]
});
