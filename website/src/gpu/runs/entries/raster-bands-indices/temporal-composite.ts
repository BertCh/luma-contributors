// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTemporal composites twelve dates of a vegetation index per cell. Clouds make the
// per-date values invalid (QA validity 0), and the composite skips them: mean and median average
// what remains, maximum is the classic max-value composite that suppresses clouds. The operation
// is a construction prop; cloud cover rewrites the validity mask.

import {GPURasterTemporal} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './temporal-composite.parameters';
import {getSwipeResult, getCellAt, getCurtainX} from './support/gestures';
import {createStackInput} from './support/stacks';

const OPERATIONS = ['mean', 'minimum', 'maximum', 'median'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['operation'],
  caption: 'Twelve cloudy dates in; one composite value per cell out.',
  createInputs: ({random, aspect}) => createStackInput(random, aspect, 48),
  build(builder, inputs, values) {
    const {cellCount, sliceCount} = inputs;
    const stackValues = inputs.observations;
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
    const firstDate = builder.allocate('first-date', 'float32', cellCount);
    return {
      outputs: {firstDate, composite},
      update: current => {
        validity.write(
          Uint32Array.from(inputs.clouds, (cloud, index) =>
            Number.isFinite(stackValues[index]) && cloud >= current.cloudCover ? 1 : 0
          )
        );
        // The first date as a single cloudy scene: its clouded cells are missing.
        firstDate.write(
          Float32Array.from({length: cellCount}, (_, cell) =>
            inputs.clouds[cell * sliceCount] >= current.cloudCover
              ? stackValues[cell * sliceCount]
              : Number.NaN
          )
        );
      }
    };
  },
  // The pointer is a curtain between one cloudy date and the composite of all twelve.
  interaction: {
    kind: 'swipe',
    hint: 'swipe one date and composite',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, values, world}) => {
      const {index} = getCellAt(inputs, world, pointer);
      let valid = 0;
      for (let slice = 0; slice < inputs.sliceCount; slice++) {
        const observation = index * inputs.sliceCount + slice;
        if (
          Number.isFinite(inputs.observations[observation]) &&
          inputs.clouds[observation] >= values.cloudCover
        ) {
          valid++;
        }
      }
      const readout = `${OPERATIONS[Math.round(values.operation)]}: ${valid} of 12 dates valid`;
      return getSwipeResult(pointer, readout);
    }
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = getCurtainX(interaction, world);
    const look = {
      layer: 'raster',
      columns: inputs.columns,
      rows: inputs.rows,
      range: [0, 0.9]
    } as const;
    return [
      {...look, values: outputs.firstDate, clip: {side: 'left', x: curtain}},
      {...look, values: outputs.composite, smoothing: true, clip: {side: 'right', x: curtain}}
    ];
  }
});
