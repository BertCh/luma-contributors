// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterArithmetic on two synthetic bands (near-infrared and red): the output raster is the
// per-cell result of the chosen operation. Sliders write the operation's per-frame parameter
// buffer, so changing them re-runs the kernel without recompiling.

import {
  getGPURasterArithmeticParameterValues,
  GPURasterArithmetic,
  type GPURasterBinaryOperation
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {parameters} from './raster-arithmetic.parameters';

const OPERATIONS: GPURasterBinaryOperation[] = [
  'normalizedDifference',
  'subtract',
  'multiply',
  'maximum',
  'absoluteDifference'
];

export default defineOperationRun({
  parameters,
  caption: 'NIR and red bands in; the arithmetic result out, per cell.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery, values) {
    const cellCount = imagery.values.length;
    const nearInfrared = builder.upload('nir', 'float32', imagery.nearInfrared);
    const red = builder.upload('red', 'float32', imagery.red);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPURasterArithmetic({
        cellCount,
        a: nearInfrared.view,
        b: red.view,
        parameters: parameterBuffer.view,
        output: {values: result.view}
      })
    );
    return {
      outputs: {nearInfrared, result},
      update: current =>
        parameterBuffer.write(
          getGPURasterArithmeticParameterValues({
            operation: OPERATIONS[Math.round(current.operation)],
            scaleB: current.scaleB,
            clampMaximum: current.clampMaximum
          })
        )
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.nearInfrared,
      columns: inputs.columns,
      rows: inputs.rows
    },
    {
      layer: 'raster',
      values: outputs.result,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: values.operation === 0 ? 'diverging' : 'sequential',
      smoothing: true
    }
  ]
});
