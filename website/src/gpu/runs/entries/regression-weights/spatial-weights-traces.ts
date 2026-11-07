// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialWeightsTraces of rook-contiguous tract weights. Top: tr((W/s)^q) for q = 1..maximumPower
// as a line (exact, or a stochastic estimate whose noise shrinks with `probeCount`). Bottom: the
// power sums of the same operation as bars.

import {GPUSpatialWeightsTraces} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {parameters} from './spatial-weights-traces.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Power traces of W (top) and power sums (bottom), from the GPU.',
  createInputs: ({random, world}) => createTracts(random, world, 10),
  build(builder, tracts, values) {
    const weights = uploadSpatialWeights(builder, tracts.contiguity);
    const powerTraces = builder.allocate('power-traces', 'float32', values.maximumPower);
    const powerSums = builder.allocate('power-sums', 'float32', values.maximumPower);
    const standardErrors = builder.allocate('standard-errors', 'float32', values.maximumPower);
    const infinityNorm = builder.allocate('infinity-norm', 'float32', 1);
    builder.graph.add(
      new GPUSpatialWeightsTraces({
        weights,
        maximumPower: values.maximumPower,
        mode: values.mode === 0 ? 'exact' : 'stochastic',
        probeCount: values.mode === 0 ? undefined : values.probeCount,
        output: {
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          powerTraceStandardErrors: standardErrors.view,
          infinityNorm: infinityNorm.view
        }
      })
    );
    return {outputs: {powerTraces, powerSums}};
  },
  draw: ({outputs}) => [
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.powerTraces,
      rectangle: [0.07, 0.56, 0.95, 0.92],
      radius: 2.2
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.powerSums,
      ramp: 'sequential',
      rectangle: [0.07, 0.08, 0.95, 0.46]
    }
  ]
});
