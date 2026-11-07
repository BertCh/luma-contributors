// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterConditional: where(elevation compared to a threshold, near-infrared, red). The
// comparison and thresholds are per-frame parameters, so sliders re-run without recompiling.

import {
  getGPURasterConditionalParameterValues,
  GPURasterConditional,
  type GPURasterComparison
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {parameters} from './raster-conditional.parameters';
import {drawRaster} from './support/bands';

const COMPARISONS: GPURasterComparison[] = ['<', '<=', '>', '>=', 'between'];

export default defineOperationRun({
  parameters,
  caption: 'Where the condition holds take NIR, elsewhere red.',
  createInputs: ({random, world}) => createImageryBands(random, world),
  build(builder, imagery) {
    const cellCount = imagery.values.length;
    const condition = builder.upload('condition', 'float32', imagery.values);
    const a = builder.upload('a', 'float32', imagery.nearInfrared);
    const b = builder.upload('b', 'float32', imagery.red);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const result = builder.allocate('result', 'float32', cellCount);
    builder.graph.add(
      new GPURasterConditional({
        cellCount,
        conditionValues: condition.view,
        a: a.view,
        b: b.view,
        parameters: parameterBuffer.view,
        output: {values: result.view}
      })
    );
    return {
      outputs: {condition, result},
      update: current =>
        parameterBuffer.write(
          getGPURasterConditionalParameterValues({
            comparison: COMPARISONS[Math.round(current.comparison)],
            threshold: current.threshold,
            upperThreshold: Math.max(current.threshold, current.upperThreshold)
          })
        )
    };
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.condition, {context: true}),
    drawRaster(inputs, outputs.result, {ramp: 'sequential', range: [0, 0.6]})
  ]
});
