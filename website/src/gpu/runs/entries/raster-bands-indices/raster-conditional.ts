// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterConditional: where(NDVI compared to a threshold, near-infrared, red). The
// comparison and thresholds are per-frame parameters, so sliders re-run without recompiling.

import {
  getGPURasterConditionalParameterValues,
  GPURasterConditional,
  type GPURasterComparison
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {bindAxes, formatNumber} from '../../interaction';
import {parameters} from './raster-conditional.parameters';
import {drawRaster} from './support/bands';

const HELD_CATEGORIES = [
  [0, 0, 0, 0],
  [1, 1, 1, 0.28]
] as const;

const COMPARISONS: GPURasterComparison[] = ['<', '<=', '>', '>=', 'between'];

const PAD = bindAxes(parameters, {
  kind: 'level',
  hint: 'move the NDVI thresholds',
  x: 'threshold',
  y: 'upperThreshold'
});

export default defineOperationRun({
  parameters,
  caption: 'Where NDVI meets the threshold show NIR reflectance, elsewhere red.',
  createInputs: ({world}) => {
    const imagery = createImageryBands(world);
    const ndvi = imagery.nearInfrared.map(
      (nir, index) => (nir - imagery.red[index]) / Math.max(0.001, nir + imagery.red[index])
    );
    return {...imagery, ndvi};
  },
  build(builder, imagery) {
    const cellCount = imagery.values.length;
    const ndvi = imagery.ndvi;
    const held = builder.allocate('held', 'uint32', cellCount);
    const condition = builder.upload('condition', 'float32', ndvi);
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
      outputs: {condition, result, held},
      update: current => {
        const comparison = COMPARISONS[Math.round(current.comparison)];
        const upper = Math.max(current.threshold, current.upperThreshold);
        parameterBuffer.write(
          getGPURasterConditionalParameterValues({
            comparison,
            threshold: current.threshold,
            upperThreshold: upper
          })
        );
        // Where the condition held, for display: the cells that took the near-infrared branch.
        held.write(
          Uint32Array.from(ndvi, value => {
            switch (comparison) {
              case '<':
                return value < current.threshold ? 1 : 0;
              case '<=':
                return value <= current.threshold ? 1 : 0;
              case '>':
                return value > current.threshold ? 1 : 0;
              case '>=':
                return value >= current.threshold ? 1 : 0;
              default:
                return value >= current.threshold && value <= upper ? 1 : 0;
            }
          })
        );
      }
    };
  },
  // The pointer is a pad over the two thresholds: threshold across, upperThreshold up.
  interaction: {
    ...PAD,
    resolve: (pointer, state) => {
      const base = PAD.resolve(pointer, state);
      const current = {...state.values, ...base.values};
      const {outputs, peek} = state;
      const held = peek(outputs.held);
      let count = 0;
      if (held) for (const value of held) count += value;
      const share = held ? Math.round((100 * count) / held.length) : 0;
      return {
        ...base,
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `NDVI ${formatNumber(current.threshold)} to ${formatNumber(Math.max(current.threshold, current.upperThreshold))} | ${share}% take NIR`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawRaster(inputs, outputs.condition, {context: true}),
    drawRaster(inputs, outputs.result, {ramp: 'sequential', range: [0, 0.6]}),
    drawRaster(inputs, outputs.held, {ramp: 'categorical', categories: HELD_CATEGORIES})
  ]
});
