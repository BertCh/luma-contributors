// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeneralizedGeographicallyWeightedRegression fits a local Poisson or logistic model at every
// tract center. Left: the local slope of the first predictor, which drifts from west to east in
// the data. Right: the fitted mean (counts) or probability.

import {GPUGeneralizedGeographicallyWeightedRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {parameters} from './generalized-gwr.parameters';
import {
  createRegressionInputs,
  drawTractMap,
  DUAL_MAP_SHARE,
  uploadRegressionMap
} from './shared/regression';
import {findTractUnder, getKernelWeight, getTractCenter, getTractOutline} from './shared/probes';
import {addTableColumn} from './shared/weights';

/** Samples a Poisson count by inversion with a fixed uniform draw. */
function samplePoisson(mean: number, uniform: number): number {
  let probability = Math.exp(-mean);
  let cumulative = probability;
  let count = 0;
  while (uniform > cumulative && count < 60) {
    count++;
    probability *= mean / count;
    cumulative += probability;
  }
  return count;
}

export default defineOperationRun({
  parameters,
  caption: 'Local slope of the first predictor (left) and fitted mean or probability (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 10, DUAL_MAP_SHARE),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const isPoisson = values.family === 0;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    builder.graph.add(
      new GPUGeneralizedGeographicallyWeightedRegression({
        family: isPoisson ? 'poisson' : 'logistic',
        kernel: values.kernel === 0 ? 'gaussian' : 'bisquare',
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        parameters: parameterBuffer.view,
        maximumIterations: 30,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: {
            codes: builder.allocate('codes', 'uint32', rowCount).view,
            iterationCount: builder.allocate('iteration-count', 'uint32', rowCount).view
          }
        }
      })
    );
    return {
      outputs: {
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * 1.14)
        ],
        slope: addTableColumn(builder, 'local-slope', coefficients, 3, 1, rowCount),
        fitted,
        response
      },
      update: current => {
        response.write(
          Float32Array.from({length: rowCount}, (_, tract) => {
            const east = inputs.tracts.centers[tract * 2] / inputs.mapWidth;
            const slope = 0.2 + current.drift * east;
            const linear =
              (isPoisson ? 0.3 : -0.8) +
              slope * inputs.predictors[tract * 2] * (isPoisson ? 1 : 2.5) -
              0.8 * inputs.predictors[tract * 2 + 1];
            return isPoisson
              ? samplePoisson(Math.exp(linear), inputs.uniforms[tract])
              : inputs.uniforms[tract] < 1 / (1 + Math.exp(-linear))
                ? 1
                : 0;
          })
        );
        parameterBuffer.write(Float32Array.of(current.bandwidth, 1e-4));
      }
    };
  },
  rebuildOn: ['family', 'kernel'],
  // The pointer picks a tract on either map: the regression point, its kernel footprint at the bandwidth
  // and the tracts that carry most of the kernel weight.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const offsetX = inputs.mapWidth * 1.14;
      const onRight = pointer.x > offsetX - 0.02;
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y, onRight ? offsetX : 0);
      const slope = peek(outputs.slope);
      const fitted = peek(outputs.fitted);
      if (tract < 0 || !slope || !fitted) return {};
      const center = getTractCenter(inputs.tracts, tract);
      const kernel = values.kernel === 0 ? 'gaussian' : 'bisquare';
      const heavy: {tract: number; weight: number}[] = [];
      for (let other = 0; other < inputs.tracts.count; other++) {
        if (other === tract) continue;
        const weight = getKernelWeight(
          kernel,
          Math.hypot(
            inputs.tracts.centers[other * 2] - center[0],
            inputs.tracts.centers[other * 2 + 1] - center[1]
          ),
          values.bandwidth
        );
        if (weight > 0.5) heavy.push({tract: other, weight});
      }
      heavy.sort((first, second) => second.weight - first.weight);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: center, radius: values.bandwidth, dashed: true},
        getTractOutline(inputs.tracts, tract),
        getTractOutline(inputs.tracts, tract, offsetX),
        ...heavy
          .slice(0, 12)
          .map(({tract: other}) => getTractOutline(inputs.tracts, other, 0, true))
      ];
      return {
        focus: tract,
        glyphs,
        readout: `slope ${slope[tract].toFixed(1)} · ${values.family === 0 ? 'mean' : 'p'} ${fitted[tract].toFixed(2)} · bw ${values.bandwidth}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const isPoisson = values.family === 0;
    const slopeRange = isPoisson ? ([0, 2] as const) : ([0, 4] as const);
    const fittedRange = isPoisson ? ([0, 6] as const) : ([0, 1] as const);
    return [
      ...drawTractMap(inputs, outputs.meshes[0], outputs.slope, {
        ramp: 'sequential',
        range: slopeRange
      }),
      ...drawTractMap(inputs, outputs.meshes[1], outputs.fitted, {
        ramp: 'sequential',
        range: fittedRange
      }),
      // The observed counts or 0/1 outcomes at the tract centers, on the fitted scale.
      {
        layer: 'dots',
        positions: outputs.meshes[1].centers,
        values: outputs.response,
        ramp: 'sequential',
        range: fittedRange,
        radius: 3
      }
    ];
  }
});
