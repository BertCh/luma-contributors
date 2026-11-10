// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicallyWeightedRegression fits a separate regression at every tract center. Left: the
// local slope of the first predictor, which drifts from west to east in the data. Right: the
// local R2. A small bandwidth follows the drift but is noisy; a large one flattens it.

import {
  GPUGeographicallyWeightedRegression,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {parameters} from './geographically-weighted-regression.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  DUAL_MAP_SHARE,
  getDriftingCoefficients,
  uploadRegressionMap
} from './shared/regression';
import {findTractUnder, getKernelWeight, getTractCenter, getTractOutline} from './shared/probes';
import {addTableColumn} from './shared/weights';

const LADDER_CAPACITY = 4;

export default defineOperationRun({
  parameters,
  caption: 'Local slope of the first predictor (left) and local R2 (right).',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 10, DUAL_MAP_SHARE),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      getGPUGeographicallyWeightedRegressionParameterLength(LADDER_CAPACITY)
    );
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const localR2 = builder.allocate('local-r2', 'float32', rowCount);
    builder.graph.add(
      new GPUGeographicallyWeightedRegression({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        parameters: parameterBuffer.view,
        maximumBandwidthCount: LADDER_CAPACITY,
        maximumNeighborCount: 64,
        refinement: false,
        output: {
          coefficients: coefficients.view,
          localR2: localR2.view,
          selectedBandwidth: builder.allocate('selected-bandwidth', 'float32', 2).view,
          summary: builder.allocate(
            'summary',
            'float32',
            GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
          ).view
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
        localR2
      },
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            coefficients: getDriftingCoefficients(inputs, current.drift)
          })
        );
        parameterBuffer.write(
          getGPUGeographicallyWeightedRegressionParameterValues(
            {
              bandwidths: [current.bandwidth],
              kernel: current.kernel === 0 ? 'gaussian' : 'bisquare'
            },
            LADDER_CAPACITY
          )
        );
      }
    };
  },
  // The pointer picks a tract on either map: the regression point. The ring is its kernel footprint at the
  // bandwidth, and the tracts that carry most of the kernel weight are outlined.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const offsetX = inputs.mapWidth * 1.14;
      const onRight = pointer.x > offsetX - 0.02;
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y, onRight ? offsetX : 0);
      const slope = peek(outputs.slope);
      const localR2 = peek(outputs.localR2);
      if (tract < 0 || !slope || !localR2) return {};
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
        {
          shape: 'ring',
          at: getTractCenter(inputs.tracts, tract, offsetX),
          radius: values.bandwidth,
          dashed: true
        },
        getTractOutline(inputs.tracts, tract),
        getTractOutline(inputs.tracts, tract, offsetX),
        ...heavy
          .slice(0, 12)
          .map(({tract: other}) => getTractOutline(inputs.tracts, other, 0, true))
      ];
      const trueSlope = 0.5 + values.drift * (center[0] / inputs.mapWidth);
      return {
        focus: tract,
        glyphs,
        readout: `slope ${slope[tract].toFixed(1)} (true ${trueSlope.toFixed(1)}) · R2 ${localR2[tract].toFixed(2)} · bw ${values.bandwidth}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.meshes[0], outputs.slope, {
      ramp: 'sequential',
      range: [0, 3.2]
    }),
    ...drawTractMap(inputs, outputs.meshes[1], outputs.localR2, {ramp: 'sequential', range: [0, 1]})
  ]
});
