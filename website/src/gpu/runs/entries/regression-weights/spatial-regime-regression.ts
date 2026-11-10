// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialRegimeRegression fits a separate intercept and slopes in every regime (vertical bands
// of tracts). Left: the fitted value of each tract, which jumps at regime boundaries. Right: the
// coefficients regime by regime (bars) with the true values (dots).

import {GPUSpatialRegimeRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './spatial-regime-regression.parameters';
import {
  createCoefficientPanel,
  createRegressionInputs,
  createResponse,
  drawCoefficientPanel,
  drawTractMap,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';
import {findTractUnder, getTractOutline} from './shared/probes';

/** True coefficients of regime `regime`, drifting away from the base set by `contrast`. */
function getRegimeCoefficients(regime: number, contrast: number): [number, number, number] {
  return [
    TRUE_COEFFICIENTS[0] + 0.6 * regime * contrast,
    TRUE_COEFFICIENTS[1] - 1.1 * regime * contrast,
    TRUE_COEFFICIENTS[2] + 0.9 * regime * contrast
  ];
}

export default defineOperationRun({
  parameters,
  caption: 'Fitted values per regime and the per-regime coefficients against the truth.',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const regimeCount = values.regimeCount;
    const regimeOf = (tract: number) =>
      Math.min(
        regimeCount - 1,
        Math.floor((inputs.tracts.centers[tract * 2] / inputs.mapWidth) * regimeCount)
      );
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const regimes = builder.upload(
      'regimes',
      'uint32',
      Uint32Array.from({length: rowCount}, (_, tract) => regimeOf(tract))
    );
    const coefficients = builder.allocate('coefficients', 'float32', regimeCount * 3);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    builder.graph.add(
      new GPUSpatialRegimeRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        regimes: regimes.view,
        regimeCount,
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          status: builder.allocate('status', 'uint32', regimeCount).view,
          counts: builder.allocate('counts', 'uint32', regimeCount).view
        }
      })
    );
    const panel = createCoefficientPanel(
      builder,
      coefficients,
      1,
      Array.from({length: regimeCount * 3}, () => 0)
    );
    return {
      outputs: {mesh: uploadRegressionMap(builder, inputs), fitted, panel},
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: current.noise,
            coefficients: tract => getRegimeCoefficients(regimeOf(tract), current.contrast)
          })
        );
        panel.truth.write(
          Float32Array.from(
            {length: regimeCount * 3},
            (_, slot) => getRegimeCoefficients(Math.floor(slot / 3), current.contrast)[slot % 3]
          )
        );
      }
    };
  },
  rebuildOn: ['regimeCount'],
  // The pointer picks a tract: its regime band is boxed on the map, with that regime's x1 slope read from the bars.
  interaction: {
    kind: 'probe',
    hint: 'hover a regime',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const estimates = peek(outputs.panel.estimates);
      if (tract < 0 || !estimates) return {};
      const regimeCount = values.regimeCount;
      const regime = Math.min(
        regimeCount - 1,
        Math.floor((inputs.tracts.centers[tract * 2] / inputs.mapWidth) * regimeCount)
      );
      const width = inputs.mapWidth / regimeCount;
      const truth = getRegimeCoefficients(regime, values.contrast)[1];
      return {
        focus: regime,
        glyphs: [
          {shape: 'box', minimum: [regime * width, 0.02], maximum: [(regime + 1) * width, 0.98]},
          getTractOutline(inputs.tracts, tract)
        ],
        readout: `regime ${regime + 1} of ${regimeCount} · x1 slope ${estimates[regime * 3 + 1].toFixed(2)} (true ${truth.toFixed(2)})`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential', range: [-2, 4]}),
    ...drawCoefficientPanel(outputs.panel)
  ]
});
