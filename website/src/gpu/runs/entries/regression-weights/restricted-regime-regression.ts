// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURestrictedRegimeRegression fits two regimes (west and east) with chosen coefficients shared.
// Left: fitted values per tract. Right: the restricted model's coefficients (a shared coefficient is
// one bar, a regime-specific one is two) and below them the Chow F statistic against the fully
// varying model, which grows when a shared coefficient is truly different between regimes.

import {GPURestrictedRegimeRegression} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createLegend, mergeText} from '../../text';
import {parameters} from './restricted-regime-regression.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawSignedBars,
  drawTractMap,
  drawZeroLine,
  MAP_SHARE,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

const COEFFICIENT_RECTANGLE = [MAP_SHARE + 0.08, 0.42, 0.97, 0.9] as const;

export default defineOperationRun({
  parameters,
  caption: 'Restricted regimes: coefficients (right), Chow F below, fitted map (left).',
  createInputs: ({random, world}) => {
    const inputs = createRegressionInputs(random, world);
    // The west-east break the model assumes; the data's true break is at the middle of the map.
    return {...inputs, boundary: inputs.mapWidth / 2};
  },
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const common = [
      values.commonIntercept === 1,
      values.commonFirst === 1,
      values.commonSecond === 1
    ];
    const coefficientCount = common.reduce((total, isCommon) => total + (isCommon ? 1 : 2), 0);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const regimes = builder.allocate('regimes', 'uint32', rowCount);
    const coefficients = builder.allocate('coefficients', 'float32', coefficientCount);
    const fitted = builder.allocate('fitted', 'float32', rowCount);
    const chow = builder.allocate('chow', 'float32', 3);
    builder.graph.add(
      new GPURestrictedRegimeRegression({
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        regimes: regimes.view,
        regimeCount: 2,
        common,
        output: {
          coefficients: coefficients.view,
          covariance: builder.allocate('covariance', 'float32', coefficientCount ** 2).view,
          fitted: fitted.view,
          chow: chow.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        fitted,
        coefficients,
        regimes,
        zeros: builder.upload('zeros', 'float32', new Float32Array(64)),
        chowStatistic: addTableColumn(builder, 'chow', chow, 3, 0, 1)
      },
      update: current => {
        regimes.write(
          Uint32Array.from({length: rowCount}, (_, tract) =>
            inputs.tracts.centers[tract * 2] < inputs.boundary ? 0 : 1
          )
        );
        response.write(
          createResponse(inputs, {
            noise: 0.4,
            coefficients: tract =>
              inputs.regimes[tract] === 0
                ? TRUE_COEFFICIENTS
                : [
                    TRUE_COEFFICIENTS[0] + 0.6 * current.contrast,
                    TRUE_COEFFICIENTS[1] - 1.1 * current.contrast,
                    TRUE_COEFFICIENTS[2] + 0.9 * current.contrast
                  ]
          })
        );
      }
    };
  },
  rebuildOn: ['commonIntercept', 'commonFirst', 'commonSecond'],
  // The pointer along x moves the west-east break the model assumes; the Chow F peaks where it meets the true break.
  interaction: {
    kind: 'level',
    hint: 'move the regime break',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      // The canvas width spans the allowed breaks, so every position moves the line.
      const boundary = Math.round((0.15 + 0.7 * pointer.unitX) * inputs.mapWidth * 50) / 50;
      const changed = boundary !== inputs.boundary;
      inputs.boundary = boundary;
      const chow = peek(outputs.chowStatistic);
      return {
        rerun: changed,
        glyphs: [{shape: 'curtain', x: boundary}],
        readout: `break at x ${(boundary / inputs.mapWidth).toFixed(2)} · Chow F ${chow ? chow[0].toFixed(0) : '-'}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    const titles = mergeText({
      labels: [
        {
          text: 'coefficients (a shared one is one bar)',
          at: [(MAP_SHARE + 0.08) * world.width, 0.9 * world.height],
          anchor: 'bottom-left',
          offset: [0, 3],
          role: 'label',
          priority: 4
        },
        {
          text: 'Chow F',
          at: [(MAP_SHARE + 0.08) * world.width, 0.34 * world.height],
          anchor: 'bottom-left',
          offset: [0, 3],
          role: 'label',
          priority: 4
        }
      ]
    });
    return [
      ...drawTractMap(inputs, outputs.mesh, outputs.fitted, {ramp: 'sequential', range: [-2, 4]}),
      // The two regimes the model assumes, tinted over the fitted values; the tint follows the break.
      {
        layer: 'polygons',
        positions: outputs.mesh.vertices,
        indices: outputs.mesh.indices,
        owners: outputs.mesh.owners,
        values: outputs.regimes,
        triangleCount: inputs.tracts.triangleCount,
        ramp: 'categorical',
        alpha: 0.4
      },
      ...drawSignedBars(outputs.coefficients, COEFFICIENT_RECTANGLE, 3),
      drawZeroLine(outputs.zeros, COEFFICIENT_RECTANGLE),
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.chowStatistic,
        color: [0.95, 0.6, 0.3, 0.95],
        range: [0, 60],
        rectangle: [MAP_SHARE + 0.08, 0.08, 0.97, 0.34]
      },
      {layer: 'text', ...titles},
      {
        layer: 'text',
        ...createLegend({
          corner: 'top-left',
          title: 'regime tint',
          entries: [
            {label: 'west', swatch: 0},
            {label: 'east', swatch: 1}
          ]
        })
      }
    ];
  }
});
