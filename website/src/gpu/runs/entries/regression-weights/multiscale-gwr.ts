// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultiscaleGeographicallyWeightedRegression gives every coefficient its own bandwidth. Left:
// the intercept surface, which truly rises from south to north (a small bandwidth follows it; a
// large one flattens it). Middle: the surface of the first slope, which truly drifts from west to
// east (a small bandwidth follows it). Right: the surface of the second slope, which is constant
// (a large bandwidth keeps it flat; a small one adds noise).

import {
  GPUMultiscaleGeographicallyWeightedRegression,
  decodeGPUMultiscaleGeographicallyWeightedRegressionStatus
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {bindAxes} from '../../interaction';
import {createLegend, mergeText} from '../../text';
import {parameters} from './multiscale-gwr.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawTractMap,
  type RegressionInputs,
  TRUE_COEFFICIENTS,
  uploadRegressionMap
} from './shared/regression';
import {addTableColumn} from './shared/weights';

const SLOPE_DRIFT = 2.5;
const THREE_MAP_SHARE = 0.32;
const MAP_GAP = 1.04;
const INTERCEPT_DRIFT = 2;
const MAP_TITLES = ['intercept', 'x1', 'x2'] as const;

/** True coefficients: a south to north intercept, a west to east first slope, a constant second slope. */
function getScaledCoefficients(
  inputs: RegressionInputs
): (tract: number) => readonly [number, number, number] {
  const {centers, count} = inputs.tracts;
  let south = Infinity;
  let north = -Infinity;
  for (let tract = 0; tract < count; tract++) {
    south = Math.min(south, centers[tract * 2 + 1]);
    north = Math.max(north, centers[tract * 2 + 1]);
  }
  return tract => [
    0.5 + INTERCEPT_DRIFT * ((centers[tract * 2 + 1] - south) / (north - south || 1)),
    0.5 + SLOPE_DRIFT * (centers[tract * 2] / inputs.mapWidth),
    TRUE_COEFFICIENTS[2]
  ];
}

export default defineOperationRun({
  parameters,
  caption:
    'Three coefficient surfaces fitted at their own scales: an intercept rising south to north, a slope drifting west to east and a constant slope.',
  createInputs: ({random, world}) => createRegressionInputs(random, world, 8, THREE_MAP_SHARE),
  build(builder, inputs) {
    const rowCount = inputs.tracts.count;
    const positions = builder.upload('positions', 'float32x2', inputs.tracts.centers);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const coefficients = builder.allocate('coefficients', 'float32', rowCount * 3);
    const codes = builder.allocate('codes', 'uint32', rowCount);
    builder.graph.add(
      new GPUMultiscaleGeographicallyWeightedRegression({
        positions: positions.view,
        predictors: predictors.view,
        predictorCount: 2,
        response: response.view,
        kernel: 'bisquare',
        maximumIterations: 50,
        parameters: parameterBuffer.view,
        output: {
          coefficients: coefficients.view,
          fitted: builder.allocate('fitted', 'float32', rowCount).view,
          status: {
            codes: codes.view,
            iterationCount: builder.allocate('iteration-count', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {
        codes,
        meshes: [
          uploadRegressionMap(builder, inputs),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * MAP_GAP),
          uploadRegressionMap(builder, inputs, inputs.mapWidth * MAP_GAP * 2)
        ],
        intercept: addTableColumn(builder, 'intercept', coefficients, 3, 0, rowCount),
        first: addTableColumn(builder, 'first-slope', coefficients, 3, 1, rowCount),
        second: addTableColumn(builder, 'second-slope', coefficients, 3, 2, rowCount)
      },
      update: current => {
        response.write(
          createResponse(inputs, {
            noise: 0.3,
            coefficients: getScaledCoefficients(inputs)
          })
        );
        parameterBuffer.write(
          Float32Array.of(
            1e-2,
            current['interceptBandwidth'],
            current['firstBandwidth'],
            current['secondBandwidth']
          )
        );
      }
    };
  },
  // x sets the bandwidth of the drifting slope, y the bandwidth of the constant intercept.
  interaction: bindAxes<unknown, {codes: RunBuffer<'uint32'>}>(parameters, {
    kind: 'pad',
    hint: 'intercept and x1 bandwidth',
    x: 'firstBandwidth',
    y: 'interceptBandwidth',
    decorate: (values, pointer, {outputs, peek}) => {
      const codes = peek(outputs.codes);
      let converged = 0;
      let excluded = 0;
      let iterationLimit = 0;
      let failed = 0;
      for (let row = 0; codes && row < codes.length; row++) {
        switch (decodeGPUMultiscaleGeographicallyWeightedRegressionStatus(codes[row])) {
          case 'converged':
            converged++;
            break;
          case 'excluded':
            excluded++;
            break;
          case 'iteration-limit':
            iterationLimit++;
            break;
          default:
            failed++;
            break;
        }
      }
      return {
        glyphs: [{shape: 'crosshair', at: [pointer.x, pointer.y]}],
        readout: `x1 bandwidth ${values['firstBandwidth']} · intercept ${values['interceptBandwidth']} · ${converged} converged · ${excluded} excluded · ${iterationLimit} iteration-limit estimate${iterationLimit === 1 ? '' : 's'} displayed · ${failed} failed`
      };
    }
  }),
  // All surfaces on one diverging scale centred on zero: the constant slope (-1.5) is clearly negative.
  draw: ({inputs, outputs, theme}) => {
    let north = -Infinity;
    for (let tract = 0; tract < inputs.tracts.count; tract++) {
      north = Math.max(north, inputs.tracts.centers[tract * 2 + 1]);
    }
    const key = createLegend({
      title: 'coefficient',
      corner: 'bottom-left',
      entries: [
        {label: '−3', swatch: [...theme.diverging[0]]},
        {label: '0', swatch: [...theme.diverging[2]]},
        {label: '+3', swatch: [...theme.diverging[5]]}
      ]
    });
    const titles = mergeText(key, {
      labels: MAP_TITLES.map((title, index) => ({
        text: title,
        at: [inputs.mapWidth * (MAP_GAP * index + 0.5), north] as const,
        offset: [0, 22] as const,
        role: 'label' as const,
        priority: 3
      }))
    });
    return [
      ...drawTractMap(inputs, outputs.meshes[0], outputs.intercept, {
        ramp: 'diverging',
        range: [-3, 3]
      }),
      ...drawTractMap(inputs, outputs.meshes[1], outputs.first, {
        ramp: 'diverging',
        range: [-3, 3]
      }),
      ...drawTractMap(inputs, outputs.meshes[2], outputs.second, {
        ramp: 'diverging',
        range: [-3, 3]
      }),
      {layer: 'text', ...titles}
    ];
  }
});
