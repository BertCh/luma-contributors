// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialLagImpacts turns a spatial-lag fit and its power traces into direct, indirect and total
// impacts per predictor. The fit and traces come from GPUSpatialLagMaximumLikelihood in the same
// graph. Right: three bar charts (direct, indirect, total) with one bar per predictor; the indirect
// impact (spillover) grows with rho. Left: the change in the response when the tract under the
// pointer gets a unit increase of the first predictor, with the estimated rho and slope.

import {
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {type CsrWeights} from '../../inputs';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {parameters} from './spatial-lag-impacts.parameters';
import {
  createRegressionInputs,
  createResponse,
  drawSignedBars,
  drawTractMap,
  drawZeroLine,
  MAP_SHARE,
  uploadRegressionMap,
  uploadStandardizedWeights
} from './shared/regression';
import {findTractUnder, getTractCenter, getTractOutline} from './shared/probes';
import {addTableColumn} from './shared/weights';

/** Column `shocked` of (I - rho W)^-1 by its Neumann series: the response of every tract to a unit shock there. */
function getShockResponse(
  weights: CsrWeights,
  shocked: number,
  rho: number,
  scale: number
): Float32Array {
  let term = new Float32Array(weights.rowCount);
  term[shocked] = 1;
  const total = Float32Array.from(term);
  for (let power = 1; power < 40; power++) {
    const next = new Float32Array(weights.rowCount);
    for (let row = 0; row < weights.rowCount; row++) {
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        next[row] += weights.weights[slot] * term[weights.neighbors[slot]] * rho;
      }
    }
    term = next;
    for (let row = 0; row < total.length; row++) total[row] += term[row];
  }
  return total.map(value => value * scale);
}

const IMPACT_LIMIT = 5;
const SHOCK_LIMIT = 1.2;

const IMPACT_NAMES = ['direct impact', 'indirect impact', 'total impact'] as const;

const IMPACT_COLORS = [
  [0.4, 0.65, 1, 0.95],
  [1, 0.6, 0.25, 0.95],
  [0.55, 0.9, 0.5, 0.95]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Direct, indirect and total impacts of each predictor (right), residuals (left).',
  createInputs: ({random, world}) => createRegressionInputs(random, world),
  build(builder, inputs, values) {
    const rowCount = inputs.tracts.count;
    const maximumPower = values.maximumPower;
    const weights = uploadStandardizedWeights(builder, inputs.weights);
    const predictors = builder.upload('predictors', 'float32', inputs.predictors);
    const response = builder.allocate('response', 'float32', rowCount);
    const table = builder.allocate('table', 'float32', 16);
    const summary = builder.allocate('summary', 'float32', 32);
    const status = builder.allocate('status', 'uint32', 1);
    const residuals = builder.allocate('residuals', 'float32', rowCount);
    const powerTraces = builder.allocate('power-traces', 'float32', maximumPower);
    const powerSums = builder.allocate('power-sums', 'float32', maximumPower);
    const infinityNorm = builder.allocate('infinity-norm', 'float32', 1);
    builder.graph.add(
      new GPUSpatialLagMaximumLikelihood({
        weights: weights.input,
        predictors: predictors.view,
        response: response.view,
        predictorCount: 2,
        maximumPower,
        output: {
          status: {code: status.view},
          table: table.view,
          summary: summary.view,
          residuals: residuals.view,
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          infinityNorm: infinityNorm.view
        }
      })
    );
    const impacts = builder.allocate('impacts', 'float32', 6);
    builder.graph.add(
      new GPUSpatialLagImpacts({
        coefficients: table.view,
        coefficientStride: 4,
        rhoIndex: 3,
        predictorCount: 2,
        rowCount,
        maximumPower,
        traces: {
          powerTraces: powerTraces.view,
          powerSums: powerSums.view,
          infinityNorm: infinityNorm.view
        },
        status: status.view,
        output: {impacts: impacts.view}
      })
    );
    const impactMap = builder.allocate('impact-map', 'float32', rowCount);
    impactMap.write(getShockResponse(inputs.weights, 0, 0.5, 2));
    return {
      outputs: {
        mesh: uploadRegressionMap(builder, inputs),
        residuals,
        table,
        impactMap,
        zeros: builder.upload('zeros', 'float32', new Float32Array(64)),
        components: [0, 1, 2].map(column =>
          addTableColumn(builder, 'impacts', impacts, 3, column, 2)
        )
      },
      update: current =>
        response.write(createResponse(inputs, {noise: 0.4, lagRho: current.strength}))
    };
  },
  rebuildOn: ['maximumPower'],
  // The pointer picks the tract that gets a unit increase of x1: the map shows how far the response moves,
  // at the tract itself (direct) and in the tracts around it (spillover), with the fitted rho and slope.
  interaction: {
    kind: 'place',
    hint: 'shock a tract',
    rest: [0.3, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const tract = findTractUnder(inputs.tracts, pointer.x, pointer.y);
      const table = peek(outputs.table);
      if (tract < 0 || !table) return {};
      const rho = table[3 * 4];
      const slope = table[1 * 4];
      const change = getShockResponse(inputs.weights, tract, rho, slope);
      outputs.impactMap.write(change);
      const direct = change[tract];
      const spillover = change.reduce((sum, value) => sum + value, 0) - direct;
      return {
        focus: tract,
        glyphs: [
          getTractOutline(inputs.tracts, tract),
          {shape: 'ring', at: getTractCenter(inputs.tracts, tract), radius: 0.05, dashed: true}
        ],
        readout: `shock at tract ${tract} · direct ${direct.toFixed(2)} · spillover ${spillover.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, world}) => {
    const panels = outputs.components.map((_, index) => {
      const rectangle = [MAP_SHARE + 0.08, 0.7 - index * 0.31, 0.97, 0.94 - index * 0.31] as const;
      const axes = createChartAxes({
        rectangle: getChartRectangle(world, rectangle),
        x: {categories: ['x1', 'x2']},
        y: {range: [-IMPACT_LIMIT, IMPACT_LIMIT], ticks: [-4, 0, 4]}
      });
      const title = {
        labels: [
          {
            text: IMPACT_NAMES[index],
            at: [(MAP_SHARE + 0.08) * world.width, (0.94 - index * 0.31) * world.height] as const,
            anchor: 'bottom-left' as const,
            offset: [0, 3] as const,
            swatch: IMPACT_COLORS[index].slice(0, 3) as unknown as readonly [
              number,
              number,
              number
            ],
            role: 'label' as const,
            priority: 4
          }
        ]
      };
      return {axes, text: mergeText(axes, title)};
    });
    return [
      ...panels.map(panel => ({layer: 'text' as const, ...panel.axes.grid})),
      ...drawTractMap(inputs, outputs.mesh, outputs.impactMap, {
        ramp: 'sequential',
        range: [0, SHOCK_LIMIT]
      }),
      ...outputs.components.flatMap((values, index) => {
        const rectangle = [
          MAP_SHARE + 0.08,
          0.7 - index * 0.31,
          0.97,
          0.94 - index * 0.31
        ] as const;
        return [
          ...drawSignedBars(values, rectangle, IMPACT_LIMIT, IMPACT_COLORS[index]),
          drawZeroLine(outputs.zeros, rectangle)
        ];
      }),
      ...panels.map(panel => ({layer: 'text' as const, ...panel.text}))
    ];
  }
});
