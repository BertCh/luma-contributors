// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGaussianPredictiveScores: the closed-form CRPS of a normal predictive distribution at every
// station. Observation: Helene day-two rain (log residual). Predictive mean: day-one rain plus
// `bias`; standard deviation `spread` (scenario). Left: stations coloured by CRPS. Right: the mean
// CRPS over stations for every (spread, bias), one evaluation of the same compiled graph per cell
// (ring: the minimum). CRPS is a proper score, so the minimum sits at the honest spread and bias of
// the forecast errors: a narrower or biased forecast is punished. The pointer is the predictive distribution being scored.

import {GPUGaussianPredictiveScores} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {createForecastScene} from './_lib/forecast';
import {
  createGridMesh,
  createPanelPad,
  findMaximum,
  getAxisSamples,
  getFrameGlyph,
  getPanelPoint,
  sweepCompiledGraph,
  type PanelAxis
} from './_lib/surface';
import {parameters} from './gaussian-predictive-scores.parameters';

const STATION_COUNT = 360;
const COLUMNS = 16;
const ROWS = 14;
const SPREAD_AXIS: PanelAxis = {parameter: parameters[0], logarithmic: true};
const BIAS_AXIS: PanelAxis = {parameter: parameters[1]};
/** The surface is drawn from its minimum up to this much worse. */
const SURFACE_DEPTH = 2;

type Inputs = ReturnType<typeof createInputs>;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createForecastScene(random, world, STATION_COUNT);
  return {
    ...scene,
    mesh: createGridMesh(scene.panel, COLUMNS, ROWS),
    spreadSamples: getAxisSamples(SPREAD_AXIS, COLUMNS),
    biasSamples: getAxisSamples(BIAS_AXIS, ROWS),
    surfaceCache: {key: '', values: new Float32Array(0), minimum: {index: -1, value: Number.NaN}}
  };
}

/** Predictive means and variances per station: baseline plus bias; sd grows with wet baseline. */
function getPredictive(inputs: Inputs, spread: number, bias: number, wetSpread: number) {
  const means = new Float32Array(inputs.count);
  const variances = new Float32Array(inputs.count);
  for (let station = 0; station < inputs.count; station++) {
    means[station] = inputs.baseline[station] + bias;
    const sd = spread * Math.exp(wetSpread * inputs.baseline[station]);
    variances[station] = sd * sd;
  }
  return {means, variances};
}

function addScores(
  builder: RunBuilder,
  id: string,
  inputs: Inputs,
  observations: RunBuffer<'float32'>
) {
  const means = builder.allocate(`${id}-means`, 'float32', inputs.count);
  const variances = builder.allocate(`${id}-variances`, 'float32', inputs.count);
  const crps = builder.allocate(`${id}-crps`, 'float32', inputs.count);
  builder.graph.add(
    new GPUGaussianPredictiveScores({
      id,
      means: means.view,
      variances: variances.view,
      observations: observations.view,
      output: {crps: crps.view}
    })
  );
  return {means, variances, crps};
}

function build(builder: RunBuilder, inputs: Inputs) {
  const observations = builder.upload('observations', 'float32', inputs.residuals);
  const main = addScores(builder, 'gaussian-scores', inputs, observations);
  // A second instance over the same observations is evaluated once per surface cell after the run.
  const sweep = addScores(builder, 'gaussian-scores-sweep', inputs, observations);
  return {
    outputs: {
      crps: main.crps,
      sweep,
      sweepTarget: builder.allocate('sweep-target', 'float32', COLUMNS * ROWS * inputs.count),
      positions: builder.upload('positions', 'float32x2', inputs.positions),
      meshPositions: builder.upload('mesh-positions', 'float32x2', inputs.mesh.positions),
      meshIndices: builder.upload('mesh-indices', 'uint32', inputs.mesh.indices),
      meshOwners: builder.upload('mesh-owners', 'uint32', inputs.mesh.owners)
    },
    update(values: Record<string, number>) {
      const predictive = getPredictive(inputs, values.spread, values.bias, values.wetSpread);
      main.means.write(predictive.means);
      main.variances.write(predictive.variances);
    }
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Gaussian CRPS of Helene day-two rain at Carolina stations; mean = day-one rain + bias, sd = spread (scenario). Right: mean CRPS over (spread, bias).',
  createInputs,
  build,
  async afterRun({device, compiled, inputs, outputs, values, upload}) {
    const cache = inputs.surfaceCache;
    const key = String(values.wetSpread);
    if (cache.key !== key) {
      const scores = await sweepCompiledGraph({
        device,
        compiled,
        pointCount: COLUMNS * ROWS,
        prepare: point => {
          const predictive = getPredictive(
            inputs,
            inputs.spreadSamples[point % COLUMNS],
            inputs.biasSamples[Math.floor(point / COLUMNS)],
            values.wetSpread
          );
          outputs.sweep.means.write(predictive.means);
          outputs.sweep.variances.write(predictive.variances);
        },
        source: outputs.sweep.crps,
        destination: outputs.sweepTarget,
        length: inputs.count
      });
      const means = Float32Array.from({length: COLUMNS * ROWS}, (_, point) => {
        let sum = 0;
        for (let station = 0; station < inputs.count; station++)
          sum += scores[point * inputs.count + station];
        return sum / inputs.count;
      });
      cache.key = key;
      cache.values = means;
      const negated = findMaximum(Float32Array.from(means, value => -value));
      cache.minimum = {index: negated.index, value: -negated.value};
    }
    const surface = Float32Array.from(cache.values, value =>
      Number.isFinite(value) ? Math.min(value - cache.minimum.value, SURFACE_DEPTH) : Number.NaN
    );
    return {surface: upload('surface', 'float32', surface)};
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'predictive sd and bias',
    rectangle: ({inputs}) => inputs.panel,
    x: SPREAD_AXIS,
    y: BIAS_AXIS,
    decorate(values, {inputs, outputs, peek}) {
      const glyphs: RunGlyph[] = [
        getFrameGlyph(inputs.panel),
        {
          shape: 'crosshair',
          at: getPanelPoint(inputs.panel, SPREAD_AXIS, BIAS_AXIS, values.spread, values.bias)
        }
      ];
      const {minimum} = inputs.surfaceCache;
      if (minimum.index >= 0) {
        const bestSpread = inputs.spreadSamples[minimum.index % COLUMNS];
        const bestBias = inputs.biasSamples[Math.floor(minimum.index / COLUMNS)];
        glyphs.push({
          shape: 'ring',
          at: getPanelPoint(inputs.panel, SPREAD_AXIS, BIAS_AXIS, bestSpread, bestBias),
          radius: 0.03
        });
      }
      const scores = peek(outputs.crps);
      let mean = 0;
      if (scores) {
        for (let station = 0; station < inputs.count; station++) mean += scores[station];
        mean /= inputs.count;
      }
      return {
        glyphs,
        spotlight: null,
        readout: scores
          ? `sd ${formatNumber(values.spread, 1)} · bias ${formatNumber(values.bias, 1)} · CRPS ${formatNumber(mean, 2)}${
              minimum.index >= 0 ? ` (best ${formatNumber(minimum.value, 2)})` : ''
            }`
          : `sd ${values.spread} · bias ${values.bias}`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.crps,
      ramp: 'sequential',
      range: [0, 2],
      radius: 5
    },
    ...(extra
      ? [
          {
            layer: 'polygons' as const,
            positions: outputs.meshPositions,
            indices: outputs.meshIndices,
            owners: outputs.meshOwners,
            values: extra.surface,
            triangleCount: inputs.mesh.triangleCount,
            ramp: 'sequential' as const,
            range: [0, SURFACE_DEPTH] as const,
            bands: 12
          }
        ]
      : [])
  ]
});
