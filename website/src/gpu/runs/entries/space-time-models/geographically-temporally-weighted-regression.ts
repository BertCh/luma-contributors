// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGeographicallyTemporallyWeightedRegression fits a local regression at every tract in every
// period, weighting observations by their space-time distance sqrt(ds^2 + timeScale dt^2). Left:
// the four periods, each showing the local slope of the response on poverty share (one fixed color
// range; singular rows muted). The slope truly sweeps west to east through the periods. A small
// timeScale lets neighboring periods borrow strength; a large one fits each period on its own; zero
// ignores time (plain GWR). Right: the AICc over (bandwidth, timeScale), one evaluation of a second
// instance of the same graph per cell, as the difference from the best cell (ring). The pointer
// moves through the surface without a recompile.

import {
  getGPUGTWRParameterValues,
  GPUGeographicallyTemporallyWeightedRegression
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {addTableColumn} from '../regression-weights/shared/weights';
import {
  createSpaceTimeScene,
  drawSpaceTimeTiles,
  PERIOD_TEMPORAL_CONTEXT,
  uploadSpaceTimeMesh,
  uploadSpaceTimeObservations,
  type SpaceTimeScene
} from './_lib/gtwr-data';
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
import {parameters} from './geographically-temporally-weighted-regression.parameters';

const SLOPE_RANGE = [0, 3.4] as const;
const SUMMARY_LENGTH = 12;
const AICC_SLOT = 2;
const R_SQUARED_SLOT = 3;
const MAP_SHARE = 0.62;
const COLUMNS = 14;
const ROWS = 12;
/** The surface is drawn as AICc minus the best, down to this many units. */
const SURFACE_DEPTH = 120;
const BANDWIDTH_AXIS: PanelAxis = {parameter: parameters[0], logarithmic: true};
const TIME_SCALE_AXIS: PanelAxis = {parameter: parameters[1]};

type Inputs = ReturnType<typeof createInputs>;

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createSpaceTimeScene(
    random,
    {width: world.width * MAP_SHARE, height: world.height},
    {tileColumns: 2, tileRows: 2, periodsFillGrid: true}
  );
  const panel = {x0: world.width * 0.66, y0: 0.16, x1: world.width * 0.96, y1: 0.84};
  return {
    ...scene,
    panel,
    mesh: createGridMesh(panel, COLUMNS, ROWS),
    bandwidthSamples: getAxisSamples(BANDWIDTH_AXIS, COLUMNS),
    timeScaleSamples: getAxisSamples(TIME_SCALE_AXIS, ROWS),
    // The surface does not depend on any slider, so it is evaluated once.
    surface: {ready: false, values: new Float32Array(0), best: {index: -1, value: Number.NaN}}
  };
}

function addRegression(
  builder: RunBuilder,
  id: string,
  scene: SpaceTimeScene,
  observations: ReturnType<typeof uploadSpaceTimeObservations>
) {
  const parameterBuffer = builder.allocate(`${id}-parameters`, 'float32', 4);
  const coefficients = builder.allocate(`${id}-coefficients`, 'float32', scene.rowCount * 2);
  const summary = builder.allocate(`${id}-summary`, 'float32', SUMMARY_LENGTH);
  builder.graph.add(
    new GPUGeographicallyTemporallyWeightedRegression({
      id,
      positions: observations.positions.view,
      times: observations.times.view,
      temporalContext: PERIOD_TEMPORAL_CONTEXT,
      predictors: observations.predictors.view,
      predictorCount: 1,
      response: observations.response.view,
      kernel: 'gaussian',
      parameters: parameterBuffer.view,
      output: {
        coefficients: coefficients.view,
        localStatus: builder.allocate(`${id}-local-status`, 'uint32', scene.rowCount).view,
        fitted: builder.allocate(`${id}-fitted`, 'float32', scene.rowCount).view,
        hatDiagonal: builder.allocate(`${id}-hat-diagonal`, 'float32', scene.rowCount).view,
        summary: summary.view
      }
    })
  );
  return {parameterBuffer, coefficients, summary};
}

function build(builder: RunBuilder, scene: Inputs) {
  const observations = uploadSpaceTimeObservations(builder, scene);
  const main = addRegression(builder, 'gtwr', scene, observations);
  // A second instance over the same data is evaluated once per surface cell after the run.
  const sweep = addRegression(builder, 'gtwr-sweep', scene, observations);
  return {
    outputs: {
      mesh: uploadSpaceTimeMesh(builder, scene, 0),
      slope: addTableColumn(builder, 'local-slope', main.coefficients, 2, 1, scene.rowCount),
      summary: main.summary,
      sweepParameters: sweep.parameterBuffer,
      sweepSummary: sweep.summary,
      sweepTarget: builder.allocate('sweep-target', 'float32', COLUMNS * ROWS * SUMMARY_LENGTH),
      meshPositions: builder.upload('mesh-positions', 'float32x2', scene.mesh.positions),
      meshIndices: builder.upload('mesh-indices', 'uint32', scene.mesh.indices),
      meshOwners: builder.upload('mesh-owners', 'uint32', scene.mesh.owners)
    },
    update: (current: Record<string, number>) =>
      main.parameterBuffer.write(
        getGPUGTWRParameterValues({bandwidth: current.bandwidth, timeScale: current.timeScale})
      )
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Local slope of a response simulated on real Chicago tract poverty shares in four periods (the slope sweeps west to east); right: AICc over (bandwidth, timeScale).',
  createInputs,
  build,
  async afterRun({device, compiled, inputs, outputs, upload}) {
    const cache = inputs.surface;
    if (!cache.ready) {
      const sums = await sweepCompiledGraph({
        device,
        compiled,
        pointCount: COLUMNS * ROWS,
        prepare: point =>
          outputs.sweepParameters.write(
            getGPUGTWRParameterValues({
              bandwidth: inputs.bandwidthSamples[point % COLUMNS],
              timeScale: inputs.timeScaleSamples[Math.floor(point / COLUMNS)]
            })
          ),
        source: outputs.sweepSummary,
        destination: outputs.sweepTarget,
        length: SUMMARY_LENGTH
      });
      const aicc = Float32Array.from(
        {length: COLUMNS * ROWS},
        (_, point) => sums[point * SUMMARY_LENGTH + AICC_SLOT]
      );
      // Cells that failed (singular fits) have a non-finite AICc and stay muted.
      cache.values = aicc;
      cache.best = findMaximum(aicc.map(value => -value));
      cache.ready = true;
    }
    const best = -cache.best.value;
    const relative = Float32Array.from(cache.values, value =>
      Number.isFinite(value) ? Math.max(best - value, -SURFACE_DEPTH) : Number.NaN
    );
    return {surface: upload('surface', 'float32', relative)};
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'bandwidth and timeScale',
    rectangle: ({inputs}) => inputs.panel,
    x: BANDWIDTH_AXIS,
    y: TIME_SCALE_AXIS,
    decorate(values, {inputs, outputs, peek}) {
      const glyphs: RunGlyph[] = [
        getFrameGlyph(inputs.panel),
        {
          shape: 'crosshair',
          at: getPanelPoint(
            inputs.panel,
            BANDWIDTH_AXIS,
            TIME_SCALE_AXIS,
            values.bandwidth,
            values.timeScale
          )
        }
      ];
      const {best} = inputs.surface;
      let bestText = '';
      if (best.index >= 0) {
        glyphs.push({
          shape: 'ring',
          at: getPanelPoint(
            inputs.panel,
            BANDWIDTH_AXIS,
            TIME_SCALE_AXIS,
            inputs.bandwidthSamples[best.index % COLUMNS],
            inputs.timeScaleSamples[Math.floor(best.index / COLUMNS)]
          ),
          radius: 0.03
        });
        bestText = ` (best ${formatNumber(-best.value, 0)})`;
      }
      const summary = peek(outputs.summary);
      const timeText =
        values.timeScale === 0 ? 'tau 0: time ignored' : `tau ${formatNumber(values.timeScale)}`;
      const fit = summary
        ? ` · AICc ${formatNumber(summary[AICC_SLOT], 0)}${bestText} · R2 ${formatNumber(summary[R_SQUARED_SLOT], 2)}`
        : '';
      return {
        glyphs,
        spotlight: null,
        readout: `h ${formatNumber(values.bandwidth)} · ${timeText}${fit}`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    ...drawSpaceTimeTiles(inputs, 0, outputs.mesh, outputs.slope, SLOPE_RANGE),
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
            range: [-SURFACE_DEPTH, 0] as const,
            bands: 12
          }
        ]
      : [])
  ]
});
