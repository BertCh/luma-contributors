// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVecchiaSpaceTimeLogLikelihood: the Gaussian-process log-likelihood of Helene log rainfall
// residuals at Carolina stations on both days, as 2 x 220 space-time observations. Rows run day 0
// then day 1 (each in max-min order), and every row is conditioned on its nearest earlier rows in
// the space-time metric (links drawn for the ten worst terms, including its own station a day before). Left: each
// observation's term of the sum at the current covariance (a station's two days drawn side by
// side). Right: the likelihood surface over (spatialRange, temporalRange), one evaluation of the
// same compiled graph per cell, relative to its maximum (ring). The pointer is the covariance
// being evaluated.

import {
  buildVecchiaConditioningSets,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues,
  getVecchiaLogLikelihoodResult,
  getVecchiaMaxMinOrdering,
  GPUVecchiaSpaceTimeLogLikelihood,
  GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH,
  type GPUVecchiaSpaceTimeLogLikelihoodProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder, type RunBuffer} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {createRainScene, packObservations} from './_lib/rain';
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
import {parameters} from './vecchia-space-time-log-likelihood.parameters';

const STATION_COUNT = 220;
const ROW_COUNT = STATION_COUNT * 2;
const NEIGHBOR_COUNT = 8;
const COLUMNS = 16;
const ROWS = 14;
const SECONDS_PER_DAY = 86400;
/** Neighbor metric: map units per day. */
const TIME_SCALE = 0.08 / SECONDS_PER_DAY;
/** Both days drawn side by side: horizontal offset of each day from the station, in map units. */
const DAY_OFFSET = 0.008;
const TEMPORAL_CONTEXT: GPUVecchiaSpaceTimeLogLikelihoodProps['temporalContext'] = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};
const SPATIAL_AXIS: PanelAxis = {parameter: parameters[0], logarithmic: true};
const TEMPORAL_AXIS: PanelAxis = {parameter: parameters[1], logarithmic: true};
/** The surface is drawn relative to its maximum, down to this many log-likelihood units. */
const SURFACE_DEPTH = 120;
/** The ramp reaches further down than the clip, so a merely low cell keeps a visible color. */
const RAMP_DEPTH = SURFACE_DEPTH * 1.7;
/** Stations whose conditioning links are drawn: the worst terms at the current covariance. */
const LINKED_SITE_COUNT = 10;

function getCovariance(
  values: Record<string, number>,
  spatialRange: number,
  temporalRange: number
) {
  return {
    type: 'separable' as const,
    sill: values.sill,
    spatial: {
      model: 'exponential' as const,
      nugget: values.nugget,
      sill: 1 - values.nugget,
      range: spatialRange
    },
    temporal: {
      model: 'exponential' as const,
      nugget: 0.05,
      sill: 0.95,
      range: temporalRange * SECONDS_PER_DAY
    }
  };
}

function getParameterValues(
  values: Record<string, number>,
  spatialRange: number,
  temporalRange: number
) {
  return getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
    getCovariance(values, spatialRange, temporalRange),
    undefined,
    TIME_SCALE
  );
}

type Inputs = ReturnType<typeof createInputs>;
type Uploaded = {
  positions: RunBuffer<'float32x2'>;
  times: RunBuffer<'float32'>;
  observations: RunBuffer<'float32x4'>;
  offsets: RunBuffer<'uint32'>;
  neighbors: RunBuffer<'uint32'>;
};

function addLikelihood(builder: RunBuilder, id: string, scene: Uploaded) {
  const parameterBuffer = builder.allocate(
    `${id}-parameters`,
    'float32',
    GPU_VECCHIA_SPACE_TIME_PARAMETER_LENGTH
  );
  const siteTerms = builder.allocate(`${id}-site-terms`, 'float32', ROW_COUNT);
  const logLikelihood = builder.allocate(`${id}-log-likelihood`, 'float32', 2);
  const status = builder.allocate(`${id}-status`, 'uint32', 3);
  builder.graph.add(
    new GPUVecchiaSpaceTimeLogLikelihood({
      id,
      temporalContext: TEMPORAL_CONTEXT,
      positions: scene.positions.view,
      times: scene.times.view,
      observations: scene.observations.view,
      neighborhoodOffsets: scene.offsets.view,
      neighborhoods: scene.neighbors.view,
      neighborCount: NEIGHBOR_COUNT,
      parameters: parameterBuffer.view,
      output: {
        conditionals: builder.allocate(`${id}-conditionals`, 'float32x4', ROW_COUNT).view,
        conditionalStatus: builder.allocate(`${id}-conditional-status`, 'uint32', ROW_COUNT).view,
        siteTerms: siteTerms.view,
        siteStatus: builder.allocate(`${id}-site-status`, 'uint32', ROW_COUNT).view,
        logLikelihood: logLikelihood.view,
        status: status.view
      }
    })
  );
  return {parameterBuffer, siteTerms, logLikelihood, status};
}

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createRainScene(random, world, STATION_COUNT);
  const order = getVecchiaMaxMinOrdering(scene.positions);
  const positions = new Float32Array(ROW_COUNT * 2);
  const drawnPositions = new Float32Array(ROW_COUNT * 2);
  const times = new Float32Array(ROW_COUNT);
  const residuals = new Float32Array(ROW_COUNT);
  for (let day = 0; day < 2; day++) {
    for (let rank = 0; rank < STATION_COUNT; rank++) {
      const station = order[rank];
      const row = day * STATION_COUNT + rank;
      const x = scene.positions[station * 2];
      const y = scene.positions[station * 2 + 1];
      positions.set([x, y], row * 2);
      drawnPositions.set([x + (day === 0 ? -DAY_OFFSET : DAY_OFFSET), y], row * 2);
      times[row] = day * SECONDS_PER_DAY;
      residuals[row] = day === 0 ? scene.previousResiduals[station] : scene.residuals[station];
    }
  }
  const sets = buildVecchiaConditioningSets({
    positions,
    neighborCount: NEIGHBOR_COUNT,
    times,
    timeScale: TIME_SCALE
  });
  return {
    ...scene,
    positions,
    drawnPositions,
    times,
    residuals,
    sets,
    mesh: createGridMesh(scene.panel, COLUMNS, ROWS),
    spatialSamples: getAxisSamples(SPATIAL_AXIS, COLUMNS),
    temporalSamples: getAxisSamples(TEMPORAL_AXIS, ROWS),
    surfaceCache: {key: '', values: new Float32Array(0), maximum: {index: -1, value: Number.NaN}}
  };
}

/** Links from the stations with the lowest finite terms to the rows each is conditioned on. */
async function getWorstLinks(inputs: Inputs, outputs: Outputs): Promise<Float32Array> {
  const bytes = await outputs.siteTerms.buffer.readAsync();
  const terms = new Float32Array(bytes.buffer, bytes.byteOffset, ROW_COUNT);
  const worst = Array.from({length: ROW_COUNT}, (_, row) => row)
    .filter(row => Number.isFinite(terms[row]))
    .sort((first, second) => terms[first] - terms[second])
    .slice(0, LINKED_SITE_COUNT);
  const segments = new Float32Array(LINKED_SITE_COUNT * NEIGHBOR_COUNT * 4);
  worst.forEach((row, rank) => {
    for (let entry = inputs.sets.offsets[row]; entry < inputs.sets.offsets[row + 1]; entry++) {
      const other = inputs.sets.neighbors[entry];
      segments.set(
        [
          inputs.drawnPositions[row * 2],
          inputs.drawnPositions[row * 2 + 1],
          inputs.drawnPositions[other * 2],
          inputs.drawnPositions[other * 2 + 1]
        ],
        (rank * NEIGHBOR_COUNT + entry - inputs.sets.offsets[row]) * 4
      );
    }
  });
  return segments;
}

function build(builder: RunBuilder, inputs: Inputs) {
  const uploaded: Uploaded = {
    positions: builder.upload('positions', 'float32x2', inputs.positions),
    times: builder.upload('times', 'float32', inputs.times),
    observations: builder.upload('observations', 'float32x4', packObservations(inputs.residuals)),
    offsets: builder.upload('offsets', 'uint32', inputs.sets.offsets),
    neighbors: builder.upload('neighbors', 'uint32', inputs.sets.neighbors)
  };
  const main = addLikelihood(builder, 'vecchia-space-time', uploaded);
  // A second instance over the same data is evaluated once per surface cell after the run.
  const sweep = addLikelihood(builder, 'vecchia-space-time-sweep', uploaded);
  return {
    outputs: {
      ...main,
      sweepParameters: sweep.parameterBuffer,
      sweepLogLikelihood: sweep.logLikelihood,
      sweepTarget: builder.allocate('sweep-target', 'float32', COLUMNS * ROWS * 2),
      drawnPositions: builder.upload('drawn-positions', 'float32x2', inputs.drawnPositions),
      meshPositions: builder.upload('mesh-positions', 'float32x2', inputs.mesh.positions),
      meshIndices: builder.upload('mesh-indices', 'uint32', inputs.mesh.indices),
      meshOwners: builder.upload('mesh-owners', 'uint32', inputs.mesh.owners)
    },
    update: (values: Record<string, number>) =>
      main.parameterBuffer.write(
        getParameterValues(values, values.spatialRange, values.temporalRange)
      )
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Space-time Vecchia log-likelihood of Helene log rainfall residuals (Carolina stations, two days); right: the (spatialRange, temporalRange) surface. Each day is detrended separately, so the days are nearly uncorrelated and the optimum sits at the lowest temporalRange.',
  createInputs,
  build,
  async afterRun({device, compiled, inputs, outputs, values, upload}) {
    const links = await getWorstLinks(inputs, outputs);
    const cache = inputs.surfaceCache;
    const key = `${values.sill}|${values.nugget}`;
    if (cache.key !== key) {
      const sums = await sweepCompiledGraph({
        device,
        compiled,
        pointCount: COLUMNS * ROWS,
        prepare: point =>
          outputs.sweepParameters.write(
            getParameterValues(
              values,
              inputs.spatialSamples[point % COLUMNS],
              inputs.temporalSamples[Math.floor(point / COLUMNS)]
            )
          ),
        source: outputs.sweepLogLikelihood,
        destination: outputs.sweepTarget,
        length: 2
      });
      const totals = Float32Array.from(
        {length: COLUMNS * ROWS},
        (_, point) => sums[point * 2] + sums[point * 2 + 1]
      );
      cache.key = key;
      cache.values = totals;
      cache.maximum = findMaximum(totals);
    }
    const relative = Float32Array.from(cache.values, value =>
      Number.isFinite(value) ? Math.max(value - cache.maximum.value, -SURFACE_DEPTH) : Number.NaN
    );
    return {
      surface: upload('surface', 'float32', relative),
      links: upload('links', 'float32x4', links),
      maximum: cache.maximum
    };
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'search space and time ranges',
    rectangle: ({inputs}) => inputs.panel,
    x: SPATIAL_AXIS,
    y: TEMPORAL_AXIS,
    decorate(values, {inputs, outputs, peek}) {
      const at = getPanelPoint(
        inputs.panel,
        SPATIAL_AXIS,
        TEMPORAL_AXIS,
        values.spatialRange,
        values.temporalRange
      );
      const glyphs: RunGlyph[] = [getFrameGlyph(inputs.panel), {shape: 'crosshair', at}];
      const {maximum} = inputs.surfaceCache;
      if (maximum.index >= 0) {
        const bestSpatial = inputs.spatialSamples[maximum.index % COLUMNS];
        const bestTemporal = inputs.temporalSamples[Math.floor(maximum.index / COLUMNS)];
        glyphs.push({
          shape: 'dot',
          at: getPanelPoint(inputs.panel, SPATIAL_AXIS, TEMPORAL_AXIS, bestSpatial, bestTemporal)
        });
      }
      const total = peek(outputs.logLikelihood);
      const status = peek(outputs.status);
      const result = total && status ? getVecchiaLogLikelihoodResult(total, status) : null;
      return {
        glyphs,
        spotlight: null,
        readout:
          status && status[0] > 0
            ? `${status[0]} sites failed: no likelihood`
            : result
              ? `logL ${formatNumber(result.logLikelihood, 1)} · spatialRange ${values.spatialRange}`
              : `spatialRange ${values.spatialRange}`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    ...(extra ? [{layer: 'lines' as const, context: true, segments: extra.links, alpha: 0.7}] : []),
    {
      layer: 'dots',
      positions: outputs.drawnPositions,
      values: outputs.siteTerms,
      ramp: 'sequential',
      range: [-3, 0.5],
      radius: 3.5
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
            range: [-RAMP_DEPTH, 0] as const,
            bands: 14
          }
        ]
      : [])
  ]
});
