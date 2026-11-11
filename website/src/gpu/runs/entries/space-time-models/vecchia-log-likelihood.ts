// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVecchiaLogLikelihood: the Gaussian-process log-likelihood of Helene rainfall (log residuals at
// Carolina stations), each station conditioned on its nearest earlier stations in a max-min order
// (the faint lines). Left: each station's term of the sum at the current covariance. Right: the
// likelihood surface over (range, sill), one evaluation of the same compiled graph per cell,
// relative to its maximum (ring). The pointer is the covariance being evaluated.

import {
  buildVecchiaConditioningSets,
  getGPUVecchiaLogLikelihoodParameterValues,
  getVecchiaLogLikelihoodResult,
  getVecchiaMaxMinOrdering,
  GPUVecchiaLogLikelihood,
  GPU_VECCHIA_PARAMETER_LENGTH,
  type GPUVecchiaCovarianceModel
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuilder, type RunBuffer} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {
  countFailedCells,
  createNeighborhoodSegments,
  createRainScene,
  createWorstStationSegments,
  packObservations
} from './_lib/rain';
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
import {parameters} from './vecchia-log-likelihood.parameters';

const STATION_COUNT = 420;
const NEIGHBOR_COUNT = 10;
const COLUMNS = 16;
const ROWS = 14;
const MODELS: GPUVecchiaCovarianceModel[] = ['exponential', 'matern-3/2'];
const RANGE_AXIS: PanelAxis = {parameter: parameters[0], logarithmic: true};
const SILL_AXIS: PanelAxis = {parameter: parameters[1], logarithmic: true};
/** The surface is drawn relative to its maximum, down to this many log-likelihood units. */
const SURFACE_DEPTH = 40;
/** Stations whose conditioning links are drawn: the ones with the lowest terms. */
const LINKED_STATION_COUNT = 10;

function getCovariance(values: Record<string, number>, range: number, sill: number) {
  const model = MODELS[values.model];
  // Matern 3/2 needs a nugget of at least 1e-3 of the sill in float32.
  const nugget = model === 'exponential' ? values.nugget : Math.max(values.nugget, 0.0011 * sill);
  return {model, range, sill, nugget};
}

/** ` · nugget x` when the floor raised the slider's nugget, so the readout shows what was evaluated. */
function getNuggetNote(values: Record<string, number>): string {
  const effective = getCovariance(values, values.range, values.sill).nugget;
  return effective !== values.nugget ? ` · nugget ${formatNumber(effective, 4)}` : '';
}

function addLikelihood(builder: RunBuilder, id: string, inputs: Inputs, scene: Uploaded) {
  const rowCount = inputs.count;
  const parameterBuffer = builder.allocate(
    `${id}-parameters`,
    'float32',
    GPU_VECCHIA_PARAMETER_LENGTH
  );
  const siteTerms = builder.allocate(`${id}-site-terms`, 'float32', rowCount);
  const logLikelihood = builder.allocate(`${id}-log-likelihood`, 'float32', 2);
  const status = builder.allocate(`${id}-status`, 'uint32', 3);
  builder.graph.add(
    new GPUVecchiaLogLikelihood({
      id,
      positions: scene.positions.view,
      observations: scene.observations.view,
      neighborhoodOffsets: scene.offsets.view,
      neighborhoods: scene.neighbors.view,
      neighborCount: NEIGHBOR_COUNT,
      parameters: parameterBuffer.view,
      output: {
        conditionals: builder.allocate(`${id}-conditionals`, 'float32x4', rowCount).view,
        conditionalStatus: builder.allocate(`${id}-conditional-status`, 'uint32', rowCount).view,
        siteTerms: siteTerms.view,
        siteStatus: builder.allocate(`${id}-site-status`, 'uint32', rowCount).view,
        logLikelihood: logLikelihood.view,
        status: status.view
      }
    })
  );
  return {parameterBuffer, siteTerms, logLikelihood, status};
}

type Inputs = ReturnType<typeof createInputs>;
type Uploaded = {
  positions: RunBuffer<'float32x2'>;
  observations: RunBuffer<'float32x4'>;
  offsets: RunBuffer<'uint32'>;
  neighbors: RunBuffer<'uint32'>;
};

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const scene = createRainScene(random, world, STATION_COUNT);
  const sets = buildVecchiaConditioningSets({
    positions: scene.positions,
    neighborCount: NEIGHBOR_COUNT,
    order: getVecchiaMaxMinOrdering(scene.positions)
  });
  return {
    ...scene,
    sets,
    segments: createNeighborhoodSegments(scene.positions, sets.offsets, sets.neighbors),
    mesh: createGridMesh(scene.panel, COLUMNS, ROWS),
    rangeSamples: getAxisSamples(RANGE_AXIS, COLUMNS),
    sillSamples: getAxisSamples(SILL_AXIS, ROWS),
    surfaceCache: {
      key: '',
      values: new Float32Array(0),
      maximum: {index: -1, value: Number.NaN}
    }
  };
}

function build(builder: RunBuilder, inputs: Inputs) {
  const uploaded: Uploaded = {
    positions: builder.upload('positions', 'float32x2', inputs.positions),
    observations: builder.upload('observations', 'float32x4', packObservations(inputs.residuals)),
    offsets: builder.upload('offsets', 'uint32', inputs.sets.offsets),
    neighbors: builder.upload('neighbors', 'uint32', inputs.sets.neighbors)
  };
  const main = addLikelihood(builder, 'vecchia', inputs, uploaded);
  // A second instance over the same data is evaluated once per surface cell after the run.
  const sweep = addLikelihood(builder, 'vecchia-sweep', inputs, uploaded);
  return {
    outputs: {
      ...main,
      sweepParameters: sweep.parameterBuffer,
      sweepLogLikelihood: sweep.logLikelihood,
      sweepTarget: builder.allocate('sweep-target', 'float32', COLUMNS * ROWS * 2),
      positions: uploaded.positions,
      meshPositions: builder.upload('mesh-positions', 'float32x2', inputs.mesh.positions),
      meshIndices: builder.upload('mesh-indices', 'uint32', inputs.mesh.indices),
      meshOwners: builder.upload('mesh-owners', 'uint32', inputs.mesh.owners)
    },
    update: (values: Record<string, number>) =>
      main.parameterBuffer.write(
        getGPUVecchiaLogLikelihoodParameterValues(getCovariance(values, values.range, values.sill))
      )
  };
}

type Outputs = ReturnType<typeof build>['outputs'];

export default defineOperationRun({
  parameters,
  caption:
    'Vecchia log-likelihood of Helene log rainfall at Carolina stations; right: the (range, sill) surface.',
  createInputs,
  build,
  async afterRun({device, compiled, inputs, outputs, values, upload}) {
    const cache = inputs.surfaceCache;
    const key = `${values.model}|${values.nugget}`;
    if (cache.key !== key) {
      const sums = await sweepCompiledGraph({
        device,
        compiled,
        pointCount: COLUMNS * ROWS,
        prepare: point =>
          outputs.sweepParameters.write(
            getGPUVecchiaLogLikelihoodParameterValues(
              getCovariance(
                values,
                inputs.rangeSamples[point % COLUMNS],
                inputs.sillSamples[Math.floor(point / COLUMNS)]
              )
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
    // Low values take the low end of the ramp; only failed evaluations (NaN) stay muted.
    const relative = Float32Array.from(cache.values, value =>
      Number.isFinite(value) ? Math.max(value - cache.maximum.value, -SURFACE_DEPTH) : Number.NaN
    );
    const bytes = await outputs.siteTerms.buffer.readAsync();
    const terms = new Float32Array(bytes.buffer, bytes.byteOffset, inputs.count);
    return {
      surface: upload('surface', 'float32', relative),
      links: upload(
        'links',
        'float32x4',
        createWorstStationSegments(
          inputs.segments,
          inputs.sets.offsets,
          terms,
          LINKED_STATION_COUNT
        )
      ),
      failedCells: countFailedCells(cache.values),
      maximum: cache.maximum
    };
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'search range and sill',
    rectangle: ({inputs}) => inputs.panel,
    x: RANGE_AXIS,
    y: SILL_AXIS,
    decorate(values, {inputs, outputs, peek}) {
      const at = getPanelPoint(inputs.panel, RANGE_AXIS, SILL_AXIS, values.range, values.sill);
      const glyphs: RunGlyph[] = [getFrameGlyph(inputs.panel), {shape: 'crosshair', at}];
      const {maximum} = inputs.surfaceCache;
      let best = '';
      const failedCells = countFailedCells(inputs.surfaceCache.values);
      if (maximum.index >= 0) {
        const bestRange = inputs.rangeSamples[maximum.index % COLUMNS];
        const bestSill = inputs.sillSamples[Math.floor(maximum.index / COLUMNS)];
        glyphs.push({
          shape: 'ring',
          at: getPanelPoint(inputs.panel, RANGE_AXIS, SILL_AXIS, bestRange, bestSill),
          radius: 0.03
        });
        best = ` (best ${formatNumber(maximum.value, 1)})`;
      }
      const total = peek(outputs.logLikelihood);
      const status = peek(outputs.status);
      const result = total && status ? getVecchiaLogLikelihoodResult(total, status) : null;
      return {
        glyphs,
        spotlight: null,
        readout: !result
          ? `range ${values.range} · sill ${values.sill}${getNuggetNote(values)}`
          : Number.isFinite(result.logLikelihood)
            ? `logL ${formatNumber(result.logLikelihood, 1)}${best} · range ${values.range} · sill ${values.sill}${getNuggetNote(values)}`
            : `sites failed: no likelihood · range ${values.range} · sill ${values.sill}${failedCells > 0 ? ` · ${failedCells} cells failed` : ''}`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    ...(extra ? [{layer: 'lines' as const, context: true, segments: extra.links, alpha: 0.7}] : []),
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.siteTerms,
      ramp: 'sequential',
      range: [-3, 0.5],
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
            range: [-SURFACE_DEPTH, 0] as const,
            bands: 12
          }
        ]
      : [])
  ]
});
