// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUVecchiaFisherScoring: one evaluation gives the Vecchia log-likelihood of Helene log rainfall,
// its score (gradient in log sill, log range, log nugget) and the expected information. The arrow
// is the Fisher scoring step from the pointer, theta + I^-1 score, drawn on the (range, sill)
// likelihood surface (one evaluation of the same compiled graph per cell; ring: its maximum).
// Left: each station's log-likelihood term.

import {
  buildVecchiaConditioningSets,
  getGPUVecchiaLogLikelihoodParameterValues,
  getVecchiaFisherEvaluation,
  getVecchiaMaxMinOrdering,
  GPUVecchiaFisherScoring,
  GPU_VECCHIA_FISHER_CELL_COUNT,
  GPU_VECCHIA_PARAMETER_LENGTH,
  type GPUVecchiaCovarianceModel
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer, type RunBuilder} from '../../define-run';
import {formatNumber, type RunGlyph} from '../../interaction';
import {
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
import {parameters} from './vecchia-fisher-scoring.parameters';

const STATION_COUNT = 420;
const NEIGHBOR_COUNT = 10;
const COLUMNS = 16;
const ROWS = 14;
const MODELS: GPUVecchiaCovarianceModel[] = ['exponential', 'matern-3/2'];
const RANGE_AXIS: PanelAxis = {parameter: parameters[0], logarithmic: true};
const SILL_AXIS: PanelAxis = {parameter: parameters[1], logarithmic: true};
const SURFACE_DEPTH = 40;
/** Stations whose conditioning links are drawn: the ones with the lowest terms. */
const LINKED_STATION_COUNT = 10;
/** Steps longer than this (in log units) are shortened for drawing. */
const LARGEST_DRAWN_STEP = 2.5;

function getCovariance(values: Record<string, number>, range: number, sill: number) {
  const model = MODELS[values.model];
  const nugget =
    model === 'exponential'
      ? Math.max(values.nugget, 1e-4)
      : Math.max(values.nugget, 0.0011 * sill);
  return {model, range, sill, nugget};
}

/** ` · nugget x` when the floor raised the slider's nugget, so the readout shows what was evaluated. */
function getNuggetNote(values: Record<string, number>): string {
  const effective = getCovariance(values, values.range, values.sill).nugget;
  return effective !== values.nugget ? ` · nugget ${formatNumber(effective, 4)}` : '';
}

type Uploaded = {
  positions: RunBuffer<'float32x2'>;
  observations: RunBuffer<'float32x4'>;
  offsets: RunBuffer<'uint32'>;
  neighbors: RunBuffer<'uint32'>;
};

function addScoring(builder: RunBuilder, id: string, rowCount: number, scene: Uploaded) {
  const parameterBuffer = builder.allocate(
    `${id}-parameters`,
    'float32',
    GPU_VECCHIA_PARAMETER_LENGTH
  );
  const siteTerms = builder.allocate(
    `${id}-site-terms`,
    'float32',
    rowCount * GPU_VECCHIA_FISHER_CELL_COUNT
  );
  const totals = builder.allocate(`${id}-totals`, 'float32', 2 * GPU_VECCHIA_FISHER_CELL_COUNT);
  const status = builder.allocate(`${id}-status`, 'uint32', 3);
  builder.graph.add(
    new GPUVecchiaFisherScoring({
      id,
      positions: scene.positions.view,
      observations: scene.observations.view,
      neighborhoodOffsets: scene.offsets.view,
      neighborhoods: scene.neighbors.view,
      neighborCount: NEIGHBOR_COUNT,
      parameters: parameterBuffer.view,
      output: {
        siteTerms: siteTerms.view,
        siteStatus: builder.allocate(`${id}-site-status`, 'uint32', rowCount).view,
        totals: totals.view,
        status: status.view
      }
    })
  );
  return {parameterBuffer, siteTerms, totals, status};
}

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

type Inputs = ReturnType<typeof createInputs>;

function build(builder: RunBuilder, inputs: Inputs) {
  const uploaded: Uploaded = {
    positions: builder.upload('positions', 'float32x2', inputs.positions),
    observations: builder.upload('observations', 'float32x4', packObservations(inputs.residuals)),
    offsets: builder.upload('offsets', 'uint32', inputs.sets.offsets),
    neighbors: builder.upload('neighbors', 'uint32', inputs.sets.neighbors)
  };
  const main = addScoring(builder, 'fisher', inputs.count, uploaded);
  // A second instance over the same data is evaluated once per surface cell after the run.
  const sweep = addScoring(builder, 'fisher-sweep', inputs.count, uploaded);
  return {
    outputs: {
      ...main,
      sweepParameters: sweep.parameterBuffer,
      sweepTotals: sweep.totals,
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

/** The Fisher scoring step `I^-1 score` in (log sill, log range, log nugget), or null. */
function getFisherStep(totals: ArrayLike<number>, status: ArrayLike<number>) {
  const evaluation = getVecchiaFisherEvaluation(totals, status);
  if (!Number.isFinite(evaluation.logLikelihood)) return null;
  // The pad is a slice at fixed nugget, so solve the (log sill, log range) sub-system of I step =
  // score; dropping the nugget component of the full three-parameter step would be wrong here.
  const [i00, i01, , i11] = evaluation.information;
  const [sillScore, rangeScore] = evaluation.score;
  const determinant = i00 * i11 - i01 * i01;
  const step = [
    (i11 * sillScore - i01 * rangeScore) / determinant,
    (i00 * rangeScore - i01 * sillScore) / determinant
  ];
  return step.every(Number.isFinite) ? {evaluation, step} : null;
}

export default defineOperationRun({
  parameters,
  caption:
    'Fisher scoring on the Vecchia likelihood of Helene log rainfall: the arrow is one step from the pointer.',
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
        // Cell 0 of the totals, `[high, low]`, is the log-likelihood.
        source: outputs.sweepTotals,
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
    const bytes = await outputs.siteTerms.buffer.readAsync();
    const siteRows = new Float32Array(
      bytes.buffer,
      bytes.byteOffset,
      inputs.count * GPU_VECCHIA_FISHER_CELL_COUNT
    );
    const terms = Float32Array.from(
      {length: inputs.count},
      (_, row) => siteRows[row * GPU_VECCHIA_FISHER_CELL_COUNT]
    );
    return {
      surface: upload('surface', 'float32', relative),
      terms: upload('terms', 'float32', terms),
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
      maximum: cache.maximum
    };
  },
  interaction: createPanelPad<Inputs, Outputs>({
    hint: 'start a scoring step',
    rectangle: ({inputs}) => inputs.panel,
    x: RANGE_AXIS,
    y: SILL_AXIS,
    decorate(values, {inputs, outputs, peek}) {
      const at = getPanelPoint(inputs.panel, RANGE_AXIS, SILL_AXIS, values.range, values.sill);
      const glyphs: RunGlyph[] = [getFrameGlyph(inputs.panel), {shape: 'dot', at}];
      const {maximum} = inputs.surfaceCache;
      if (maximum.index >= 0) {
        glyphs.push({
          shape: 'ring',
          at: getPanelPoint(
            inputs.panel,
            RANGE_AXIS,
            SILL_AXIS,
            inputs.rangeSamples[maximum.index % COLUMNS],
            inputs.sillSamples[Math.floor(maximum.index / COLUMNS)]
          ),
          radius: 0.03
        });
      }
      const totals = peek(outputs.totals);
      const status = peek(outputs.status);
      const fisher = totals && status ? getFisherStep(totals, status) : null;
      if (!fisher)
        return {
          glyphs,
          spotlight: null,
          readout: `range ${values.range} · sill ${values.sill}${getNuggetNote(values)}`
        };
      const [sillStep, rangeStep] = fisher.step;
      const length = Math.hypot(sillStep, rangeStep);
      const shrink = length > LARGEST_DRAWN_STEP ? LARGEST_DRAWN_STEP / length : 1;
      const nextRange = values.range * Math.exp(rangeStep * shrink);
      const nextSill = values.sill * Math.exp(sillStep * shrink);
      glyphs.push({
        shape: 'arrow',
        from: at,
        to: getPanelPoint(inputs.panel, RANGE_AXIS, SILL_AXIS, nextRange, nextSill)
      });
      return {
        glyphs,
        spotlight: null,
        readout: `logL ${formatNumber(fisher.evaluation.logLikelihood, 1)}${shrink < 1 ? ' · step clipped' : ''} · step to range ${formatNumber(values.range * Math.exp(rangeStep), 2)}, sill ${formatNumber(values.sill * Math.exp(sillStep), 2)}${getNuggetNote(values)}`
      };
    }
  }),
  draw: ({inputs, outputs, extra}) => [
    ...(extra
      ? [
          {layer: 'lines' as const, context: true, segments: extra.links, alpha: 0.7},
          {
            layer: 'dots' as const,
            positions: outputs.positions,
            values: extra.terms,
            ramp: 'sequential' as const,
            range: [-3, 0.5] as const,
            radius: 5
          },
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
