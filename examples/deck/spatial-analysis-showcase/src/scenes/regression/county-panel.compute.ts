// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUSpatialErrorMaximumLikelihood,
  GPUSpatialLagImpacts,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialPanelErrorMaximumLikelihood,
  GPUSpatialPanelLagMaximumLikelihood,
  GPUSpatialPanelRegression,
  GPUInstrumentalVariablesCovariance
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type GraphDataView} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type CountyPanelOptions = {
  estimator:
    | 'panel-2sls-pooled'
    | 'panel-2sls-unit'
    | 'lag-ml-random'
    | 'lag-ml-unit'
    | 'error-ml-unit'
    | 'cross-lag-ml-1990'
    | 'cross-error-ml-1990';
  period: '1970' | '1980' | '1990';
  map: 'observed' | 'residual';
  covariance: 'HC0' | 'HC1' | 'HAC-50' | 'HAC-100';
};

type CountyPanelCovarianceMethod = CountyPanelOptions['covariance'];

const ENTITY_COUNT = 21;
const PERIOD_COUNT = 3;
const ROW_COUNT = ENTITY_COUNT * PERIOD_COUNT;
const PREDICTOR_COUNT = 2;
const SUMMARY_LENGTH = 32;
const PERIOD_INDEX: Record<CountyPanelOptions['period'], number> = {1970: 0, 1980: 1, 1990: 2};
const MODEL_INDEX: Record<CountyPanelOptions['estimator'], number> = {
  'panel-2sls-pooled': 0,
  'panel-2sls-unit': 1,
  'lag-ml-random': 2,
  'lag-ml-unit': 3,
  'error-ml-unit': 4,
  'cross-lag-ml-1990': 5,
  'cross-error-ml-1990': 6
};
const COVARIANCE_METHODS: CountyPanelCovarianceMethod[] = ['HC0', 'HC1', 'HAC-50', 'HAC-100'];
const COVARIANCE_LABELS: Record<CountyPanelCovarianceMethod, string> = {
  HC0: 'HC0',
  HC1: 'HC1 · 63/59 correction',
  'HAC-50': 'Gaussian space-time · 50 km',
  'HAC-100': 'Gaussian space-time · 100 km'
};

type ModelBuffers = {
  residuals: Buffer;
  table: Buffer;
  summary: Buffer;
  status: Buffer;
};

/** Historical county outcomes and five explicitly different balanced spatial-panel fits. */
export async function createCountyPanel(
  ctx: SceneContext<CountyPanelOptions>
): Promise<SceneInstance<CountyPanelOptions>> {
  const dataset = ctx.datasets.get('ncovr-new-jersey');
  const frame = {kind: 'azimuthal-equidistant' as const, origin: [-74.55, 40.05] as const};
  const projection = createAnalysisFrameProjection(frame);
  const resources = new SpatialAnalysisResources(ctx.device, 'county-panel');
  const features = dataset.geojson?.features ?? [];
  if (features.length !== ENTITY_COUNT)
    throw new Error('NCOVR common-support geometry must contain all 21 New Jersey units');
  const mesh = buildPolygonMesh(dataset.geojson!, projection.project);
  const polygonBuffers = createPolygonMeshBuffers(resources, mesh, 'counties');

  const field = (name: string) => dataset.column<Float32Array>(name);
  const outcome = Float32Array.from([
    ...field('unemployment1970'),
    ...field('unemployment1980'),
    ...field('unemployment1990')
  ]);
  const poverty = [
    field('familyPoverty1969'),
    field('familyPoverty1979'),
    field('familyPoverty1989')
  ];
  const density = [field('logDensity1970'), field('logDensity1980'), field('logDensity1990')];
  const predictors = new Float32Array(ROW_COUNT * PREDICTOR_COUNT);
  const means = [0, 0];
  for (let period = 0; period < PERIOD_COUNT; period++) {
    for (let entity = 0; entity < ENTITY_COUNT; entity++) {
      const row = period * ENTITY_COUNT + entity;
      predictors[row * PREDICTOR_COUNT] = poverty[period][entity];
      predictors[row * PREDICTOR_COUNT + 1] = density[period][entity];
      means[0] += poverty[period][entity] / ROW_COUNT;
      means[1] += density[period][entity] / ROW_COUNT;
    }
  }
  const deviations = [0, 0];
  for (let row = 0; row < ROW_COUNT; row++) {
    for (let column = 0; column < PREDICTOR_COUNT; column++) {
      const difference = predictors[row * PREDICTOR_COUNT + column] - means[column];
      deviations[column] += (difference * difference) / ROW_COUNT;
    }
  }
  for (let row = 0; row < ROW_COUNT; row++) {
    for (let column = 0; column < PREDICTOR_COUNT; column++) {
      predictors[row * PREDICTOR_COUNT + column] =
        (predictors[row * PREDICTOR_COUNT + column] - means[column]) /
        Math.sqrt(deviations[column]);
    }
  }

  const queenOffsets = dataset.column<Uint32Array>('queenOffsets');
  const queenNeighbors = dataset.column<Uint32Array>('queenNeighbors');
  const neighborWeights = new Float32Array(queenNeighbors.length);
  for (let row = 0; row < ENTITY_COUNT; row++) {
    const degree = queenOffsets[row + 1] - queenOffsets[row];
    if (degree > 0) neighborWeights.fill(1 / degree, queenOffsets[row], queenOffsets[row + 1]);
  }
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'county-panel-models'});
  const importedViews = new Map<Buffer, GraphDataView>();
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    const existing = importedViews.get(buffer);
    if (existing) return existing as GraphDataView<Format>;
    const imported = importGraphBuffer(graph, name, buffer, format, length);
    importedViews.set(buffer, imported);
    return imported;
  };
  const offsetsBuffer = resources.createBuffer('queen-offsets', queenOffsets);
  const neighborsBuffer = resources.createBuffer('queen-neighbors', queenNeighbors);
  const weightsBuffer = resources.createBuffer('queen-row-weights', neighborWeights);
  const weights = {
    spatialContext: projection.spatialContext,
    information: {
      rowCount: ENTITY_COUNT,
      transform: 'row' as const,
      symmetric: false,
      provenance: {source: 'NCOVR common-support geometry', criterion: 'queen'}
    },
    offsets: view('queen-offsets', offsetsBuffer, 'uint32', ENTITY_COUNT + 1),
    neighbors: view('queen-neighbors', neighborsBuffer, 'uint32', queenNeighbors.length),
    weights: view('queen-weights', weightsBuffer, 'float32', neighborWeights.length)
  };
  const outcomeBuffer = resources.createBuffer('panel-unemployment', outcome);
  const predictorBuffer = resources.createBuffer('panel-standardized-predictors', predictors);
  const responseView = view('response', outcomeBuffer, 'float32', ROW_COUNT);
  const predictorsView = view('predictors', predictorBuffer, 'float32', predictors.length);

  const makeModel = (name: string): ModelBuffers => ({
    residuals: resources.createBuffer(`${name}-residuals`, ROW_COUNT * 4),
    table: resources.createBuffer(`${name}-table`, (PREDICTOR_COUNT + 2) * 4 * 4),
    summary: resources.createBuffer(`${name}-summary`, SUMMARY_LENGTH * 4),
    status: resources.createBuffer(`${name}-status`, 4)
  });
  const pooled = makeModel('pooled-2sls');
  const fixed = makeModel('unit-2sls');
  const randomLag = makeModel('random-lag-ml');
  const unitLag = makeModel('unit-lag-ml');
  const unitError = makeModel('unit-error-ml');
  const make2SlsOutput = (name: string, target: ModelBuffers) => ({
    status: {code: view(`${name}-status`, target.status, 'uint32', 1)},
    table: view(`${name}-table`, target.table, 'float32', (PREDICTOR_COUNT + 2) * 4),
    summary: view(`${name}-summary`, target.summary, 'float32', SUMMARY_LENGTH),
    residuals: view(`${name}-residuals`, target.residuals, 'float32', ROW_COUNT)
  });
  graph.add(
    new GPUSpatialPanelRegression({
      id: 'pooled-panel-spatial-2sls',
      weights,
      periodCount: PERIOD_COUNT,
      effects: 'pooled',
      predictorCount: PREDICTOR_COUNT,
      predictors: predictorsView,
      response: responseView,
      instrumentOrder: 1,
      output: make2SlsOutput('pooled-2sls', pooled)
    })
  );
  graph.add(
    new GPUSpatialPanelRegression({
      id: 'unit-fixed-effects-panel-spatial-2sls',
      weights,
      periodCount: PERIOD_COUNT,
      effects: 'unit',
      predictorCount: PREDICTOR_COUNT,
      predictors: predictorsView,
      response: responseView,
      instrumentOrder: 1,
      output: make2SlsOutput('unit-2sls', fixed)
    })
  );

  // The pooled-IV sandwich uses the same row-standardized Queen operator as the fitted model.
  // Z contains X and W y; H contains X and W X, with the intercept implicit in both tables.
  const laggedPanelBuffer = resources.createBuffer('pooled-iv-lagged-panel', ROW_COUNT * 3 * 4);
  const laggedPanelView = view(
    'pooled-iv-lagged-panel',
    laggedPanelBuffer,
    'float32',
    ROW_COUNT * 3
  );
  addKernelPass(graph, {
    id: 'pooled-iv-queen-lags',
    invocationCount: ROW_COUNT,
    bindings: [
      {name: 'response', view: responseView, type: 'f32', access: 'read'},
      {name: 'predictors', view: predictorsView, type: 'f32', access: 'read'},
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
      {name: 'lagged', view: laggedPanelView, type: 'f32', access: 'read_write'}
    ],
    body: `let period = index / ${ENTITY_COUNT}u; let entity = index % ${ENTITY_COUNT}u;
var responseLag = 0.0; var firstPredictorLag = 0.0; var secondPredictorLag = 0.0;
for (var slot = offsets[offsetsOffset + entity]; slot < offsets[offsetsOffset + entity + 1u]; slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let neighborRow = period * ${ENTITY_COUNT}u + neighbor; let weight = weights[weightsOffset + slot];
 responseLag += weight * response[responseOffset + neighborRow];
 firstPredictorLag += weight * predictors[predictorsOffset + neighborRow * ${PREDICTOR_COUNT}u];
 secondPredictorLag += weight * predictors[predictorsOffset + neighborRow * ${PREDICTOR_COUNT}u + 1u];
}
lagged[laggedOffset + index * 3u] = responseLag;
lagged[laggedOffset + index * 3u + 1u] = firstPredictorLag;
lagged[laggedOffset + index * 3u + 2u] = secondPredictorLag;`
  });
  const ivDesignBuffer = resources.createBuffer('pooled-iv-structural-design', ROW_COUNT * 3 * 4);
  const ivInstrumentBuffer = resources.createBuffer('pooled-iv-instruments', ROW_COUNT * 4 * 4);
  const ivDesignView = view(
    'pooled-iv-structural-design',
    ivDesignBuffer,
    'float32',
    ROW_COUNT * 3
  );
  const ivInstrumentView = view(
    'pooled-iv-instruments',
    ivInstrumentBuffer,
    'float32',
    ROW_COUNT * 4
  );
  addKernelPass(graph, {
    id: 'pooled-iv-design-tables',
    invocationCount: ROW_COUNT,
    bindings: [
      {name: 'predictors', view: predictorsView, type: 'f32', access: 'read'},
      {name: 'lagged', view: laggedPanelView, type: 'f32', access: 'read'},
      {name: 'design', view: ivDesignView, type: 'f32', access: 'read_write'},
      {name: 'instruments', view: ivInstrumentView, type: 'f32', access: 'read_write'}
    ],
    body: `let firstPredictor = predictors[predictorsOffset + index * ${PREDICTOR_COUNT}u]; let secondPredictor = predictors[predictorsOffset + index * ${PREDICTOR_COUNT}u + 1u];
design[designOffset + index * 3u] = firstPredictor; design[designOffset + index * 3u + 1u] = secondPredictor; design[designOffset + index * 3u + 2u] = lagged[laggedOffset + index * 3u];
instruments[instrumentsOffset + index * 4u] = firstPredictor; instruments[instrumentsOffset + index * 4u + 1u] = secondPredictor; instruments[instrumentsOffset + index * 4u + 2u] = lagged[laggedOffset + index * 3u + 1u]; instruments[instrumentsOffset + index * 4u + 3u] = lagged[laggedOffset + index * 3u + 2u];`
  });
  const pooledResidualView = view(
    'pooled-2sls-residuals-for-covariance',
    pooled.residuals,
    'float32',
    ROW_COUNT
  );
  const makeHacKernel = (bandwidthMeters: number, suffix: string) => {
    const coordinates = mesh.labelPoints.map(([longitude, latitude]) =>
      projection.project(longitude, latitude)
    );
    const offsets = new Uint32Array(ROW_COUNT + 1);
    const neighbors = new Uint32Array(ROW_COUNT * ROW_COUNT);
    const kernelWeights = new Float32Array(ROW_COUNT * ROW_COUNT);
    for (let row = 0; row < ROW_COUNT; row++) {
      offsets[row] = row * ROW_COUNT;
      const rowEntity = row % ENTITY_COUNT;
      const rowPeriod = Math.floor(row / ENTITY_COUNT);
      for (let column = 0; column < ROW_COUNT; column++) {
        const columnEntity = column % ENTITY_COUNT;
        const columnPeriod = Math.floor(column / ENTITY_COUNT);
        const distanceSquared =
          (coordinates[rowEntity][0] - coordinates[columnEntity][0]) ** 2 +
          (coordinates[rowEntity][1] - coordinates[columnEntity][1]) ** 2;
        const periodDifference = Math.abs(rowPeriod - columnPeriod) * 10;
        const spatialExponent = distanceSquared / (2 * bandwidthMeters ** 2);
        const temporalExponent = periodDifference ** 2 / (2 * 10 ** 2);
        const slot = row * ROW_COUNT + column;
        neighbors[slot] = column;
        kernelWeights[slot] = Math.exp(-spatialExponent - temporalExponent);
      }
    }
    offsets[ROW_COUNT] = ROW_COUNT * ROW_COUNT;
    const kernelOffsetsBuffer = resources.createBuffer(`pooled-iv-hac-${suffix}-offsets`, offsets);
    const kernelNeighborsBuffer = resources.createBuffer(
      `pooled-iv-hac-${suffix}-neighbors`,
      neighbors
    );
    const kernelWeightsBuffer = resources.createBuffer(
      `pooled-iv-hac-${suffix}-weights`,
      kernelWeights
    );
    return {
      spatialContext: projection.spatialContext,
      information: {
        rowCount: ROW_COUNT,
        transform: 'kernel' as const,
        symmetric: true,
        hasSelfWeights: true,
        provenance: {source: 'Authored Gaussian space-time sensitivity', criterion: 'full dense'}
      },
      offsets: view(
        `pooled-iv-hac-${suffix}-offsets`,
        kernelOffsetsBuffer,
        'uint32',
        ROW_COUNT + 1
      ),
      neighbors: view(
        `pooled-iv-hac-${suffix}-neighbors`,
        kernelNeighborsBuffer,
        'uint32',
        ROW_COUNT * ROW_COUNT
      ),
      weights: view(
        `pooled-iv-hac-${suffix}-weights`,
        kernelWeightsBuffer,
        'float32',
        ROW_COUNT * ROW_COUNT
      )
    };
  };
  const covarianceOutputs = new Map<
    CountyPanelCovarianceMethod,
    {covariance: Buffer; status: Buffer}
  >();
  for (const method of COVARIANCE_METHODS) {
    const covarianceBuffer = resources.createBuffer(`pooled-iv-${method}-covariance`, 16 * 4);
    const statusBuffer = resources.createBuffer(`pooled-iv-${method}-status`, 4);
    const kernel =
      method === 'HAC-50'
        ? makeHacKernel(50000, '50km')
        : method === 'HAC-100'
          ? makeHacKernel(100000, '100km')
          : undefined;
    const covarianceMethod: 'HC0' | 'HC1' | 'HAC' =
      method === 'HC0' ? 'HC0' : method === 'HC1' ? 'HC1' : 'HAC';
    graph.add(
      new GPUInstrumentalVariablesCovariance({
        id: `pooled-panel-iv-covariance-${method}`,
        design: ivDesignView,
        coefficientCount: 3,
        instruments: ivInstrumentView,
        instrumentCount: 4,
        residuals: pooledResidualView,
        method: covarianceMethod,
        kernel,
        output: {
          covariance: view(`pooled-iv-${method}-covariance`, covarianceBuffer, 'float32', 16),
          status: view(`pooled-iv-${method}-status`, statusBuffer, 'uint32', 1)
        }
      })
    );
    covarianceOutputs.set(method, {covariance: covarianceBuffer, status: statusBuffer});
  }

  const makeLagOutput = (name: string, target: ModelBuffers, hausman = false) => {
    const covariance = resources.createBuffer(`${name}-covariance`, (PREDICTOR_COUNT + 2) ** 2 * 4);
    const fitted = resources.createBuffer(`${name}-fitted`, ROW_COUNT * 4);
    const hausmanTests = hausman ? resources.createBuffer(`${name}-hausman-tests`, 12) : null;
    const hausmanStatus = hausman ? resources.createBuffer(`${name}-hausman-status`, 4) : null;
    return {
      output: {
        coefficients: view(`${name}-coefficients`, target.table, 'float32', PREDICTOR_COUNT + 3),
        covariance: view(`${name}-covariance`, covariance, 'float32', (PREDICTOR_COUNT + 2) ** 2),
        summary: view(`${name}-summary`, target.summary, 'float32', 6),
        status: view(`${name}-status`, target.status, 'uint32', 1),
        fitted: view(`${name}-fitted`, fitted, 'float32', ROW_COUNT),
        residuals: view(`${name}-residuals`, target.residuals, 'float32', ROW_COUNT)
      },
      hausman:
        hausman && hausmanTests && hausmanStatus
          ? {
              tests: view(`${name}-hausman-tests`, hausmanTests, 'float32', 3),
              status: view(`${name}-hausman-status`, hausmanStatus, 'uint32', 1)
            }
          : null,
      hausmanTests,
      hausmanStatus
    };
  };
  const randomLagOutput = makeLagOutput('random-lag-ml', randomLag, true);
  graph.add(
    new GPUSpatialPanelLagMaximumLikelihood({
      id: 'random-effects-panel-lag-ml',
      weights,
      periodCount: PERIOD_COUNT,
      predictors: predictorsView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      effects: 'random',
      maximumIterations: 48,
      maximumSearchIterations: 40,
      tolerance: 1e-5,
      output: randomLagOutput.output,
      hausman: randomLagOutput.hausman ?? undefined
    })
  );
  graph.add(
    new GPUSpatialPanelLagMaximumLikelihood({
      id: 'unit-fixed-effects-panel-lag-ml',
      weights,
      periodCount: PERIOD_COUNT,
      predictors: predictorsView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      effects: 'unit',
      maximumIterations: 48,
      maximumSearchIterations: 40,
      tolerance: 1e-5,
      output: makeLagOutput('unit-lag-ml', unitLag).output
    })
  );
  const errorStatus = resources.createBuffer('unit-error-ml-converged', 4);
  const errorConvergedView = view('unit-error-ml-converged', errorStatus, 'uint32', 1);
  const errorDiagnostics = resources.createBuffer('unit-error-ml-diagnostics', 24);
  const errorDiagnosticStatus = resources.createBuffer('unit-error-ml-diagnostic-status', 4);
  graph.add(
    new GPUSpatialPanelErrorMaximumLikelihood({
      id: 'unit-fixed-effects-panel-error-ml',
      weights,
      periodCount: PERIOD_COUNT,
      predictors: predictorsView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      maximumPower: 64,
      traceMode: 'exact',
      output: {
        status: {
          code: view('unit-error-ml-status', unitError.status, 'uint32', 1),
          converged: view('unit-error-ml-converged', errorStatus, 'uint32', 1)
        },
        table: view('unit-error-ml-table', unitError.table, 'float32', (PREDICTOR_COUNT + 2) * 4),
        summary: view('unit-error-ml-summary', unitError.summary, 'float32', 22),
        residuals: view('unit-error-ml-residuals', unitError.residuals, 'float32', ROW_COUNT)
      },
      diagnostics: {
        tests: view('unit-error-ml-diagnostics', errorDiagnostics, 'float32', 6),
        status: view('unit-error-ml-diagnostic-status', errorDiagnosticStatus, 'uint32', 1)
      }
    })
  );

  // The 1990 cross-section is a separate, labelled comparison; it does not borrow panel effects.
  const crossResponseBuffer = resources.createBuffer(
    'cross-section-unemployment-1990',
    field('unemployment1990')
  );
  const crossPredictorValues = new Float32Array(ENTITY_COUNT * PREDICTOR_COUNT);
  for (let entity = 0; entity < ENTITY_COUNT; entity++) {
    crossPredictorValues[entity * PREDICTOR_COUNT] = field('familyPoverty1989')[entity];
    crossPredictorValues[entity * PREDICTOR_COUNT + 1] = field('logDensity1990')[entity];
  }
  for (let column = 0; column < PREDICTOR_COUNT; column++) {
    let mean = 0;
    for (let entity = 0; entity < ENTITY_COUNT; entity++)
      mean += crossPredictorValues[entity * PREDICTOR_COUNT + column] / ENTITY_COUNT;
    let variance = 0;
    for (let entity = 0; entity < ENTITY_COUNT; entity++)
      variance +=
        (crossPredictorValues[entity * PREDICTOR_COUNT + column] - mean) ** 2 / ENTITY_COUNT;
    for (let entity = 0; entity < ENTITY_COUNT; entity++)
      crossPredictorValues[entity * PREDICTOR_COUNT + column] =
        (crossPredictorValues[entity * PREDICTOR_COUNT + column] - mean) / Math.sqrt(variance);
  }
  const crossResponseView = view(
    'cross-section-response',
    crossResponseBuffer,
    'float32',
    ENTITY_COUNT
  );
  const crossPredictorBuffer = resources.createBuffer(
    'cross-section-predictors',
    crossPredictorValues
  );
  const crossPredictorView = view(
    'cross-section-predictors',
    crossPredictorBuffer,
    'float32',
    crossPredictorValues.length
  );
  const crossLag = makeModel('cross-lag-ml-1990');
  const crossError = makeModel('cross-error-ml-1990');
  const crossLagConverged = resources.createBuffer('cross-lag-ml-converged', 4);
  const crossErrorConverged = resources.createBuffer('cross-error-ml-converged', 4);
  const crossLagTraces = {
    powerTraces: view(
      'cross-lag-traces',
      resources.createBuffer('cross-lag-traces', 64 * 4),
      'float32',
      64
    ),
    powerSums: view(
      'cross-lag-power-sums',
      resources.createBuffer('cross-lag-power-sums', 64 * 4),
      'float32',
      64
    ),
    infinityNorm: view(
      'cross-lag-infinity-norm',
      resources.createBuffer('cross-lag-infinity-norm', 4),
      'float32',
      1
    )
  };
  const crossLagTable = view(
    'cross-lag-table',
    crossLag.table,
    'float32',
    (PREDICTOR_COUNT + 2) * 4
  );
  graph.add(
    new GPUSpatialLagMaximumLikelihood({
      id: 'cross-sectional-spatial-lag-ml-1990',
      weights,
      predictors: crossPredictorView,
      predictorCount: PREDICTOR_COUNT,
      response: crossResponseView,
      maximumPower: 64,
      traceMode: 'exact',
      output: {
        status: {
          code: view('cross-lag-status', crossLag.status, 'uint32', 1),
          converged: view('cross-lag-converged', crossLagConverged, 'uint32', 1)
        },
        table: crossLagTable,
        summary: view('cross-lag-summary', crossLag.summary, 'float32', 22),
        residuals: view('cross-lag-residuals', crossLag.residuals, 'float32', ENTITY_COUNT),
        ...crossLagTraces
      }
    })
  );
  graph.add(
    new GPUSpatialErrorMaximumLikelihood({
      id: 'cross-sectional-spatial-error-ml-1990',
      weights,
      predictors: crossPredictorView,
      predictorCount: PREDICTOR_COUNT,
      response: crossResponseView,
      maximumPower: 64,
      traceMode: 'exact',
      output: {
        status: {
          code: view('cross-error-status', crossError.status, 'uint32', 1),
          converged: view('cross-error-converged', crossErrorConverged, 'uint32', 1)
        },
        table: view('cross-error-table', crossError.table, 'float32', (PREDICTOR_COUNT + 2) * 4),
        summary: view('cross-error-summary', crossError.summary, 'float32', 22),
        residuals: view('cross-error-residuals', crossError.residuals, 'float32', ENTITY_COUNT)
      }
    })
  );
  const impactsBuffer = resources.createBuffer('cross-lag-impacts-1990', PREDICTOR_COUNT * 3 * 4);
  graph.add(
    new GPUSpatialLagImpacts({
      id: 'cross-sectional-lag-impacts-1990',
      coefficients: crossLagTable,
      coefficientStride: 4,
      rhoIndex: PREDICTOR_COUNT + 1,
      predictorCount: PREDICTOR_COUNT,
      rowCount: ENTITY_COUNT,
      maximumPower: 64,
      traces: crossLagTraces,
      status: view('cross-lag-status-for-impacts', crossLag.status, 'uint32', 1),
      output: {impacts: view('cross-lag-impacts', impactsBuffer, 'float32', PREDICTOR_COUNT * 3)}
    })
  );

  const mapValues = resources.createBuffer('display-values', ENTITY_COUNT * 4);
  const periodParameter = resources.createBuffer('display-period', new Uint32Array([0, 0, 0]));
  const mapView = view('display-values', mapValues, 'float32', ENTITY_COUNT);
  const periodView = view('display-parameters', periodParameter, 'uint32', 3);
  const modelResidualViews = [
    pooled,
    fixed,
    randomLag,
    unitLag,
    unitError,
    crossLag,
    crossError
  ].map((model, index) =>
    view(
      `display-residual-source-${index}`,
      model.residuals,
      'float32',
      index < 5 ? ROW_COUNT : ENTITY_COUNT
    )
  );
  const modelStatusViews = [pooled, fixed, randomLag, unitLag, unitError, crossLag, crossError].map(
    (model, index) => view(`display-status-source-${index}`, model.status, 'uint32', 1)
  );
  const crossLagConvergedView = view(
    'cross-lag-converged-for-display',
    crossLagConverged,
    'uint32',
    1
  );
  const crossErrorConvergedView = view(
    'cross-error-converged-for-display',
    crossErrorConverged,
    'uint32',
    1
  );
  addKernelPass(graph, {
    id: 'select-observed-period-values',
    invocationCount: ENTITY_COUNT,
    bindings: [
      {name: 'response', view: responseView, type: 'f32', access: 'read'},
      {name: 'parameters', view: periodView, type: 'u32', access: 'read'},
      {name: 'output', view: mapView, type: 'f32', access: 'read_write'}
    ],
    body: `if (parameters[parametersOffset + 2u] == 0u) { let row = parameters[parametersOffset] * ${ENTITY_COUNT}u + index; output[outputOffset + index] = response[responseOffset + row]; }`
  });
  const convergedByModel: (GraphDataView<'uint32'> | null)[] = [
    null,
    null,
    null,
    null,
    errorConvergedView,
    crossLagConvergedView,
    crossErrorConvergedView
  ];
  for (let modelIndex = 0; modelIndex < modelResidualViews.length; modelIndex++) {
    const convergedView = convergedByModel[modelIndex];
    addKernelPass(graph, {
      id: `select-estimator-residuals-${modelIndex}`,
      invocationCount: ENTITY_COUNT,
      bindings: [
        {name: 'residual', view: modelResidualViews[modelIndex], type: 'f32', access: 'read'},
        {name: 'status', view: modelStatusViews[modelIndex], type: 'u32', access: 'read'},
        ...(convergedView
          ? [
              {
                name: 'converged',
                view: convergedView,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'parameters', view: periodView, type: 'u32', access: 'read'},
        {name: 'output', view: mapView, type: 'f32', access: 'read_write'}
      ],
      declarations:
        'fn getNaN() -> f32 { var bits: u32 = 0x7fc00000u; return bitcast<f32>(bits); }',
      body: `if (parameters[parametersOffset + 2u] == 0u || parameters[parametersOffset + 1u] != ${modelIndex}u) { return; }
let sourceRow = select(index, parameters[parametersOffset] * ${ENTITY_COUNT}u + index, ${modelIndex < 5 ? 'true' : 'false'});
let valid = status[statusOffset] == 0u${convergedView ? ' && converged[convergedOffset] == 1u' : ''};
output[outputOffset + index] = select(residual[residualOffset + sourceRow], getNaN(), !valid);`
    });
  }

  const compiled = resources.track(await graph.compileAsync());
  let covarianceDirty = true;
  const covarianceReaders = new Map<CountyPanelCovarianceMethod, SummaryReader>();
  for (const method of COVARIANCE_METHODS) {
    const output = covarianceOutputs.get(method)!;
    covarianceReaders.set(
      method,
      new SummaryReader(
        resources,
        `pooled-iv-covariance-${method}`,
        [
          {buffer: output.covariance, size: 16 * 4},
          {buffer: output.status, size: 4}
        ],
        bytes => {
          if (ctx.options.estimator !== 'panel-2sls-pooled' || ctx.options.covariance !== method)
            return;
          const status = new Uint32Array(bytes, 16 * 4, 1)[0];
          if (status !== 0) {
            ctx.setReadout(
              'ivCovarianceStatus',
              `${COVARIANCE_LABELS[method]} unavailable (GPU covariance status ${status}); coefficients are unchanged.`
            );
            ctx.setReadout('ivCovarianceMatrix', 'Matrix unavailable for this covariance status.');
            ctx.setReadout('ivCovarianceStandardErrors', 'Standard errors unavailable.');
            ctx.setChart('ivCovarianceChart', null);
            return;
          }
          const covarianceValues = new Float32Array(bytes, 0, 16);
          if (!Array.from(covarianceValues).every(Number.isFinite)) {
            ctx.setReadout(
              'ivCovarianceStatus',
              'GPU covariance returned non-finite matrix entries.'
            );
            ctx.setReadout(
              'ivCovarianceMatrix',
              'Matrix unavailable because an entry is non-finite.'
            );
            ctx.setReadout('ivCovarianceStandardErrors', 'Standard errors unavailable.');
            ctx.setChart('ivCovarianceChart', null);
            return;
          }
          const coefficientNames = ['Intercept', 'Poverty', 'Density', 'Wy'];
          const standardErrors = coefficientNames.map((coefficientName, index) => {
            const variance = covarianceValues[index * 4 + index];
            return variance >= 0 ? Math.sqrt(variance) : Number.NaN;
          });
          ctx.setReadout(
            'ivCovarianceStatus',
            `${COVARIANCE_LABELS[method]} succeeded; estimates unchanged. This is a covariance sensitivity, not a significance or causal decision.`
          );
          ctx.setReadout(
            'ivCovarianceStandardErrors',
            coefficientNames
              .map(
                (name, index) =>
                  `${name} ${Number.isFinite(standardErrors[index]) ? standardErrors[index].toFixed(4) : 'invalid diagonal'}`
              )
              .join(' · ')
          );
          ctx.setReadout(
            'ivCovarianceMatrix',
            covarianceValues
              .reduce<string[]>((rows, value, index) => {
                if (index % 4 === 0) rows.push('');
                rows[rows.length - 1] += `${index % 4 ? '  ' : ''}${value.toFixed(5)}`;
                return rows;
              }, [])
              .map((row, index) => `${coefficientNames[index]}: ${row}`)
              .join(' | ')
          );
          ctx.setChart('ivCovarianceChart', {
            kind: 'bars',
            values: standardErrors,
            labels: coefficientNames,
            title: `${COVARIANCE_LABELS[method]} standard errors`
          });
        }
      )
    );
  }
  let mapDirty = true;
  let mapGeneration = 0;
  let mapReadGeneration = 0;
  let latestMapValues = new Float32Array(ENTITY_COUNT).fill(Number.NaN);
  let lastRequestedEstimator: CountyPanelOptions['estimator'] = ctx.options.estimator;
  const readers = new Map<CountyPanelOptions['estimator'], SummaryReader>();
  const modelByKey = new Map<CountyPanelOptions['estimator'], ModelBuffers>([
    ['panel-2sls-pooled', pooled],
    ['panel-2sls-unit', fixed],
    ['lag-ml-random', randomLag],
    ['lag-ml-unit', unitLag],
    ['error-ml-unit', unitError],
    ['cross-lag-ml-1990', crossLag],
    ['cross-error-ml-1990', crossError]
  ]);
  for (const [key, model] of modelByKey) {
    const coefficientBytes = model.table.byteLength;
    readers.set(
      key,
      new SummaryReader(
        resources,
        `summary-${key}`,
        [
          {buffer: model.table, size: coefficientBytes},
          {buffer: model.status, size: 4},
          ...(key === 'error-ml-unit' ? [{buffer: errorStatus, size: 4}] : []),
          ...(key === 'cross-lag-ml-1990' ? [{buffer: crossLagConverged, size: 4}] : []),
          ...(key === 'cross-error-ml-1990' ? [{buffer: crossErrorConverged, size: 4}] : [])
        ],
        bytes => {
          if (ctx.options.estimator !== key) return;
          const values = new Float32Array(bytes);
          const status = new Uint32Array(bytes, coefficientBytes, 1)[0];
          const converged =
            !['error-ml-unit', 'cross-lag-ml-1990', 'cross-error-ml-1990'].includes(key) ||
            new Uint32Array(bytes, coefficientBytes + 4, 1)[0] === 1;
          const valid = status === 0 && converged;
          ctx.setReadout(
            'fitStatus',
            valid
              ? 'solver status 0; convergence status valid'
              : `invalid or incomplete fit (code ${status}; converged ${converged})`
          );
          if (!valid) {
            ctx.setReadout(
              'povertyCoefficient',
              'not interpreted: fit status is invalid or incomplete'
            );
            ctx.setReadout(
              'densityCoefficient',
              'not interpreted: fit status is invalid or incomplete'
            );
            ctx.setReadout(
              'lagOrErrorParameter',
              'not interpreted: fit status is invalid or incomplete'
            );
          } else if (key === 'lag-ml-random' || key === 'lag-ml-unit') {
            ctx.setReadout('povertyCoefficient', values[1]);
            ctx.setReadout('densityCoefficient', values[2]);
            ctx.setReadout('lagOrErrorParameter', values[3]);
          } else {
            ctx.setReadout(
              'povertyCoefficient',
              Number.isFinite(values[4]) ? values[4] : 'not reported by this fit'
            );
            ctx.setReadout(
              'densityCoefficient',
              Number.isFinite(values[8]) ? values[8] : 'not reported by this fit'
            );
            ctx.setReadout(
              'lagOrErrorParameter',
              Number.isFinite(values[12]) ? values[12] : 'not reported by this fit'
            );
          }
        }
      )
    );
  }
  const hausmanReader = new SummaryReader(
    resources,
    'random-lag-hausman',
    [
      {buffer: randomLagOutput.hausmanTests!, size: 12},
      {buffer: randomLagOutput.hausmanStatus!, size: 4}
    ],
    bytes => {
      if (ctx.options.estimator !== 'lag-ml-random') return;
      const status = new Uint32Array(bytes, 12, 1)[0];
      if (status === 0) {
        const test = new Float32Array(bytes, 0, 3);
        ctx.setReadout(
          'hausmanStatus',
          `valid comparison: statistic ${test[0].toFixed(3)}, df ${test[1].toFixed(0)} (predictors + 2), p ${test[2].toFixed(3)}`
        );
      } else if (status === 1)
        ctx.setReadout('hausmanStatus', 'invalid: one or both fits did not succeed');
      else
        ctx.setReadout(
          'hausmanStatus',
          'invalid covariance difference; do not interpret the test statistic'
        );
    }
  );
  const errorDiagnosticReader = new SummaryReader(
    resources,
    'panel-error-diagnostics',
    [
      {buffer: errorDiagnostics, size: 24},
      {buffer: errorDiagnosticStatus, size: 4}
    ],
    bytes => {
      if (ctx.options.estimator !== 'error-ml-unit') return;
      const status = new Uint32Array(bytes, 24, 1)[0];
      if (status !== 0) {
        ctx.setReadout('panelDiagnostics', `invalid diagnostics status ${status}`);
        return;
      }
      const values = new Float32Array(bytes, 0, 6);
      ctx.setReadout(
        'panelDiagnostics',
        `pooled untransformed LM screen (not fitted unit-error residual diagnostics): LM error ${values[0].toFixed(2)} (p ${values[1].toFixed(3)}); robust LM lag ${values[2].toFixed(2)} (p ${values[3].toFixed(3)}); robust LM error ${values[4].toFixed(2)} (p ${values[5].toFixed(3)})`
      );
    }
  );
  const impactsReader = new SummaryReader(
    resources,
    'cross-lag-impacts-readback',
    [
      {buffer: impactsBuffer, size: PREDICTOR_COUNT * 3 * 4},
      {buffer: crossLag.status, size: 4},
      {buffer: crossLagConverged, size: 4}
    ],
    bytes => {
      if (ctx.options.estimator !== 'cross-lag-ml-1990') return;
      const status = new Uint32Array(bytes, 24, 2);
      if (status[0] !== 0 || status[1] !== 1) {
        ctx.setReadout(
          'crossImpacts',
          `not interpreted: solver status ${status[0]}, converged ${status[1]}`
        );
        return;
      }
      const values = new Float32Array(bytes, 0, PREDICTOR_COUNT * 3);
      ctx.setReadout(
        'crossImpacts',
        `poverty direct ${values[0].toFixed(2)}, indirect ${values[1].toFixed(2)}, total ${values[2].toFixed(2)}; log density direct ${values[3].toFixed(2)}, indirect ${values[4].toFixed(2)}, total ${values[5].toFixed(2)}`
      );
    }
  );
  const names = (dataset.manifest as {names?: Array<{name: string}>}).names ?? [];
  const nameList = names.length
    ? names.map(item => item.name)
    : features.map(feature =>
        String(feature.properties?.NAME ?? feature.properties?.name ?? 'County unit')
      );
  const mapReader = new SummaryReader(
    resources,
    'county-map-readback',
    [{buffer: mapValues, size: ENTITY_COUNT * 4}],
    bytes => {
      if (!mapDirty && mapReadGeneration === mapGeneration) {
        const values = new Float32Array(bytes);
        latestMapValues = values;
        const validCount = Array.from(values).filter(Number.isFinite).length;
        ctx.setReadout('mappedUnits', validCount);
      }
    }
  );
  const getLayers = (): Layer[] => {
    const residualMap = ctx.options.map === 'residual';
    return [
      new SpatialAnalysisPolygonLayer({
        id: 'county-panel-map',
        analysisFrame: frame,
        triangles: polygonBuffers.triangles,
        features: polygonBuffers.triangleFeatures,
        vertexCount: polygonBuffers.vertexCount,
        values: mapValues,
        colormap: residualMap ? 'diverging' : 'cividis',
        valueRange: residualMap ? [-5, 5] : [0, 15],
        noDataColor: [220, 220, 220, 210],
        opacity: 0.88
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'county-panel-outlines',
        analysisFrame: frame,
        segments: polygonBuffers.outline,
        instanceCount: polygonBuffers.outlineCount,
        color: [58, 63, 75, 150],
        widthPixels: 0.65
      })
    ];
  };
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder: CommandEncoder) {
      const modelIndex = MODEL_INDEX[ctx.options.estimator];
      periodParameter.write(
        new Uint32Array([
          ctx.options.estimator.startsWith('cross-') ? 2 : PERIOD_INDEX[ctx.options.period],
          modelIndex,
          ctx.options.map === 'observed' ? 0 : 1
        ])
      );
      const shouldReadCovariance = mapDirty || covarianceDirty;
      if (mapDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!mapReader.isPending) mapReadGeneration = mapGeneration;
        const reader = readers.get(ctx.options.estimator)!;
        reader.request(commandEncoder);
        if (ctx.options.estimator === 'lag-ml-random') hausmanReader.request(commandEncoder);
        if (ctx.options.estimator === 'error-ml-unit')
          errorDiagnosticReader.request(commandEncoder);
        if (ctx.options.estimator === 'cross-lag-ml-1990') impactsReader.request(commandEncoder);
        mapReader.request(commandEncoder);
        mapDirty = false;
        lastRequestedEstimator = ctx.options.estimator;
      } else {
        if (!mapReader.isPending) mapReadGeneration = mapGeneration;
        readers.get(lastRequestedEstimator)?.flush(commandEncoder);
        if (lastRequestedEstimator === 'lag-ml-random') hausmanReader.flush(commandEncoder);
        if (lastRequestedEstimator === 'error-ml-unit') errorDiagnosticReader.flush(commandEncoder);
        if (lastRequestedEstimator === 'cross-lag-ml-1990') impactsReader.flush(commandEncoder);
        mapReader.flush(commandEncoder);
      }
      if (shouldReadCovariance && ctx.options.estimator === 'panel-2sls-pooled') {
        covarianceReaders.get(ctx.options.covariance)?.request(commandEncoder);
        covarianceDirty = false;
      }
      for (const reader of covarianceReaders.values()) reader.flush(commandEncoder);
    },
    getLayers,
    setOption(id) {
      if (id === 'covariance') {
        covarianceDirty = true;
        ctx.setReadout('ivCovarianceStatus', 'pending selected covariance readback');
        ctx.setReadout('ivCovarianceStandardErrors', 'pending');
        ctx.setReadout('ivCovarianceMatrix', 'pending');
        ctx.setChart('ivCovarianceChart', null);
        return;
      }
      if (id === 'period' || id === 'estimator' || id === 'map') {
        periodParameter.write(
          new Uint32Array([
            ctx.options.estimator.startsWith('cross-') ? 2 : PERIOD_INDEX[ctx.options.period],
            MODEL_INDEX[ctx.options.estimator],
            ctx.options.map === 'observed' ? 0 : 1
          ])
        );
        mapDirty = true;
        mapGeneration++;
        mapValues.write(new Float32Array(ENTITY_COUNT).fill(Number.NaN));
        latestMapValues.fill(Number.NaN);
        ctx.setReadout('mappedUnits', 'pending');
        ctx.setReadout('fitStatus', 'pending GPU fit');
        ctx.setReadout('povertyCoefficient', 'pending GPU fit');
        ctx.setReadout('densityCoefficient', 'pending GPU fit');
        ctx.setReadout('lagOrErrorParameter', 'pending GPU fit');
        ctx.setReadout('hausmanStatus', 'pending or not applicable');
        ctx.setReadout('panelDiagnostics', 'pending or not applicable');
        ctx.setReadout('crossImpacts', 'pending or not applicable');
        ctx.setReadout('ivCovarianceStatus', 'pending or only available for pooled Spatial 2SLS');
        ctx.setReadout('ivCovarianceStandardErrors', 'pending');
        ctx.setReadout('ivCovarianceMatrix', 'pending');
        ctx.setChart('ivCovarianceChart', null);
        covarianceDirty = true;
        ctx.requestLayers();
      }
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      let row = -1;
      let nearestDistance = Infinity;
      for (let index = 0; index < mesh.labelPoints.length; index++) {
        const [longitude, latitude] = mesh.labelPoints[index];
        const [centroidX, centroidY] = projection.project(longitude, latitude);
        const distance = (centroidX - x) ** 2 + (centroidY - y) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          row = index;
        }
      }
      if (row < 0 || row >= ENTITY_COUNT || nearestDistance > 30_000 ** 2) return null;
      const suffix = ctx.options.map === 'observed' ? 'observed unemployment' : 'model residual';
      const period = ctx.options.estimator.startsWith('cross-')
        ? '1990 cross-section'
        : ctx.options.period;
      const mappedValue = latestMapValues[row];
      const valueText = Number.isFinite(mappedValue)
        ? `${mappedValue.toFixed(2)} ${ctx.options.map === 'observed' ? '%' : 'percentage points'}`
        : 'no valid mapped output';
      return `Nearest county-label match: ${nameList[row] ?? `NCOVR unit ${row + 1}`}; ${suffix} ${valueText}; ${period}. The current frame is a historical common-support unit.`;
    },
    destroy() {
      for (const reader of readers.values()) reader.stop();
      hausmanReader.stop();
      errorDiagnosticReader.stop();
      impactsReader.stop();
      mapReader.stop();
      resources.destroy();
    }
  };
}
