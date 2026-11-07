// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  getGPUNeighborSearchParameterValues,
  GPULocalCollinearity,
  GPUMultiscaleGeographicallyWeightedRegression,
  GPUNeighborSearch,
  GPUSeeminglyUnrelatedRegression
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {createNamedReader} from './b6-reader';

export type MeuseMetalModelFormOptions = {
  display: 'local-distance-coefficient' | 'local-metal-condition';
  bandwidthProfile: 'compact' | 'broad';
};

const K_NEIGHBORS = 12;
const PREDICTOR_COUNT = 3;
const EQUATION_COUNT = 3;
const METAL_NAMES = ['log zinc', 'log copper', 'log lead'] as const;
const BANDWIDTH_PROFILES = {
  compact: Float32Array.of(1e-4, 3500, 1800, 1800, 2200),
  broad: Float32Array.of(1e-4, 4500, 2800, 2800, 3200)
} as const;

function standardize(values: Float32Array): Float32Array {
  let sum = 0;
  for (const value of values) sum += value;
  const mean = sum / values.length;
  let squared = 0;
  for (const value of values) squared += (value - mean) ** 2;
  const deviation = Math.sqrt(squared / values.length) || 1;
  return Float32Array.from(values, value => (value - mean) / deviation);
}

function makeTable(columns: readonly Float32Array[], rowCount: number): Float32Array {
  const normalized = columns.map(standardize);
  const result = new Float32Array(rowCount * columns.length);
  for (let row = 0; row < rowCount; row++) {
    for (let column = 0; column < columns.length; column++) {
      result[row * columns.length + column] = normalized[column][row];
    }
  }
  return result;
}

function getFiniteRange(values: Float32Array): readonly [number, number] {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return [minimum, maximum];
}

/** Joint metal regression, local predictor collinearity and a fixed-bandwidth MGWR fit. */
export async function createMeuseMetalModelForm(
  ctx: SceneContext<MeuseMetalModelFormOptions>
): Promise<SceneInstance<MeuseMetalModelFormOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const sourceRows = dataset.column<Float32Array>('logZinc').length;
  const rawColumns = [
    dataset.column<Float32Array>('logZinc'),
    dataset.column<Float32Array>('logCopper'),
    dataset.column<Float32Array>('logLead')
  ];
  const elevationSource = dataset.column<Float32Array>('sampleElevation');
  const organicMatterSource = dataset.column<Float32Array>('sampleOrganicMatter');
  const distanceSource = dataset.column<Float32Array>('sampleDistanceMeters');
  const positionSource = dataset.column<Float32Array>('samplePositionRD');
  const studyRows: number[] = [];
  for (let row = 0; row < sourceRows; row++) {
    if (
      Number.isFinite(elevationSource[row]) &&
      Number.isFinite(organicMatterSource[row]) &&
      Number.isFinite(distanceSource[row]) &&
      rawColumns.every(column => Number.isFinite(column[row]))
    ) {
      studyRows.push(row);
    }
  }
  const rowCount = studyRows.length;
  const responseColumns = rawColumns.map(column =>
    Float32Array.from(studyRows, row => column[row])
  );
  const environmentalColumns = [
    Float32Array.from(studyRows, row => elevationSource[row]),
    Float32Array.from(studyRows, row => organicMatterSource[row]),
    Float32Array.from(studyRows, row => distanceSource[row])
  ];
  const environmentalPredictors = makeTable(environmentalColumns, rowCount);
  const metalPredictors = makeTable(responseColumns, rowCount);
  const equationPredictors = new Float32Array(rowCount * PREDICTOR_COUNT * EQUATION_COUNT);
  const equationResponses = new Float32Array(rowCount * EQUATION_COUNT);
  for (let equation = 0; equation < EQUATION_COUNT; equation++) {
    equationResponses.set(responseColumns[equation], equation * rowCount);
    equationPredictors.set(environmentalPredictors, equation * rowCount * PREDICTOR_COUNT);
  }

  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (const row of studyRows) {
    minimumX = Math.min(minimumX, positionSource[row * 2]);
    maximumX = Math.max(maximumX, positionSource[row * 2]);
    minimumY = Math.min(minimumY, positionSource[row * 2 + 1]);
    maximumY = Math.max(maximumY, positionSource[row * 2 + 1]);
  }
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const positions = new Float32Array(rowCount * 2);
  const allMapPositions = dataset.projectColumn('samplePosition', dataset.defaultOrigin);
  const mapPositions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    const sourceRow = studyRows[row];
    positions[row * 2] = positionSource[sourceRow * 2] - centerX;
    positions[row * 2 + 1] = positionSource[sourceRow * 2 + 1] - centerY;
    mapPositions[row * 2] = allMapPositions[sourceRow * 2];
    mapPositions[row * 2 + 1] = allMapPositions[sourceRow * 2 + 1];
  }
  const positionX = Float32Array.from({length: rowCount}, (_, row) => positions[row * 2]);
  const positionY = Float32Array.from({length: rowCount}, (_, row) => positions[row * 2 + 1]);
  const bounds = [
    Math.min(...positionX),
    Math.min(...positionY),
    Math.max(...positionX),
    Math.max(...positionY)
  ] as const;

  const resources = new SpatialAnalysisResources(device, 'meuse-metal-model-form');
  const positionsBuffer = resources.createBuffer('positions', positions);
  const environmentalPredictorsBuffer = resources.createBuffer(
    'environmental-predictors',
    environmentalPredictors
  );
  const metalPredictorsBuffer = resources.createBuffer('metal-predictors', metalPredictors);
  const equationPredictorsBuffer = resources.createBuffer('sur-predictors', equationPredictors);
  const equationResponsesBuffer = resources.createBuffer('sur-responses', equationResponses);
  const neighborParametersBuffer = resources.createBuffer(
    'neighbor-parameters',
    getGPUNeighborSearchParameterValues({bounds, weightKind: 'binary', rowStandardize: true})
  );
  const neighborOffsetsBuffer = resources.createBuffer('neighbor-offsets', (rowCount + 1) * 4);
  const neighborIdsBuffer = resources.createBuffer('neighbor-ids', rowCount * K_NEIGHBORS * 4);
  const neighborWeightsBuffer = resources.createBuffer(
    'neighbor-weights',
    rowCount * K_NEIGHBORS * 4
  );
  const neighborDistancesBuffer = resources.createBuffer(
    'neighbor-distances',
    rowCount * K_NEIGHBORS * 4
  );
  const neighborOverflowBuffer = resources.createBuffer('neighbor-overflow', 4);
  const neighborRequiredBuffer = resources.createBuffer('neighbor-required', 4);
  const neighborInvalidBuffer = resources.createBuffer('neighbor-invalid', 4);
  const collinearityVifBuffer = resources.createBuffer(
    'local-metal-vif',
    rowCount * EQUATION_COUNT * 4
  );
  const collinearityConditionBuffer = resources.createBuffer('local-metal-condition', rowCount * 4);
  const collinearityStatusBuffer = resources.createBuffer(
    'local-metal-condition-status',
    rowCount * 4
  );
  const surCoefficientBuffer = resources.createBuffer(
    'sur-coefficients',
    EQUATION_COUNT * (PREDICTOR_COUNT + 1) * 4
  );
  const surCovarianceBuffer = resources.createBuffer(
    'sur-covariance',
    (EQUATION_COUNT * (PREDICTOR_COUNT + 1)) ** 2 * 4
  );
  const surResidualCovarianceBuffer = resources.createBuffer(
    'sur-residual-covariance',
    EQUATION_COUNT ** 2 * 4
  );
  const surResidualsBuffer = resources.createBuffer('sur-residuals', EQUATION_COUNT * rowCount * 4);
  const surStatusBuffer = resources.createBuffer('sur-status', 4);
  const mgwrParametersBuffer = resources.createBuffer(
    'mgwr-parameters',
    BANDWIDTH_PROFILES[ctx.options.bandwidthProfile]
  );
  const mgwrCoefficientBuffer = resources.createBuffer(
    'mgwr-coefficients',
    rowCount * (PREDICTOR_COUNT + 1) * 4
  );
  const mgwrFittedBuffer = resources.createBuffer('mgwr-fitted', rowCount * 4);
  const mgwrStatusBuffer = resources.createBuffer('mgwr-status', rowCount * 4);
  const mgwrIterationsBuffer = resources.createBuffer('mgwr-iterations', 4);
  const mappedDistanceCoefficientBuffer = resources.createBuffer(
    'mapped-distance-coefficient',
    rowCount * 4,
    Buffer.VERTEX
  );
  const mappedConditionBuffer = resources.createBuffer(
    'mapped-local-metal-condition',
    rowCount * 4,
    Buffer.VERTEX
  );
  const generationInputBuffer = resources.createBuffer('profile-generation-input', 4);
  const generationOutputBuffer = resources.createBuffer('profile-generation-output', 4);
  const mapPositionsBuffer = resources.createBuffer('map-positions', mapPositions, Buffer.VERTEX);

  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-metal-model-form'});
  const positionsView = importGraphBuffer(
    graph,
    'positions',
    positionsBuffer,
    'float32x2',
    rowCount
  );
  const environmentalView = importGraphBuffer(
    graph,
    'environmental-predictors',
    environmentalPredictorsBuffer,
    'float32',
    rowCount * PREDICTOR_COUNT
  );
  const metalView = importGraphBuffer(
    graph,
    'metal-predictors',
    metalPredictorsBuffer,
    'float32',
    rowCount * EQUATION_COUNT
  );
  const equationPredictorsView = importGraphBuffer(
    graph,
    'sur-predictors',
    equationPredictorsBuffer,
    'float32',
    rowCount * EQUATION_COUNT * PREDICTOR_COUNT
  );
  const equationResponsesView = importGraphBuffer(
    graph,
    'sur-responses',
    equationResponsesBuffer,
    'float32',
    rowCount * EQUATION_COUNT
  );
  const neighborParametersView = importGraphBuffer(
    graph,
    'neighbor-parameters',
    neighborParametersBuffer,
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const offsetsView = importGraphBuffer(
    graph,
    'neighbor-offsets',
    neighborOffsetsBuffer,
    'uint32',
    rowCount + 1
  );
  const neighborIdsView = importGraphBuffer(
    graph,
    'neighbor-ids',
    neighborIdsBuffer,
    'uint32',
    rowCount * K_NEIGHBORS
  );
  const neighborWeightsView = importGraphBuffer(
    graph,
    'neighbor-weights',
    neighborWeightsBuffer,
    'float32',
    rowCount * K_NEIGHBORS
  );
  const neighborDistancesView = importGraphBuffer(
    graph,
    'neighbor-distances',
    neighborDistancesBuffer,
    'float32',
    rowCount * K_NEIGHBORS
  );
  const neighborOverflowView = importGraphBuffer(
    graph,
    'neighbor-overflow',
    neighborOverflowBuffer,
    'uint32',
    1
  );
  const neighborRequiredView = importGraphBuffer(
    graph,
    'neighbor-required',
    neighborRequiredBuffer,
    'uint32',
    1
  );
  const neighborInvalidView = importGraphBuffer(
    graph,
    'neighbor-invalid',
    neighborInvalidBuffer,
    'uint32',
    1
  );
  const collinearityVifView = importGraphBuffer(
    graph,
    'local-metal-vif',
    collinearityVifBuffer,
    'float32',
    rowCount * EQUATION_COUNT
  );
  const collinearityConditionView = importGraphBuffer(
    graph,
    'local-metal-condition',
    collinearityConditionBuffer,
    'float32',
    rowCount
  );
  const collinearityStatusView = importGraphBuffer(
    graph,
    'local-metal-condition-status',
    collinearityStatusBuffer,
    'uint32',
    rowCount
  );
  const surCoefficientView = importGraphBuffer(
    graph,
    'sur-coefficients',
    surCoefficientBuffer,
    'float32',
    EQUATION_COUNT * (PREDICTOR_COUNT + 1)
  );
  const surCovarianceView = importGraphBuffer(
    graph,
    'sur-covariance',
    surCovarianceBuffer,
    'float32',
    (EQUATION_COUNT * (PREDICTOR_COUNT + 1)) ** 2
  );
  const surResidualCovarianceView = importGraphBuffer(
    graph,
    'sur-residual-covariance',
    surResidualCovarianceBuffer,
    'float32',
    EQUATION_COUNT ** 2
  );
  const surResidualsView = importGraphBuffer(
    graph,
    'sur-residuals',
    surResidualsBuffer,
    'float32',
    EQUATION_COUNT * rowCount
  );
  const surStatusView = importGraphBuffer(graph, 'sur-status', surStatusBuffer, 'uint32', 1);
  const mgwrParametersView = importGraphBuffer(
    graph,
    'mgwr-parameters',
    mgwrParametersBuffer,
    'float32',
    5
  );
  const mgwrCoefficientView = importGraphBuffer(
    graph,
    'mgwr-coefficients',
    mgwrCoefficientBuffer,
    'float32',
    rowCount * (PREDICTOR_COUNT + 1)
  );
  const mgwrFittedView = importGraphBuffer(
    graph,
    'mgwr-fitted',
    mgwrFittedBuffer,
    'float32',
    rowCount
  );
  const mgwrStatusView = importGraphBuffer(
    graph,
    'mgwr-status',
    mgwrStatusBuffer,
    'uint32',
    rowCount
  );
  const mgwrIterationsView = importGraphBuffer(
    graph,
    'mgwr-iterations',
    mgwrIterationsBuffer,
    'uint32',
    1
  );
  const mappedCoefficientView = importGraphBuffer(
    graph,
    'mapped-distance-coefficient',
    mappedDistanceCoefficientBuffer,
    'float32',
    rowCount
  );
  const mappedConditionView = importGraphBuffer(
    graph,
    'mapped-local-metal-condition',
    mappedConditionBuffer,
    'float32',
    rowCount
  );
  const generationInputView = importGraphBuffer(
    graph,
    'profile-generation-input',
    generationInputBuffer,
    'uint32',
    1
  );
  const generationOutputView = importGraphBuffer(
    graph,
    'profile-generation-output',
    generationOutputBuffer,
    'uint32',
    1
  );
  const weightsSearch = new GPUNeighborSearch({
    id: 'meuse-local-metal-neighbor-search',
    mode: 'knn',
    k: K_NEIGHBORS,
    positions: positionsView,
    parameters: neighborParametersView,
    gridSize: [64, 64],
    weightTransform: 'row',
    rowStandardize: true,
    output: {
      offsets: offsetsView,
      neighbors: neighborIdsView,
      weights: neighborWeightsView,
      distances: neighborDistancesView,
      status: {
        overflow: neighborOverflowView,
        requiredCount: neighborRequiredView,
        invalidCount: neighborInvalidView
      }
    }
  });
  graph.add(weightsSearch);
  graph.add(
    new GPULocalCollinearity({
      id: 'meuse-local-metal-collinearity',
      weights: weightsSearch.outputWeights,
      predictors: metalView,
      predictorCount: EQUATION_COUNT,
      output: {
        varianceInflationFactors: collinearityVifView,
        conditionNumbers: collinearityConditionView,
        status: collinearityStatusView
      }
    })
  );
  graph.add(
    new GPUSeeminglyUnrelatedRegression({
      id: 'meuse-metal-sur',
      equationCount: EQUATION_COUNT,
      predictorCount: PREDICTOR_COUNT,
      rowCount,
      predictors: equationPredictorsView,
      responses: equationResponsesView,
      output: {
        coefficients: surCoefficientView,
        covariance: surCovarianceView,
        residualCovariance: surResidualCovarianceView,
        residuals: surResidualsView,
        status: surStatusView
      }
    })
  );
  graph.add(
    new GPUMultiscaleGeographicallyWeightedRegression({
      id: 'meuse-zinc-environmental-mgwr',
      positions: positionsView,
      predictors: environmentalView,
      predictorCount: PREDICTOR_COUNT,
      response: importGraphBuffer(
        graph,
        'mgwr-zinc-response',
        resources.createBuffer('mgwr-zinc-response', responseColumns[0]),
        'float32',
        rowCount
      ),
      parameters: mgwrParametersView,
      maximumIterations: 16,
      output: {
        coefficients: mgwrCoefficientView,
        fitted: mgwrFittedView,
        status: {codes: mgwrStatusView, iterationCount: mgwrIterationsView}
      }
    })
  );
  addKernelPass(graph, {
    id: 'meuse-copy-distance-coefficient',
    invocationCount: rowCount,
    bindings: [
      {name: 'coefficients', view: mgwrCoefficientView, type: 'f32', access: 'read'},
      {name: 'status', view: mgwrStatusView, type: 'u32', access: 'read'},
      {name: 'mapped', view: mappedCoefficientView, type: 'f32', access: 'read_write'}
    ],
    body: `let coefficient = coefficients[coefficientsOffset + index * ${PREDICTOR_COUNT + 1}u + 3u]; let finite = (bitcast<u32>(coefficient) & 0x7f800000u) != 0x7f800000u; mapped[mappedOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), coefficient, status[statusOffset + index] == 0u && finite);`
  });
  addKernelPass(graph, {
    id: 'meuse-mask-local-metal-condition',
    invocationCount: rowCount,
    bindings: [
      {name: 'condition', view: collinearityConditionView, type: 'f32', access: 'read'},
      {name: 'status', view: collinearityStatusView, type: 'u32', access: 'read'},
      {name: 'mapped', view: mappedConditionView, type: 'f32', access: 'read_write'}
    ],
    body: `let value = condition[conditionOffset + index]; let finite = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; mapped[mappedOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), value, status[statusOffset + index] == 0u && finite);`
  });
  addKernelPass(graph, {
    id: 'meuse-model-form-generation-snapshot',
    invocationCount: 1,
    bindings: [
      {name: 'generation', view: generationInputView, type: 'u32', access: 'read'},
      {name: 'snapshot', view: generationOutputView, type: 'u32', access: 'read_write'}
    ],
    body: 'snapshot[snapshotOffset] = generation[generationOffset];'
  });
  const compiled = resources.track(await graph.compileAsync());
  let latestCondition: Float32Array | null = null;
  let latestCollinearityStatus: Uint32Array | null = null;
  let latestMappedCoefficient: Float32Array | null = null;
  let latestMgwrStatus: Uint32Array | null = null;
  let latestSurStatus: number | null = null;
  let generation = 0;
  let graphDirty = true;
  const reader = createNamedReader(
    resources,
    'model-results',
    [
      {name: 'surStatus', buffer: surStatusBuffer, bytes: 4},
      {
        name: 'surCoefficients',
        buffer: surCoefficientBuffer,
        bytes: EQUATION_COUNT * (PREDICTOR_COUNT + 1) * 4
      },
      {
        name: 'surResidualCovariance',
        buffer: surResidualCovarianceBuffer,
        bytes: EQUATION_COUNT ** 2 * 4
      },
      {name: 'collinearityCondition', buffer: collinearityConditionBuffer, bytes: rowCount * 4},
      {name: 'collinearityStatus', buffer: collinearityStatusBuffer, bytes: rowCount * 4},
      {name: 'mgwrStatus', buffer: mgwrStatusBuffer, bytes: rowCount * 4},
      {name: 'mgwrIterations', buffer: mgwrIterationsBuffer, bytes: 4},
      {
        name: 'mappedDistanceCoefficient',
        buffer: mappedDistanceCoefficientBuffer,
        bytes: rowCount * 4
      },
      {name: 'mappedCondition', buffer: mappedConditionBuffer, bytes: rowCount * 4},
      {name: 'neighborStatus', buffer: neighborOverflowBuffer, bytes: 4},
      {name: 'neighborInvalid', buffer: neighborInvalidBuffer, bytes: 4},
      {name: 'generation', buffer: generationOutputBuffer, bytes: 4}
    ],
    get => {
      if (get('generation').u32[0] !== generation) return;
      latestSurStatus = get('surStatus').u32[0];
      latestCondition = get('mappedCondition').f32;
      latestCollinearityStatus = get('collinearityStatus').u32;
      latestMappedCoefficient = get('mappedDistanceCoefficient').f32;
      latestMgwrStatus = get('mgwrStatus').u32;
      const coefficientStatus = latestCollinearityStatus;
      let validConditions = 0;
      let invalidConditions = 0;
      let maximumCondition = 0;
      for (let row = 0; row < rowCount; row++) {
        if (coefficientStatus[row] === 0 && Number.isFinite(latestCondition[row])) {
          validConditions++;
          maximumCondition = Math.max(maximumCondition, latestCondition[row]);
        } else invalidConditions++;
      }
      ctx.setReadout(
        'collinearityStatus',
        `${validConditions}/${rowCount} valid local predictor-correlation diagnostics · ${invalidConditions} unsupported neighborhoods · max condition index ${validConditions ? maximumCondition.toFixed(1) : '—'}`
      );
      const surCoefficients = get('surCoefficients').f32;
      const residualCovariance = get('surResidualCovariance').f32;
      if (latestSurStatus === 0) {
        const equations = METAL_NAMES.map((name, equation) => {
          const offset = equation * (PREDICTOR_COUNT + 1);
          return `${name}: intercept ${surCoefficients[offset].toFixed(2)}, elevation ${surCoefficients[offset + 1].toFixed(2)}, organic matter ${surCoefficients[offset + 2].toFixed(2)}, river distance ${surCoefficients[offset + 3].toFixed(2)}`;
        });
        const correlations: string[] = [];
        for (let first = 0; first < EQUATION_COUNT; first++) {
          for (let second = first + 1; second < EQUATION_COUNT; second++) {
            const numerator = residualCovariance[first * EQUATION_COUNT + second];
            const denominator = Math.sqrt(
              Math.max(0, residualCovariance[first * EQUATION_COUNT + first]) *
                Math.max(0, residualCovariance[second * EQUATION_COUNT + second])
            );
            correlations.push(
              `${METAL_NAMES[first]}/${METAL_NAMES[second]} residual r ${denominator ? (numerator / denominator).toFixed(2) : '—'}`
            );
          }
        }
        ctx.setReadout(
          'surStatus',
          `One-step feasible SUR converged · ${rowCount} common rows · ${equations.join(' · ')}`
        );
        ctx.setReadout('surResiduals', correlations.join(' · '));
      } else {
        ctx.setReadout(
          'surStatus',
          `SUR status ${latestSurStatus}; coefficient system is unavailable.`
        );
        ctx.setReadout('surResiduals', null);
      }
      let converged = 0;
      for (const status of latestMgwrStatus) if (status === 0) converged++;
      const iterationCount = get('mgwrIterations').u32[0];
      const distanceCoefficients = latestMappedCoefficient.filter(Number.isFinite);
      const [coefficientMinimum, coefficientMaximum] = getFiniteRange(distanceCoefficients);
      const bandwidths = BANDWIDTH_PROFILES[ctx.options.bandwidthProfile];
      ctx.setReadout(
        'mgwrStatus',
        `${converged}/${rowCount} local rows converged after ${iterationCount} backfit sweeps · authored bandwidths intercept/elev/OM/dist ${Array.from(
          bandwidths.slice(1),
          value => `${(value / 1000).toFixed(1)} km`
        ).join('/')}` +
          (distanceCoefficients.length
            ? ` · distance coefficient range ${coefficientMinimum.toFixed(3)} to ${coefficientMaximum.toFixed(3)}`
            : ' · distance coefficient unavailable')
      );
      const neighborStatus = get('neighborStatus').u32[0];
      const neighborInvalid = get('neighborInvalid').u32[0];
      ctx.setReadout(
        'neighborStatus',
        `12-nearest row-standardized weights · overflow ${neighborStatus} · invalid coordinates ${neighborInvalid}`
      );
      ctx.requestLayers();
      ctx.refreshTooltip();
    }
  );

  ctx.setReadout(
    'studyCount',
    `${rowCount}/${sourceRows} complete cases; two source rows omit organic matter`
  );
  ctx.setReadout('neighborStatus', 'Local 12-neighbor weights pending GPU run.');
  ctx.setReadout('collinearityStatus', 'Local metal condition indices pending GPU run.');
  ctx.setReadout('surStatus', 'Aligned three-equation SUR fit pending GPU run.');
  ctx.setReadout('surResiduals', null);
  ctx.setReadout('mgwrStatus', 'Fixed-profile MGWR fit pending GPU run.');

  function getTooltip(event: ScenePointerEvent) {
    if (!event.coordinate) return null;
    const origin = dataset.defaultOrigin;
    const projected = dataset
      .getProjection(origin)
      .project(event.coordinate[0], event.coordinate[1]);
    let row = -1;
    let nearestDistance = Infinity;
    for (let candidate = 0; candidate < rowCount; candidate++) {
      const distance = Math.hypot(
        mapPositions[candidate * 2] - projected[0],
        mapPositions[candidate * 2 + 1] - projected[1]
      );
      if (distance < nearestDistance) {
        row = candidate;
        nearestDistance = distance;
      }
    }
    if (row < 0 || nearestDistance > 60) return null;
    const values = responseColumns.map(column => column[row].toFixed(2));
    const condition = latestCondition?.[row];
    const coefficient = latestMappedCoefficient?.[row];
    const collinearityText =
      latestCollinearityStatus?.[row] === 0 && condition !== undefined && Number.isFinite(condition)
        ? `Local log-metal condition index: ${condition.toFixed(2)}`
        : 'Local log-metal condition index unavailable';
    const coefficientText =
      latestMgwrStatus?.[row] === 0 && coefficient !== undefined && Number.isFinite(coefficient)
        ? `MGWR river-distance coefficient: ${coefficient.toFixed(3)}`
        : 'MGWR coefficient unavailable';
    return `Measured log metals · Zn ${values[0]}, Cu ${values[1]}, Pb ${values[2]} · ${Math.round(nearestDistance)} m from site\n${collinearityText}\n${coefficientText}`;
  }

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        reader.request(commandEncoder);
        graphDirty = false;
      } else reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      if (
        latestCondition === null ||
        latestCollinearityStatus === null ||
        latestMappedCoefficient === null ||
        latestMgwrStatus === null
      ) {
        return [];
      }
      const values =
        ctx.options.display === 'local-distance-coefficient'
          ? mappedDistanceCoefficientBuffer
          : mappedConditionBuffer;
      return [
        new SpatialAnalysisPointLayer({
          id: 'meuse-metal-model-points',
          coordinateOrigin: [dataset.defaultOrigin[0], dataset.defaultOrigin[1], 0],
          positions: mapPositionsBuffer,
          values,
          valueFormat: 'float32',
          valueRange: ctx.options.display === 'local-distance-coefficient' ? [-0.4, 0.4] : [0, 40],
          colormap: ctx.options.display === 'local-distance-coefficient' ? 'diverging' : 'magma',
          instanceCount: rowCount,
          radiusPixels: 6,
          color: [255, 255, 255, 255],
          outlineColor: [16, 18, 22, 230]
        })
      ];
    },
    setOption(id, value) {
      if (id === 'display') {
        ctx.refreshLegends();
        ctx.requestLayers();
      } else if (id === 'bandwidthProfile') {
        const profile = value === 'broad' ? 'broad' : 'compact';
        mgwrParametersBuffer.write(BANDWIDTH_PROFILES[profile]);
        generation++;
        generationInputBuffer.write(Uint32Array.of(generation));
        reader.markStale();
        graphDirty = true;
        latestCondition = null;
        latestCollinearityStatus = null;
        latestMappedCoefficient = null;
        latestMgwrStatus = null;
        latestSurStatus = null;
        ctx.setReadout('collinearityStatus', 'Local metal condition map pending profile refresh.');
        ctx.setReadout('surStatus', 'Joint SUR summary pending profile refresh.');
        ctx.setReadout('surResiduals', null);
        ctx.setReadout('mgwrStatus', 'MGWR map and summary pending profile refresh.');
        ctx.requestLayers();
      }
    },
    getTooltip,
    destroy() {
      reader.release();
      resources.destroy();
    }
  };
}
