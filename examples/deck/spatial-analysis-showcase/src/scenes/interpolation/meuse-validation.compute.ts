// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPULocalKriging,
  GPULocalRBF,
  GPUNestedInterpolationValidation,
  type GPUNestedInterpolationCandidate,
  type GPULocalRBFProps
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';

export type MeuseValidationOptions = {
  response: 'zinc' | 'copper' | 'lead' | 'cadmium';
  estimator: 'ordinary' | 'universal' | 'external-drift' | 'rbf';
  rbfKernel:
    | 'thin-plate-spline'
    | 'cubic'
    | 'gaussian'
    | 'linear'
    | 'quintic'
    | 'multiquadric'
    | 'inverse-multiquadric'
    | 'inverse-quadratic';
  display: 'prediction' | 'variance' | 'oof-residual';
};

const OUTER_FOLD_COUNT = 3;
const INNER_FOLD_COUNT = 2;
const OUTER_BLOCK_METERS = 1000;
const INNER_BLOCK_METERS = 500;
const SEED = 31;
const NEIGHBOR_COUNT = 16;
const CANDIDATES: readonly GPUNestedInterpolationCandidate[] = [
  {
    mode: 'ordinary-kriging',
    covariance: {model: 'exponential', range: 300, sill: 0.35, nugget: 0.05}
  },
  {
    mode: 'ordinary-kriging',
    covariance: {model: 'exponential', range: 900, sill: 0.35, nugget: 0.05}
  },
  {
    mode: 'ordinary-kriging',
    covariance: {model: 'spherical', range: 900, sill: 0.35, nugget: 0.05}
  },
  {mode: 'rbf', kernel: 'thin-plate-spline', degree: 1, epsilon: 1 / 300, smoothing: 0.02},
  {mode: 'rbf', kernel: 'gaussian', degree: 0, epsilon: 1 / 300, smoothing: 0.08}
];

function hashSpatialBlock(
  coordinateX: number,
  coordinateY: number,
  minimumX: number,
  minimumY: number,
  size: number
): number {
  const tileColumn = Math.floor((coordinateX - minimumX) / size);
  const tileRow = Math.floor((coordinateY - minimumY) / size);
  return (Math.imul(tileColumn, 73856093) ^ Math.imul(tileRow, 19349663) ^ SEED) >>> 0;
}

function makeSpatialFolds(positions: Float32Array) {
  const sampleCount = positions.length / 2;
  let minimumX = Infinity;
  let minimumY = Infinity;
  for (let row = 0; row < sampleCount; row++) {
    minimumX = Math.min(minimumX, positions[row * 2]);
    minimumY = Math.min(minimumY, positions[row * 2 + 1]);
  }
  const training = new Uint32Array(sampleCount * OUTER_FOLD_COUNT);
  const testing = new Uint32Array(sampleCount * OUTER_FOLD_COUNT);
  const innerBlocks = new Uint32Array(sampleCount);
  for (let row = 0; row < sampleCount; row++) {
    const coordinateX = positions[row * 2];
    const coordinateY = positions[row * 2 + 1];
    const outerFold =
      hashSpatialBlock(coordinateX, coordinateY, minimumX, minimumY, OUTER_BLOCK_METERS) %
      OUTER_FOLD_COUNT;
    innerBlocks[row] = hashSpatialBlock(
      coordinateX,
      coordinateY,
      minimumX,
      minimumY,
      INNER_BLOCK_METERS
    );
    for (let fold = 0; fold < OUTER_FOLD_COUNT; fold++) {
      testing[fold * sampleCount + row] = Number(outerFold === fold);
      training[fold * sampleCount + row] = Number(outerFold !== fold);
    }
  }
  return {training, testing, innerBlocks};
}

function makeDrift(values: Float32Array): Float32Array {
  const drift = new Float32Array(values.length * 2);
  for (let row = 0; row < values.length; row++) {
    drift[row * 2] = 1;
    drift[row * 2 + 1] = Math.sqrt(Math.max(0, values[row]));
  }
  return drift;
}

/** Nested held-out selection is kept separate from the all-observation display surface. */
export async function createMeuseValidation(
  ctx: SceneContext<MeuseValidationOptions>
): Promise<SceneInstance<MeuseValidationOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const sampleCount = dataset.column('logZinc').length;
  const gridCount = dataset.column('gridDist').length;
  const origin = dataset.defaultOrigin;
  const sampleMapPositions = dataset.projectColumn('samplePosition', origin);
  const gridMapPositions = dataset.projectColumn('gridPosition', origin);
  const samplePositionSource = dataset.column('samplePositionRD');
  const gridPositionSource = dataset.column('gridPositionRD');
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < sampleCount; row++) {
    minimumX = Math.min(minimumX, samplePositionSource[row * 2]);
    maximumX = Math.max(maximumX, samplePositionSource[row * 2]);
    minimumY = Math.min(minimumY, samplePositionSource[row * 2 + 1]);
    maximumY = Math.max(maximumY, samplePositionSource[row * 2 + 1]);
  }
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const samplePositions = Float32Array.from(
    samplePositionSource,
    (value, index) => value - (index % 2 ? centerY : centerX)
  );
  const gridPositions = Float32Array.from(
    gridPositionSource,
    (value, index) => value - (index % 2 ? centerY : centerX)
  );
  const sampleDistance = dataset.column('sampleDist');
  const gridDistance = dataset.column('gridDist');
  const allValues = {
    zinc: dataset.column('logZinc'),
    copper: dataset.column('logCopper'),
    lead: dataset.column('logLead'),
    cadmium: dataset.column('logCadmium')
  };
  const response = Float32Array.from(allValues[ctx.options.response]);
  const folds = makeSpatialFolds(samplePositions);
  const resources = new SpatialAnalysisResources(device, 'meuse-validation');
  const samplesBuffer = resources.createBuffer('sample-positions', samplePositions);
  const valuesBuffer = resources.createBuffer('sample-log-metal', response);
  const observationsData = new Float32Array(sampleCount * 4);
  for (let row = 0; row < sampleCount; row++) {
    observationsData[row * 4] = response[row];
    observationsData[row * 4 + 1] = 0;
    observationsData[row * 4 + 2] = 0;
    observationsData[row * 4 + 3] = 1;
  }
  const observationsBuffer = resources.createBuffer('sample-observations', observationsData);
  const sampleMapPositionsBuffer = resources.createBuffer(
    'sample-map-positions',
    sampleMapPositions,
    Buffer.VERTEX
  );
  const gridMapPositionsBuffer = resources.createBuffer(
    'grid-map-positions',
    gridMapPositions,
    Buffer.VERTEX
  );
  const observedMapValuesBuffer = resources.createBuffer(
    'observed-map-values',
    response,
    Buffer.VERTEX
  );
  const gridBuffer = resources.createBuffer('grid-positions', gridPositions);
  const sampleDriftBuffer = resources.createBuffer(
    'sample-external-drift',
    makeDrift(sampleDistance)
  );
  const gridDriftBuffer = resources.createBuffer('grid-external-drift', makeDrift(gridDistance));
  const trainingBuffer = resources.createBuffer('outer-training', folds.training);
  const testingBuffer = resources.createBuffer('outer-testing', folds.testing);
  const innerBlocksBuffer = resources.createBuffer('inner-blocks', folds.innerBlocks);
  const validationVersionInput = resources.createBuffer(
    'validation-version-input',
    Uint32Array.of(0)
  );
  const validationVersionOutput = resources.createBuffer('validation-version-output', 4);

  const nestedGraph = new GPUCommandGraph<void>(device, {id: 'meuse-nested-validation'});
  const nestedOutputs = {
    selected: resources.createBuffer('selected-candidates', OUTER_FOLD_COUNT * 4),
    scores: resources.createBuffer(
      'candidate-scores',
      OUTER_FOLD_COUNT * CANDIDATES.length * 3 * 4
    ),
    predictions: resources.createBuffer('oof-predictions', sampleCount * OUTER_FOLD_COUNT * 4),
    metrics: resources.createBuffer('fold-metrics', OUTER_FOLD_COUNT * 6 * 4),
    status: resources.createBuffer('fold-status', OUTER_FOLD_COUNT * 4),
    oofResiduals: resources.createBuffer('oof-residuals', sampleCount * 4, Buffer.VERTEX),
    oofFoldIds: resources.createBuffer('oof-fold-ids', sampleCount * 4)
  };
  const nestedPositionsView = importGraphBuffer(
    nestedGraph,
    'sample-positions',
    samplesBuffer,
    'float32x2',
    sampleCount
  );
  const nestedResponseView = importGraphBuffer(
    nestedGraph,
    'response',
    valuesBuffer,
    'float32',
    sampleCount
  );
  const nestedTrainingView = importGraphBuffer(
    nestedGraph,
    'outer-training',
    trainingBuffer,
    'uint32',
    sampleCount * OUTER_FOLD_COUNT
  );
  const nestedTestingView = importGraphBuffer(
    nestedGraph,
    'outer-testing',
    testingBuffer,
    'uint32',
    sampleCount * OUTER_FOLD_COUNT
  );
  const nestedPredictionView = importGraphBuffer(
    nestedGraph,
    'predictions',
    nestedOutputs.predictions,
    'float32',
    sampleCount * OUTER_FOLD_COUNT
  );
  nestedGraph.add(
    new GPUNestedInterpolationValidation({
      id: 'meuse-nested',
      positions: nestedPositionsView,
      response: nestedResponseView,
      training: nestedTrainingView,
      testing: nestedTestingView,
      foldCount: OUTER_FOLD_COUNT,
      innerBlocks: importGraphBuffer(
        nestedGraph,
        'inner-blocks',
        innerBlocksBuffer,
        'uint32',
        sampleCount
      ),
      innerFoldCount: INNER_FOLD_COUNT,
      innerBufferDistance: 250,
      seed: SEED,
      neighborCount: NEIGHBOR_COUNT,
      candidates: CANDIDATES,
      output: {
        selected: importGraphBuffer(
          nestedGraph,
          'selected',
          nestedOutputs.selected,
          'uint32',
          OUTER_FOLD_COUNT
        ),
        scores: importGraphBuffer(
          nestedGraph,
          'scores',
          nestedOutputs.scores,
          'float32',
          OUTER_FOLD_COUNT * CANDIDATES.length * 3
        ),
        predictions: nestedPredictionView,
        metrics: importGraphBuffer(
          nestedGraph,
          'metrics',
          nestedOutputs.metrics,
          'float32',
          OUTER_FOLD_COUNT * 6
        ),
        status: importGraphBuffer(
          nestedGraph,
          'status',
          nestedOutputs.status,
          'uint32',
          OUTER_FOLD_COUNT
        )
      }
    })
  );
  addKernelPass(nestedGraph, {
    id: 'meuse-gather-oof-residuals',
    invocationCount: sampleCount,
    bindings: [
      {name: 'testing', view: nestedTestingView, type: 'u32', access: 'read'},
      {name: 'predictions', view: nestedPredictionView, type: 'f32', access: 'read'},
      {name: 'response', view: nestedResponseView, type: 'f32', access: 'read'},
      {
        name: 'residuals',
        view: importGraphBuffer(
          nestedGraph,
          'oof-residuals',
          nestedOutputs.oofResiduals,
          'float32',
          sampleCount
        ),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'foldIds',
        view: importGraphBuffer(
          nestedGraph,
          'oof-fold-ids',
          nestedOutputs.oofFoldIds,
          'uint32',
          sampleCount
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: `var foldId = 0xffffffffu; var predictionValue = bitcast<f32>(0x7fc00000u | (index & 0u)); for (var fold = 0u; fold < ${OUTER_FOLD_COUNT}u; fold++) { let offset = fold * ${sampleCount}u + index; if (testing[testingOffset + offset] != 0u) { foldId = fold; predictionValue = predictions[predictionsOffset + offset]; break; } } let observedValue = response[responseOffset + index]; let predictionBits = bitcast<u32>(predictionValue); let observedBits = bitcast<u32>(observedValue); let predictionFinite = (predictionBits & 0x7f800000u) != 0x7f800000u; let observedFinite = (observedBits & 0x7f800000u) != 0x7f800000u; let valid = foldId != 0xffffffffu && predictionFinite && observedFinite; residuals[residualsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), observedValue - predictionValue, valid); foldIds[foldIdsOffset + index] = foldId;`
  });
  addKernelPass(nestedGraph, {
    id: 'meuse-snapshot-validation-version',
    invocationCount: 1,
    bindings: [
      {
        name: 'versionInput',
        view: importGraphBuffer(
          nestedGraph,
          'validation-version-input',
          validationVersionInput,
          'uint32',
          1
        ),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'versionOutput',
        view: importGraphBuffer(
          nestedGraph,
          'validation-version-output',
          validationVersionOutput,
          'uint32',
          1
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: 'versionOutput[versionOutputOffset] = versionInput[versionInputOffset];'
  });
  const nestedCompiled = resources.track(await nestedGraph.compileAsync());

  const surfaceValues = resources.createBuffer('surface-log-prediction', gridCount * 4);
  const surfaceVariance = resources.createBuffer('surface-auxiliary-diagnostic', gridCount * 4);
  const packedValues = resources.createBuffer('surface-packed', gridCount * 4 * 4);
  const surfaceStatus = resources.createBuffer('surface-status', gridCount * 4);
  const surfaceGenerationInput = resources.createBuffer(
    'surface-generation-input',
    Uint32Array.of(0)
  );
  const surfaceGenerationOutput = resources.createBuffer('surface-generation-output', 8);
  let validationVersion = 0;
  let surfaceModelVersion = 0;
  let nestedDirty = true;
  let surfaceDirty = true;
  let latestOofResiduals: Float32Array | null = null;
  let latestOofFoldIds: Uint32Array | null = null;
  let latestSurfaceStatusCounts: {
    valid: number;
    noNeighborhood: number;
    unreliable: number;
  } | null = null;
  let surfacePending = true;
  let activeSurfaceCompiled: CompiledGPUCommandGraph<void> | null = null;
  let activeSurfaceModelToken: Buffer | null = null;
  let destroyed = false;

  function updateSurfaceStatusReadout(): void {
    if (ctx.options.display === 'oof-residual') {
      if (!latestOofResiduals) {
        ctx.setReadout(
          'surfaceStatus',
          'Out-of-fold log-metal residuals are pending GPU validation.'
        );
        return;
      }
      const validResidualCount = latestOofResiduals.reduce(
        (count, value) => count + Number(Number.isFinite(value)),
        0
      );
      ctx.setReadout(
        'surfaceStatus',
        `Out-of-fold log-metal residuals · ${validResidualCount} held-out sample predictions · training rows and unsupported predictions are blank`
      );
      return;
    }
    if (surfacePending || !latestSurfaceStatusCounts) {
      ctx.setReadout('surfaceStatus', 'The selected full-data surface is pending GPU execution.');
      return;
    }

    const {valid, noNeighborhood, unreliable} = latestSurfaceStatusCounts;
    if (ctx.options.estimator === 'rbf') {
      const displayName =
        ctx.options.display === 'variance'
          ? 'unitless solver pivot diagnostic'
          : 'log-metal prediction';
      ctx.setReadout(
        'surfaceStatus',
        `RBF ${displayName} · ${valid.toLocaleString()} valid grid cells · ${noNeighborhood.toLocaleString()} without usable neighborhoods · ${unreliable.toLocaleString()} rank failures`
      );
      return;
    }

    const displayName =
      ctx.options.display === 'variance'
        ? 'Conditional point prediction variance (squared log-concentration units)'
        : 'Log-metal prediction';
    ctx.setReadout(
      'surfaceStatus',
      `${displayName} · ${valid.toLocaleString()} valid grid cells · ${noNeighborhood.toLocaleString()} with insufficient support · ${unreliable.toLocaleString()} singular or unreliable solves`
    );
  }

  async function compileSurface(options: MeuseValidationOptions): Promise<void> {
    const requestedVersion = ++surfaceModelVersion;
    surfacePending = true;
    latestSurfaceStatusCounts = null;
    updateSurfaceStatusReadout();
    ctx.requestLayers();
    const modelToken = resources.createBuffer(
      `surface-model-token-${requestedVersion}`,
      Uint32Array.of(requestedVersion)
    );
    surfaceGenerationInput.write(Uint32Array.of(validationVersion));
    ctx.setStatus(
      `Compiling ${options.estimator === 'rbf' ? options.rbfKernel : options.estimator} surface…`
    );
    const surfaceGraph = new GPUCommandGraph<void>(device, {
      id: `meuse-surface-${requestedVersion}`
    });
    const outputValues = importGraphBuffer(
      surfaceGraph,
      'surface-values',
      packedValues,
      'float32x4',
      gridCount
    );
    const outputStatus = importGraphBuffer(
      surfaceGraph,
      'surface-status',
      surfaceStatus,
      'uint32',
      gridCount
    );
    const surfacePositionView = importGraphBuffer(
      surfaceGraph,
      'grid-positions',
      gridBuffer,
      'float32x2',
      gridCount
    );
    const samplePositionView = importGraphBuffer(
      surfaceGraph,
      'sample-positions',
      samplesBuffer,
      'float32x2',
      sampleCount
    );
    const surfaceResponseView = importGraphBuffer(
      surfaceGraph,
      'response',
      valuesBuffer,
      'float32',
      sampleCount
    );
    const sampleDriftView = importGraphBuffer(
      surfaceGraph,
      'sample-drift',
      sampleDriftBuffer,
      'float32',
      sampleCount * 2
    );
    const gridDriftView = importGraphBuffer(
      surfaceGraph,
      'grid-drift',
      gridDriftBuffer,
      'float32',
      gridCount * 2
    );
    if (options.estimator === 'rbf') {
      const degree: NonNullable<GPULocalRBFProps['degree']> =
        options.rbfKernel === 'quintic'
          ? 2
          : ['thin-plate-spline', 'cubic'].includes(options.rbfKernel)
            ? 1
            : 0;
      surfaceGraph.add(
        new GPULocalRBF({
          id: 'meuse-direct-rbf',
          positions: samplePositionView,
          values: surfaceResponseView,
          queries: surfacePositionView,
          kernel: options.rbfKernel,
          degree,
          epsilon: 1 / 300,
          smoothing: 0.02,
          neighborCount: NEIGHBOR_COUNT,
          output: {values: outputValues, status: outputStatus}
        })
      );
    } else {
      const mode = options.estimator;
      surfaceGraph.add(
        new GPULocalKriging({
          id: 'meuse-direct-kriging',
          positions: samplePositionView,
          observations: importGraphBuffer(
            surfaceGraph,
            'observations',
            observationsBuffer,
            'float32x4',
            sampleCount
          ),
          supports: surfacePositionView,
          mode,
          covariance: {model: 'exponential', range: 900, sill: 0.35, nugget: 0.05},
          neighborCount: NEIGHBOR_COUNT,
          ...(mode === 'external-drift'
            ? {driftCount: 2, sampleDrift: sampleDriftView, queryDrift: gridDriftView}
            : {}),
          output: {values: outputValues, status: outputStatus}
        })
      );
    }
    const predictionLayerValues = importGraphBuffer(
      surfaceGraph,
      'prediction-display',
      surfaceValues,
      'float32',
      gridCount
    );
    const auxiliaryLayerValues = importGraphBuffer(
      surfaceGraph,
      'auxiliary-display',
      surfaceVariance,
      'float32',
      gridCount
    );
    addKernelPass(surfaceGraph, {
      id: 'meuse-unpack-surface',
      invocationCount: gridCount,
      bindings: [
        {name: 'packed', view: outputValues, type: 'f32', access: 'read'},
        {name: 'status', view: outputStatus, type: 'u32', access: 'read'},
        {name: 'prediction', view: predictionLayerValues, type: 'f32', access: 'read_write'},
        {name: 'auxiliary', view: auxiliaryLayerValues, type: 'f32', access: 'read_write'}
      ],
      body: `let valid = status[statusOffset + index] == 0u; let value = packed[packedOffset + index * 4u]; let auxiliaryValue = packed[packedOffset + index * 4u + 1u]; let nan = bitcast<f32>(0x7fc00000u | (index & 0u)); prediction[predictionOffset + index] = select(nan, value, valid); auxiliary[auxiliaryOffset + index] = select(nan, auxiliaryValue, valid);`
    });
    addKernelPass(surfaceGraph, {
      id: 'meuse-snapshot-generation',
      invocationCount: 1,
      bindings: [
        {
          name: 'generationInput',
          view: importGraphBuffer(
            surfaceGraph,
            'surface-generation-input',
            surfaceGenerationInput,
            'uint32',
            1
          ),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'modelToken',
          view: importGraphBuffer(surfaceGraph, 'surface-model-token', modelToken, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'generationOutput',
          view: importGraphBuffer(
            surfaceGraph,
            'surface-generation-output',
            surfaceGenerationOutput,
            'uint32x2',
            1
          ),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: 'generationOutput[generationOutputOffset] = generationInput[generationInputOffset]; generationOutput[generationOutputOffset + 1u] = modelToken[modelTokenOffset];'
    });
    const compiled = await surfaceGraph.compileAsync();
    if (destroyed || requestedVersion !== surfaceModelVersion) {
      compiled.destroy();
      resources.release(modelToken);
      return;
    }
    if (activeSurfaceCompiled) resources.release(activeSurfaceCompiled);
    if (activeSurfaceModelToken) resources.release(activeSurfaceModelToken);
    activeSurfaceCompiled = resources.track(compiled);
    activeSurfaceModelToken = modelToken;
    surfaceDirty = true;
    ctx.setStatus('');
    ctx.requestLayers();
  }
  await compileSurface(ctx.options);
  const validationReader = new SummaryReader(
    resources,
    'meuse-nested-metrics',
    [
      {buffer: nestedOutputs.selected, size: OUTER_FOLD_COUNT * 4},
      {buffer: nestedOutputs.scores, size: OUTER_FOLD_COUNT * CANDIDATES.length * 3 * 4},
      {buffer: nestedOutputs.metrics, size: OUTER_FOLD_COUNT * 6 * 4},
      {buffer: nestedOutputs.status, size: OUTER_FOLD_COUNT * 4},
      {buffer: nestedOutputs.oofResiduals, size: sampleCount * 4},
      {buffer: nestedOutputs.oofFoldIds, size: sampleCount * 4},
      {buffer: validationVersionOutput, size: 4}
    ],
    bytes => {
      const selected = new Uint32Array(bytes, 0, OUTER_FOLD_COUNT);
      const scoresOffset = OUTER_FOLD_COUNT * 4;
      const scores = new Float32Array(
        bytes,
        scoresOffset,
        OUTER_FOLD_COUNT * CANDIDATES.length * 3
      );
      const offset = scoresOffset + OUTER_FOLD_COUNT * CANDIDATES.length * 3 * 4;
      const metrics = new Float32Array(bytes, offset, OUTER_FOLD_COUNT * 6);
      const status = new Uint32Array(bytes, offset + OUTER_FOLD_COUNT * 6 * 4, OUTER_FOLD_COUNT);
      const residualOffset = offset + OUTER_FOLD_COUNT * 6 * 4 + OUTER_FOLD_COUNT * 4;
      const generationOffset = residualOffset + sampleCount * 8;
      if (new Uint32Array(bytes, generationOffset, 1)[0] !== validationVersion) return;
      if (status.some(value => value !== 0)) {
        ctx.setReadout(
          'validationStatus',
          `Some outer folds are unavailable (status ${Array.from(status).join(', ')}).`
        );
        ctx.setChart('foldErrors', null);
        ctx.setChart('candidateScores', null);
        return;
      }
      const counts = Array.from({length: OUTER_FOLD_COUNT}, (_, fold) => metrics[fold * 6]);
      const errors = Array.from({length: OUTER_FOLD_COUNT}, (_, fold) =>
        Math.sqrt(metrics[fold * 6 + 2])
      );
      const allCount = counts.reduce((sum, value) => sum + value, 0);
      const pooledRmse = Math.sqrt(
        counts.reduce((sum, value, fold) => sum + value * errors[fold] ** 2, 0) /
          Math.max(1, allCount)
      );
      const candidateNames = [
        'OK exp · 300 m',
        'OK exp · 900 m',
        'OK spherical · 900 m',
        'RBF thin plate',
        'RBF Gaussian'
      ];
      latestOofResiduals = new Float32Array(
        bytes.slice(residualOffset, residualOffset + sampleCount * 4)
      );
      latestOofFoldIds = new Uint32Array(
        bytes.slice(residualOffset + sampleCount * 4, residualOffset + sampleCount * 8)
      );
      updateSurfaceStatusReadout();
      ctx.refreshTooltip();
      ctx.requestLayers();
      ctx.setReadout(
        'validationStatus',
        'Nested GPU validation complete · outer outcomes were not used to select each model.'
      );
      ctx.setReadout(
        'validationRmse',
        `${pooledRmse.toFixed(3)} log-metal units · ${allCount} held-out rows`
      );
      ctx.setReadout(
        'selectedModels',
        Array.from(
          selected,
          (index, fold) => `F${fold + 1}: ${candidateNames[index] ?? 'no candidate'}`
        ).join(' · ')
      );
      ctx.setChart('foldErrors', {
        kind: 'bars',
        title: 'Outer-fold RMSE after inner-only selection',
        labels: errors.map((_, fold) => `Fold ${fold + 1}`),
        values: errors,
        description:
          'RMSE in natural-log metal units. Each outer test block is held out while two-fold inner spatial validation selects among five fixed, authored candidates.'
      });
      const scoreLabels: string[] = [];
      const scoreValues: number[] = [];
      for (let fold = 0; fold < OUTER_FOLD_COUNT; fold++) {
        for (let candidate = 0; candidate < CANDIDATES.length; candidate++) {
          const scoreOffset = (fold * CANDIDATES.length + candidate) * 3;
          const mse = scores[scoreOffset];
          const scoredCount = scores[scoreOffset + 1];
          const missingCount = scores[scoreOffset + 2];
          const candidateName = candidateNames[candidate].replace(' · ', ' ');
          const complete = scoredCount > 0 && missingCount === 0 && Number.isFinite(mse);
          scoreLabels.push(`F${fold + 1} ${candidateName}${complete ? '' : ' · n/a'}`);
          scoreValues.push(complete ? Math.sqrt(mse) : 0);
        }
      }
      ctx.setChart('candidateScores', {
        kind: 'bars',
        title: 'Inner-selection RMSE by outer training fold',
        labels: scoreLabels,
        values: scoreValues,
        description:
          'Natural-log metal units. Each bar is an inner-only candidate score within one outer-training split; n/a rows have no complete inner score. The held-out outer scores are shown separately.'
      });
    }
  );
  const surfaceReader = new SummaryReader(
    resources,
    'meuse-surface-status',
    [
      {buffer: surfaceStatus, size: gridCount * 4},
      {buffer: surfaceGenerationOutput, size: 8}
    ],
    bytes => {
      const generation = new Uint32Array(bytes, gridCount * 4, 2);
      if (generation[0] !== validationVersion || generation[1] !== surfaceModelVersion) return;
      const status = new Uint32Array(bytes, 0, gridCount);
      let predictedCount = 0;
      let noNeighborhoodCount = 0;
      let unreliableCount = 0;
      for (const value of status) {
        if (value === 0) predictedCount++;
        else if (value === 1) noNeighborhoodCount++;
        else if (value === 2) unreliableCount++;
      }
      latestSurfaceStatusCounts = {
        valid: predictedCount,
        noNeighborhood: noNeighborhoodCount,
        unreliable: unreliableCount
      };
      if (surfacePending) {
        surfacePending = false;
        ctx.requestLayers();
      }
      updateSurfaceStatusReadout();
    }
  );
  const riverVertexCount =
    (dataset.geojson?.features[0]?.geometry as {coordinates: [number, number][]} | undefined)
      ?.coordinates.length ?? 0;
  const riverSegments = new Float32Array(Math.max(0, riverVertexCount - 1) * 4);
  const riverCoordinates =
    (dataset.geojson?.features[0]?.geometry as {coordinates: [number, number][]} | undefined)
      ?.coordinates ?? [];
  const projection = dataset.getProjection(origin);
  for (let index = 0; index + 1 < riverCoordinates.length; index++) {
    const first = projection.project(...riverCoordinates[index]);
    const second = projection.project(...riverCoordinates[index + 1]);
    riverSegments.set([first[0], first[1], second[0], second[1]], index * 4);
  }
  const riverBuffer = resources.createBuffer('river-outline', riverSegments);
  ctx.setReadout('sampleCount', sampleCount);
  ctx.setReadout('gridCount', gridCount);
  ctx.setReadout(
    'validationDesign',
    'Three outer spatial folds · 1 km block tiles · inner 500 m tiles · 250 m inner buffer · seed 31'
  );
  ctx.setReadout('validationStatus', 'Nested validation pending GPU execution.');
  updateSurfaceStatusReadout();
  ctx.setChart('foldErrors', null);
  ctx.setChart('candidateScores', null);

  return {
    getCompiledGraphs: () =>
      activeSurfaceCompiled
        ? ([nestedCompiled, activeSurfaceCompiled] as CompiledGPUCommandGraph<never>[])
        : ([nestedCompiled] as CompiledGPUCommandGraph<never>[]),
    encode(commandEncoder) {
      if (nestedDirty) {
        nestedCompiled.encode(commandEncoder, {parameters: undefined});
        validationReader.request(commandEncoder);
        nestedDirty = false;
      } else {
        validationReader.flush(commandEncoder);
      }
      if (surfaceDirty && activeSurfaceCompiled) {
        activeSurfaceCompiled.encode(commandEncoder, {parameters: undefined});
        surfaceReader.request(commandEncoder);
        surfaceDirty = false;
      } else {
        surfaceReader.flush(commandEncoder);
      }
    },
    setOption(id, _value, state) {
      ctx.requestLayers();
      if (id === 'response') {
        const currentResponse = Float32Array.from(allValues[state.response]);
        valuesBuffer.write(currentResponse);
        observedMapValuesBuffer.write(currentResponse);
        for (let row = 0; row < sampleCount; row++) {
          observationsData[row * 4] = currentResponse[row];
        }
        observationsBuffer.write(observationsData);
        validationVersion++;
        validationVersionInput.write(Uint32Array.of(validationVersion));
        validationReader.markStale();
        latestOofResiduals = null;
        latestOofFoldIds = null;
        nestedDirty = true;
        surfacePending = true;
        latestSurfaceStatusCounts = null;
        surfaceGenerationInput.write(Uint32Array.of(validationVersion));
        surfaceReader.markStale();
        surfaceDirty = true;
        ctx.refreshLegends();
        updateSurfaceStatusReadout();
        ctx.setReadout('validationStatus', 'Recomputing nested validation for the selected metal.');
      } else if (id === 'estimator' || id === 'rbfKernel') {
        const requestedVersion = surfaceModelVersion + 1;
        void compileSurface(state).catch(error => {
          if (!destroyed) {
            const message = error instanceof Error ? error.message : String(error);
            ctx.setStatus(`Surface compilation failed: ${message}`);
            if (surfaceModelVersion === requestedVersion) {
              ctx.setReadout(
                'surfaceStatus',
                'Surface model compilation failed; no current surface is shown.'
              );
            }
          }
        });
      } else if (id === 'display') {
        ctx.refreshLegends();
        updateSurfaceStatusReadout();
        if (state.display === 'oof-residual' && !latestOofResiduals) {
          ctx.setReadout('validationStatus', 'Out-of-fold residuals are pending GPU validation.');
        }
      }
      ctx.refreshTooltip();
    },
    getLayers(): Layer[] {
      const values = ctx.options.display === 'prediction' ? surfaceValues : surfaceVariance;
      const layers: Layer[] = [];
      if (ctx.options.display !== 'oof-residual' && !surfacePending) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-prediction-grid',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: gridMapPositionsBuffer,
            values,
            valueFormat: 'float32',
            instanceCount: gridCount,
            radiusPixels: 3,
            colormap: ctx.options.display === 'prediction' ? 'viridis' : 'magma',
            valueRange:
              ctx.options.display === 'prediction'
                ? MEUSE_LOG_RESPONSE_RANGES[ctx.options.response]
                : ctx.options.estimator === 'rbf'
                  ? [0, 1]
                  : [0, 0.6],
            noDataColor: [0, 0, 0, 0],
            color: [255, 255, 255, 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'meuse-river-outline',
          coordinateOrigin: [origin[0], origin[1], 0],
          segments: riverBuffer,
          instanceCount: riverSegments.length / 4,
          widthPixels: 2,
          color: [49, 113, 151, 230]
        })
      );
      if (
        ctx.options.display === 'prediction' ||
        (ctx.options.display === 'oof-residual' && latestOofResiduals)
      ) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-observed-samples',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: sampleMapPositionsBuffer,
            values:
              ctx.options.display === 'oof-residual'
                ? nestedOutputs.oofResiduals
                : observedMapValuesBuffer,
            valueFormat: 'float32',
            valueRange:
              ctx.options.display === 'oof-residual'
                ? [-1.5, 1.5]
                : MEUSE_LOG_RESPONSE_RANGES[ctx.options.response],
            colormap: ctx.options.display === 'oof-residual' ? 'diverging' : 'viridis',
            instanceCount: sampleCount,
            radiusPixels: 4.5,
            outlineColor: [16, 18, 22, 255],
            outlineWidthPixels: 1
          })
        );
      }
      return layers;
    },
    getTooltip(event: ScenePointerEvent): string | null {
      if (!event.coordinate) return null;
      const [coordinateX, coordinateY] = projection.project(
        event.coordinate[0],
        event.coordinate[1]
      );
      let nearestIndex = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < sampleCount; row++) {
        const distance = Math.hypot(
          sampleMapPositions[row * 2] - coordinateX,
          sampleMapPositions[row * 2 + 1] - coordinateY
        );
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestIndex = row;
        }
      }
      if (nearestIndex < 0 || nearestDistance > 85) return null;
      const responseName = ctx.options.response;
      const capitalizedName = `${responseName.charAt(0).toUpperCase()}${responseName.slice(1)}`;
      const measured = dataset.column(`sample${capitalizedName}`)[nearestIndex];
      const lines = [
        `Meuse field sample · ${responseName}`,
        `Observed ${measured.toFixed(2)} mg/kg`,
        `Nearest sample ${Math.round(nearestDistance)} m away`
      ];
      const residual = latestOofResiduals?.[nearestIndex];
      const fold = latestOofFoldIds?.[nearestIndex];
      if (
        residual !== undefined &&
        Number.isFinite(residual) &&
        fold !== undefined &&
        fold !== 0xffffffff
      ) {
        const predictedMedian = Math.exp(allValues[responseName][nearestIndex] - residual);
        lines.push(`OOF fold ${fold + 1} · prediction median ${predictedMedian.toFixed(2)} mg/kg`);
        lines.push(`OOF residual ${residual >= 0 ? '+' : ''}${residual.toFixed(3)} log units`);
      } else {
        lines.push('Nested held-out result pending');
      }
      return lines.join('\n');
    },
    destroy() {
      destroyed = true;
      validationReader.stop();
      surfaceReader.stop();
      resources.destroy();
    }
  };
}
