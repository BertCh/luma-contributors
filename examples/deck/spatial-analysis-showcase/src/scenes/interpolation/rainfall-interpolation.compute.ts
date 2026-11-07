// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, Viewport} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  evaluateVariogramModel,
  fitVariogramModel,
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPUVariogram,
  GPUVariogramModelFit,
  type VariogramModel,
  type VariogramModelType,
  type VariogramModelWeighting
} from '@luma-contributors/graph/gpu-dataframe';
import {makeGPURasterMetadata} from '@luma-contributors/graph/gpu-raster';
import {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  getGPUKrigingParameterValues,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  GPU_KRIGING_PARAMETER_LENGTH,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  GPUKriging,
  type GPUFocalStatisticsShape,
  GPUSpatialCrossValidation
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {addInterpolationDiagnosticsRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {
  type AnalysisFrame,
  createAnalysisFrameProjection,
  getViewportFrameBounds,
  projectLngLatToFrame
} from '../../engine/analysis-frame';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {formatSignificant, formatSparkline, hashInteger} from './b7-format';
import {
  getRainfallSurfaceStyle,
  RAINFALL_VARIABLES,
  type RainfallFocalStatistic,
  type RainfallSurface,
  type RainfallVariable
} from './b7-rainfall-style';
import {B7TessellatedRasterLayer} from './b7-tessellated-raster-layer';
import {toRadians} from '@math.gl/core';
import {getWebMercatorMetersPerPixel} from '../../engine/web-mercator';
import {getViewRasterGrid} from '../../../../spatial-analysis-shared/geometry-helpers';

/** Option state of the rainfall scene. */
export type RainfallOptions = {
  variable: RainfallVariable;
  holdOut: boolean;
  validationSplit: 'random' | 'spatial';
  surface: RainfallSurface;
  power: number;
  neighborCount: number;
  krigingNeighborCount: number;
  searchRadiusKm: number;
  minimumStations: number;
  variogramModel: VariogramModelType;
  weighting: VariogramModelWeighting;
  robust: boolean;
  maximumLagKm: number;
  direction: 'all' | '0' | '1' | '2' | '3';
  azimuth: number;
  focalStatistic: RainfallFocalStatistic;
  focalRadius: number;
  focalShape: GPUFocalStatisticsShape;
  focalMinimumCount: number;
  keepHoles: boolean;
  ramp: RampName;
  opacity: number;
  showStations: boolean;
};

/** Raster that follows the camera. Compile-time. */
const RASTER_WIDTH = 384;
const RASTER_HEIGHT = 240;

/** The raster grid of the viewport bounds at this scene's resolution. */
function getViewGrid(bounds: readonly [number, number, number, number]) {
  return getViewRasterGrid(bounds, RASTER_WIDTH, RASTER_HEIGHT);
}
const CELL_COUNT = RASTER_WIDTH * RASTER_HEIGHT;

/** Compile-time capacities of the per-frame neighbor limits. */
const MAXIMUM_IDW_NEIGHBORS = 32;
const MAXIMUM_KRIGING_NEIGHBORS = 16;
const MAXIMUM_FOCAL_RADIUS = 8;
const INDEX_GRID_SIZE: readonly [number, number] = [64, 64];
/** Variogram lags and direction sectors. Compile-time. */
const LAG_COUNT = 16;
const SECTOR_COUNT = 4;
const BIN_COUNT = LAG_COUNT * SECTOR_COUNT;
const VARIOGRAM_GRID_SIZE: readonly [number, number] = [32, 32];
const HOLD_OUT_MODULUS = 10;
const SPATIAL_FOLD_COUNT = 5;
const SPATIAL_BLOCK_SIZE_METERS = 250_000;
/** Milliseconds the camera and controls stay still before the surfaces are read back. */
const SETTLE_MILLISECONDS = 450;
const STATION_NAME_LIMIT_PIXELS = 11;
const VARIOGRAM_FIT_MODELS: readonly VariogramModelType[] = [
  'spherical',
  'exponential',
  'gaussian'
];
const VARIOGRAM_FIT_WEIGHTINGS: readonly VariogramModelWeighting[] = ['cressie', 'pairs', 'none'];

/**
 * Analysis frame of the gauges, the grid and every distance (IDW search radius, kriging range,
 * variogram lags): azimuthal equidistant about the centre of the GHCN box [-92, 24, -74, 40].
 * Distances from the centre are exact and the scale error is at most +0.65% at the box corners
 * (the Mercator-local frame this replaced was -8.0% to +9.5%). The layers draw in this frame.
 */
const RAINFALL_FRAME: AnalysisFrame = {kind: 'azimuthal-equidistant', origin: [-83, 32]};

type Bins = {distances: number[]; gammas: number[]; robust: number[]; pairs: number[]};

/**
 * Interpolates GHCN-Daily gauge readings onto a raster that follows the camera.
 *
 * One compiled graph holds `GPUInverseDistanceWeighting`, `GPUKriging` (variance output), a
 * kernel pass that picks the displayed surface and `GPUFocalStatistics`. A second small graph runs
 * `GPUVariogram` (4 direction sectors, 16 lags) whenever the stations, hold-out mask or maximum lag
 * change; the CPU fits a model to the read-back bins and writes it into the kriging parameters. A
 * third graph uses `GPUSpatialCrossValidation` only for spatial mode and writes one fold's masks
 * back for interpolation; a stale readback cannot replace a newer split. Maps stay GPU-bound; the
 * CPU reads bounded raster summaries for validation metrics and tooltips.
 */
export async function createRainfallInterpolation(
  ctx: SceneContext<RainfallOptions>
): Promise<SceneInstance<RainfallOptions>> {
  const stations = ctx.datasets.get('ghcn-stations');
  const {device} = ctx;
  const projection = createAnalysisFrameProjection(RAINFALL_FRAME);
  const positions = projectLngLatToFrame(projection, stations.column('position'));
  const stationCount = positions.length / 2;
  const valuesByVariable: Record<RainfallVariable, Float32Array> = {
    prcp: stations.column('prcp'),
    prcpPrevDay: stations.column('prcpPrevDay'),
    tmax: stations.column('tmax')
  };
  const elevation = stations.column('elevation');
  const stationTable = parseCsv(
    await fetchText(getDataFileUrl('ghcn-stations', 'stations.csv'), ctx.signal)
  );
  ctx.signal.throwIfAborted();
  const stationNames = stationTable.rows.map(row => row[1] ?? '');
  const stationStates = stationTable.rows.map(row => row[2] ?? '');
  const stationIdentifiers = stationTable.rows.map(row => row[0] ?? '');

  const stationBounds: [number, number, number, number] = [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ];
  for (let station = 0; station < stationCount; station++) {
    stationBounds[0] = Math.min(stationBounds[0], positions[station * 2]);
    stationBounds[1] = Math.min(stationBounds[1], positions[station * 2 + 1]);
    stationBounds[2] = Math.max(stationBounds[2], positions[station * 2]);
    stationBounds[3] = Math.max(stationBounds[3], positions[station * 2 + 1]);
  }
  const indexBounds: [number, number, number, number] = [
    stationBounds[0] - 1000,
    stationBounds[1] - 1000,
    stationBounds[2] + 1000,
    stationBounds[3] + 1000
  ];

  const resources = new SpatialAnalysisResources(device, 'rainfall');
  const positionsBuffer = resources.createBuffer('positions', positions);
  const valuesBuffer = resources.createBuffer('values', valuesByVariable[ctx.options.variable]);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(stationCount).fill(1));
  const spatialValidationMask = resources.createBuffer(
    'spatial-validation-mask',
    new Uint32Array(stationCount)
  );
  const spatialBlockIds = new Uint32Array(stationCount);
  const spatialBlockColumnCount =
    Math.ceil((stationBounds[2] - stationBounds[0]) / SPATIAL_BLOCK_SIZE_METERS) + 1;
  for (let station = 0; station < stationCount; station++) {
    const tileX = Math.floor(
      (positions[station * 2] - stationBounds[0]) / SPATIAL_BLOCK_SIZE_METERS
    );
    const tileY = Math.floor(
      (positions[station * 2 + 1] - stationBounds[1]) / SPATIAL_BLOCK_SIZE_METERS
    );
    spatialBlockIds[station] = tileX + spatialBlockColumnCount * tileY;
  }
  const spatialBlocks = resources.createBuffer('spatial-validation-blocks', spatialBlockIds);
  const spatialFolds = resources.createBuffer('spatial-validation-folds', stationCount * 4);
  const spatialTraining = resources.createBuffer(
    'spatial-validation-training',
    stationCount * SPATIAL_FOLD_COUNT * 4
  );
  const spatialTesting = resources.createBuffer(
    'spatial-validation-testing',
    stationCount * SPATIAL_FOLD_COUNT * 4
  );
  const spatialCounts = resources.createBuffer(
    'spatial-validation-counts',
    SPATIAL_FOLD_COUNT * 3 * 4
  );
  const spatialStatus = resources.createBuffer('spatial-validation-status', SPATIAL_FOLD_COUNT * 4);
  const spatialValidationGraph = new GPUCommandGraph<void>(device, {
    id: 'rainfall-spatial-validation'
  });
  spatialValidationGraph.add(
    new GPUSpatialCrossValidation({
      id: 'rainfall-spatial-folds',
      positions: importGraphBuffer(
        spatialValidationGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        stationCount
      ),
      blocks: importGraphBuffer(
        spatialValidationGraph,
        'blocks',
        spatialBlocks,
        'uint32',
        stationCount
      ),
      mask: importGraphBuffer(
        spatialValidationGraph,
        'valid',
        spatialValidationMask,
        'uint32',
        stationCount
      ),
      foldCount: SPATIAL_FOLD_COUNT,
      seed: 1,
      output: {
        folds: importGraphBuffer(
          spatialValidationGraph,
          'folds',
          spatialFolds,
          'uint32',
          stationCount
        ),
        training: importGraphBuffer(
          spatialValidationGraph,
          'training',
          spatialTraining,
          'uint32',
          stationCount * SPATIAL_FOLD_COUNT
        ),
        testing: importGraphBuffer(
          spatialValidationGraph,
          'testing',
          spatialTesting,
          'uint32',
          stationCount * SPATIAL_FOLD_COUNT
        ),
        counts: importGraphBuffer(
          spatialValidationGraph,
          'counts',
          spatialCounts,
          'uint32',
          SPATIAL_FOLD_COUNT * 3
        ),
        status: importGraphBuffer(
          spatialValidationGraph,
          'status',
          spatialStatus,
          'uint32',
          SPATIAL_FOLD_COUNT
        )
      }
    })
  );
  const spatialValidationCompiled: CompiledGPUCommandGraph<void> = resources.track(
    await spatialValidationGraph.compileAsync()
  );
  const heldOutIds = resources.createBuffer('held-out-ids', new Uint32Array(stationCount));
  const idwSurface = resources.createBuffer('idw-surface', CELL_COUNT * 4);
  const idwCounts = resources.createBuffer('idw-counts', CELL_COUNT * 4);
  const krigingSurface = resources.createBuffer('kriging-surface', CELL_COUNT * 4);
  const krigingVariance = resources.createBuffer('kriging-variance', CELL_COUNT * 4);
  const selectedSurface = resources.createBuffer('selected-surface', CELL_COUNT * 4);
  const auxiliarySurface = resources.createBuffer('auxiliary-surface', CELL_COUNT * 4);
  const focalStatistics = {
    mean: resources.createBuffer('focal-mean', CELL_COUNT * 4),
    min: resources.createBuffer('focal-min', CELL_COUNT * 4),
    max: resources.createBuffer('focal-max', CELL_COUNT * 4),
    range: resources.createBuffer('focal-range', CELL_COUNT * 4),
    standardDeviation: resources.createBuffer('focal-standard-deviation', CELL_COUNT * 4)
  } satisfies Record<RainfallFocalStatistic, Buffer>;
  const semivarianceBuffer = resources.createBuffer('semivariances', BIN_COUNT * 4);
  const robustBuffer = resources.createBuffer('robust-semivariances', BIN_COUNT * 4);
  const pairCountBuffer = resources.createBuffer('pair-counts', BIN_COUNT * 4);
  const distanceBuffer = resources.createBuffer('mean-distances', BIN_COUNT * 4);
  const statisticsBuffer = resources.createBuffer('variogram-statistics', 5 * 4);
  const fitInputBuffers = {
    distances: resources.createBuffer('fit-distances', new Float32Array(LAG_COUNT)),
    matheron: resources.createBuffer('fit-matheron', new Float32Array(LAG_COUNT)),
    robust: resources.createBuffer('fit-robust', new Float32Array(LAG_COUNT)),
    pairCounts: resources.createBuffer('fit-pair-counts', new Uint32Array(LAG_COUNT))
  };
  const fitOutputBuffers = VARIOGRAM_FIT_WEIGHTINGS.flatMap(weighting =>
    [false, true].map(robust => ({
      weighting,
      robust,
      result: undefined as
        | {parameters: Float32Array; diagnostics: Float32Array; status: Uint32Array}
        | undefined,
      parameters: resources.createBuffer(
        `fit-${weighting}-${robust ? 'robust' : 'matheron'}-parameters`,
        3 * 16
      ),
      diagnostics: resources.createBuffer(
        `fit-${weighting}-${robust ? 'robust' : 'matheron'}-diagnostics`,
        3 * 16
      ),
      status: resources.createBuffer(
        `fit-${weighting}-${robust ? 'robust' : 'matheron'}-status`,
        3 * 4
      )
    }))
  );
  const leaveOneOutPredictionBuffer = resources.createBuffer(
    'leave-one-out-predictions',
    stationCount * 4
  );
  const leaveOneOutVarianceBuffer = resources.createBuffer(
    'leave-one-out-prediction-variance',
    stationCount * 4
  );
  const leaveOneOutResidualBuffer = resources.createBuffer(
    'leave-one-out-residuals',
    stationCount * 4
  );
  const leaveOneOutStandardizedBuffer = resources.createBuffer(
    'leave-one-out-standardized-residuals',
    stationCount * 4
  );
  const leaveOneOutValidCountBuffer = resources.createBuffer('leave-one-out-valid-count', 4);
  const leaveOneOutStandardizedCountBuffer = resources.createBuffer(
    'leave-one-out-standardized-count',
    4
  );
  const leaveOneOutBiasBuffer = resources.createBuffer('leave-one-out-bias', 4);
  const leaveOneOutMaeBuffer = resources.createBuffer('leave-one-out-mae', 4);
  const leaveOneOutRmseBuffer = resources.createBuffer('leave-one-out-rmse', 4);
  const leaveOneOutKrigingInvalidBuffer = resources.createBuffer(
    'leave-one-out-kriging-invalid',
    4
  );
  const leaveOneOutKrigingUncertainBuffer = resources.createBuffer(
    'leave-one-out-kriging-uncertain',
    4
  );
  const leaveOneOutSummaryInvalidBuffer = resources.createBuffer(
    'leave-one-out-summary-invalid',
    4
  );
  const idwParameters = resources.createParameterBuffer(
    'idw-parameters',
    'float32',
    GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH
  );
  const krigingParameters = resources.createParameterBuffer(
    'kriging-parameters',
    'float32',
    GPU_KRIGING_PARAMETER_LENGTH
  );
  const focalParameters = resources.createParameterBuffer(
    'focal-parameters',
    'float32',
    GPU_FOCAL_STATISTICS_PARAMETER_LENGTH
  );
  const variogramParameters = resources.createParameterBuffer(
    'variogram-parameters',
    'float32',
    GPU_VARIOGRAM_PARAMETER_LENGTH
  );
  const modeParameters = resources.createParameterBuffer('mode-parameters', 'float32', 4);

  // --- Main graph -----------------------------------------------------------------------------
  const graph = new GPUCommandGraph<void>(device, {id: 'rainfall'});
  const positionsView = importGraphBuffer(
    graph,
    'positions',
    positionsBuffer,
    'float32x2',
    stationCount
  );
  const valuesView = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', stationCount);
  const maskView = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', stationCount);
  const idwSurfaceView = importGraphBuffer(graph, 'idw-surface', idwSurface, 'float32', CELL_COUNT);
  const krigingSurfaceView = importGraphBuffer(
    graph,
    'kriging-surface',
    krigingSurface,
    'float32',
    CELL_COUNT
  );
  const krigingVarianceView = importGraphBuffer(
    graph,
    'kriging-variance',
    krigingVariance,
    'float32',
    CELL_COUNT
  );
  const idwCountsView = importGraphBuffer(graph, 'idw-counts', idwCounts, 'uint32', CELL_COUNT);
  const selectedView = importGraphBuffer(graph, 'selected', selectedSurface, 'float32', CELL_COUNT);
  graph.add(
    new GPUInverseDistanceWeighting({
      spatialContext: projection.spatialContext,
      id: 'idw',
      positions: positionsView,
      values: valuesView,
      mask: maskView,
      parameters: idwParameters.importToGraph(graph),
      grid: makeGPURasterMetadata({width: RASTER_WIDTH, height: RASTER_HEIGHT}),
      indexGridSize: INDEX_GRID_SIZE,
      indexBounds,
      maximumNeighborCount: MAXIMUM_IDW_NEIGHBORS,
      output: {values: idwSurfaceView, counts: idwCountsView}
    })
  );
  graph.add(
    new GPUKriging({
      spatialContext: projection.spatialContext,
      id: 'kriging',
      positions: positionsView,
      values: valuesView,
      mask: maskView,
      parameters: krigingParameters.importToGraph(graph),
      grid: makeGPURasterMetadata({width: RASTER_WIDTH, height: RASTER_HEIGHT}),
      indexGridSize: INDEX_GRID_SIZE,
      indexBounds,
      maximumNeighborCount: MAXIMUM_KRIGING_NEIGHBORS,
      output: {values: krigingSurfaceView, variance: krigingVarianceView}
    })
  );
  // Surface selector. A parameter word chooses what feeds the focal statistics (IDW, kriging, or
  // their difference) and what the auxiliary map shows (kriging standard error or IDW support).
  addKernelPass(graph, {
    id: 'select-surface',
    invocationCount: CELL_COUNT,
    bindings: [
      {name: 'idw', view: idwSurfaceView, type: 'f32', access: 'read'},
      {name: 'kriging', view: krigingSurfaceView, type: 'f32', access: 'read'},
      {name: 'variance', view: krigingVarianceView, type: 'f32', access: 'read'},
      {name: 'counts', view: idwCountsView, type: 'u32', access: 'read'},
      {name: 'mode', view: modeParameters.importToGraph(graph), type: 'f32', access: 'read'},
      {name: 'selected', view: selectedView, type: 'f32', access: 'read_write'},
      {
        name: 'auxiliary',
        view: importGraphBuffer(graph, 'auxiliary', auxiliarySurface, 'float32', CELL_COUNT),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: /* wgsl */ `
  var nanBits: u32 = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let a = idw[idwOffset + index];
  let b = kriging[krigingOffset + index];
  let v = variance[varianceOffset + index];
  let aInvalid = (bitcast<u32>(a) & 0x7fffffffu) >= 0x7f800000u;
  let bInvalid = (bitcast<u32>(b) & 0x7fffffffu) >= 0x7f800000u;
  let vInvalid = (bitcast<u32>(v) & 0x7fffffffu) >= 0x7f800000u;
  let surface = u32(mode[modeOffset]);
  var selectedValue = a;
  var invalid = aInvalid;
  if (surface == 1u) {
    selectedValue = b;
    invalid = bInvalid;
  } else if (surface == 2u) {
    selectedValue = b - a;
    invalid = aInvalid || bInvalid;
  }
  selected[selectedOffset + index] = select(selectedValue, nan, invalid);
  if (u32(mode[modeOffset + 1u]) == 1u) {
    auxiliary[auxiliaryOffset + index] = f32(counts[countsOffset + index]);
  } else {
    auxiliary[auxiliaryOffset + index] = select(sqrt(max(v, 0.0)), nan, vInvalid);
  }`
  });
  graph.add(
    new GPUFocalStatistics({
      id: 'focal',
      values: selectedView,
      grid: makeGPURasterMetadata({width: RASTER_WIDTH, height: RASTER_HEIGHT}),
      maximumRadius: MAXIMUM_FOCAL_RADIUS,
      parameters: focalParameters.importToGraph(graph),
      output: {
        mean: importGraphBuffer(graph, 'focal-mean', focalStatistics.mean, 'float32', CELL_COUNT),
        min: importGraphBuffer(graph, 'focal-min', focalStatistics.min, 'float32', CELL_COUNT),
        max: importGraphBuffer(graph, 'focal-max', focalStatistics.max, 'float32', CELL_COUNT),
        range: importGraphBuffer(
          graph,
          'focal-range',
          focalStatistics.range,
          'float32',
          CELL_COUNT
        ),
        standardDeviation: importGraphBuffer(
          graph,
          'focal-standard-deviation',
          focalStatistics.standardDeviation,
          'float32',
          CELL_COUNT
        )
      }
    })
  );
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());

  // --- Variogram graph ------------------------------------------------------------------------
  const variogramOverflowBuffer = resources.createBuffer('variogram-overflow', 4);
  const variogramGraph = new GPUCommandGraph<void>(device, {id: 'rainfall-variogram'});
  variogramGraph.add(
    new GPUVariogram({
      spatialContext: projection.spatialContext,
      id: 'variogram',
      positions: importGraphBuffer(
        variogramGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        stationCount
      ),
      values: importGraphBuffer(variogramGraph, 'values', valuesBuffer, 'float32', stationCount),
      mask: importGraphBuffer(variogramGraph, 'mask', maskBuffer, 'uint32', stationCount),
      parameters: variogramParameters.importToGraph(variogramGraph),
      gridSize: VARIOGRAM_GRID_SIZE,
      lagCount: LAG_COUNT,
      directionCount: SECTOR_COUNT,
      output: {
        semivariances: importGraphBuffer(
          variogramGraph,
          'semivariances',
          semivarianceBuffer,
          'float32',
          BIN_COUNT
        ),
        robustSemivariances: importGraphBuffer(
          variogramGraph,
          'robust',
          robustBuffer,
          'float32',
          BIN_COUNT
        ),
        pairCounts: importGraphBuffer(
          variogramGraph,
          'pair-counts',
          pairCountBuffer,
          'uint32',
          BIN_COUNT
        ),
        meanDistances: importGraphBuffer(
          variogramGraph,
          'mean-distances',
          distanceBuffer,
          'float32',
          BIN_COUNT
        ),
        statistics: importGraphBuffer(variogramGraph, 'statistics', statisticsBuffer, 'float32', 5),
        // 1 when a bin's pair count saturated; the fit would be distorted, so it is skipped.
        status: {
          overflow: importGraphBuffer(
            variogramGraph,
            'variogram-overflow',
            variogramOverflowBuffer,
            'uint32',
            1
          )
        }
      }
    })
  );
  const variogramCompiled: CompiledGPUCommandGraph<void> = resources.track(
    await variogramGraph.compileAsync()
  );

  // --- GPU model-fit comparison and conditional leave-one-out diagnostics -------------------
  const fitGraph = new GPUCommandGraph<void>(device, {id: 'rainfall-variogram-fit-comparison'});
  const fitDistancesView = importGraphBuffer(
    fitGraph,
    'fit-distances',
    fitInputBuffers.distances,
    'float32',
    LAG_COUNT
  );
  const fitMatheronView = importGraphBuffer(
    fitGraph,
    'fit-matheron',
    fitInputBuffers.matheron,
    'float32',
    LAG_COUNT
  );
  const fitRobustView = importGraphBuffer(
    fitGraph,
    'fit-robust',
    fitInputBuffers.robust,
    'float32',
    LAG_COUNT
  );
  const fitPairCountsView = importGraphBuffer(
    fitGraph,
    'fit-pair-counts',
    fitInputBuffers.pairCounts,
    'uint32',
    LAG_COUNT
  );
  for (const fitOutput of fitOutputBuffers) {
    fitGraph.add(
      new GPUVariogramModelFit({
        id: `fit-${fitOutput.weighting}-${fitOutput.robust ? 'robust' : 'matheron'}`,
        distances: fitDistancesView,
        semivariances: fitOutput.robust ? fitRobustView : fitMatheronView,
        pairCounts: fitPairCountsView,
        models: VARIOGRAM_FIT_MODELS,
        weighting: fitOutput.weighting,
        rangeSteps: 200,
        maximumIterations: 60,
        output: {
          parameters: importGraphBuffer(
            fitGraph,
            `fit-${fitOutput.weighting}-${fitOutput.robust ? 'robust' : 'matheron'}-parameters`,
            fitOutput.parameters,
            'float32x4',
            3
          ),
          diagnostics: importGraphBuffer(
            fitGraph,
            `fit-${fitOutput.weighting}-${fitOutput.robust ? 'robust' : 'matheron'}-diagnostics`,
            fitOutput.diagnostics,
            'float32x4',
            3
          ),
          status: importGraphBuffer(
            fitGraph,
            `fit-${fitOutput.weighting}-${fitOutput.robust ? 'robust' : 'matheron'}-status`,
            fitOutput.status,
            'uint32',
            3
          )
        }
      })
    );
  }
  const fitCompiled: CompiledGPUCommandGraph<void> = resources.track(await fitGraph.compileAsync());

  const leaveOneOutGraph = new GPUCommandGraph<void>(device, {
    id: 'rainfall-conditional-leave-one-out'
  });
  const leaveOneOutRecipe = addInterpolationDiagnosticsRecipe(leaveOneOutGraph, {
    id: 'rainfall-leave-one-out',
    kriging: {
      spatialContext: projection.spatialContext,
      positions: importGraphBuffer(
        leaveOneOutGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        stationCount
      ),
      values: importGraphBuffer(leaveOneOutGraph, 'values', valuesBuffer, 'float32', stationCount),
      mask: importGraphBuffer(leaveOneOutGraph, 'mask', maskBuffer, 'uint32', stationCount),
      parameters: krigingParameters.importToGraph(leaveOneOutGraph),
      grid: makeGPURasterMetadata({width: RASTER_WIDTH, height: RASTER_HEIGHT}),
      indexGridSize: INDEX_GRID_SIZE,
      indexBounds,
      maximumNeighborCount: MAXIMUM_KRIGING_NEIGHBORS
    },
    outputs: {
      predictions: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-predictions',
        leaveOneOutPredictionBuffer,
        'float32',
        stationCount
      ),
      predictionVariance: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-variance',
        leaveOneOutVarianceBuffer,
        'float32',
        stationCount
      ),
      residuals: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-residuals',
        leaveOneOutResidualBuffer,
        'float32',
        stationCount
      ),
      standardizedResiduals: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-standardized',
        leaveOneOutStandardizedBuffer,
        'float32',
        stationCount
      ),
      validCount: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-valid-count',
        leaveOneOutValidCountBuffer,
        'uint32',
        1
      ),
      standardizedCount: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-standardized-count',
        leaveOneOutStandardizedCountBuffer,
        'uint32',
        1
      ),
      bias: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-bias',
        leaveOneOutBiasBuffer,
        'float32',
        1
      ),
      meanAbsoluteError: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-mae',
        leaveOneOutMaeBuffer,
        'float32',
        1
      ),
      rootMeanSquaredError: importGraphBuffer(
        leaveOneOutGraph,
        'leave-one-out-rmse',
        leaveOneOutRmseBuffer,
        'float32',
        1
      )
    }
  });
  const leaveOneOutKrigingStatus = leaveOneOutRecipe.status.stages[0].status;
  const leaveOneOutSummaryStatus = leaveOneOutRecipe.status.stages[1].status;
  addKernelPass(leaveOneOutGraph, {
    id: 'rainfall-leave-one-out-status-copy',
    invocationCount: 1,
    bindings: [
      {
        name: 'krigingInvalid',
        view: leaveOneOutKrigingStatus.invalidCount!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'krigingUncertain',
        view: leaveOneOutKrigingStatus.uncertainCount!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'summaryInvalid',
        view: leaveOneOutSummaryStatus.invalidCount!,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'invalidOutput',
        view: importGraphBuffer(
          leaveOneOutGraph,
          'leave-one-out-kriging-invalid',
          leaveOneOutKrigingInvalidBuffer,
          'uint32',
          1
        ),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'uncertainOutput',
        view: importGraphBuffer(
          leaveOneOutGraph,
          'leave-one-out-kriging-uncertain',
          leaveOneOutKrigingUncertainBuffer,
          'uint32',
          1
        ),
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'summaryOutput',
        view: importGraphBuffer(
          leaveOneOutGraph,
          'leave-one-out-summary-invalid',
          leaveOneOutSummaryInvalidBuffer,
          'uint32',
          1
        ),
        type: 'u32',
        access: 'read_write'
      }
    ],
    body: `invalidOutput[invalidOutputOffset] = krigingInvalid[krigingInvalidOffset];
uncertainOutput[uncertainOutputOffset] = krigingUncertain[krigingUncertainOffset];
summaryOutput[summaryOutputOffset] = summaryInvalid[summaryInvalidOffset];`
  });
  const leaveOneOutCompiled: CompiledGPUCommandGraph<void> = resources.track(
    await leaveOneOutGraph.compileAsync()
  );

  // --- State ----------------------------------------------------------------------------------
  let destroyed = false;
  let variogramOverflowed = false;
  let measuring = false;
  let dirtyRaster = true;
  let spatialValidationStale = false;
  let splitVersion = 0;
  let requestedSplitVersion = 0;
  let variogramStale = true;
  let splitSeed = 1;
  let heldOutCount = 0;
  let trainingCount = 0;
  let lastChangeTime = performance.now();
  let readStale = true;
  let surfaceVersion = 0;
  let requestedSurfaceVersion = 0;
  let variogramVersion = 0;
  let requestedVariogramVersion = 0;
  let fitVersion = 0;
  let requestedFitVersion = 0;
  let fitStale = false;
  let leaveOneOutVersion = 0;
  let requestedLeaveOneOutVersion = 0;
  let leaveOneOutStale = true;
  let leaveOneOutRead: {
    predictions: Float32Array;
    predictionVariance: Float32Array;
    residuals: Float32Array;
    standardizedResiduals: Float32Array;
    validCount: number;
    standardizedCount: number;
    bias: number;
    meanAbsoluteError: number;
    rootMeanSquaredError: number;
    krigingInvalid: number;
    krigingUncertain: number;
    summaryInvalid: number;
  } | null = null;
  let viewBounds: [number, number, number, number] = [...stationBounds];
  let encodedBounds: [number, number, number, number] = [...stationBounds];
  let bins: {sectors: Bins[]; omni: Bins; statistics: Float32Array} | null = null;
  let fit: VariogramModel | null = null;
  let heldOut = new Uint8Array(stationCount);
  let heldOutList: number[] = [];
  let surfaceRead: {
    extent: [number, number, number, number];
    idw: Float32Array;
    kriging: Float32Array;
    variance: Float32Array;
  } | null = null;
  let requestedExtent: [number, number, number, number] = [...stationBounds];
  let lastLegendKey = '';

  const markChanged = () => {
    lastChangeTime = performance.now();
    readStale = true;
    surfaceVersion++;
  };

  const spatialValidationReader = new SummaryReader(
    resources,
    'rainfall-spatial-validation',
    [
      {buffer: spatialTraining, size: stationCount * SPATIAL_FOLD_COUNT * 4},
      {buffer: spatialTesting, size: stationCount * SPATIAL_FOLD_COUNT * 4},
      {buffer: spatialCounts, size: SPATIAL_FOLD_COUNT * 3 * 4},
      {buffer: spatialStatus, size: SPATIAL_FOLD_COUNT * 4}
    ],
    bytes => {
      if (
        requestedSplitVersion !== splitVersion ||
        !ctx.options.holdOut ||
        ctx.options.validationSplit !== 'spatial'
      ) {
        return;
      }
      const training = new Uint32Array(bytes, 0, stationCount);
      const testing = new Uint32Array(bytes, stationCount * SPATIAL_FOLD_COUNT * 4, stationCount);
      const countsOffset = stationCount * SPATIAL_FOLD_COUNT * 8;
      const counts = new Uint32Array(bytes, countsOffset, SPATIAL_FOLD_COUNT * 3);
      const status = new Uint32Array(
        bytes,
        countsOffset + SPATIAL_FOLD_COUNT * 3 * 4,
        SPATIAL_FOLD_COUNT
      );
      if (status[0] !== 0 || counts[0] === 0 || counts[1] === 0) {
        ctx.setReadout('validationVerdict', 'spatial fold has no usable training or test gauges');
        ctx.setReadout('validationSupport', 'spatial fold unavailable');
        ctx.setChart('validationChart', null);
        return;
      }
      heldOut = Uint8Array.from(testing, flag => (flag ? 1 : 0));
      heldOutList = [];
      for (let station = 0; station < stationCount; station++) {
        if (heldOut[station]) heldOutList.push(station);
      }
      trainingCount = counts[0];
      heldOutCount = counts[1];
      maskBuffer.write(training);
      heldOutIds.write(Uint32Array.from(heldOutList));
      ctx.setReadout('training', formatCount(trainingCount));
      ctx.setReadout(
        'heldOut',
        `${formatCount(heldOutCount)} · spatial fold 1/${SPATIAL_FOLD_COUNT}`
      );
      ctx.setReadout(
        'validationDesign',
        `${SPATIAL_BLOCK_SIZE_METERS / 1000} km tiles · one of ${SPATIAL_FOLD_COUNT} folds · deterministic assignment`
      );
      variogramStale = true;
      dirtyRaster = true;
      markChanged();
      ctx.requestLayers();
    }
  );

  function getValues(): Float32Array {
    return valuesByVariable[ctx.options.variable];
  }

  function writeSplit(): void {
    variogramVersion++;
    surfaceRead = null;
    heldOut = new Uint8Array(stationCount);
    heldOutList = [];
    const values = getValues();
    trainingCount = 0;
    for (let station = 0; station < stationCount; station++) {
      const finite = Number.isFinite(values[station]);
      const out =
        ctx.options.holdOut &&
        ctx.options.validationSplit === 'random' &&
        finite &&
        hashInteger(station * 31 + splitSeed) % HOLD_OUT_MODULUS === 0;
      if (out) {
        heldOut[station] = 1;
        heldOutList.push(station);
      } else if (finite) {
        trainingCount++;
      }
    }
    if (ctx.options.holdOut && ctx.options.validationSplit === 'spatial') {
      splitVersion++;
      spatialBlocks.write(
        Uint32Array.from(spatialBlockIds, block => block ^ hashInteger(splitSeed))
      );
      spatialValidationMask.write(
        Uint32Array.from(values, value => (Number.isFinite(value) ? 1 : 0))
      );
      spatialValidationStale = true;
      heldOut.fill(0);
      heldOutList = [];
      heldOutCount = 0;
      trainingCount = 0;
      maskBuffer.write(new Uint32Array(stationCount));
      ctx.setReadout('training', 'assigning spatial folds…');
      ctx.setReadout('heldOut', 'assigning spatial folds…');
      ctx.setReadout('validationSupport', 'assigning spatial folds…');
      ctx.setReadout('validationVerdict', 'waiting for spatial fold');
      ctx.setChart('validationChart', null);
    } else {
      heldOutCount = heldOutList.length;
      maskBuffer.write(Uint32Array.from(heldOut, flag => (flag ? 0 : 1)));
    }
    heldOutIds.write(Uint32Array.from(heldOutList));
    ctx.setReadout('stations', `${formatCount(stationCount)} gauges`);
    if (!(ctx.options.holdOut && ctx.options.validationSplit === 'spatial')) {
      ctx.setReadout('training', formatCount(trainingCount));
      ctx.setReadout('heldOut', ctx.options.holdOut ? formatCount(heldOutCount) : 'none');
      ctx.setReadout(
        'validationDesign',
        ctx.options.holdOut ? 'random gauge holdout · 10%' : 'holdout disabled'
      );
    }
  }

  function getSampleVariance(): number {
    const variance = bins?.statistics[2];
    return variance && variance > 0 ? variance : 1;
  }

  function writeParameters(): void {
    const o = ctx.options;
    const radius = o.searchRadiusKm * 1000;
    idwParameters.write(
      getGPUInverseDistanceWeightingParameterValues(
        {
          searchRadius: radius,
          power: o.power,
          neighborCount: o.neighborCount,
          minimumNeighborCount: o.minimumStations
        },
        getViewGrid(viewBounds)
      )
    );
    const total = getSampleVariance();
    const model = fit ?? {
      model: o.variogramModel,
      nugget: 0.1 * total,
      sill: 0.9 * total,
      range: 150_000
    };
    const krigingParameterValues = getGPUKrigingParameterValues(
      {
        searchRadius: radius,
        neighborCount: o.krigingNeighborCount,
        minimumNeighborCount: Math.max(o.minimumStations, 3),
        variogram: {
          model: model.model,
          nugget: Math.max(model.nugget, 1e-4 * total),
          sill: Math.max(model.sill, 1e-4 * total),
          range: Math.max(model.range, 1000)
        }
      },
      getViewGrid(viewBounds)
    );
    krigingParameters.write(krigingParameterValues);
    ctx.setReadout(
      'krigingParameters',
      `sample-variance scale ${total.toPrecision(7)} · ${model.model} variogram in GPU float32 buffer: nugget ${krigingParameterValues[7]!.toPrecision(7)}, sill ${krigingParameterValues[8]!.toPrecision(7)}, range ${krigingParameterValues[9]!.toPrecision(7)} m · floors nugget/sill 1e−4 × variance, range 1,000 m`
    );
    focalParameters.write(
      getGPUFocalStatisticsParameterValues({
        radius: o.focalRadius,
        shape: o.focalShape,
        minimumCount: o.focalMinimumCount,
        propagateCenterNoData: o.keepHoles
      })
    );
    modeParameters.write(
      Float32Array.of(
        o.surface === 'kriging' ? 1 : o.surface === 'difference' ? 2 : 0,
        o.surface === 'support' ? 1 : 0,
        0,
        0
      )
    );
  }

  function writeVariogramParameters(): void {
    variogramParameters.write(
      getGPUVariogramParameterValues({
        bounds: indexBounds,
        maximumDistance: ctx.options.maximumLagKm * 1000,
        azimuthOffset: toRadians(ctx.options.azimuth)
      })
    );
  }

  // --- Variogram read-back and CPU fit --------------------------------------------------------
  const variogramReader = new SummaryReader(
    resources,
    'rainfall-variogram',
    [
      {buffer: semivarianceBuffer, size: BIN_COUNT * 4},
      {buffer: robustBuffer, size: BIN_COUNT * 4},
      {buffer: pairCountBuffer, size: BIN_COUNT * 4},
      {buffer: distanceBuffer, size: BIN_COUNT * 4},
      {buffer: statisticsBuffer, size: 5 * 4},
      {buffer: variogramOverflowBuffer, size: 4}
    ],
    bytes => {
      if (requestedVariogramVersion !== variogramVersion) {
        variogramStale = true;
        return;
      }
      const floats = new Float32Array(bytes);
      const counts = new Uint32Array(bytes);
      const read = (sector: number): Bins => {
        const result: Bins = {distances: [], gammas: [], robust: [], pairs: []};
        for (let lag = 0; lag < LAG_COUNT; lag++) {
          const bin = sector * LAG_COUNT + lag;
          result.gammas.push(floats[bin]);
          result.robust.push(floats[BIN_COUNT + bin]);
          result.pairs.push(counts[2 * BIN_COUNT + bin]);
          result.distances.push(floats[3 * BIN_COUNT + bin]);
        }
        return result;
      };
      const sectors = Array.from({length: SECTOR_COUNT}, (_, sector) => read(sector));
      const omni: Bins = {distances: [], gammas: [], robust: [], pairs: []};
      for (let lag = 0; lag < LAG_COUNT; lag++) {
        let pairs = 0;
        let robustPairs = 0;
        let gamma = 0;
        let robustRootMeanSum = 0;
        let distance = 0;
        for (const sector of sectors) {
          const n = sector.pairs[lag];
          if (n > 0 && Number.isFinite(sector.gammas[lag])) {
            pairs += n;
            gamma += n * sector.gammas[lag];
            distance += n * sector.distances[lag];
          }
          const sectorRobust = sector.robust[lag];
          if (n > 0 && Number.isFinite(sectorRobust) && sectorRobust >= 0) {
            // GPUVariogram uses gamma_CH = mean(sqrt(abs(delta)))^4 / (2 * (0.457 + 0.494 / N)).
            // Recover each sector's root mean, pool its pair-level terms, then reapply the pooled correction.
            const sectorRootMean = (2 * (0.457 + 0.494 / n) * sectorRobust) ** 0.25;
            robustPairs += n;
            robustRootMeanSum += n * sectorRootMean;
          }
        }
        omni.pairs.push(pairs);
        omni.gammas.push(pairs > 0 ? gamma / pairs : Number.NaN);
        omni.robust.push(
          robustPairs > 0
            ? (robustRootMeanSum / robustPairs) ** 4 / (2 * (0.457 + 0.494 / robustPairs))
            : Number.NaN
        );
        omni.distances.push(pairs > 0 ? distance / pairs : Number.NaN);
      }
      variogramOverflowed = counts[4 * BIN_COUNT + 5] !== 0;
      bins = {sectors, omni, statistics: floats.slice(4 * BIN_COUNT, 4 * BIN_COUNT + 5)};
      refit();
    }
  );

  const variogramFitReader = new SummaryReader(
    resources,
    'rainfall-variogram-model-fit',
    fitOutputBuffers.flatMap(fitOutput => [
      {buffer: fitOutput.parameters, size: 3 * 16},
      {buffer: fitOutput.diagnostics, size: 3 * 16},
      {buffer: fitOutput.status, size: 3 * 4}
    ]),
    bytes => {
      if (requestedFitVersion !== fitVersion || destroyed) return;
      const floats = new Float32Array(bytes);
      const statuses = new Uint32Array(bytes);
      let byteOffset = 0;
      for (const fitOutput of fitOutputBuffers) {
        fitOutput.result = {
          parameters: floats.slice(byteOffset / 4, byteOffset / 4 + 12),
          diagnostics: floats.slice(byteOffset / 4 + 12, byteOffset / 4 + 24),
          status: statuses.slice(byteOffset / 4 + 24, byteOffset / 4 + 27)
        };
        byteOffset += 3 * 16 + 3 * 16 + 3 * 4;
      }
      updateFitComparison();
    }
  );

  const leaveOneOutReader = new SummaryReader(
    resources,
    'rainfall-conditional-leave-one-out',
    [
      {buffer: leaveOneOutPredictionBuffer, size: stationCount * 4},
      {buffer: leaveOneOutVarianceBuffer, size: stationCount * 4},
      {buffer: leaveOneOutResidualBuffer, size: stationCount * 4},
      {buffer: leaveOneOutStandardizedBuffer, size: stationCount * 4},
      {buffer: leaveOneOutValidCountBuffer, size: 4},
      {buffer: leaveOneOutStandardizedCountBuffer, size: 4},
      {buffer: leaveOneOutBiasBuffer, size: 4},
      {buffer: leaveOneOutMaeBuffer, size: 4},
      {buffer: leaveOneOutRmseBuffer, size: 4},
      {buffer: leaveOneOutKrigingInvalidBuffer, size: 4},
      {buffer: leaveOneOutKrigingUncertainBuffer, size: 4},
      {buffer: leaveOneOutSummaryInvalidBuffer, size: 4}
    ],
    bytes => {
      if (requestedLeaveOneOutVersion !== leaveOneOutVersion || destroyed) return;
      const floats = new Float32Array(bytes);
      const counts = new Uint32Array(bytes);
      leaveOneOutRead = {
        predictions: floats.slice(0, stationCount),
        predictionVariance: floats.slice(stationCount, stationCount * 2),
        residuals: floats.slice(stationCount * 2, stationCount * 3),
        standardizedResiduals: floats.slice(stationCount * 3, stationCount * 4),
        validCount: counts[stationCount * 4],
        standardizedCount: counts[stationCount * 4 + 1],
        bias: floats[stationCount * 4 + 2],
        meanAbsoluteError: floats[stationCount * 4 + 3],
        rootMeanSquaredError: floats[stationCount * 4 + 4],
        krigingInvalid: counts[stationCount * 4 + 5],
        krigingUncertain: counts[stationCount * 4 + 6],
        summaryInvalid: counts[stationCount * 4 + 7]
      };
      updateLeaveOneOutReadouts();
    }
  );

  function fitBins(source: Bins): VariogramModel | null {
    if (variogramOverflowed) return null;
    try {
      const distances = Float32Array.from(source.distances);
      const semivariances = Float32Array.from(ctx.options.robust ? source.robust : source.gammas);
      const pairCounts = Uint32Array.from(source.pairs);
      return fitVariogramModel(
        {
          distances,
          semivariances,
          pairCounts
        },
        {model: ctx.options.variogramModel, weighting: ctx.options.weighting}
      );
    } catch {
      return null;
    }
  }

  /** Fits the model on the CPU to the omnidirectional bins and refreshes the variogram readouts. */
  function refit(): void {
    if (!bins || destroyed) return;
    const o = ctx.options;
    const unit = RAINFALL_VARIABLES[o.variable].squaredUnit;
    for (const fitOutput of fitOutputBuffers) fitOutput.result = undefined;
    fitInputBuffers.distances.write(Float32Array.from(bins.omni.distances));
    fitInputBuffers.matheron.write(Float32Array.from(bins.omni.gammas));
    fitInputBuffers.robust.write(Float32Array.from(bins.omni.robust));
    fitInputBuffers.pairCounts.write(Uint32Array.from(bins.omni.pairs));
    if (!variogramOverflowed) {
      fitStale = true;
      fitVersion++;
    } else {
      fitStale = false;
    }
    fit = fitBins(bins.omni);
    clearLeaveOneOutDiagnostics();
    const lags = o.robust ? bins.omni.robust : bins.omni.gammas;
    const modelValues = bins.omni.distances.map(distance =>
      fit ? evaluateVariogramModel(fit, distance) : Number.NaN
    );
    const scale = Math.max(
      ...lags.filter(Number.isFinite),
      ...modelValues.filter(Number.isFinite),
      1e-9
    );
    ctx.setReadout('empirical', formatSparkline(lags, scale));
    ctx.setReadout('fitted', formatSparkline(modelValues, scale));
    ctx.setReadout(
      'lagAxis',
      `0 to ${o.maximumLagKm} km in ${LAG_COUNT} lags; top of the chart ${formatSignificant(scale)} ${unit}`
    );
    ctx.setReadout(
      'variogramModel',
      variogramOverflowed
        ? 'pair counts saturated (overflow): not fitted'
        : fit
          ? `nugget ${formatSignificant(fit.nugget)} · partial sill ${formatSignificant(fit.sill)} ${unit} · range ${formatSignificant(fit.range / 1000)} km`
          : 'not fitted'
    );
    ctx.setReadout('sampleVariance', `${formatSignificant(bins.statistics[2])} ${unit}`);
    updateFitComparison();
    updateDirectionReadout();
    dirtyRaster = true;
    markChanged();
    ctx.requestLayers();
  }

  function updateFitComparison(): void {
    if (!bins || destroyed) return;
    const empiricalBins = bins.omni;
    const o = ctx.options;
    const unit = RAINFALL_VARIABLES[o.variable].squaredUnit;
    const chosenOutput = fitOutputBuffers.find(
      output => output.weighting === o.weighting && output.robust === o.robust
    );
    const modelIndex = VARIOGRAM_FIT_MODELS.indexOf(o.variogramModel);
    const gpuResult = chosenOutput?.result;
    const gpuStatus = gpuResult?.status[modelIndex];
    const gpuParameters = gpuResult?.parameters;
    const gpuModelFit =
      gpuParameters && gpuStatus !== undefined && (gpuStatus === 0 || gpuStatus === 3)
        ? {
            model: o.variogramModel,
            nugget: gpuParameters[modelIndex * 4],
            sill: gpuParameters[modelIndex * 4 + 1],
            range: gpuParameters[modelIndex * 4 + 2],
            residual: gpuParameters[modelIndex * 4 + 3]
          }
        : null;
    const lags = o.robust ? bins.omni.robust : bins.omni.gammas;
    const modelValues = bins.omni.distances.map(distance =>
      fit ? evaluateVariogramModel(fit, distance) : Number.NaN
    );
    const gpuModelValues = bins.omni.distances.map(distance =>
      gpuModelFit ? evaluateVariogramModel(gpuModelFit, distance) : Number.NaN
    );
    const gpuDiagnostics = gpuResult?.diagnostics;
    const diagnosticOffset = modelIndex * 4;
    const gpuParameterDescription = gpuModelFit
      ? `GPU nugget ${formatSignificant(gpuModelFit.nugget)} ${unit} · sill ${formatSignificant(gpuModelFit.sill)} ${unit} · range ${formatSignificant(gpuModelFit.range / 1000)} km · weighted residual ${formatSignificant(gpuModelFit.residual)}. `
      : '';
    const diagnosticsDescription =
      gpuDiagnostics && (gpuStatus === 0 || gpuStatus === 3)
        ? `GPU diagnostics: ${gpuDiagnostics[diagnosticOffset]} usable bins · SSE ${formatSignificant(gpuDiagnostics[diagnosticOffset + 1])} · AIC ${formatSignificant(gpuDiagnostics[diagnosticOffset + 2])} · relative log-range bracket ${formatSignificant(gpuDiagnostics[diagnosticOffset + 3])}.`
        : '';
    const cpuGpuDifference =
      fit && gpuModelFit
        ? `Δ range ${formatSignificant((gpuModelFit.range - fit.range) / 1000)} km · Δ nugget ${formatSignificant(gpuModelFit.nugget - fit.nugget)} ${unit} · Δ sill ${formatSignificant(gpuModelFit.sill - fit.sill)} ${unit}`
        : '';
    const rawCpuFit = fit
      ? `CPU [${[fit.nugget, fit.sill, fit.range, fit.residual].map(value => value.toPrecision(7)).join(', ')}]`
      : 'CPU fit unavailable';
    const rawGpuFit = gpuModelFit
      ? `GPU [${[gpuModelFit.nugget, gpuModelFit.sill, gpuModelFit.range, gpuModelFit.residual].map(value => value.toPrecision(7)).join(', ')}]`
      : 'GPU fit unavailable';
    const statusDescription = variogramOverflowed
      ? 'Pair-count overflow: CPU and GPU model fits withheld. '
      : gpuStatus === 0
        ? `GPU fit status 0 (interior); ${cpuGpuDifference}. `
        : gpuStatus === 3
          ? `GPU fit status 3 (boundary or iteration budget): candidate only; range is not identified. ${cpuGpuDifference}. `
          : gpuStatus === 1
            ? 'GPU fit status 1 (too few usable bins); fit withheld. '
            : gpuStatus === 2
              ? 'GPU fit status 2 (numerical failure); fit withheld. '
              : 'GPU fit pending. ';
    if (gpuStatus !== undefined && gpuStatus !== 0) ctx.setStatus(statusDescription);
    ctx.setReadout(
      'fitComparison',
      `${statusDescription}${gpuParameterDescription}${diagnosticsDescription} Raw parameters: ${rawCpuFit} · ${rawGpuFit} (nugget, sill, range m, weighted residual). AIC is a descriptive bin-residual diagnostic, not a spatial likelihood or a comparison across different bins.`
    );
    ctx.setReadout(
      'variogramInputBins',
      [
        'distance_m, semivariance, pair_count',
        ...empiricalBins.distances.map(
          (distance, lag) =>
            `${distance.toPrecision(7)}, ${lags[lag].toPrecision(7)}, ${empiricalBins.pairs[lag]}`
        )
      ].join('\n')
    );
    ctx.setChart('variogramChart', {
      kind: 'line',
      title: 'Empirical bins and CPU/GPU model fits',
      xLabel: 'Gauge separation (km)',
      yLabel: `Semivariance (${unit})`,
      series: [
        {
          label: o.robust ? 'Robust empirical' : 'Empirical',
          x: bins.omni.distances.map(distance => distance / 1000),
          y: lags,
          points: true
        },
        {
          label: `${o.variogramModel} CPU`,
          x: bins.omni.distances.map(distance => distance / 1000),
          y: modelValues,
          dashed: true
        },
        {
          label: `${o.variogramModel} GPU`,
          x: bins.omni.distances.map(distance => distance / 1000),
          y: gpuModelValues,
          dashed: true
        }
      ],
      table: true,
      guides: [{y: bins.statistics[2], label: 'sample variance'}],
      markers:
        fit && Number.isFinite(fit.range)
          ? [{x: fit.range / 1000, label: `range ${formatSignificant(fit.range / 1000)} km`}]
          : undefined,
      description:
        'GPU-binned omnidirectional semivariance, CPU and GPU fits to the same displayed bins, and sample-variance reference. Show values exposes the actual input bins.'
    });
  }

  function clearLeaveOneOutDiagnostics(): void {
    leaveOneOutVersion++;
    leaveOneOutRead = null;
    leaveOneOutStale = !ctx.options.holdOut;
    ctx.setStatus(
      ctx.options.holdOut
        ? 'Conditional LOO paused during hold-out validation'
        : 'Conditional LOO updating'
    );
    ctx.setReadout(
      'leaveOneOutStatus',
      ctx.options.holdOut ? 'paused while a separate hold-out split is active' : 'updating…'
    );
    ctx.setReadout('leaveOneOutMetrics', 'updating…');
    ctx.setReadout('leaveOneOutValues', 'waiting for actual fitted-model diagnostics');
    ctx.setChart('leaveOneOutChart', null);
  }

  function clearVariogramComparison(): void {
    fitVersion++;
    fitStale = false;
    fit = null;
    for (const fitOutput of fitOutputBuffers) fitOutput.result = undefined;
    ctx.setReadout('fitComparison', 'waiting for updated empirical bins');
    ctx.setReadout('variogramModel', 'waiting for updated empirical bins');
    ctx.setReadout('krigingParameters', 'waiting for current fitted variogram and sample variance');
    ctx.setReadout('empirical', '...');
    ctx.setReadout('fitted', '...');
    ctx.setChart('variogramChart', null);
  }

  function updateLeaveOneOutReadouts(): void {
    if (!leaveOneOutRead) return;
    const diagnosticRead = leaveOneOutRead;
    const observedValues = getValues();
    const unit = RAINFALL_VARIABLES[ctx.options.variable].unit;
    const valid = diagnosticRead.validCount;
    const standardized = diagnosticRead.standardizedCount;
    const finiteGaugeCount = observedValues.reduce(
      (count, value) => count + Number(Number.isFinite(value)),
      0
    );
    const status = `Conditional LOO gauges ${valid}/${finiteGaugeCount} valid; standardized scores require positive prediction variance (${standardized}/${valid}). Kriging raster output cells: ${diagnosticRead.krigingInvalid} invalid or unsupported, ${diagnosticRead.krigingUncertain} uncertain; LOO summary status: ${diagnosticRead.summaryInvalid} invalid.`;
    ctx.setStatus(status);
    ctx.setReadout('leaveOneOutStatus', status);
    const fixedProbeValues = stationIdentifiers
      .slice(0, 3)
      .map(
        (stationIdentifier, stationIndex) =>
          `${stationIdentifier}: observed ${observedValues[stationIndex].toPrecision(7)}, predicted ${diagnosticRead.predictions[stationIndex].toPrecision(7)}, variance ${diagnosticRead.predictionVariance[stationIndex].toPrecision(7)}, residual ${diagnosticRead.residuals[stationIndex].toPrecision(7)}, standardized ${diagnosticRead.standardizedResiduals[stationIndex].toPrecision(7)}`
      )
      .join(' · ');
    ctx.setReadout(
      'leaveOneOutMetrics',
      `bias ${formatSignificant(diagnosticRead.bias)} ${unit} · MAE ${formatSignificant(diagnosticRead.meanAbsoluteError)} ${unit} · RMSE ${formatSignificant(diagnosticRead.rootMeanSquaredError)} ${unit} · n=${valid}; fixed source-ID probes: ${fixedProbeValues}`
    );
    if (ctx.options.holdOut) {
      ctx.setChart('leaveOneOutChart', null);
      return;
    }
    const observed: number[] = [];
    const predicted: number[] = [];
    const tableRows: string[] = [
      'source_station_id, observed, predicted, prediction variance, residual, standardized residual'
    ];
    for (let station = 0; station < stationCount; station++) {
      const observedValue = observedValues[station];
      const predictedValue = diagnosticRead.predictions[station];
      const predictionVariance = diagnosticRead.predictionVariance[station];
      const residual = diagnosticRead.residuals[station];
      const standardizedResidual = diagnosticRead.standardizedResiduals[station];
      if (Number.isFinite(observedValue) && Number.isFinite(predictedValue)) {
        observed.push(observedValue);
        predicted.push(predictedValue);
      }
      tableRows.push(
        `${stationIdentifiers[station]}, ${observedValue.toPrecision(7)}, ${predictedValue.toPrecision(7)}, ${predictionVariance.toPrecision(7)}, ${residual.toPrecision(7)}, ${standardizedResidual.toPrecision(7)}`
      );
    }
    ctx.setReadout('leaveOneOutValues', tableRows.join('\n'));
    ctx.setChart('leaveOneOutChart', {
      kind: 'scatter',
      title: 'Observed versus leave-one-out predicted rainfall',
      xLabel: `Observed (${unit})`,
      yLabel: `LOO prediction (${unit})`,
      x: observed,
      y: predicted,
      diagonal: true,
      table: true,
      description:
        'Each point compares an actual gauge observation with its leave-one-out prediction in the same units; the diagonal marks equality. Show values exposes both actual numbers. The full station table also includes prediction variance, residual and standardized residual; standardized values are available only for positive prediction variance. This is conditional diagnostics because the variogram was fitted on all gauges before leave-one-out prediction.'
    });
  }

  function updateDirectionReadout(): void {
    if (!bins) return;
    const o = ctx.options;
    if (o.direction === 'all') {
      ctx.setReadout('sectorRange', 'omnidirectional (choose a sector to compare directions)');
      return;
    }
    const sector = Number(o.direction);
    const from = (o.azimuth + (sector * 180) / SECTOR_COUNT) % 180;
    const to = from + 180 / SECTOR_COUNT;
    const sectorFit = fitBins(bins.sectors[sector]);
    ctx.setReadout(
      'sectorRange',
      sectorFit
        ? `${from.toFixed(0)}° to ${to.toFixed(0)}° from east: range ${formatSignificant(sectorFit.range / 1000)} km, sill ${formatSignificant(sectorFit.nugget + sectorFit.sill)} (omnidirectional ${formatSignificant((fit?.range ?? Number.NaN) / 1000)} km)`
        : `${from.toFixed(0)}° to ${to.toFixed(0)}° from east: too few pairs`
    );
  }

  // --- Surface read-back: hold-out validation and tooltips ------------------------------------
  const surfaceReader = new SummaryReader(
    resources,
    'rainfall-surfaces',
    [
      {buffer: idwSurface, size: CELL_COUNT * 4},
      {buffer: krigingSurface, size: CELL_COUNT * 4},
      {buffer: krigingVariance, size: CELL_COUNT * 4}
    ],
    bytes => {
      if (requestedSurfaceVersion !== surfaceVersion) {
        readStale = true;
        return;
      }
      surfaceRead = {
        extent: requestedExtent,
        idw: new Float32Array(bytes, 0, CELL_COUNT),
        kriging: new Float32Array(bytes, CELL_COUNT * 4, CELL_COUNT),
        variance: new Float32Array(bytes, CELL_COUNT * 8, CELL_COUNT)
      };
      validateHeldOut();
    }
  );

  /** Bilinear sample of a read-back raster at planar meters; NaN when outside or nodata. */
  function sampleRaster(
    raster: Float32Array,
    extent: readonly [number, number, number, number],
    x: number,
    y: number
  ): number {
    const cellWidth = (extent[2] - extent[0]) / RASTER_WIDTH;
    const cellHeight = (extent[3] - extent[1]) / RASTER_HEIGHT;
    const column = (x - extent[0]) / cellWidth - 0.5;
    const row = (y - extent[1]) / cellHeight - 0.5;
    if (column < -0.5 || row < -0.5 || column > RASTER_WIDTH - 0.5 || row > RASTER_HEIGHT - 0.5) {
      return Number.NaN;
    }
    const column0 = Math.max(0, Math.min(RASTER_WIDTH - 2, Math.floor(column)));
    const row0 = Math.max(0, Math.min(RASTER_HEIGHT - 2, Math.floor(row)));
    const fx = Math.max(0, Math.min(1, column - column0));
    const fy = Math.max(0, Math.min(1, row - row0));
    let sum = 0;
    let weight = 0;
    for (const [dx, dy, w] of [
      [0, 0, (1 - fx) * (1 - fy)],
      [1, 0, fx * (1 - fy)],
      [0, 1, (1 - fx) * fy],
      [1, 1, fx * fy]
    ] as const) {
      const value = raster[(row0 + dy) * RASTER_WIDTH + column0 + dx];
      if (Number.isFinite(value) && w > 0) {
        sum += w * value;
        weight += w;
      }
    }
    return weight > 0.5 ? sum / weight : Number.NaN;
  }

  function validateHeldOut(): void {
    if (!surfaceRead || !ctx.options.holdOut) {
      for (const id of ['validationIdw', 'validationKriging', 'validationVerdict'])
        ctx.setReadout(id, ctx.options.holdOut ? '...' : 'turn on "Hold out 10% of gauges"');
      ctx.setReadout(
        'validationSupport',
        ctx.options.holdOut ? 'waiting for surface readback' : 'holdout disabled'
      );
      if (!ctx.options.holdOut) ctx.setReadout('validationDesign', 'holdout disabled');
      ctx.setChart('validationChart', null);
      return;
    }
    const values = getValues();
    const unit = RAINFALL_VARIABLES[ctx.options.variable].unit;
    const scores = {idw: {sum: 0, abs: 0, bias: 0, n: 0}, kriging: {sum: 0, abs: 0, bias: 0, n: 0}};
    const cellWidth = (surfaceRead.extent[2] - surfaceRead.extent[0]) / RASTER_WIDTH;
    const cellHeight = (surfaceRead.extent[3] - surfaceRead.extent[1]) / RASTER_HEIGHT;
    let visibleHeldOutCount = 0;
    for (const station of heldOutList) {
      const x = positions[station * 2];
      const y = positions[station * 2 + 1];
      if (
        x < surfaceRead.extent[0] - cellWidth / 2 ||
        x > surfaceRead.extent[2] + cellWidth / 2 ||
        y < surfaceRead.extent[1] - cellHeight / 2 ||
        y > surfaceRead.extent[3] + cellHeight / 2
      ) {
        continue;
      }
      visibleHeldOutCount++;
      for (const [name, raster] of [
        ['idw', surfaceRead.idw],
        ['kriging', surfaceRead.kriging]
      ] as const) {
        const predicted = sampleRaster(raster, surfaceRead.extent, x, y);
        if (!Number.isFinite(predicted)) continue;
        const error = predicted - values[station];
        scores[name].sum += error * error;
        scores[name].abs += Math.abs(error);
        scores[name].bias += error;
        scores[name].n++;
      }
    }
    const describe = (score: typeof scores.idw) =>
      score.n > 0
        ? `RMSE ${formatSignificant(Math.sqrt(score.sum / score.n))} ${unit} · MAE ${formatSignificant(score.abs / score.n)} · bias ${formatSignificant(score.bias / score.n)} (n=${score.n})`
        : 'no held-out gauge on screen';
    ctx.setReadout('validationIdw', describe(scores.idw));
    ctx.setReadout('validationKriging', describe(scores.kriging));
    ctx.setReadout(
      'validationSupport',
      `held out ${heldOutList.length} · visible ${visibleHeldOutCount} · IDW ${scores.idw.n} scored / ${visibleHeldOutCount - scores.idw.n} unsupported · kriging ${scores.kriging.n} scored / ${visibleHeldOutCount - scores.kriging.n} unsupported`
    );
    if (scores.idw.n > 0 && scores.kriging.n > 0) {
      const idwMetrics = {
        rmse: Math.sqrt(scores.idw.sum / scores.idw.n),
        mae: scores.idw.abs / scores.idw.n,
        bias: scores.idw.bias / scores.idw.n
      };
      const krigingMetrics = {
        rmse: Math.sqrt(scores.kriging.sum / scores.kriging.n),
        mae: scores.kriging.abs / scores.kriging.n,
        bias: scores.kriging.bias / scores.kriging.n
      };
      ctx.setChart('validationChart', {
        kind: 'dumbbell',
        title:
          ctx.options.validationSplit === 'spatial'
            ? 'Spatial-fold prediction errors'
            : 'Random-holdout prediction errors',
        xLabel: unit,
        aLabel: `IDW (n=${scores.idw.n})`,
        bLabel: `Kriging (n=${scores.kriging.n})`,
        rows: [
          {label: 'RMSE', a: idwMetrics.rmse, b: krigingMetrics.rmse, highlight: true},
          {label: 'MAE', a: idwMetrics.mae, b: krigingMetrics.mae},
          {label: 'Bias', a: idwMetrics.bias, b: krigingMetrics.bias}
        ],
        description:
          'IDW and kriging error metrics at held-out gauges inside the current raster extent; lower RMSE and MAE are better, while bias is best near zero.'
      });
    } else {
      ctx.setChart('validationChart', null);
    }
    if (scores.idw.n > 5 && scores.kriging.n > 5) {
      const idwRmse = Math.sqrt(scores.idw.sum / scores.idw.n);
      const krigingRmse = Math.sqrt(scores.kriging.sum / scores.kriging.n);
      const change = ((krigingRmse - idwRmse) / idwRmse) * 100;
      ctx.setReadout(
        'validationVerdict',
        `kriging RMSE is ${Math.abs(change).toFixed(1)}% ${change <= 0 ? 'lower' : 'higher'} than IDW at the held-out gauges on screen`
      );
    } else {
      ctx.setReadout('validationVerdict', 'zoom out so more held-out gauges are visible');
    }
  }

  // --- Timing ---------------------------------------------------------------------------------
  async function measureGraphs(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('graphTime', 'measuring...');
    try {
      const options = {
        parameters: undefined,
        completionBuffer: auxiliarySurface,
        signal: ctx.signal
      };
      const main = await measureCompiledGraph(device, compiled, options);
      const variogram = await measureCompiledGraph(device, variogramCompiled, {
        ...options,
        completionBuffer: statisticsBuffer
      });
      if (destroyed) return;
      ctx.setReadout(
        'graphTime',
        `${compiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(main)}`
      );
      ctx.setReadout(
        'variogramTime',
        `${variogramCompiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(variogram)}`
      );
    } catch {
      if (!destroyed) ctx.setReadout('graphTime', 'interrupted');
    } finally {
      measuring = false;
    }
  }

  writeSplit();
  writeVariogramParameters();
  ctx.setReadout('empirical', '...');
  ctx.setReadout('fitted', '...');
  ctx.setReadout('krigingParameters', 'waiting for current fitted variogram and sample variance');

  // --- Tooltip --------------------------------------------------------------------------------
  function getTooltip(coordinate: readonly [number, number]): string | null {
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    const zoom = ctx.getViewport()?.zoom ?? 5;
    const metersPerPixel = getWebMercatorMetersPerPixel(coordinate[1], zoom);
    const limit = STATION_NAME_LIMIT_PIXELS * metersPerPixel;
    const values = getValues();
    const meta = RAINFALL_VARIABLES[ctx.options.variable];
    let nearest = -1;
    let nearestDistance = limit * limit;
    for (let station = 0; station < stationCount; station++) {
      if (!Number.isFinite(values[station])) continue;
      const dx = positions[station * 2] - x;
      const dy = positions[station * 2 + 1] - y;
      const distance = dx * dx + dy * dy;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = station;
      }
    }
    if (nearest >= 0) {
      const state = stationStates[nearest] ? `, ${stationStates[nearest]}` : '';
      return `${stationNames[nearest]}${state}\n${formatSignificant(values[nearest])} ${meta.unit} observed · elevation ${formatSignificant(elevation[nearest], 4)} m${heldOut[nearest] ? '\nheld out: not used by the surfaces' : ''}`;
    }
    if (!surfaceRead) return null;
    const idw = sampleRaster(surfaceRead.idw, surfaceRead.extent, x, y);
    const kriging = sampleRaster(surfaceRead.kriging, surfaceRead.extent, x, y);
    const variance = sampleRaster(surfaceRead.variance, surfaceRead.extent, x, y);
    if (!Number.isFinite(idw) && !Number.isFinite(kriging)) return null;
    const parts = [`IDW ${formatSignificant(idw)} ${meta.unit}`];
    if (Number.isFinite(kriging)) {
      parts.push(
        `kriging ${formatSignificant(kriging)} ± ${formatSignificant(Math.sqrt(Math.max(variance, 0)))} ${meta.unit}`
      );
    }
    return parts.join('\n');
  }

  return {
    getCompiledGraphs: () =>
      [
        compiled,
        variogramCompiled,
        fitCompiled,
        leaveOneOutCompiled,
        spatialValidationCompiled
      ] as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'variable':
          valuesBuffer.write(getValues());
          writeSplit();
          variogramStale = true;
          dirtyRaster = true;
          clearVariogramComparison();
          clearLeaveOneOutDiagnostics();
          break;
        case 'holdOut':
          writeSplit();
          variogramStale = true;
          dirtyRaster = true;
          clearVariogramComparison();
          clearLeaveOneOutDiagnostics();
          break;
        case 'validationSplit':
          writeSplit();
          variogramStale = true;
          dirtyRaster = true;
          clearVariogramComparison();
          clearLeaveOneOutDiagnostics();
          break;
        case 'variogramModel':
        case 'weighting':
        case 'robust':
          refit();
          break;
        case 'maximumLagKm':
        case 'azimuth':
          variogramVersion++;
          writeVariogramParameters();
          variogramStale = true;
          clearVariogramComparison();
          clearLeaveOneOutDiagnostics();
          break;
        case 'direction':
          updateDirectionReadout();
          break;
        case 'ramp':
        case 'opacity':
        case 'showStations':
        case 'focalStatistic':
          break;
        default:
          dirtyRaster = true;
          if (['searchRadiusKm', 'krigingNeighborCount', 'minimumStations'].includes(id)) {
            clearLeaveOneOutDiagnostics();
          }
      }
      void state;
      markChanged();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reseed') {
        splitSeed++;
        writeSplit();
        variogramStale = true;
        dirtyRaster = true;
        clearVariogramComparison();
        clearLeaveOneOutDiagnostics();
        markChanged();
        ctx.requestLayers();
      } else if (id === 'measure') {
        void measureGraphs();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => (event.coordinate ? getTooltip(event.coordinate) : null),

    encode(commandEncoder, frame) {
      encodeFrame(commandEncoder, frame.viewport);
    },

    getLayers() {
      return buildLayers();
    },

    destroy() {
      destroyed = true;
      variogramReader.stop();
      variogramFitReader.stop();
      leaveOneOutReader.stop();
      surfaceReader.stop();
      spatialValidationReader.stop();
      resources.destroy();
    }
  };

  function encodeFrame(
    commandEncoder: Parameters<SceneInstance<RainfallOptions>['encode']>[0],
    viewport: Viewport
  ) {
    const next = getViewportFrameBounds(viewport, projection);
    if (next.some((value, index) => value !== viewBounds[index])) {
      viewBounds = next;
      dirtyRaster = true;
      markChanged();
      ctx.setReadout(
        'cellSize',
        `${formatSignificant((viewBounds[2] - viewBounds[0]) / RASTER_WIDTH / 1000)} km × ${formatSignificant((viewBounds[3] - viewBounds[1]) / RASTER_HEIGHT / 1000)} km`
      );
    }
    if (variogramStale && !variogramReader.isPending) {
      variogramCompiled.encode(commandEncoder, {parameters: undefined});
      requestedVariogramVersion = variogramVersion;
      variogramReader.request(commandEncoder);
      variogramStale = false;
    }
    variogramReader.flush(commandEncoder);
    if (fitStale && !variogramFitReader.isPending) {
      fitCompiled.encode(commandEncoder, {parameters: undefined});
      requestedFitVersion = fitVersion;
      variogramFitReader.request(commandEncoder);
      fitStale = false;
    }
    variogramFitReader.flush(commandEncoder);
    if (
      spatialValidationStale &&
      ctx.options.holdOut &&
      ctx.options.validationSplit === 'spatial' &&
      !spatialValidationReader.isPending
    ) {
      spatialValidationCompiled.encode(commandEncoder, {parameters: undefined});
      requestedSplitVersion = splitVersion;
      spatialValidationReader.request(commandEncoder);
      spatialValidationStale = false;
    }
    spatialValidationReader.flush(commandEncoder);
    if (dirtyRaster) {
      dirtyRaster = false;
      writeParameters();
      compiled.encode(commandEncoder, {parameters: undefined});
      encodedBounds = [...viewBounds];
    }
    if (
      leaveOneOutStale &&
      !ctx.options.holdOut &&
      fit !== null &&
      !variogramStale &&
      !variogramReader.isPending &&
      !variogramOverflowed &&
      !leaveOneOutReader.isPending
    ) {
      leaveOneOutCompiled.encode(commandEncoder, {parameters: undefined});
      requestedLeaveOneOutVersion = leaveOneOutVersion;
      leaveOneOutReader.request(commandEncoder);
      leaveOneOutStale = false;
    } else if (leaveOneOutStale && !ctx.options.holdOut) {
      ctx.setReadout(
        'leaveOneOutStatus',
        variogramOverflowed
          ? 'unavailable: variogram pair-count overflow prevents an actual fitted model'
          : fit === null
            ? 'waiting for a valid CPU variogram fit; fallback kriging parameters are not used for LOO'
            : 'waiting for current empirical bins or a previous LOO readback to finish'
      );
      ctx.setStatus(
        variogramOverflowed
          ? 'Conditional LOO unavailable: variogram pair-count overflow'
          : fit === null
            ? 'Conditional LOO waiting for an actual CPU variogram fit'
            : 'Conditional LOO waiting for current empirical bins or a previous readback'
      );
    }
    leaveOneOutReader.flush(commandEncoder);
    if (readStale && performance.now() - lastChangeTime > SETTLE_MILLISECONDS) {
      if (!surfaceReader.isPending) {
        requestedExtent = [...encodedBounds];
        requestedSurfaceVersion = surfaceVersion;
        surfaceReader.request(commandEncoder);
        readStale = false;
      }
    } else {
      surfaceReader.flush(commandEncoder);
    }
  }

  function buildLayers(): Layer[] {
    const o = ctx.options;
    const style = getRainfallSurfaceStyle(o);
    const meta = RAINFALL_VARIABLES[o.variable];
    const dark = ctx.theme() === 'dark';
    const isAuxiliary = o.surface === 'error' || o.surface === 'support';
    let range = style.range;
    if (!range) {
      const total = fit ? fit.nugget + fit.sill : getSampleVariance();
      range = [0, Math.sqrt(Math.max(total, 1e-9))];
    }
    const legendKey = `${range[0]}:${range[1]}`;
    if (legendKey !== lastLegendKey) {
      lastLegendKey = legendKey;
      ctx.setLegendExtent('surface', range);
    }
    const statistic = o.focalRadius > 0 ? o.focalStatistic : 'mean';
    const layers: Layer[] = [
      new B7TessellatedRasterLayer({
        id: `rainfall-${o.surface}-${statistic}`,
        analysisFrame: RAINFALL_FRAME,
        gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
        bounds: idwParameters.buffer,
        rowOrigin: 'south',
        values: isAuxiliary ? auxiliarySurface : focalStatistics[statistic],
        valueFormat: 'float32',
        colormap: style.ramp,
        valueRange: range,
        noDataColor: [0, 0, 0, 0],
        color: [255, 255, 255, Math.round(255 * o.opacity)]
      })
    ];
    if (o.showStations) {
      const stationRamp = o.ramp;
      const stationRange = meta.range;
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'rainfall-station-halo',
          analysisFrame: RAINFALL_FRAME,
          positions: positionsBuffer,
          values: valuesBuffer,
          valueFormat: 'float32',
          // A one-color ramp: stations without a value (NaN) stay invisible.
          colormap: 'grayscale',
          valueRange: dark ? [-1e30, -1e29] : [1e30, 1e31],
          noDataColor: [0, 0, 0, 0],
          instanceCount: stationCount,
          radiusPixels: 2.6,
          color: [255, 255, 255, dark ? 130 : 190]
        }),
        new SpatialAnalysisPointLayer({
          id: 'rainfall-stations',
          analysisFrame: RAINFALL_FRAME,
          positions: positionsBuffer,
          values: valuesBuffer,
          valueFormat: 'float32',
          colormap: stationRamp,
          valueRange: stationRange,
          noDataColor: [0, 0, 0, 0],
          instanceCount: stationCount,
          radiusPixels: 1.7,
          color: [255, 255, 255, 255]
        })
      );
      if (o.holdOut && heldOutCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'rainfall-held-out-ring',
            analysisFrame: RAINFALL_FRAME,
            positions: positionsBuffer,
            ids: heldOutIds,
            instanceCount: heldOutCount,
            radiusPixels: 5.4,
            color: dark ? [255, 255, 255, 255] : [20, 20, 20, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'rainfall-held-out',
            analysisFrame: RAINFALL_FRAME,
            positions: positionsBuffer,
            ids: heldOutIds,
            values: valuesBuffer,
            valueFormat: 'float32',
            colormap: stationRamp,
            valueRange: stationRange,
            noDataColor: [0, 0, 0, 0],
            instanceCount: heldOutCount,
            radiusPixels: 3.6,
            color: [255, 255, 255, 255]
          })
        );
      }
    }
    return layers;
  }
}
