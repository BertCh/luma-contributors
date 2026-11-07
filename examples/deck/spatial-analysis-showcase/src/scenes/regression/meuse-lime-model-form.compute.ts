// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  GPUProbitRegression
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {createNamedReader} from './b6-reader';

export type MeuseLimeModelFormOptions = {
  display: 'local-probability' | 'lime-observed';
};

const BANDWIDTHS = [750, 1000, 1500, 2200, 3500] as const;
const PREDICTOR_COUNT = 3;
const MAP_VALUE_RANGES = {
  'local-probability': [0, 1] as const,
  'lime-observed': [0, 1] as const
};

function standardize(values: Float32Array): Float32Array {
  let sum = 0;
  for (const value of values) sum += value;
  const mean = sum / values.length;
  let squared = 0;
  for (const value of values) squared += (value - mean) ** 2;
  const deviation = Math.sqrt(squared / values.length) || 1;
  return Float32Array.from(values, value => (value - mean) / deviation);
}

function makePredictorTable(columns: readonly Float32Array[], rowCount: number): Float32Array {
  const normalized = columns.map(standardize);
  const values = new Float32Array(rowCount * columns.length);
  for (let row = 0; row < rowCount; row++) {
    for (let column = 0; column < columns.length; column++) {
      values[row * columns.length + column] = normalized[column][row];
    }
  }
  return values;
}

function describeProbit(coefficients: Float32Array, standardErrors: Float32Array): string {
  if (coefficients.length < PREDICTOR_COUNT + 1 || !Number.isFinite(coefficients[0])) {
    return 'Probit model unavailable';
  }
  const labels = ['intercept', 'log zinc', 'elevation', 'organic matter'];
  return labels
    .map(
      (label, index) =>
        `${label} ${coefficients[index].toFixed(3)} (SE ${
          Number.isFinite(standardErrors[index]) ? standardErrors[index].toFixed(3) : '—'
        })`
    )
    .join(' · ');
}

/** GPU probit and geographically weighted logistic comparisons for measured lime presence. */
export async function createMeuseLimeModelForm(
  ctx: SceneContext<MeuseLimeModelFormOptions>
): Promise<SceneInstance<MeuseLimeModelFormOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const responseBytes = dataset.column<Uint8Array>('sampleLime');
  const sourceRowCount = responseBytes.length;
  const logZinc = dataset.column<Float32Array>('logZinc');
  const elevation = dataset.column<Float32Array>('sampleElevation');
  const organicMatter = dataset.column<Float32Array>('sampleOrganicMatter');
  const studyRows: number[] = [];
  for (let row = 0; row < sourceRowCount; row++) {
    if (
      Number.isFinite(logZinc[row]) &&
      Number.isFinite(elevation[row]) &&
      Number.isFinite(organicMatter[row])
    ) {
      studyRows.push(row);
    }
  }
  const rowCount = studyRows.length;
  const response = Float32Array.from(studyRows, row => responseBytes[row]);
  const predictors = makePredictorTable(
    [logZinc, elevation, organicMatter].map(column =>
      Float32Array.from(studyRows, row => column[row])
    ),
    rowCount
  );
  const samplePositionSource = dataset.column<Float32Array>('samplePositionRD');
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (const sourceRow of studyRows) {
    minimumX = Math.min(minimumX, samplePositionSource[sourceRow * 2]);
    maximumX = Math.max(maximumX, samplePositionSource[sourceRow * 2]);
    minimumY = Math.min(minimumY, samplePositionSource[sourceRow * 2 + 1]);
    maximumY = Math.max(maximumY, samplePositionSource[sourceRow * 2 + 1]);
  }
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const positions = new Float32Array(rowCount * 2);
  const allMapPositions = dataset.projectColumn('samplePosition', dataset.defaultOrigin);
  const mapPositions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    const sourceRow = studyRows[row];
    positions[row * 2] = samplePositionSource[sourceRow * 2] - centerX;
    positions[row * 2 + 1] = samplePositionSource[sourceRow * 2 + 1] - centerY;
    mapPositions[row * 2] = allMapPositions[sourceRow * 2];
    mapPositions[row * 2 + 1] = allMapPositions[sourceRow * 2 + 1];
  }
  const origin = dataset.defaultOrigin;
  const resources = new SpatialAnalysisResources(device, 'meuse-lime-model-form');
  const positionsBuffer = resources.createBuffer('positions', positions);
  const predictorsBuffer = resources.createBuffer('predictors', predictors);
  const responseBuffer = resources.createBuffer('lime-response', response);
  const mapPositionsBuffer = resources.createBuffer('map-positions', mapPositions, Buffer.VERTEX);
  const observedBuffer = resources.createBuffer('lime-observed-map', response, Buffer.VERTEX);
  const localFitBuffer = resources.createBuffer('local-logistic-fitted', rowCount * 4);
  const localCoefficientsBuffer = resources.createBuffer(
    'local-logistic-coefficients',
    rowCount * (PREDICTOR_COUNT + 1) * 4
  );
  const localStatusBuffer = resources.createBuffer('local-logistic-status', rowCount * 4);
  const localIterationsBuffer = resources.createBuffer('local-logistic-iterations', rowCount * 4);
  const localParametersBuffer = resources.createBuffer(
    'local-logistic-parameters',
    Float32Array.of(2000, 1e-4)
  );
  const probitCoefficientsBuffer = resources.createBuffer(
    'probit-coefficients',
    (PREDICTOR_COUNT + 1) * 4
  );
  const probitErrorsBuffer = resources.createBuffer(
    'probit-standard-errors',
    (PREDICTOR_COUNT + 1) * 4
  );
  const probitFittedBuffer = resources.createBuffer('probit-fitted', rowCount * 4);
  const probitEffectsBuffer = resources.createBuffer(
    'probit-marginal-effects',
    PREDICTOR_COUNT * 4
  );
  const probitSummaryBuffer = resources.createBuffer('probit-summary', 4 * 4);
  const probitStatusBuffer = resources.createBuffer('probit-status', 4);
  const selectionScoresBuffer = resources.createBuffer('bandwidth-scores', BANDWIDTHS.length * 4);
  const selectionBandwidthBuffer = resources.createBuffer('selected-bandwidth', 4);
  const selectionStatusBuffer = resources.createBuffer('bandwidth-selection-status', 4);

  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-lime-model-form'});
  const positionView = importGraphBuffer(
    graph,
    'positions',
    positionsBuffer,
    'float32x2',
    rowCount
  );
  const predictorView = importGraphBuffer(
    graph,
    'predictors',
    predictorsBuffer,
    'float32',
    rowCount * PREDICTOR_COUNT
  );
  const responseView = importGraphBuffer(
    graph,
    'lime-response',
    responseBuffer,
    'float32',
    rowCount
  );
  const fitView = importGraphBuffer(graph, 'local-fitted', localFitBuffer, 'float32', rowCount);
  const coefficientView = importGraphBuffer(
    graph,
    'local-coefficients',
    localCoefficientsBuffer,
    'float32',
    rowCount * (PREDICTOR_COUNT + 1)
  );
  const statusView = importGraphBuffer(
    graph,
    'local-status',
    localStatusBuffer,
    'uint32',
    rowCount
  );
  const iterationView = importGraphBuffer(
    graph,
    'local-iterations',
    localIterationsBuffer,
    'uint32',
    rowCount
  );
  const parameterView = importGraphBuffer(
    graph,
    'local-parameters',
    localParametersBuffer,
    'float32',
    2
  );
  const probitCoefficientView = importGraphBuffer(
    graph,
    'probit-coefficients',
    probitCoefficientsBuffer,
    'float32',
    PREDICTOR_COUNT + 1
  );
  const probitErrorsView = importGraphBuffer(
    graph,
    'probit-errors',
    probitErrorsBuffer,
    'float32',
    PREDICTOR_COUNT + 1
  );
  const probitFittedView = importGraphBuffer(
    graph,
    'probit-fitted',
    probitFittedBuffer,
    'float32',
    rowCount
  );
  const probitEffectsView = importGraphBuffer(
    graph,
    'probit-effects',
    probitEffectsBuffer,
    'float32',
    PREDICTOR_COUNT
  );
  const probitSummaryView = importGraphBuffer(
    graph,
    'probit-summary',
    probitSummaryBuffer,
    'float32',
    4
  );
  const probitStatusView = importGraphBuffer(
    graph,
    'probit-status',
    probitStatusBuffer,
    'uint32',
    1
  );
  const selectionScoresView = importGraphBuffer(
    graph,
    'selection-scores',
    selectionScoresBuffer,
    'float32',
    BANDWIDTHS.length
  );
  const selectionBandwidthView = importGraphBuffer(
    graph,
    'selection-bandwidth',
    selectionBandwidthBuffer,
    'float32',
    1
  );
  const selectionStatusView = importGraphBuffer(
    graph,
    'selection-status',
    selectionStatusBuffer,
    'uint32',
    1
  );
  graph.add(
    new GPUProbitRegression({
      id: 'meuse-lime-probit',
      predictors: predictorView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      output: {
        coefficients: probitCoefficientView,
        standardErrors: probitErrorsView,
        fitted: probitFittedView,
        marginalEffects: probitEffectsView,
        summary: probitSummaryView,
        status: probitStatusView
      }
    })
  );
  graph.add(
    new GPUGeneralizedGeographicallyWeightedRegression({
      id: 'meuse-lime-bandwidth-selection',
      family: 'logistic',
      positions: positionView,
      predictors: predictorView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      parameters: parameterView,
      kernel: 'bisquare',
      bandwidthMode: 'fixed',
      bandwidthSelection: {
        bandwidths: BANDWIDTHS,
        output: {
          scores: selectionScoresView,
          bandwidth: selectionBandwidthView,
          status: selectionStatusView
        }
      },
      output: {
        coefficients: coefficientView,
        fitted: fitView,
        status: {codes: statusView, iterationCount: iterationView}
      }
    })
  );
  const compiled = resources.track(await graph.compileAsync());
  let activeBandwidth: number | null = null;
  let latestFitted: Float32Array | null = null;
  let latestStatus: Uint32Array | null = null;
  let latestProbitStatus: number | null = null;
  let latestSelectionStatus: number | null = null;
  let graphDirty = true;
  const reader = createNamedReader(
    resources,
    'model-results',
    [
      {
        name: 'probitCoefficients',
        buffer: probitCoefficientsBuffer,
        bytes: (PREDICTOR_COUNT + 1) * 4
      },
      {name: 'probitErrors', buffer: probitErrorsBuffer, bytes: (PREDICTOR_COUNT + 1) * 4},
      {name: 'probitSummary', buffer: probitSummaryBuffer, bytes: 4 * 4},
      {name: 'probitStatus', buffer: probitStatusBuffer, bytes: 4},
      {name: 'selectionScores', buffer: selectionScoresBuffer, bytes: BANDWIDTHS.length * 4},
      {name: 'selectionBandwidth', buffer: selectionBandwidthBuffer, bytes: 4},
      {name: 'selectionStatus', buffer: selectionStatusBuffer, bytes: 4},
      {name: 'localFitted', buffer: localFitBuffer, bytes: rowCount * 4},
      {name: 'localStatus', buffer: localStatusBuffer, bytes: rowCount * 4}
    ],
    get => {
      const probitStatus = get('probitStatus').u32[0];
      const selectionStatus = get('selectionStatus').u32[0];
      latestProbitStatus = probitStatus;
      latestSelectionStatus = selectionStatus;
      const summary = get('probitSummary').f32;
      const coefficientText = describeProbit(
        get('probitCoefficients').f32,
        get('probitErrors').f32
      );
      ctx.setReadout(
        'probitStatus',
        probitStatus === 0
          ? `${summary[0].toFixed(0)} rows · log likelihood ${summary[1].toFixed(2)} · ${summary[3].toFixed(0)} Fisher-scoring iterations`
          : `Model status ${probitStatus} · check separation or design rank`
      );
      ctx.setReadout(
        'probitCoefficients',
        probitStatus === 0 ? coefficientText : 'Probit coefficients unavailable'
      );
      const scores = get('selectionScores').f32;
      activeBandwidth = get('selectionBandwidth').f32[0];
      if (selectionStatus === 0 && Number.isFinite(activeBandwidth)) {
        const unavailable = BANDWIDTHS.filter((_, index) => !Number.isFinite(scores[index]));
        ctx.setReadout(
          'selectionStatus',
          `${activeBandwidth} m fixed bisquare bandwidth selected by minimum LOO deviance; unavailable candidates: ${unavailable.length ? unavailable.map(value => `${value} m`).join(', ') : 'none'}; scores are selection criteria, not independent validation`
        );
        ctx.setChart('bandwidthScores', {
          kind: 'bars',
          title: 'Leave-one-out mean deviance by fixed bandwidth',
          labels: BANDWIDTHS.map((value, index) =>
            Number.isFinite(scores[index]) ? `${value} m` : `${value} m · unavailable`
          ),
          values: Array.from(scores),
          description:
            'Lower finite mean deviance is preferred among the five authored fixed-distance candidates. A candidate with an unavailable score had at least one failed or non-finite leave-one-out fit and is not eligible for selection; the winning score is a selection criterion, not independent validation.'
        });
      } else {
        activeBandwidth = null;
        ctx.setReadout('selectionStatus', `Bandwidth selection status ${selectionStatus}`);
      }
      const localFitted = get('localFitted').f32;
      const localStatus = get('localStatus').u32;
      latestFitted = localFitted;
      latestStatus = localStatus;
      let converged = 0;
      let failed = 0;
      let positive = 0;
      for (let row = 0; row < rowCount; row++) {
        if (localStatus[row] === 0 && Number.isFinite(localFitted[row])) {
          converged++;
          positive += localFitted[row];
        } else failed++;
      }
      if (selectionStatus !== 0 || activeBandwidth === null) {
        latestFitted = null;
        latestStatus = null;
      }
      ctx.setReadout(
        'localStatus',
        `${converged}/${rowCount} selected-bandwidth local fits converged · ${failed} failed or excluded · mean in-sample fitted probability ${converged ? (positive / converged).toFixed(3) : '—'}`
      );
      ctx.requestLayers();
      ctx.refreshTooltip();
    }
  );
  const limeCount = response.reduce((sum, value) => sum + value, 0);
  ctx.setReadout(
    'sampleCount',
    `${rowCount}/${sourceRowCount} complete-case sites · ${limeCount} lime-present · ${sourceRowCount - rowCount} omitted for missing organic matter`
  );
  ctx.setReadout('probitStatus', 'Global probit fit pending GPU run.');
  ctx.setReadout('selectionStatus', 'LOO bandwidth selection pending GPU run.');
  ctx.setReadout('localStatus', 'Local logistic fit pending GPU run.');
  ctx.setReadout('probitCoefficients', null);
  ctx.setChart('bandwidthScores', null);

  function makeTooltip(event: ScenePointerEvent) {
    if (!event.coordinate || !latestFitted || !latestStatus) return null;
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
    const source = response[row] ? 'lime present' : 'lime absent';
    const prediction = latestFitted?.[row];
    const status = latestStatus?.[row];
    const localText =
      status === 0 && prediction !== undefined && Number.isFinite(prediction)
        ? `Local logistic in-sample fitted probability: ${prediction.toFixed(3)} (bandwidth ${activeBandwidth} m)`
        : 'Local logistic probability unavailable at this site';
    return `${source}\n${localText}\nGlobal probit: ${latestProbitStatus === 0 ? 'converged' : 'pending or unavailable'}; bandwidth selector: ${latestSelectionStatus === 0 ? 'LOO deviance' : 'pending or unavailable'}`;
  }

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        reader.request(commandEncoder);
        graphDirty = false;
      } else {
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const localFitsReady =
        latestSelectionStatus === 0 &&
        activeBandwidth !== null &&
        latestStatus !== null &&
        latestStatus.every(status => status === 0) &&
        latestFitted !== null &&
        latestFitted.every(Number.isFinite);
      if (ctx.options.display === 'local-probability' && !localFitsReady) return [];
      const values = ctx.options.display === 'local-probability' ? localFitBuffer : observedBuffer;
      return [
        new SpatialAnalysisPointLayer({
          id: 'meuse-lime-model-points',
          coordinateOrigin: [origin[0], origin[1], 0],
          positions: mapPositionsBuffer,
          values,
          valueFormat: 'float32',
          valueRange: MAP_VALUE_RANGES[ctx.options.display],
          colormap: ctx.options.display === 'local-probability' ? 'cividis' : 'diverging',
          instanceCount: rowCount,
          radiusPixels: 6,
          color: [255, 255, 255, 255],
          outlineColor: [16, 18, 22, 230]
        })
      ];
    },
    setOption(id) {
      if (id === 'display') {
        ctx.refreshLegends();
        ctx.requestLayers();
      }
    },
    getTooltip: makeTooltip,
    destroy() {
      reader.release();
      resources.destroy();
    }
  };
}
