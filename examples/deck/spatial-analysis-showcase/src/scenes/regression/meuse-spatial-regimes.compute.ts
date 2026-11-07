// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPURestrictedRegimeRegression,
  GPUSpatialRegimeRegression
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {createNamedReader} from './b6-reader';
import {toRadians} from '@math.gl/core';

export type MeuseSpatialRegimesOptions = {
  display: 'lime-group' | 'observed' | 'fitted' | 'residual';
  profile: 'pooled' | 'varying-intercept' | 'intercept-distance' | 'fully-varying';
};

const PREDICTOR_COUNT = 3;
const REGIME_COUNT = 2;
const PROFILE_LABELS = [
  'Pooled: all coefficients common',
  'Varying intercept',
  'Varying intercept and river distance',
  'Fully varying by measured lime'
] as const;
const PROFILE_INDEX: Record<MeuseSpatialRegimesOptions['profile'], number> = {
  pooled: 0,
  'varying-intercept': 1,
  'intercept-distance': 2,
  'fully-varying': 3
};
const COMMON_PROFILES = [
  [true, true, true, true],
  [false, true, true, true],
  [false, true, true, false],
  [false, false, false, false]
] as const;

function getDisplayIndex(display: MeuseSpatialRegimesOptions['display']): number {
  switch (display) {
    case 'lime-group':
      return 3;
    case 'observed':
      return 0;
    case 'fitted':
      return 1;
    case 'residual':
      return 2;
  }
}

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

type ProfileBuffers = {
  coefficientCount: number;
  coefficients: Buffer;
  covariance: Buffer;
  fitted: Buffer;
  chow: Buffer;
  stability: Buffer;
  status: Buffer;
};

/** Fits nested OLS restrictions against the measured Meuse lime field as a two-level regime. */
export async function createMeuseSpatialRegimes(
  ctx: SceneContext<MeuseSpatialRegimesOptions>
): Promise<SceneInstance<MeuseSpatialRegimesOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const sourceRowCount = dataset.column<Float32Array>('sampleDistanceMeters').length;
  const sourceResponse = dataset.column<Float32Array>('logZinc');
  const sourceElevation = dataset.column<Float32Array>('sampleElevation');
  const sourceOrganicMatter = dataset.column<Float32Array>('sampleOrganicMatter');
  const sourceDistance = dataset.column<Float32Array>('sampleDistanceMeters');
  const sourceRegimes = dataset.column<Uint8Array>('sampleLime');
  const studyRows: number[] = [];
  for (let row = 0; row < sourceRowCount; row++) {
    if (
      Number.isFinite(sourceResponse[row]) &&
      Number.isFinite(sourceElevation[row]) &&
      Number.isFinite(sourceOrganicMatter[row]) &&
      Number.isFinite(sourceDistance[row]) &&
      sourceRegimes[row] <= 1
    ) {
      studyRows.push(row);
    }
  }
  const rowCount = studyRows.length;
  const response = Float32Array.from(studyRows, row => sourceResponse[row]);
  const regimes = Uint32Array.from(studyRows, row => sourceRegimes[row]);
  const predictors = makePredictorTable(
    [sourceElevation, sourceOrganicMatter, sourceDistance].map(column =>
      Float32Array.from(studyRows, row => column[row])
    ),
    rowCount
  );
  const groupCounts = [0, 0];
  for (const regime of regimes) groupCounts[regime]++;

  const samplePositions = dataset.projectColumn('samplePosition', dataset.defaultOrigin);
  const mapPositions = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) {
    const sourceRow = studyRows[row];
    mapPositions[row * 2] = samplePositions[sourceRow * 2];
    mapPositions[row * 2 + 1] = samplePositions[sourceRow * 2 + 1];
  }

  const resources = new SpatialAnalysisResources(device, 'meuse-spatial-regimes');
  const predictorsBuffer = resources.createBuffer('regime-predictors', predictors);
  const responseBuffer = resources.createBuffer('log-zinc-response', response);
  const regimesBuffer = resources.createBuffer('measured-lime-regimes', regimes);
  const mapPositionsBuffer = resources.createBuffer('map-positions', mapPositions, Buffer.VERTEX);
  const mapValuesBuffer = resources.createBuffer('mapped-log-zinc', rowCount * 4, Buffer.VERTEX);
  const mapControlBuffer = resources.createBuffer(
    'map-control',
    Uint32Array.of(getDisplayIndex(ctx.options.display), 0)
  );
  const generationInputBuffer = resources.createBuffer('generation-input', Uint32Array.of(0));
  const generationOutputBuffer = resources.createBuffer('generation-output', 4);
  const profiles = COMMON_PROFILES.map((common, index) => {
    const coefficientCount = common.reduce(
      (total, isCommon) => total + (isCommon ? 1 : REGIME_COUNT),
      0
    );
    return {
      coefficientCount,
      coefficients: resources.createBuffer(`profile-${index}-coefficients`, coefficientCount * 4),
      covariance: resources.createBuffer(
        `profile-${index}-covariance`,
        coefficientCount * coefficientCount * 4
      ),
      fitted: resources.createBuffer(`profile-${index}-fitted`, rowCount * 4),
      chow: resources.createBuffer(`profile-${index}-chow`, 3 * 4),
      stability: resources.createBuffer(
        `profile-${index}-stability`,
        (PREDICTOR_COUNT + 2) * 3 * 4
      ),
      status: resources.createBuffer(`profile-${index}-status`, 4)
    } satisfies ProfileBuffers;
  });
  const directCoefficientsBuffer = resources.createBuffer(
    'fully-varying-direct-coefficients',
    REGIME_COUNT * (PREDICTOR_COUNT + 1) * 4
  );
  const directFittedBuffer = resources.createBuffer('fully-varying-direct-fitted', rowCount * 4);
  const directStatusBuffer = resources.createBuffer(
    'fully-varying-direct-status',
    REGIME_COUNT * 4
  );
  const directCountsBuffer = resources.createBuffer(
    'fully-varying-direct-counts',
    REGIME_COUNT * 4
  );

  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-spatial-regimes'});
  const predictorsView = importGraphBuffer(
    graph,
    'predictors',
    predictorsBuffer,
    'float32',
    rowCount * PREDICTOR_COUNT
  );
  const responseView = importGraphBuffer(graph, 'response', responseBuffer, 'float32', rowCount);
  const regimesView = importGraphBuffer(graph, 'regimes', regimesBuffer, 'uint32', rowCount);
  const profileViews = profiles.map((profile, index) => ({
    coefficients: importGraphBuffer(
      graph,
      `profile-${index}-coefficients`,
      profile.coefficients,
      'float32',
      profile.coefficientCount
    ),
    covariance: importGraphBuffer(
      graph,
      `profile-${index}-covariance`,
      profile.covariance,
      'float32',
      profile.coefficientCount ** 2
    ),
    fitted: importGraphBuffer(
      graph,
      `profile-${index}-fitted`,
      profile.fitted,
      'float32',
      rowCount
    ),
    chow: importGraphBuffer(graph, `profile-${index}-chow`, profile.chow, 'float32', 3),
    stability: importGraphBuffer(
      graph,
      `profile-${index}-stability`,
      profile.stability,
      'float32',
      (PREDICTOR_COUNT + 2) * 3
    ),
    status: importGraphBuffer(graph, `profile-${index}-status`, profile.status, 'uint32', 1)
  }));
  for (let index = 0; index < COMMON_PROFILES.length; index++) {
    const common = COMMON_PROFILES[index];
    graph.add(
      new GPURestrictedRegimeRegression({
        id: `meuse-restricted-profile-${index}`,
        predictors: predictorsView,
        predictorCount: PREDICTOR_COUNT,
        response: responseView,
        regimes: regimesView,
        regimeCount: REGIME_COUNT,
        common,
        output: {
          coefficients: profileViews[index].coefficients,
          covariance: profileViews[index].covariance,
          fitted: profileViews[index].fitted,
          chow: profileViews[index].chow,
          stability: profileViews[index].stability,
          status: profileViews[index].status
        }
      })
    );
  }
  graph.add(
    new GPUSpatialRegimeRegression({
      id: 'meuse-fully-varying-direct-check',
      predictors: predictorsView,
      predictorCount: PREDICTOR_COUNT,
      response: responseView,
      regimes: regimesView,
      regimeCount: REGIME_COUNT,
      output: {
        coefficients: importGraphBuffer(
          graph,
          'direct-coefficients',
          directCoefficientsBuffer,
          'float32',
          REGIME_COUNT * (PREDICTOR_COUNT + 1)
        ),
        fitted: importGraphBuffer(graph, 'direct-fitted', directFittedBuffer, 'float32', rowCount),
        status: importGraphBuffer(
          graph,
          'direct-status',
          directStatusBuffer,
          'uint32',
          REGIME_COUNT
        ),
        counts: importGraphBuffer(
          graph,
          'direct-counts',
          directCountsBuffer,
          'uint32',
          REGIME_COUNT
        )
      }
    })
  );
  const mapControlView = importGraphBuffer(graph, 'map-control', mapControlBuffer, 'uint32', 2);
  const mapValuesView = importGraphBuffer(
    graph,
    'mapped-values',
    mapValuesBuffer,
    'float32',
    rowCount
  );
  addKernelPass(graph, {
    id: 'meuse-regime-map-clear',
    invocationCount: rowCount,
    bindings: [{name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}],
    body: 'mapped[mappedOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));'
  });
  for (let profileIndex = 0; profileIndex < profiles.length; profileIndex++) {
    addKernelPass(graph, {
      id: `meuse-regime-map-profile-${profileIndex}`,
      invocationCount: rowCount,
      bindings: [
        {name: 'control', view: mapControlView, type: 'u32', access: 'read'},
        {name: 'response', view: responseView, type: 'f32', access: 'read'},
        {name: 'regimes', view: regimesView, type: 'u32', access: 'read'},
        {
          name: 'fitted',
          view: profileViews[profileIndex].fitted,
          type: 'f32',
          access: 'read'
        },
        {
          name: 'status',
          view: profileViews[profileIndex].status,
          type: 'u32',
          access: 'read'
        },
        {name: 'mapped', view: mapValuesView, type: 'f32', access: 'read_write'}
      ],
      body: `if (control[controlOffset + 1u] == ${profileIndex}u) { var value = response[responseOffset + index]; if (control[controlOffset] == 3u) { value = f32(regimes[regimesOffset + index]); } else if (control[controlOffset] != 0u) { value = bitcast<f32>(0x7fc00000u | (index & 0u)); if (status[statusOffset] == 0u) { value = fitted[fittedOffset + index]; if (control[controlOffset] == 2u) { value = response[responseOffset + index] - value; } } } mapped[mappedOffset + index] = value; }`
    });
  }
  const generationInputView = importGraphBuffer(
    graph,
    'generation-input',
    generationInputBuffer,
    'uint32',
    1
  );
  const generationOutputView = importGraphBuffer(
    graph,
    'generation-output',
    generationOutputBuffer,
    'uint32',
    1
  );
  addKernelPass(graph, {
    id: 'meuse-regime-generation-snapshot',
    invocationCount: 1,
    bindings: [
      {name: 'generation', view: generationInputView, type: 'u32', access: 'read'},
      {name: 'snapshot', view: generationOutputView, type: 'u32', access: 'read_write'}
    ],
    body: 'snapshot[snapshotOffset] = generation[generationOffset];'
  });

  const compiled = resources.track(await graph.compileAsync());
  let currentGeneration = 0;
  let graphDirty = true;
  let latestMapValues: Float32Array | null = null;
  const readerSources = [
    {name: 'mapValues', buffer: mapValuesBuffer, bytes: rowCount * 4},
    {name: 'generation', buffer: generationOutputBuffer, bytes: 4},
    {
      name: 'directCoefficients',
      buffer: directCoefficientsBuffer,
      bytes: REGIME_COUNT * (PREDICTOR_COUNT + 1) * 4
    },
    {name: 'directFitted', buffer: directFittedBuffer, bytes: rowCount * 4},
    {name: 'directStatus', buffer: directStatusBuffer, bytes: REGIME_COUNT * 4},
    {name: 'directCounts', buffer: directCountsBuffer, bytes: REGIME_COUNT * 4},
    ...profiles.flatMap((profile, index) => [
      {
        name: `coefficients-${index}`,
        buffer: profile.coefficients,
        bytes: profile.coefficientCount * 4
      },
      {
        name: `covariance-${index}`,
        buffer: profile.covariance,
        bytes: profile.coefficientCount ** 2 * 4
      },
      {name: `fitted-${index}`, buffer: profile.fitted, bytes: rowCount * 4},
      {name: `chow-${index}`, buffer: profile.chow, bytes: 3 * 4},
      {name: `stability-${index}`, buffer: profile.stability, bytes: (PREDICTOR_COUNT + 2) * 3 * 4},
      {name: `status-${index}`, buffer: profile.status, bytes: 4}
    ])
  ];
  const reader = createNamedReader(resources, 'regime-model-results', readerSources, get => {
    if (get('generation').u32[0] !== currentGeneration) return;
    latestMapValues = get('mapValues').f32;
    const profileMetrics: {rmse: number; n: number; status: number}[] = [];
    const diagnostics: string[] = [];
    for (let index = 0; index < profiles.length; index++) {
      const status = get(`status-${index}`).u32[0];
      const fitted = get(`fitted-${index}`).f32;
      let squaredError = 0;
      let fittedCount = 0;
      for (let row = 0; row < rowCount; row++) {
        if (status === 0 && Number.isFinite(fitted[row])) {
          const residual = response[row] - fitted[row];
          squaredError += residual * residual;
          fittedCount++;
        }
      }
      const rmse = fittedCount ? Math.sqrt(squaredError / fittedCount) : Number.NaN;
      profileMetrics.push({rmse, n: fittedCount, status});
      const common = COMMON_PROFILES[index];
      const coefficients = get(`coefficients-${index}`).f32;
      const covariance = get(`covariance-${index}`).f32;
      const chow = get(`chow-${index}`).f32;
      const stability = get(`stability-${index}`).f32;
      const offsets: number[] = [];
      let next = 0;
      for (const isCommon of common) {
        offsets.push(next);
        next += isCommon ? 1 : REGIME_COUNT;
      }
      const coefficientLabels = ['intercept', 'elevation', 'organic matter', 'river distance'];
      const coefficientText =
        status !== 0
          ? `status ${status} · covariance / coefficients unavailable`
          : coefficientLabels
              .map((label, column) => {
                const offset = offsets[column];
                const terms = common[column]
                  ? [coefficients[offset]]
                  : [coefficients[offset], coefficients[offset + 1]];
                const standardErrors = terms.map((_, regime) => {
                  const diagonalIndex = common[column] ? offset : offset + regime;
                  const variance =
                    covariance[diagonalIndex * profiles[index].coefficientCount + diagonalIndex];
                  return Number.isFinite(variance) && variance >= 0
                    ? Math.sqrt(variance).toFixed(3)
                    : '—';
                });
                return `${label} ${terms.map((value, regime) => `${value.toFixed(3)} (SE ${standardErrors[regime]})`).join(' / ')}`;
              })
              .join(' · ');
      const stabilityTerms =
        status === 0
          ? coefficientLabels
              .map((label, column) => {
                const degrees = stability[column * 3 + 1];
                const pValue = stability[column * 3 + 2];
                return degrees > 0 && Number.isFinite(pValue)
                  ? `${label} p=${pValue.toFixed(3)}`
                  : null;
              })
              .filter((value): value is string => value !== null)
          : [];
      const jointP = status === 0 ? stability[(PREDICTOR_COUNT + 1) * 3 + 2] : Number.NaN;
      const chowText =
        status === 0 && Number.isFinite(chow[0])
          ? `Chow F ${chow[0].toFixed(3)} (${chow[1].toFixed(0)}, ${chow[2].toFixed(0)} df)`
          : status === 0
            ? 'Chow F unavailable/no restrictions'
            : `Chow/Wald unavailable: model status ${status}`;
      diagnostics.push(
        `${PROFILE_LABELS[index]}: ${coefficientText}; ${chowText}; ${stabilityTerms.join(', ') || 'no varying-column Wald tests'}; joint p ${Number.isFinite(jointP) ? jointP.toFixed(3) : '—'}`
      );
    }
    ctx.setReadout('profileDiagnostics', diagnostics.join('\n'));
    ctx.setReadout(
      'profileStatus',
      profileMetrics
        .map(
          (metric, index) =>
            `${PROFILE_LABELS[index]}: status ${metric.status}, ${metric.n}/${rowCount} fitted`
        )
        .join(' · ')
    );
    const validMetrics = profileMetrics
      .map((metric, index) => ({metric, index}))
      .filter(value => value.metric.n > 0);
    ctx.setChart(
      'profileRmse',
      validMetrics.length
        ? {
            kind: 'bars',
            title: 'In-sample log-zinc RMSE by coefficient restrictions',
            labels: validMetrics.map(value => PROFILE_LABELS[value.index]),
            values: validMetrics.map(value => value.metric.rmse),
            description:
              'Descriptive in-sample fit on the same 153 complete observations. These scores do not adjust for the different parameter counts and are not spatial validation.'
          }
        : null
    );
    const directStatus = get('directStatus').u32;
    const directCoefficients = get('directCoefficients').f32;
    const directFitted = get('directFitted').f32;
    const fullyVaryingMetrics = profileMetrics[3];
    let maximumCoefficientDifference = Number.NaN;
    let maximumFittedDifference = Number.NaN;
    if (directStatus[0] === 0 && directStatus[1] === 0 && fullyVaryingMetrics.status === 0) {
      const restricted = get('coefficients-3').f32;
      maximumCoefficientDifference = 0;
      for (let column = 0; column < PREDICTOR_COUNT + 1; column++) {
        for (let regime = 0; regime < REGIME_COUNT; regime++) {
          const directIndex = regime * (PREDICTOR_COUNT + 1) + column;
          const restrictedIndex = column * REGIME_COUNT + regime;
          maximumCoefficientDifference = Math.max(
            maximumCoefficientDifference,
            Math.abs(directCoefficients[directIndex] - restricted[restrictedIndex])
          );
        }
      }
      const restrictedFitted = get('fitted-3').f32;
      maximumFittedDifference = 0;
      for (let row = 0; row < rowCount; row++) {
        maximumFittedDifference = Math.max(
          maximumFittedDifference,
          Math.abs(directFitted[row] - restrictedFitted[row])
        );
      }
    }
    const directCounts = get('directCounts').u32;
    ctx.setReadout(
      'directCheck',
      `Direct spatial-regime OLS status ${Array.from(directStatus).join('/')}; GPU group counts ${Array.from(directCounts).join('/')}; max coefficient / fitted-value difference from fully varying restricted fit ${Number.isFinite(maximumCoefficientDifference) ? maximumCoefficientDifference.toExponential(2) : 'unavailable'} / ${Number.isFinite(maximumFittedDifference) ? maximumFittedDifference.toExponential(2) : 'unavailable'}`
    );
    ctx.setReadout(
      'sampleCount',
      `${rowCount} common complete rows · lime 0: ${groupCounts[0]} · lime 1: ${groupCounts[1]}`
    );
    ctx.setReadout(
      'mapStatus',
      `Mapped ${ctx.options.display} values for ${PROFILE_LABELS[PROFILE_INDEX[ctx.options.profile]]}; fixed legend ranges shown; selected fit status ${profileMetrics[PROFILE_INDEX[ctx.options.profile]].status}.`
    );
    ctx.requestLayers();
  });

  const writeMapControl = () => {
    mapControlBuffer.write(
      Uint32Array.of(
        getDisplayIndex(ctx.options.display),
        ['pooled', 'varying-intercept', 'intercept-distance', 'fully-varying'].indexOf(
          ctx.options.profile
        )
      )
    );
    currentGeneration++;
    generationInputBuffer.write(Uint32Array.of(currentGeneration));
    graphDirty = true;
    latestMapValues = null;
    ctx.setReadout('profileStatus', 'All profile and direct-fit statuses pending GPU execution.');
    ctx.setReadout(
      'profileDiagnostics',
      'Coefficients, covariance and nested-model statistics pending GPU execution.'
    );
    ctx.setReadout(
      'directCheck',
      'Direct fully varying regime OLS comparison pending GPU execution.'
    );
    ctx.setReadout('mapStatus', 'Selected map and model summary pending GPU execution.');
    ctx.setChart('profileRmse', null);
    ctx.requestLayers();
  };
  ctx.setReadout(
    'sampleCount',
    `${rowCount} complete rows · lime 0: ${groupCounts[0]} · lime 1: ${groupCounts[1]}`
  );
  ctx.setReadout(
    'profileStatus',
    'Four restricted fits and a direct fully varying fit pending GPU execution.'
  );
  ctx.setReadout(
    'profileDiagnostics',
    'Coefficients, covariance and nested-model statistics pending GPU execution.'
  );
  ctx.setReadout(
    'directCheck',
    'Direct fully varying regime OLS comparison pending GPU execution.'
  );
  ctx.setReadout('mapStatus', 'Observed log-zinc values shown while model fits are pending.');
  ctx.setChart('profileRmse', null);

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (graphDirty || frame.frameIndex < 2) {
        compiled.encode(commandEncoder, {parameters: undefined});
        graphDirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },
    getLayers: () => {
      if (!latestMapValues) return [];
      return [
        new SpatialAnalysisPointLayer({
          id: 'meuse-spatial-regime-points',
          coordinateOrigin: [dataset.defaultOrigin[0], dataset.defaultOrigin[1], 0],
          positions: mapPositionsBuffer,
          // Category encoding reads integer category IDs from the measured lime field.
          // Keep the separate float32 map buffer for fitted, observed, and residual views.
          values: ctx.options.display === 'lime-group' ? regimesBuffer : mapValuesBuffer,
          valueFormat: ctx.options.display === 'lime-group' ? 'uint32' : 'float32',
          valueRange:
            ctx.options.display === 'residual'
              ? [-1.5, 1.5]
              : ctx.options.display === 'lime-group'
                ? [0, 1]
                : [4.7, 7.6],
          colormap:
            ctx.options.display === 'lime-group'
              ? 'category'
              : ctx.options.display === 'residual'
                ? 'diverging'
                : 'cividis',
          ...(ctx.options.display === 'lime-group'
            ? {
                palette: [
                  [37, 99, 235, 255],
                  [220, 38, 38, 255]
                ] as const
              }
            : {}),
          instanceCount: rowCount,
          radiusPixels: 7,
          color: [255, 255, 255, 255],
          outlineColor: [18, 23, 28, 220]
        })
      ];
    },
    setOption() {
      writeMapControl();
    },
    getTooltip: event => {
      if (!event.coordinate || !latestMapValues) return null;
      const [longitude, latitude] = event.coordinate;
      let closestRow = -1;
      let closestDistance = Infinity;
      const mapCoordinates = dataset.column<Float32Array>('samplePosition');
      for (let row = 0; row < rowCount; row++) {
        const sourceRow = studyRows[row];
        const deltaLongitude =
          (mapCoordinates[sourceRow * 2] - longitude) * Math.cos(toRadians(latitude)) * 111320;
        const deltaLatitude = (mapCoordinates[sourceRow * 2 + 1] - latitude) * 111320;
        const distance = Math.hypot(deltaLongitude, deltaLatitude);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestRow = row;
        }
      }
      if (closestRow < 0 || closestDistance > 60) return null;
      const value = latestMapValues[closestRow];
      const displayValue =
        ctx.options.display === 'lime-group'
          ? `measured lime ${regimes[closestRow]}`
          : `${Number.isFinite(value) ? value.toFixed(3) : 'unavailable'} log units`;
      return `Nearest sample within ${closestDistance.toFixed(0)} m · ${PROFILE_LABELS[PROFILE_INDEX[ctx.options.profile]]} · ${ctx.options.display}: ${displayValue} · observed log zinc ${response[closestRow].toFixed(3)} · measured lime ${regimes[closestRow]}`;
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
