// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GPUMarkedPointStatistics} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPUInhomogeneousPointPattern,
  GPUStudyWindowPreparation,
  GPU_STUDY_WINDOW_FLAG_EMPTY,
  GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData, LineChartData} from '../chart-types';
import type {SceneContext, SceneInstance, ScenePointerEvent, TooltipContent} from '../scene';
import {
  EARTHQUAKE_PAIR_BOUNDS,
  EARTHQUAKE_PAIR_MAXIMUM_RADIUS_METERS,
  EARTHQUAKE_PAIR_RADIUS_COUNT,
  buildEarthquakePairData,
  type EarthquakePairEvent
} from './earthquake-pair-structure-data';
import {clamp} from '@math.gl/core';

const DEPTH_COLORS: readonly (readonly [number, number, number, number])[] = [
  [30, 84, 118, 255],
  [45, 144, 151, 255],
  [204, 171, 103, 255],
  [171, 83, 78, 255]
];
const INTENSITY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [199, 215, 201, 255],
  [61, 151, 145, 255],
  [211, 128, 76, 255]
];
const PAIR_TYPE_OPTIONS = [
  'all',
  'shallow-to-deep',
  'deep-to-shallow',
  'shallow-to-shallow'
] as const;
const INTENSITY_BANDWIDTHS = [75_000, 150_000, 225_000, 300_000] as const;
const PAIR_CORRELATION_BANDWIDTHS = [10_000, 25_000] as const;
const PAIR_CORRELATION_ESTIMATORS = ['annular', 'epanechnikov'] as const;
const PAIR_CORRELATION_DIVISORS = ['radius', 'distance'] as const;
const ZERO_CORRECTIONS = ['none', 'reflection'] as const;
const MARKED_RADIUS_VALUES = Float32Array.from(
  {length: EARTHQUAKE_PAIR_RADIUS_COUNT},
  (_, index) => ((index + 1) * EARTHQUAKE_PAIR_MAXIMUM_RADIUS_METERS) / EARTHQUAKE_PAIR_RADIUS_COUNT
);
const STATUS_LABELS: Record<number, string> = {
  0: 'valid',
  1: 'empty edge-corrected support or category',
  2: 'invalid input or nonfinite accumulation',
  3: 'overlap work exhausted',
  4: 'overlap unresolved'
};

export type EarthquakePairStructureOptions = {
  mode: 'marked' | 'inhomogeneous';
  depthPair: (typeof PAIR_TYPE_OPTIONS)[number];
  excludeExactTen: boolean;
  markedCorrection: 'none' | 'translation';
  inhomogeneousCorrection: 'border' | 'translation';
  intensityModel: 'constant' | 'epanechnikov';
  intensityBandwidth: (typeof INTENSITY_BANDWIDTHS)[number];
  selfContribution: 'include' | 'exclude';
  pairCorrelationEstimator: (typeof PAIR_CORRELATION_ESTIMATORS)[number];
  pairCorrelationBandwidth: (typeof PAIR_CORRELATION_BANDWIDTHS)[number];
  pairCorrelationDivisor: (typeof PAIR_CORRELATION_DIVISORS)[number];
  zeroCorrection: (typeof ZERO_CORRECTIONS)[number];
};

type MarkedView = {
  mode: 'marked';
  radii: Float32Array;
  statistics: Float32Array;
  pairCounts: Uint32Array;
  status: Uint32Array;
  summary: string;
};
type InhomogeneousView = {
  mode: 'inhomogeneous';
  radii: Float32Array;
  intensities: Float32Array;
  k: Float32Array;
  lMinusR: Float32Array;
  pairCorrelation: Float32Array;
  status: Uint32Array;
  windowSummary: Float32Array;
  summary: string;
  intensityRange: readonly [number, number] | null;
};
type PatternView = MarkedView | InhomogeneousView;
type Variant = {
  key: string;
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  dirty: boolean;
  result: PatternView;
};
type ReadbackSource = {buffer: Buffer; size: number};

function interpolateColor(
  value: number,
  minimum: number,
  maximum: number,
  colors: readonly (readonly [number, number, number, number])[]
): [number, number, number, number] {
  if (!Number.isFinite(value) || colors.length < 2) return [121, 122, 116, 255];
  const amount = maximum > minimum ? clamp((value - minimum) / (maximum - minimum), 0, 1) : 0;
  const scaled = amount * (colors.length - 1);
  const lower = Math.min(colors.length - 2, Math.floor(scaled));
  const fraction = scaled - lower;
  const lowerColor = colors[lower];
  const upperColor = colors[lower + 1];
  return [
    Math.round(lowerColor[0] + (upperColor[0] - lowerColor[0]) * fraction),
    Math.round(lowerColor[1] + (upperColor[1] - lowerColor[1]) * fraction),
    Math.round(lowerColor[2] + (upperColor[2] - lowerColor[2]) * fraction),
    Math.round(lowerColor[3] + (upperColor[3] - lowerColor[3]) * fraction)
  ];
}

function quantileRange(values: Float32Array): readonly [number, number] | null {
  const finite = Array.from(values)
    .filter(value => Number.isFinite(value) && value > 0)
    .sort((first, second) => first - second);
  if (finite.length !== values.length) return null;
  return [
    finite[Math.floor((finite.length - 1) * 0.05)],
    finite[Math.floor((finite.length - 1) * 0.95)]
  ];
}

function makeLineChart(
  title: string,
  values: ArrayLike<number>,
  radii: Float32Array,
  yLabel: string,
  description: string,
  color = 4
): LineChartData {
  return {
    kind: 'line',
    title,
    xLabel: 'Search radius (km)',
    yLabel,
    formatX: radius => radius.toFixed(0),
    formatY: value => (Number.isFinite(value) ? value.toPrecision(3) : 'undefined'),
    description,
    series: [{label: title, x: radii.map(radius => radius / 1000), y: values, color, width: 2.4}]
  };
}

function makePairCountChart(counts: Uint32Array): BarChartData {
  return {
    kind: 'bars',
    title: 'Ordered non-self pairs within radius',
    labels: Array.from(MARKED_RADIUS_VALUES, radius => `${radius / 1000} km`),
    values: counts,
    color: 4,
    description:
      'Cumulative directed non-self pair counts read from the GPU output; these are counts, not independent events.'
  };
}

function decodeFloats(bytes: ArrayBuffer, offset: number, count: number): Float32Array {
  return new Float32Array(bytes.slice(offset, offset + count * 4));
}
function decodeWords(bytes: ArrayBuffer, offset: number, count: number): Uint32Array {
  return new Uint32Array(bytes.slice(offset, offset + count * 4));
}

function readStatus(status: Uint32Array): string {
  return Array.from(
    status,
    (code, index) => `r${index + 1}: ${STATUS_LABELS[code] ?? `unknown status ${code}`}`
  ).join('; ');
}

function describeWindowFlags(flags: number): string {
  const labels = [
    ...(flags & GPU_STUDY_WINDOW_FLAG_EMPTY ? ['empty support'] : []),
    ...(flags & GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN
      ? ['uncertain polygon orientation']
      : [])
  ];
  return labels.length > 0 ? labels.join(', ') : `unrecognized flag value ${flags}`;
}

function getSelectedPair(
  value: EarthquakePairStructureOptions['depthPair']
): readonly [number, number] | undefined {
  switch (value) {
    case 'all':
      return undefined;
    case 'shallow-to-deep':
      return [0, 1];
    case 'deep-to-shallow':
      return [1, 0];
    case 'shallow-to-shallow':
      return [0, 0];
  }
}

function makeWindowPath(
  data: ReturnType<typeof buildEarthquakePairData>
): [number, number, number, number][] {
  const path: [number, number, number, number][] = [];
  for (let index = 0; index < data.rectanglePath.length - 1; index++) {
    const first = data.rectanglePath[index];
    const second = data.rectanglePath[index + 1];
    path.push([first[0], first[1], second[0], second[1]]);
  }
  return path;
}

export async function createEarthquakePairStructure(
  ctx: SceneContext<EarthquakePairStructureOptions>
): Promise<SceneInstance<EarthquakePairStructureOptions>> {
  const dataset = ctx.datasets.get('poopdeck-earthquakes');
  const data = buildEarthquakePairData(dataset);
  const resources = new SpatialAnalysisResources(ctx.device, 'earthquake-pair-structure-input');
  const positionBuffer = resources.createBuffer('all-selected-positions', data.positions);
  const markBuffer = resources.createBuffer('reported-depth-km', data.depths);
  const depthGroupBuffer = resources.createBuffer('depth-category', data.depthGroups);
  const allMaskBuffer = resources.createBuffer('all-rows-mask', data.allEventsMask);
  const exactMaskBuffer = resources.createBuffer(
    'exclude-exact-ten-km-mask',
    data.excludingTenKilometersMask
  );
  const constantIntensity = data.events.length / (data.rectangleAreaSquareKilometers * 1_000_000);
  const constantIntensityBuffer = resources.createBuffer(
    'constant-baseline-intensity',
    Float32Array.from({length: data.events.length}, () => constantIntensity)
  );
  const windowPositions = Float32Array.of(
    EARTHQUAKE_PAIR_BOUNDS.minimumX,
    EARTHQUAKE_PAIR_BOUNDS.minimumY,
    EARTHQUAKE_PAIR_BOUNDS.maximumX,
    EARTHQUAKE_PAIR_BOUNDS.minimumY,
    EARTHQUAKE_PAIR_BOUNDS.maximumX,
    EARTHQUAKE_PAIR_BOUNDS.maximumY,
    EARTHQUAKE_PAIR_BOUNDS.minimumX,
    EARTHQUAKE_PAIR_BOUNDS.maximumY,
    EARTHQUAKE_PAIR_BOUNDS.minimumX,
    EARTHQUAKE_PAIR_BOUNDS.minimumY
  );
  const windowPositionBuffer = resources.createBuffer('fixed-aeqd-window', windowPositions);
  const windowRingBuffer = resources.createBuffer('window-ring-offsets', Uint32Array.of(0, 5));
  const windowPolygonBuffer = resources.createBuffer(
    'window-polygon-offsets',
    Uint32Array.of(0, 1)
  );
  const windowFeatureBuffer = resources.createBuffer(
    'window-feature-offsets',
    Uint32Array.of(0, 1)
  );
  const cache = new Map<string, Variant>();
  const graphSwap = resources.track(new GraphSwap());
  let active: Variant | null = null;
  let revision = 0;
  let destroyed = false;
  let selectedEvent: EarthquakePairEvent | null = null;
  let displayEvents = data.events.map(event => ({
    ...event,
    intensityColor: null as [number, number, number, number] | null,
    intensityRadius: 3.2
  }));

  const areaText = `${Math.round(data.rectangleAreaSquareKilometers).toLocaleString()} km² (projected rectangle)`;
  ctx.setReadout('selectedDepth', 'Select an epicentre to inspect reported depth.');
  ctx.setReadout(
    'archiveCount',
    `${formatCount(data.sourceCount)} archived rows; type-0 earthquake filter applied`
  );
  ctx.setReadout(
    'windowSupport',
    `${formatCount(data.events.length)} earthquakes inside the fixed rectangle; ${areaText}`
  );
  ctx.setLegendExtent('reported-depth', data.depthExtent);
  ctx.setReadout(
    'depthSplit',
    `${formatCount(data.shallowCount)} below 70 km; ${formatCount(data.deepCount)} at least 70 km`
  );
  ctx.setReadout(
    'exactTenCount',
    `${formatCount(data.exactTenKilometerCount)} rows at exactly 10 km; ${formatCount(data.duplicateCoordinatePairs)} coincident coordinate pairs`
  );
  ctx.setReadout('pairStatus', 'Compile a marked pair graph to see GPU status.');
  ctx.setReadout('pairTerminal', 'GPU terminal-radius values pending.');
  ctx.setReadout('pairSummary', 'pending GPU output');
  ctx.setReadout('inhomStatus', 'Compile an intensity graph to see GPU status.');
  ctx.setReadout('inhomSummary', 'pending GPU output');
  ctx.setReadout('intensitySummary', 'pending GPU output');
  ctx.setReadout('supportEdge', 'The full projected rectangle is the study support.');

  const currentKey = (options: Readonly<EarthquakePairStructureOptions>): string => {
    if (options.mode === 'marked') {
      return `marked-${options.depthPair}-${Number(options.excludeExactTen)}-${options.markedCorrection}`;
    }
    return [
      'inhomogeneous',
      options.intensityModel,
      ...(options.intensityModel === 'epanechnikov'
        ? [options.intensityBandwidth, options.selfContribution]
        : []),
      options.inhomogeneousCorrection,
      ...(options.inhomogeneousCorrection === 'translation'
        ? [
            options.pairCorrelationEstimator,
            ...(options.pairCorrelationEstimator === 'epanechnikov'
              ? [
                  options.pairCorrelationBandwidth,
                  options.pairCorrelationDivisor,
                  options.zeroCorrection
                ]
              : [])
          ]
        : [])
    ].join('-');
  };

  const buildVariant = async (
    options: Readonly<EarthquakePairStructureOptions>
  ): Promise<Variant> => {
    const key = currentKey(options);
    const variantResources = new SpatialAnalysisResources(
      ctx.device,
      `earthquake-pair-structure-${key}`
    );
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `earthquake-pair-structure-${key}`});
    const bind = createGraphBufferImporter(graph);
    const outputSources: ReadbackSource[] = [];
    let result: PatternView;
    const createView = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      id: string,
      format: Format,
      length: number
    ): {buffer: Buffer; view: GraphDataView<Format>} => {
      const bytesPerElement = format === 'float32x2' ? 8 : 4;
      const buffer = variantResources.createBuffer(id, length * bytesPerElement);
      const view = bind(buffer, format, length);
      outputSources.push({buffer, size: buffer.byteLength});
      return {buffer, view};
    };
    const positions = bind(positionBuffer, 'float32x2', data.events.length);
    const marks = bind(markBuffer, 'float32', data.events.length);
    const maskBuffer =
      options.mode === 'marked' && options.excludeExactTen ? exactMaskBuffer : allMaskBuffer;
    const mask = bind(maskBuffer, 'uint32', data.events.length);
    const radiusBuffer = variantResources.createBuffer('radius-bins-meters', MARKED_RADIUS_VALUES);
    const radiusView = bind(radiusBuffer, 'float32', EARTHQUAKE_PAIR_RADIUS_COUNT);

    if (options.mode === 'marked') {
      const statistics = createView(
        'marked-statistics',
        'float32',
        EARTHQUAKE_PAIR_RADIUS_COUNT * 4
      );
      const pairCounts = createView('marked-pair-counts', 'uint32', EARTHQUAKE_PAIR_RADIUS_COUNT);
      const status = createView('marked-status', 'uint32', EARTHQUAKE_PAIR_RADIUS_COUNT);
      const parameters = variantResources.createBuffer(
        'marked-parameters',
        Float32Array.of(
          EARTHQUAKE_PAIR_BOUNDS.minimumX,
          EARTHQUAKE_PAIR_BOUNDS.minimumY,
          EARTHQUAKE_PAIR_BOUNDS.maximumX,
          EARTHQUAKE_PAIR_BOUNDS.maximumY,
          EARTHQUAKE_PAIR_MAXIMUM_RADIUS_METERS,
          0
        )
      );
      const parameterView = bind(parameters, 'float32', 6);
      const typePair = getSelectedPair(options.depthPair);
      graph.add(
        new GPUMarkedPointStatistics({
          id: 'marked-depth-pairs',
          spatialContext: data.projection.spatialContext,
          positions,
          ...(typePair
            ? {types: bind(depthGroupBuffer, 'uint32', data.events.length), typePair}
            : {}),
          marks,
          mask,
          parameters: parameterView,
          radiusCount: EARTHQUAKE_PAIR_RADIUS_COUNT,
          edgeCorrection: options.markedCorrection,
          output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
        })
      );
      result = {
        mode: 'marked',
        radii: Float32Array.from(MARKED_RADIUS_VALUES),
        statistics: new Float32Array(EARTHQUAKE_PAIR_RADIUS_COUNT * 4),
        pairCounts: new Uint32Array(EARTHQUAKE_PAIR_RADIUS_COUNT),
        status: new Uint32Array(EARTHQUAKE_PAIR_RADIUS_COUNT).fill(2),
        summary: 'Marked pair statistics are being calculated.'
      };
    } else {
      const intensities = createView('estimated-intensity', 'float32', data.events.length);
      const kOutput = createView('inhomogeneous-k', 'float32', EARTHQUAKE_PAIR_RADIUS_COUNT);
      const pairCorrelation = createView(
        'inhomogeneous-pair-correlation',
        'float32',
        EARTHQUAKE_PAIR_RADIUS_COUNT
      );
      const status = createView('inhomogeneous-status', 'uint32', EARTHQUAKE_PAIR_RADIUS_COUNT);
      const windowSummary = createView('study-window-summary', 'float32', 8);
      const windowPreparation = variantResources.track(
        new GPUStudyWindowPreparation({
          id: 'fixed-aeqd-rectangle',
          positions: bind(windowPositionBuffer, 'float32x2', 5),
          ringOffsets: bind(windowRingBuffer, 'uint32', 2),
          polygonOffsets: bind(windowPolygonBuffer, 'uint32', 2),
          featureOffsets: bind(windowFeatureBuffer, 'uint32', 2),
          spatialContext: data.projection.spatialContext,
          samplingGrid: [32, 32],
          output: {summary: windowSummary.view}
        })
      );
      graph.add(windowPreparation);
      graph.add(
        new GPUInhomogeneousPointPattern({
          id: 'inhomogeneous-depth-pattern',
          positions,
          studyWindow: windowPreparation,
          ...(options.intensityModel === 'constant'
            ? {suppliedIntensity: bind(constantIntensityBuffer, 'float32', data.events.length)}
            : {bandwidth: options.intensityBandwidth, selfContribution: options.selfContribution}),
          edgeCorrection: options.inhomogeneousCorrection,
          ...(options.inhomogeneousCorrection === 'translation' &&
          options.pairCorrelationEstimator === 'epanechnikov'
            ? {
                pairCorrelation: {
                  kernel: 'epanechnikov' as const,
                  bandwidth: options.pairCorrelationBandwidth,
                  divisor: options.pairCorrelationDivisor,
                  zeroCorrection: options.zeroCorrection
                }
              }
            : {}),
          radii: radiusView,
          maximumPairCount: (data.events.length * (data.events.length - 1)) / 2,
          maximumOverlapEdgePairs: 65_536,
          output: {
            intensity: intensities.view,
            k: kOutput.view,
            pairCorrelation: pairCorrelation.view,
            status: status.view
          }
        })
      );
      result = {
        mode: 'inhomogeneous',
        radii: Float32Array.from(MARKED_RADIUS_VALUES),
        intensities: new Float32Array(data.events.length),
        k: new Float32Array(EARTHQUAKE_PAIR_RADIUS_COUNT),
        lMinusR: new Float32Array(EARTHQUAKE_PAIR_RADIUS_COUNT),
        pairCorrelation: new Float32Array(EARTHQUAKE_PAIR_RADIUS_COUNT),
        status: new Uint32Array(EARTHQUAKE_PAIR_RADIUS_COUNT).fill(2),
        windowSummary: new Float32Array(8),
        summary: 'Inhomogeneous point-pattern estimates are being calculated.',
        intensityRange: null
      };
    }

    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = variantResources.track(await graph.compileAsync());
    } catch (error) {
      variantResources.destroy();
      throw error;
    }
    let builtVariant: Variant | null = null;
    const reader = new SummaryReader(
      variantResources,
      `earthquake-pair-structure-${key}`,
      outputSources,
      bytes => {
        if (
          !builtVariant ||
          active !== builtVariant ||
          destroyed ||
          currentKey(ctx.options) !== key
        )
          return;
        let offset = 0;
        const readFloat = (count: number) => {
          const value = decodeFloats(bytes, offset, count);
          offset += count * 4;
          return value;
        };
        const readWord = (count: number) => {
          const value = decodeWords(bytes, offset, count);
          offset += count * 4;
          return value;
        };
        if (options.mode === 'marked') {
          const statistics = readFloat(EARTHQUAKE_PAIR_RADIUS_COUNT * 4);
          const pairCounts = readWord(EARTHQUAKE_PAIR_RADIUS_COUNT);
          const status = readWord(EARTHQUAKE_PAIR_RADIUS_COUNT);
          const valid = status.every(code => code === 0);
          const pairLabel = options.depthPair.replaceAll('-', ' ');
          const maskLabel = options.excludeExactTen
            ? 'exact 10 km rows excluded (assumption sensitivity)'
            : 'all selected rows retained';
          const excludedExactTenCount = options.excludeExactTen ? data.exactTenKilometerCount : 0;
          const eligibleCount = data.events.length - excludedExactTenCount;
          const eligibleShallowCount = data.shallowCount - excludedExactTenCount;
          const focusCount =
            options.depthPair === 'deep-to-shallow'
              ? data.deepCount
              : options.depthPair === 'all'
                ? eligibleCount
                : eligibleShallowCount;
          const neighborCount =
            options.depthPair === 'shallow-to-deep'
              ? data.deepCount
              : options.depthPair === 'all'
                ? eligibleCount
                : eligibleShallowCount;
          const summary = `${valid ? 'GPU bins valid' : 'GPU status: ' + readStatus(status)}; ${formatCount(eligibleCount)} eligible GPU input rows; ${formatCount(data.events.length)} mapped events remain visible; ${pairLabel}; ${maskLabel}; ${options.markedCorrection} correction`;
          const terminalIndex = EARTHQUAKE_PAIR_RADIUS_COUNT - 1;
          const terminalLMinusR =
            Math.sqrt(statistics[terminalIndex * 4] / Math.PI) -
            MARKED_RADIUS_VALUES[terminalIndex];
          const lMinusR = Float32Array.from(
            MARKED_RADIUS_VALUES,
            (radius, index) => Math.sqrt(statistics[index * 4] / Math.PI) - radius
          );
          builtVariant.result = {
            mode: 'marked',
            radii: Float32Array.from(MARKED_RADIUS_VALUES),
            statistics,
            pairCounts,
            status,
            summary
          };
          ctx.setReadout('pairStatus', summary);
          ctx.setReadout(
            'pairTerminal',
            valid
              ? `At 80 km: K=${statistics[terminalIndex * 4].toPrecision(6)} m²; L−r=${terminalLMinusR.toPrecision(6)} m; g=${statistics[terminalIndex * 4 + 2].toPrecision(6)}; mark correlation=${statistics[terminalIndex * 4 + 3].toPrecision(6)}; ${formatCount(pairCounts[terminalIndex])} directed pairs.`
              : 'Terminal-radius values withheld because at least one GPU bin status is invalid.'
          );
          ctx.setReadout(
            'pairSummary',
            `GPU focus n=${formatCount(focusCount)}; neighbour n=${formatCount(neighborCount)}. ${options.depthPair === 'all' ? 'K/L normalization uses n² in this implementation.' : 'Ordered cross-category K/L normalization uses n_focus × n_neighbor.'} Depth marks affect only product-mark correlation.`
          );
          ctx.setChart(
            'markedLCurve',
            valid
              ? makeLineChart(
                  'L(r) − r',
                  lMinusR,
                  builtVariant.result.radii,
                  'L(r) − r (m)',
                  'GPU marked-point K output transformed to L(r) minus radius.'
                )
              : null
          );
          ctx.setChart(
            'markedPairCorrelation',
            valid
              ? makeLineChart(
                  'Annular pair correlation',
                  column(statistics, 2),
                  builtVariant.result.radii,
                  'g(r)',
                  'GPU annular pair-correlation output. Empty-annulus NaNs remain chart gaps.',
                  1
                )
              : null
          );
          ctx.setChart(
            'markedMarkCorrelation',
            valid
              ? makeLineChart(
                  'Product-mark correlation',
                  column(statistics, 3),
                  builtVariant.result.radii,
                  'mark correlation',
                  'GPU product-mark correlation normalized by the product of group mean depths.',
                  2
                )
              : null
          );
          ctx.setChart('pairCounts', valid ? makePairCountChart(pairCounts) : null);
        } else {
          const intensities = readFloat(data.events.length);
          const kOutput = readFloat(EARTHQUAKE_PAIR_RADIUS_COUNT);
          const pairCorrelation = readFloat(EARTHQUAKE_PAIR_RADIUS_COUNT);
          const status = readWord(EARTHQUAKE_PAIR_RADIUS_COUNT);
          const windowSummary = readFloat(8);
          const intensityRange = quantileRange(intensities);
          const windowFlags = Math.round(windowSummary[7]);
          const valid =
            status.every(code => code === 0) && intensityRange !== null && windowFlags === 0;
          const lMinusR = Float32Array.from(kOutput, (value, index) =>
            Number.isFinite(value) && value >= 0
              ? Math.sqrt(value / Math.PI) - MARKED_RADIUS_VALUES[index]
              : Number.NaN
          );
          const validValues = Array.from(intensities).filter(Number.isFinite);
          const meanIntensity =
            validValues.reduce((total, value) => total + value, 0) /
            Math.max(1, validValues.length);
          const summary = valid
            ? `GPU statuses valid at all 16 radii; ${formatCount(data.events.length)} rows retained; ${options.intensityModel === 'constant' ? `constant λ = n / area = ${constantIntensity.toExponential(3)} events/m²` : `Epanechnikov intensity bandwidth ${options.intensityBandwidth / 1000} km`}; ${options.inhomogeneousCorrection} correction`
            : `Inference withheld: ${readStatus(status)}; nonpositive/nonfinite intensities ${data.events.length - validValues.filter(value => value > 0).length}; window flags ${windowFlags === 0 ? 'none' : describeWindowFlags(windowFlags)}`;
          builtVariant.result = {
            mode: 'inhomogeneous',
            radii: Float32Array.from(MARKED_RADIUS_VALUES),
            intensities,
            k: kOutput,
            lMinusR,
            pairCorrelation,
            status,
            windowSummary,
            summary,
            intensityRange: valid ? intensityRange : null
          };
          ctx.setReadout('inhomStatus', summary);
          const terminalIndex = EARTHQUAKE_PAIR_RADIUS_COUNT - 1;
          ctx.setReadout(
            'inhomTerminal',
            valid
              ? `At 80 km: K=${kOutput[terminalIndex].toPrecision(6)} m²; L−r=${lMinusR[terminalIndex].toPrecision(6)} m; g=${pairCorrelation[terminalIndex].toPrecision(6)}. At 5 km: K=${kOutput[0].toPrecision(6)} m²; L−r=${lMinusR[0].toPrecision(6)} m; g=${pairCorrelation[0].toPrecision(6)}.`
              : 'Terminal-radius values withheld because GPU output is invalid.'
          );
          ctx.setReadout(
            'inhomSummary',
            valid
              ? `${options.inhomogeneousCorrection === 'border' ? 'Border correction erodes the support by r.' : 'Translation correction weights overlaps of the fixed rectangle.'} ${options.intensityModel === 'constant' ? 'A supplied constant N/area baseline' : 'A plug-in Epanechnikov estimate'} weights the same full input; n=${formatCount(data.events.length)}.`
              : 'GPU status is not valid; curve charts and intensity overlay are withheld without dropping source rows.'
          );
          ctx.setReadout(
            'intensitySummary',
            valid
              ? `Mean GPU intensity ${meanIntensity.toExponential(3)} events/m²; all ${formatCount(data.events.length)} output intensities are finite and positive.`
              : 'Intensity output contains an invalid value; no estimated-intensity inference is drawn.'
          );
          ctx.setReadout(
            'supportEdge',
            options.inhomogeneousCorrection === 'border'
              ? `Fixed rectangle area ${areaText}; largest radius 80 km; ${countBeyondBorder(data.events)} events are more than 80 km from every rectangle edge (support geometry calculation).`
              : `Fixed rectangle area ${areaText}; all ${formatCount(data.events.length)} input events retained for translation correction.`
          );
          ctx.setChart(
            'inhomLCurve',
            valid
              ? makeLineChart(
                  'Inhomogeneous L(r) − r',
                  lMinusR,
                  builtVariant.result.radii,
                  'L(r) − r (m)',
                  'GPU inhomogeneous K transformed to L(r) minus radius.',
                  0
                )
              : null
          );
          const effectivePairCorrelationEstimator =
            options.inhomogeneousCorrection === 'translation'
              ? options.pairCorrelationEstimator
              : 'annular';
          ctx.setChart(
            'inhomPairCorrelation',
            valid
              ? makeLineChart(
                  effectivePairCorrelationEstimator === 'annular'
                    ? 'Annular pair correlation'
                    : 'Epanechnikov pair correlation',
                  pairCorrelation,
                  builtVariant.result.radii,
                  'g(r)',
                  `GPU pair-correlation output with ${effectivePairCorrelationEstimator} estimator and ${options.inhomogeneousCorrection} edge correction.`,
                  1
                )
              : null
          );
          ctx.setChart('intensityHistogram', valid ? buildIntensityHistogram(intensities) : null);
          if (valid && intensityRange) ctx.setLegendExtent('estimated-intensity', intensityRange);
          displayEvents = data.events.map((event, index) => {
            const intensity = intensities[index];
            const fraction =
              intensityRange && intensityRange[1] > intensityRange[0]
                ? clamp(
                    (Math.log(intensity) - Math.log(intensityRange[0])) /
                      (Math.log(intensityRange[1]) - Math.log(intensityRange[0])),
                    0,
                    1
                  )
                : 0.5;
            return {
              ...event,
              intensityColor: valid ? interpolateColor(fraction, 0, 1, INTENSITY_COLORS) : null,
              intensityRadius: valid ? 3 + 4 * fraction : 3.2
            };
          });
        }
        ctx.requestLayers();
      }
    );
    builtVariant = {key, graph: compiled, resources: variantResources, reader, dirty: true, result};
    return builtVariant;
  };

  const clearReadouts = (options: Readonly<EarthquakePairStructureOptions>) => {
    ctx.setChart('markedLCurve', null);
    ctx.setChart('markedPairCorrelation', null);
    ctx.setChart('markedMarkCorrelation', null);
    ctx.setChart('pairCounts', null);
    ctx.setChart('inhomLCurve', null);
    ctx.setChart('inhomPairCorrelation', null);
    ctx.setChart('intensityHistogram', null);
    displayEvents = data.events.map(event => ({
      ...event,
      intensityColor: null,
      intensityRadius: 3.2
    }));
    if (options.mode === 'marked') {
      ctx.setReadout('pairStatus', 'Waiting for the selected marked-pair GPU graph.');
      ctx.setReadout('pairTerminal', 'GPU terminal-radius values pending.');
      ctx.setReadout('pairSummary', 'pending GPU output');
    } else {
      ctx.setReadout('inhomStatus', 'Waiting for the selected intensity GPU graph.');
      ctx.setReadout('inhomTerminal', 'GPU terminal-radius values pending.');
      ctx.setReadout('inhomSummary', 'pending GPU output');
      ctx.setReadout('intensitySummary', 'pending GPU output');
      ctx.setReadout('supportEdge', 'Support status pending GPU output.');
    }
  };

  const requestVariant = (options: Readonly<EarthquakePairStructureOptions>) => {
    const optionsSnapshot = {...options};
    const requestedKey = currentKey(optionsSnapshot);
    revision++;
    const requestedRevision = revision;
    clearReadouts(optionsSnapshot);
    ctx.requestLayers();
    void graphSwap
      .request({
        build: async () => cache.get(requestedKey) ?? buildVariant(optionsSnapshot),
        commit: next => {
          if (
            destroyed ||
            requestedRevision !== revision ||
            currentKey(ctx.options) !== requestedKey
          )
            return;
          cache.delete(requestedKey);
          cache.set(requestedKey, next);
          active = next;
          next.dirty = true;
          while (cache.size > 8) {
            const oldestKey = cache.keys().next().value;
            if (!oldestKey || oldestKey === requestedKey) break;
            const oldest = cache.get(oldestKey);
            cache.delete(oldestKey);
            if (oldest) {
              oldest.reader.release();
              oldest.resources.destroy();
            }
          }
          ctx.requestLayers();
        },
        discard: next => {
          if (!cache.has(requestedKey)) {
            next.reader.release();
            next.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && requestedRevision === revision) {
          ctx.setReadout(
            optionsSnapshot.mode === 'marked' ? 'pairStatus' : 'inhomStatus',
            `GPU graph build failed: ${String(error)}`
          );
        }
      });
  };

  const activeResult = (): PatternView | null =>
    active && currentKey(ctx.options) === active.key ? active.result : null;
  const setSelectedEvent = (event: EarthquakePairEvent) => {
    selectedEvent = event;
    ctx.setReadout(
      'selectedDepth',
      `Source row ${event.sourceRow}; reported depth ${event.depthKm.toFixed(2)} km; ${event.depthGroup === 0 ? 'below 70 km' : '70 km or deeper'}${event.exactTenKilometers ? '; exactly 10 km (not proof of a fixed-depth solution)' : ''}.`
    );
    ctx.requestLayers();
  };
  const getEventFromPointer = (event: ScenePointerEvent): EarthquakePairEvent | null => {
    const object = event.info.object;
    if (
      object &&
      typeof object === 'object' &&
      'sourceRow' in object &&
      'depthKm' in object &&
      'selectedIndex' in object &&
      'position' in object &&
      'projectedPosition' in object &&
      'depthGroup' in object &&
      'exactTenKilometers' in object
    )
      return object;
    return null;
  };
  const getLayers = (): Layer[] => {
    const layers: Layer[] = [
      new PathLayer({
        id: 'earthquake-pair-study-window',
        data: makeWindowPath(data),
        getPath: (segment: [number, number, number, number]) => [
          [segment[0], segment[1]],
          [segment[2], segment[3]]
        ],
        getColor: [49, 75, 82, 205],
        getWidth: 2,
        widthUnits: 'pixels',
        pickable: false
      }),
      new ScatterplotLayer({
        id: 'earthquake-pair-events',
        data: displayEvents,
        getPosition: (event: (typeof displayEvents)[number]) => event.position,
        getFillColor: (event: (typeof displayEvents)[number]) =>
          interpolateColor(event.depthKm, data.depthExtent[0], data.depthExtent[1], DEPTH_COLORS),
        getLineColor: (event: (typeof displayEvents)[number]) =>
          event.intensityColor ?? [237, 229, 210, 140],
        getRadius: (event: (typeof displayEvents)[number]) => event.intensityRadius,
        radiusUnits: 'pixels',
        radiusMinPixels: 2,
        radiusMaxPixels: 9,
        stroked: true,
        lineWidthMinPixels: 1,
        pickable: true
      }),
      new ScatterplotLayer({
        id: 'earthquake-pair-exact-ten-km',
        data: data.events.filter(event => event.exactTenKilometers),
        getPosition: (event: EarthquakePairEvent) => event.position,
        getFillColor: [0, 0, 0, 0],
        getLineColor: [243, 189, 98, 245],
        getRadius: 7,
        radiusUnits: 'pixels',
        stroked: true,
        lineWidthMinPixels: 1.5,
        pickable: false
      })
    ];
    if (selectedEvent) {
      layers.push(
        new ScatterplotLayer({
          id: 'earthquake-pair-selected-event',
          data: [selectedEvent],
          getPosition: (event: EarthquakePairEvent) => event.position,
          getFillColor: [255, 246, 213, 40],
          getLineColor: [38, 48, 50, 245],
          getRadius: 10,
          radiusUnits: 'pixels',
          stroked: true,
          lineWidthMinPixels: 2,
          pickable: false
        })
      );
    }
    return layers;
  };

  requestVariant(ctx.options);
  return {
    getCompiledGraphs: () => Array.from(cache.values(), variant => variant.graph),
    getLayers,
    encode: (commandEncoder, _frame) => {
      if (!active) return;
      active.graph.encode(commandEncoder, {parameters: undefined});
      if (active.dirty) {
        active.dirty = false;
        active.reader.markStale();
      }
      active.reader.flush(commandEncoder);
    },
    setOption: (_id, _value, options) => requestVariant(options),
    getTooltip: event => {
      const record = getEventFromPointer(event);
      if (!record) return null;
      const current = activeResult();
      const intensity =
        current?.mode === 'inhomogeneous' ? current.intensities[record.selectedIndex] : undefined;
      return {
        title: 'USGS ComCat earthquake',
        subtitle: `Archived row ${record.sourceRow} · type 0`,
        rows: [
          {label: 'Reported depth', value: record.depthKm.toFixed(2), unit: 'km', emphasis: true},
          {label: 'Depth group', value: record.depthGroup === 0 ? '<70 km' : '≥70 km'},
          ...(record.exactTenKilometers
            ? [{label: 'Exact 10 km', value: 'yes; not proof of fixed depth'}]
            : []),
          ...(typeof intensity === 'number' && Number.isFinite(intensity)
            ? [{label: 'GPU intensity', value: intensity.toExponential(3), unit: 'events/m²'}]
            : [])
        ],
        highlight: {kind: 'point', coordinate: record.position, tone: 'signal', pulse: false}
      } satisfies TooltipContent;
    },
    onClick: event => {
      const record = getEventFromPointer(event);
      if (!record) return false;
      setSelectedEvent(record);
      return true;
    },
    destroy: () => {
      destroyed = true;
      graphSwap.destroy();
      for (const variant of cache.values()) {
        variant.reader.stop();
        variant.resources.destroy();
      }
      resources.destroy();
    }
  };
}

function column(values: Float32Array, index: number): Float32Array {
  const output = new Float32Array(EARTHQUAKE_PAIR_RADIUS_COUNT);
  for (let row = 0; row < output.length; row++) output[row] = values[row * 4 + index];
  return output;
}

function countBeyondBorder(events: readonly EarthquakePairEvent[]): number {
  return events.reduce((count, event) => {
    const [positionX, positionY] = event.projectedPosition;
    const distance = Math.min(
      positionX - EARTHQUAKE_PAIR_BOUNDS.minimumX,
      EARTHQUAKE_PAIR_BOUNDS.maximumX - positionX,
      positionY - EARTHQUAKE_PAIR_BOUNDS.minimumY,
      EARTHQUAKE_PAIR_BOUNDS.maximumY - positionY
    );
    return count + Number(distance > EARTHQUAKE_PAIR_MAXIMUM_RADIUS_METERS);
  }, 0);
}

function buildIntensityHistogram(values: Float32Array): BarChartData {
  const bins = 12;
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const counts = new Uint32Array(bins);
  if (maximum === minimum) {
    counts[Math.floor(bins / 2)] = values.length;
    return {
      kind: 'histogram',
      title: 'GPU intensity across all selected events',
      values: counts,
      xDomain: [minimum * 0.99, maximum * 1.01],
      xLabel: 'GPU intensity (events/m²)',
      yLabel: 'Earthquakes',
      description:
        'All selected rows share the supplied constant N/area intensity; the GPU output is not spatially varying.'
    };
  }
  const step = (maximum - minimum) / bins;
  for (const value of values) {
    const index = Math.min(bins - 1, Math.floor((value - minimum) / step));
    counts[index]++;
  }
  return {
    kind: 'histogram',
    title: 'GPU intensity across all selected events',
    values: counts,
    xDomain: [minimum, maximum],
    xLabel: 'GPU intensity (events/m²)',
    yLabel: 'Earthquakes',
    description:
      'Histogram of every GPU estimated intensity value; no source rows were sampled or removed.'
  };
}
