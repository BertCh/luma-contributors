// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exit condition of wave 2 (roadmap section 8): filter, cube, Gi*, trend, Markov, Knox and a panel
 * regression on one pipeline with no CPU readback between stages. One graph, one submission; the
 * outputs are read only after it, and every expectation is computed on the CPU from the raw events
 * by code that shares nothing with the contributors under test.
 */

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {
  assertCompatibleGPUTemporalContexts,
  type GPUTemporalContext
} from '../../../src/analysis-core/contracts/index';
import {
  getGPUClassBreaksParameterValues,
  GPUClassBreaks
} from '../../../src/dataframe-analysis/column-classification/index';
import {GPUCubeLayoutTranspose} from '../../../src/dataframe-analysis/cube-layout/index';
import {GPUPeriodIndex} from '../../../src/dataframe-analysis/period-index/index';
import {
  getGPUTimeWindowFilterWordParameterValues,
  getInt64TimeWords,
  GPUTimeWindowFilter
} from '../../../src/dataframe-analysis/time-window-filter/index';
import {GPUTimeWordsRelative} from '../../../src/dataframe-analysis/time-words-relative/index';
import {GPUValueCube} from '../../../src/dataframe-analysis/value-cube/index';
import {
  GPUClassAssignment,
  GPUTransitionMatrix
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/index';
import {
  getGPUEmergingHotSpotsParameterValues,
  GPUEmergingHotSpots
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch
} from '../../../src/gpu-spatial-analysis/neighbor-search/index';
import {
  getGPUSpaceTimeParameterValues,
  GPUKnoxTest,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {GPUSpatialPanelRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-regression';
import {
  computeEmergingHotSpotCells,
  computeSpaceTimeGiStar
} from '../../gpu-spatial-analysis/emerging-hot-spots/emerging-hot-spots-oracle';
import {
  computeTransitionOracle,
  getOracleClass,
  getOracleQuantile
} from '../../gpu-spatial-analysis/distribution-dynamics/distribution-oracle';
import {computeKnoxOracle} from '../../gpu-spatial-analysis/space-time-tests/space-time-oracle';
import {
  createLatticeWeights,
  type DiagnosticsOracleWeights
} from '../../gpu-spatial-analysis/spatial-regression/spatial-regression-diagnostics-oracle';
import {fitSpatialTwoStageLeastSquaresOnCPU} from '../../gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-oracle';
import {requireTestDevice} from '../../utils/require-test-device';
import {getExactFloat32Sum} from './value-cube-oracle';
import {createValueCubeRandom, ValueCubeRig} from './value-cube-rig';

// The scene: a 6 x 6 lattice of cells, 12 daily periods from 2024-03-01, four in-window events per
// (cell, period), a rising hot block, a falling cold block, and decoy events (value 5000) outside
// the time window that would wreck every result if the filter did not remove them.
const SIDE = 6;
const CELL_COUNT = SIDE * SIDE;
const PERIOD_COUNT = 12;
const EVENTS_PER_BIN = 4;
const DAY = 86_400_000n;
const HOUR = 3_600_000n;
const ORIGIN = 1_709_251_200_000n; // 2024-03-01T00:00:00Z in epoch milliseconds
const WINDOW_START = ORIGIN + 12n * HOUR;
const WINDOW_END = ORIGIN + 11n * DAY + 12n * HOUR;
const HOT_CELLS = [0, 1, 6, 7];
const COLD_CELLS = [28, 29, 34, 35];
const DECOY_VALUE = 5000;
const CLASS_COUNT = 4;
const KNOX_SEED = 7;
const KNOX_PERMUTATIONS = 49;
const KNOX_TIME_THRESHOLD = Number(6n * HOUR);
const KNOX_RADIUS = 60;

const EVENT_CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

type Event = {
  time: bigint;
  x: number;
  y: number;
  cell: number;
  response: number;
  covariate: number;
};

function createScene(): Event[] {
  const random = createValueCubeRandom(2024);
  const events: Event[] = [];
  const eighths = (value: number) => Math.round(value * 8) / 8;
  const inCell = (cell: number) => {
    const column = cell % SIDE;
    const row = Math.floor(cell / SIDE);
    return {x: column * 100 + 10 + random() * 80, y: row * 100 + 10 + random() * 80, cell};
  };
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    for (let day = 0; day < PERIOD_COUNT; day++) {
      const covariateCenter = Math.round(random() * 32) / 4;
      const hot = HOT_CELLS.includes(cell) ? 2.5 * day : COLD_CELLS.includes(cell) ? -2 * day : 0;
      // A space-time cluster: the four events of a bin share a place and a two-hour stretch.
      const center = inCell(cell);
      const earliestHour = day === 0 ? 12.5 : 0;
      const latestHour = day === PERIOD_COUNT - 1 ? 9 : 21;
      const hour = earliestHour + random() * (latestHour - earliestHour);
      for (let index = 0; index < EVENTS_PER_BIN; index++) {
        const offsetMilliseconds = BigInt(Math.floor(random() * 2 * Number(HOUR)));
        events.push({
          time:
            ORIGIN +
            BigInt(day) * DAY +
            BigInt(Math.floor(hour * Number(HOUR))) +
            offsetMilliseconds,
          x: Math.fround(center.x + (random() - 0.5) * 16),
          y: Math.fround(center.y + (random() - 0.5) * 16),
          cell,
          response: eighths(10 + hot + 0.5 * covariateCenter + (random() - 0.5) * 2),
          covariate: eighths(covariateCenter + (random() - 0.5))
        });
      }
    }
    const decoy = (time: bigint) => {
      const place = inCell(cell);
      events.push({
        time,
        x: Math.fround(place.x),
        y: Math.fround(place.y),
        cell,
        response: DECOY_VALUE,
        covariate: DECOY_VALUE
      });
    };
    const hourOf = (low: number, high: number) =>
      BigInt(Math.floor((low + random() * (high - low)) * Number(HOUR)));
    // Day 0 before the window starts (inside period 0), day 11 after the window ends (inside
    // period 11), one day before the periods and one day after them.
    for (let index = 0; index < 2; index++) decoy(ORIGIN + hourOf(1, 10));
    for (let index = 0; index < 2; index++) decoy(ORIGIN + 11n * DAY + hourOf(13, 22));
    decoy(ORIGIN - DAY + hourOf(1, 22));
    decoy(ORIGIN + BigInt(PERIOD_COUNT) * DAY + hourOf(1, 22));
  }
  // Shuffle: rows are not grouped by cell or time.
  for (let row = events.length - 1; row > 0; row--) {
    const other = Math.floor(random() * (row + 1));
    [events[row], events[other]] = [events[other], events[row]];
  }
  return events;
}

/** Whether the CPU time window keeps the event. */
const isInWindow = (event: Event) => event.time >= WINDOW_START && event.time <= WINDOW_END;

/** Cell-major mean cube of a column by exact big-integer sums, from the in-window events only. */
function computeMeanCube(
  events: readonly Event[],
  column: 'response' | 'covariate',
  useWindow: boolean
): Float32Array {
  const bins: number[][] = Array.from({length: CELL_COUNT * PERIOD_COUNT}, () => []);
  for (const event of events) {
    const period = Number((event.time - ORIGIN) / DAY);
    const floored = event.time < ORIGIN ? -1 : period;
    if ((useWindow && !isInWindow(event)) || floored < 0 || floored >= PERIOD_COUNT) continue;
    bins[event.cell * PERIOD_COUNT + floored].push(event[column]);
  }
  return Float32Array.from(bins, values =>
    values.length > 0 ? Math.fround(getExactFloat32Sum(values) / values.length) : NaN
  );
}

function toPeriodMajor(cube: Float32Array): Float32Array {
  const result = new Float32Array(cube.length);
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    for (let period = 0; period < PERIOD_COUNT; period++) {
      result[period * CELL_COUNT + cell] = cube[cell * PERIOD_COUNT + period];
    }
  }
  return result;
}

/** Block-diagonal `I_T (x) W` of a per-period weights matrix. */
function expandPanelWeights(weights: DiagnosticsOracleWeights): DiagnosticsOracleWeights {
  const offsets = [0];
  const neighbors: number[] = [];
  const values: number[] = [];
  for (let period = 0; period < PERIOD_COUNT; period++) {
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      for (let slot = weights.offsets[cell]; slot < weights.offsets[cell + 1]; slot++) {
        neighbors.push(period * CELL_COUNT + weights.neighbors[slot]);
        values.push(weights.weights[slot]);
      }
      offsets.push(neighbors.length);
    }
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(values)
  };
}

type PipelineResult = {
  filterCount: number;
  periodOutOfRange: number;
  giZScores: number[];
  trendZ: number[];
  trendS: number[];
  category: number[];
  hotSliceCount: number[];
  transitionCounts: number[];
  transitionRowTotals: number[];
  knoxStatistics: number[];
  knoxSummary: number[];
  neighborSlots: number;
  neighborOverflow: number;
  panelTable: number[];
  panelStatus: number;
  submissions: number;
};

const HOT_SPOT_SETTINGS = {radius: 1, temporalWindow: 1} as const;

/** Builds the whole pipeline in one graph, submits once and reads only the final outputs. */
async function runPipeline(
  events: readonly Event[],
  options: {applyFilterMask: boolean; neighborSlotCapacity: number}
): Promise<PipelineResult> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new ValueCubeRig(device);
  const rowCount = events.length;
  const binCount = CELL_COUNT * PERIOD_COUNT;

  // Raw event columns: uploaded once, never read back.
  const timestamps = rig.input(
    getInt64TimeWords(BigInt64Array.from(events, event => event.time)),
    'uint32x2'
  );
  const cells = rig.input(
    Uint32Array.from(events, event => event.cell),
    'uint32'
  );
  const responses = rig.input(
    Float32Array.from(events, event => event.response),
    'float32'
  );
  const covariates = rig.input(
    Float32Array.from(events, event => event.covariate),
    'float32'
  );
  const positions = rig.input(
    Float32Array.from(events.flatMap(event => [event.x, event.y])),
    'float32x2'
  );

  // 1. Time window filter on exact Int64 words: the accepted-row mask.
  const filterIds = rig.output('uint32', rowCount);
  const filterCount = rig.output('uint32', 1);
  const filterOverflow = rig.output('uint32', 1);
  const filterMask = rig.output('uint32', rowCount);
  const filter = new GPUTimeWindowFilter({
    timestamps,
    window: rig.input(
      getGPUTimeWindowFilterWordParameterValues({start: WINDOW_START, end: WINDOW_END}),
      'uint32'
    ),
    output: {
      rows: filterIds,
      outputMask: filterMask,
      status: {count: filterCount, overflow: filterOverflow}
    }
  });

  // 2. Period index: fixed one-day periods from the origin. Not masked, so the status counts every
  // valid time outside the twelve periods.
  const periods = rig.output('uint32', rowCount);
  const periodOutOfRange = rig.output('uint32', 1);
  const periodIndex = new GPUPeriodIndex({
    temporalContext: EVENT_CONTEXT,
    periods: {kind: 'fixed', width: DAY, origin: ORIGIN, count: PERIOD_COUNT},
    timestamps,
    output: {periods, status: {invalidCount: periodOutOfRange}}
  });
  // The filter reads the same words as the period index: their contexts must agree.
  assertCompatibleGPUTemporalContexts(
    'wave two filter and period index',
    EVENT_CONTEXT,
    periodIndex.temporalContext
  );

  // 3. Value cubes (cell-major): mean response and mean covariate per (cell, period).
  const responseMeans = rig.output('float32', binCount);
  const covariateMeans = rig.output('float32', binCount);
  const mask = options.applyFilterMask ? filterMask : undefined;
  const responseCube = new GPUValueCube({
    id: 'response-cube',
    cellCount: CELL_COUNT,
    periodCount: PERIOD_COUNT,
    cells,
    periods,
    values: responses,
    mask,
    output: {means: responseMeans}
  });
  const covariateCube = new GPUValueCube({
    id: 'covariate-cube',
    cellCount: CELL_COUNT,
    periodCount: PERIOD_COUNT,
    cells,
    periods,
    values: covariates,
    mask,
    output: {means: covariateMeans}
  });

  // 4. Emerging hot spots on the response cube: Gi*, Mann-Kendall trend, categories.
  const hotSpotOutput = {
    giZScores: rig.output('float32', binCount),
    trendZ: rig.output('float32', CELL_COUNT),
    trendP: rig.output('float32', CELL_COUNT),
    trendS: rig.output('sint32', CELL_COUNT),
    category: rig.output('uint32', CELL_COUNT),
    hotSliceCount: rig.output('uint32', CELL_COUNT),
    coldSliceCount: rig.output('uint32', CELL_COUNT)
  };
  const hotSpots = new GPUEmergingHotSpots({
    values: responseMeans,
    gridWidth: SIDE,
    gridHeight: SIDE,
    sliceCount: PERIOD_COUNT,
    parameters: rig.input(getGPUEmergingHotSpotsParameterValues({...HOT_SPOT_SETTINGS}), 'float32'),
    output: hotSpotOutput
  });

  // 5. Layout: cell-major cube to the period-major layout of Markov chains and panels.
  const responsePeriodMajor = rig.output('float32', binCount);
  const covariatePeriodMajor = rig.output('float32', binCount);
  const transposeResponse = new GPUCubeLayoutTranspose({
    id: 'transpose-response',
    values: responseMeans,
    from: 'cell-major',
    cellCount: CELL_COUNT,
    periodCount: PERIOD_COUNT,
    output: responsePeriodMajor
  });
  const transposeCovariate = new GPUCubeLayoutTranspose({
    id: 'transpose-covariate',
    values: covariateMeans,
    from: 'cell-major',
    cellCount: CELL_COUNT,
    periodCount: PERIOD_COUNT,
    output: covariatePeriodMajor
  });

  // 6. Markov: pooled quantile breaks on the GPU, classes, transition counts.
  const breaks = rig.output('float32', CLASS_COUNT + 1);
  const classCount = rig.output('uint32', 1);
  const classes = rig.output('uint32', binCount);
  const transitionCounts = rig.output('uint32', CLASS_COUNT * CLASS_COUNT);
  const transitionRowTotals = rig.output('uint32', CLASS_COUNT);
  const classBreaks = new GPUClassBreaks({
    values: responsePeriodMajor,
    maximumClassCount: CLASS_COUNT,
    methods: ['quantile'],
    parameters: rig.input(
      getGPUClassBreaksParameterValues({method: 'quantile', classCount: CLASS_COUNT}, CLASS_COUNT),
      'float32'
    ),
    output: {breaks, classCount}
  });
  const classAssignment = new GPUClassAssignment({
    values: responsePeriodMajor,
    breaks,
    classCount,
    output: classes
  });
  const transitions = new GPUTransitionMatrix({
    classes,
    periodCount: PERIOD_COUNT,
    classCount: CLASS_COUNT,
    output: {counts: transitionCounts, rowTotals: transitionRowTotals}
  });

  // 7. Knox: relative times from the words, spatial pairs among the accepted events, the test.
  const relativeTimes = rig.output('float32', rowCount);
  const timeWordsRelative = new GPUTimeWordsRelative({
    timestamps,
    temporalContext: EVENT_CONTEXT,
    origin: ORIGIN,
    output: relativeTimes
  });
  // Knox compares float32 ticks after ORIGIN against a threshold in the same unit: the context it
  // reads is exactly the output context of the conversion, and not the raw words' context.
  assertCompatibleGPUTemporalContexts(
    'wave two knox times',
    timeWordsRelative.outputTemporalContext,
    {
      unit: 'millisecond',
      epoch: ORIGIN,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    }
  );
  expect(() =>
    assertCompatibleGPUTemporalContexts(
      'wave two knox times',
      EVENT_CONTEXT,
      timeWordsRelative.outputTemporalContext
    )
  ).toThrow(/incompatible temporal contexts/);
  const neighborOffsets = rig.output('uint32', rowCount + 1);
  const neighborIds = rig.output('uint32', options.neighborSlotCapacity);
  const neighborWeights = rig.output('float32', options.neighborSlotCapacity);
  const neighborOverflow = rig.output('uint32', 1);
  const neighborSearch = new GPUNeighborSearch({
    mode: 'radius',
    gridSize: [8, 8],
    positions,
    mask: filterMask,
    parameters: rig.input(
      getGPUNeighborSearchParameterValues({bounds: [-1, -1, 601, 601], radius: KNOX_RADIUS}),
      'float32'
    ),
    output: {
      offsets: neighborOffsets,
      neighbors: neighborIds,
      weights: neighborWeights,
      status: {overflow: neighborOverflow}
    }
  });
  const knoxStatistics = rig.output('uint32', 65);
  const knoxSummary = rig.output('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const knox = new GPUKnoxTest({
    pairs: {offsets: neighborOffsets, neighbors: neighborIds, weights: neighborWeights},
    times: relativeTimes,
    parameters: rig.input(
      getGPUSpaceTimeParameterValues({
        seed: KNOX_SEED,
        permutationCount: KNOX_PERMUTATIONS,
        timeThreshold: KNOX_TIME_THRESHOLD
      }),
      'uint32'
    ),
    maximumPermutations: 64,
    output: {statistics: knoxStatistics, summary: knoxSummary}
  });

  // 8. Panel regression: mean response on mean covariate with a spatial lag, entities are cells.
  const latticeWeights = createLatticeWeights(SIDE, true);
  const panelTable = rig.output('float32', 12);
  const panelSummary = rig.output('float32', 16);
  const panelResiduals = rig.output('float32', binCount);
  const panelStatus = rig.output('uint32', 1);
  const panel = new GPUSpatialPanelRegression({
    weights: {
      offsets: rig.input(latticeWeights.offsets, 'uint32'),
      neighbors: rig.input(latticeWeights.neighbors, 'uint32'),
      weights: rig.input(latticeWeights.weights, 'float32')
    },
    periodCount: PERIOD_COUNT,
    predictorCount: 1,
    predictors: covariatePeriodMajor,
    response: responsePeriodMajor,
    output: {
      table: panelTable,
      summary: panelSummary,
      residuals: panelResiduals,
      status: {code: panelStatus}
    }
  });

  // One graph, one submission. Nothing is read until it has run.
  rig.run(
    filter,
    periodIndex,
    responseCube,
    covariateCube,
    hotSpots,
    transposeResponse,
    transposeCovariate,
    classBreaks,
    classAssignment,
    transitions,
    timeWordsRelative,
    neighborSearch,
    knox,
    panel
  );
  expect(rig.submissions).toBe(1);

  const result: PipelineResult = {
    filterCount: (await rig.readUint(filterCount))[0],
    periodOutOfRange: (await rig.readUint(periodOutOfRange))[0],
    giZScores: await rig.readFloat(hotSpotOutput.giZScores),
    trendZ: await rig.readFloat(hotSpotOutput.trendZ),
    trendS: Array.from(new Int32Array((await rig.readWords(hotSpotOutput.trendS)).buffer)),
    category: await rig.readUint(hotSpotOutput.category),
    hotSliceCount: await rig.readUint(hotSpotOutput.hotSliceCount),
    transitionCounts: await rig.readUint(transitionCounts),
    transitionRowTotals: await rig.readUint(transitionRowTotals),
    knoxStatistics: await rig.readUint(knoxStatistics),
    knoxSummary: await rig.readFloat(knoxSummary),
    neighborSlots: (await rig.readUint(neighborOffsets))[rowCount],
    neighborOverflow: (await rig.readUint(neighborOverflow))[0],
    panelTable: await rig.readFloat(panelTable),
    panelStatus: (await rig.readUint(panelStatus))[0],
    submissions: rig.submissions
  };
  rig.destroy();
  return result;
}

/** Brute-force spatial pairs among the in-window events, as a CSR over all event rows. */
function buildPairs(events: readonly Event[]) {
  const offsets = [0];
  const neighbors: number[] = [];
  for (const [row, event] of events.entries()) {
    if (isInWindow(event)) {
      for (const [other, candidate] of events.entries()) {
        if (other !== row && isInWindow(candidate)) {
          if (Math.hypot(event.x - candidate.x, event.y - candidate.y) <= KNOX_RADIUS) {
            neighbors.push(other);
          }
        }
      }
    }
    offsets.push(neighbors.length);
  }
  return {offsets, neighbors};
}

describe('wave 2 composition: filter, cube, Gi*, trend, Markov, Knox and panel on one pipeline', () => {
  it('runs in one submission and matches an independent CPU computation of every stage', async () => {
    const events = createScene();
    const inWindowCount = events.filter(isInWindow).length;
    expect(inWindowCount).toBe(CELL_COUNT * PERIOD_COUNT * EVENTS_PER_BIN);
    const pairs = buildPairs(events);
    expect(pairs.neighbors.length).toBeGreaterThan(10_000);
    const actual = await runPipeline(events, {
      applyFilterMask: true,
      neighborSlotCapacity: pairs.neighbors.length + 64
    });
    expect(actual.submissions).toBe(1);

    // Stage 1 and 2: the filter keeps exactly the in-window events; the period index counts the
    // valid times outside the twelve periods (one decoy day before and one after, per cell).
    expect(actual.filterCount).toBe(inWindowCount);
    expect(actual.periodOutOfRange).toBe(2 * CELL_COUNT);

    // Stage 3 to 5, through the stages that read the cube: Gi*, trend, categories.
    const expectedMeans = computeMeanCube(events, 'response', true);
    expect(expectedMeans.every(Number.isFinite)).toBe(true);
    const packed = getGPUEmergingHotSpotsParameterValues({...HOT_SPOT_SETTINGS});
    const shape = {gridWidth: SIDE, gridHeight: SIDE, sliceCount: PERIOD_COUNT};
    const expectedZ = computeSpaceTimeGiStar({...shape, values: expectedMeans}, packed);
    actual.giZScores.forEach((z, bin) => {
      expect(Math.abs(z - expectedZ[bin]), `Gi* z of bin ${bin}`).toBeLessThan(2e-3);
    });
    const expectedCells = computeEmergingHotSpotCells(expectedZ, shape, packed);
    expect(actual.category).toEqual(expectedCells.category);
    expect(actual.trendS).toEqual(expectedCells.trendS);
    actual.trendZ.forEach((z, cell) => {
      expect(Math.abs(z - expectedCells.trendZ[cell]), `trend z of cell ${cell}`).toBeLessThan(
        2e-3
      );
    });
    expect(actual.hotSliceCount).toEqual(expectedCells.hotSliceCount);
    // Meaningful: the designed rising block is a rising trend and a hot category, the falling
    // block a falling trend, and the quiet cells far from both are not.
    for (const cell of HOT_CELLS) {
      expect(actual.trendS[cell], `hot cell ${cell}`).toBeGreaterThan(0);
      expect(actual.trendZ[cell]).toBeGreaterThan(1.96);
      expect(actual.hotSliceCount[cell]).toBeGreaterThan(0);
    }
    for (const cell of COLD_CELLS) {
      expect(actual.trendS[cell], `cold cell ${cell}`).toBeLessThan(0);
      expect(actual.trendZ[cell]).toBeLessThan(-1.96);
    }
    expect(new Set(actual.category).size).toBeGreaterThan(2);

    // Stage 6: Markov transition counts from pooled quantile classes of the period-major cube.
    const periodMajor = toPeriodMajor(expectedMeans);
    const sorted = [...periodMajor].sort((a, b) => a - b);
    const edges = Array.from({length: CLASS_COUNT + 1}, (_, edge) =>
      Math.fround(getOracleQuantile(sorted, edge / CLASS_COUNT))
    );
    const expectedClasses = Array.from(periodMajor, value =>
      getOracleClass(value, edges, CLASS_COUNT)
    );
    const expectedTransitions = computeTransitionOracle({
      classes: expectedClasses,
      rows: CELL_COUNT,
      periods: PERIOD_COUNT,
      classCount: CLASS_COUNT
    });
    expect(actual.transitionCounts).toEqual(expectedTransitions.counts);
    expect(actual.transitionRowTotals).toEqual(expectedTransitions.rowTotals);
    expect(expectedTransitions.rowTotals.reduce((sum, total) => sum + total, 0)).toBe(
      CELL_COUNT * (PERIOD_COUNT - 1)
    );
    // The designed trends make the chain move: some cells change class between periods.
    const stayed = [0, 1, 2, 3].reduce(
      (sum, state) => sum + expectedTransitions.counts[state * CLASS_COUNT + state],
      0
    );
    expect(stayed).toBeGreaterThan(0);
    expect(stayed).toBeLessThan(CELL_COUNT * (PERIOD_COUNT - 1));

    // Stage 7: Knox. Times are exact words differences rounded to float32.
    expect(actual.neighborOverflow).toBe(0);
    expect(actual.neighborSlots).toBe(pairs.neighbors.length);
    const relativeTimes = Float32Array.from(events, event =>
      Math.fround(Number(event.time - ORIGIN))
    );
    const expectedKnox = computeKnoxOracle(
      pairs,
      relativeTimes,
      KNOX_SEED,
      KNOX_PERMUTATIONS,
      KNOX_TIME_THRESHOLD
    );
    expect(actual.knoxStatistics.slice(0, KNOX_PERMUTATIONS + 1)).toEqual(expectedKnox.statistics);
    expect(actual.knoxSummary[GPU_SPACE_TIME_SUMMARY.pairCount]).toBe(expectedKnox.pairCount);
    expect(actual.knoxSummary[GPU_SPACE_TIME_SUMMARY.observed]).toBe(expectedKnox.statistics[0]);
    // Clustered events: more close pairs than the permutations give, in every permutation.
    expect(expectedKnox.statistics[0]).toBeGreaterThan(1.5 * expectedKnox.expected);
    expect(actual.knoxSummary[GPU_SPACE_TIME_SUMMARY.pseudoPGreater]).toBeCloseTo(
      (expectedKnox.greater + 1) / (KNOX_PERMUTATIONS + 1),
      6
    );
    expect(expectedKnox.greater).toBe(0);

    // Stage 8: panel regression of the mean response on the mean covariate with a spatial lag.
    expect(actual.panelStatus).toBe(0);
    const covariatePeriodMajor = toPeriodMajor(computeMeanCube(events, 'covariate', true));
    const expectedPanel = fitSpatialTwoStageLeastSquaresOnCPU(
      expandPanelWeights(createLatticeWeights(SIDE, true)),
      covariatePeriodMajor,
      periodMajor,
      1
    );
    // Table rows are intercept, covariate, rho; each row holds coefficient, standard error, z, p.
    [0, 1, 2].forEach(row => {
      const coefficient = actual.panelTable[4 * row];
      const standardError = actual.panelTable[4 * row + 1];
      expect(
        Math.abs(coefficient - expectedPanel.coefficients[row]),
        `coefficient ${row}: ${coefficient} vs ${expectedPanel.coefficients[row]}`
      ).toBeLessThan(2e-3 * (1 + Math.abs(expectedPanel.coefficients[row])));
      expect(
        Math.abs(standardError - expectedPanel.standardErrors[row]),
        `standard error ${row}: ${standardError} vs ${expectedPanel.standardErrors[row]}`
      ).toBeLessThan(5e-3 * (1 + expectedPanel.standardErrors[row]));
    });
    // The generating model has a positive covariate effect.
    expect(expectedPanel.coefficients[1]).toBeGreaterThan(0.2);
  });

  it('would produce visibly different results without the time window, so the filter is load bearing', async () => {
    const events = createScene();
    const pairs = buildPairs(events);
    const capacity = pairs.neighbors.length + 64;
    const filtered = await runPipeline(events, {
      applyFilterMask: true,
      neighborSlotCapacity: capacity
    });
    const unfiltered = await runPipeline(events, {
      applyFilterMask: false,
      neighborSlotCapacity: capacity
    });
    // The decoys sit in the first and last periods: the Gi* z of those slices and the panel move.
    const distortion = filtered.giZScores.reduce(
      (largest, z, bin) => Math.max(largest, Math.abs(z - unfiltered.giZScores[bin])),
      0
    );
    expect(distortion).toBeGreaterThan(1);
    const coefficientShift = [0, 4, 8].map(index =>
      Math.abs(filtered.panelTable[index] - unfiltered.panelTable[index])
    );
    expect(Math.max(...coefficientShift)).toBeGreaterThan(0.05);
    // The cube without the mask is also exactly what the CPU gets from all in-period events.
    const expectedMeans = computeMeanCube(events, 'response', false);
    const packed = getGPUEmergingHotSpotsParameterValues({...HOT_SPOT_SETTINGS});
    const shape = {gridWidth: SIDE, gridHeight: SIDE, sliceCount: PERIOD_COUNT};
    const expectedZ = computeSpaceTimeGiStar({...shape, values: expectedMeans}, packed);
    unfiltered.giZScores.forEach((z, bin) => {
      expect(Math.abs(z - expectedZ[bin]), `unfiltered Gi* z of bin ${bin}`).toBeLessThan(2e-3);
    });
  });
});
