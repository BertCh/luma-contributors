// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUConformalPredictionParameterValues,
  GPUConformalPredictionIntervals,
  GPUGeoConformalPredictionIntervals,
  GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED,
  GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED,
  GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED
} from '../../../src/gpu-spatial-analysis/spatial-validation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// ---- Seeded random numbers, so every simulation replays exactly. ----
function createRandom(seed: number) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return {uniform, normal};
}

// ---- Harness: compile once, rewrite the input buffers, submit again. ----
type Layout = {
  rows: number;
  queryRows: number;
  foldCount?: number;
  folds?: boolean;
  scales?: boolean;
  mask?: boolean;
  queryScales?: boolean;
  neighbors?: number;
};
type Inputs = {
  observations: number[];
  predictions: number[];
  folds?: number[];
  scales?: number[];
  mask?: number[];
  queryPredictions: number[];
  queryScales?: number[];
  alpha: number;
  selfWeight?: number;
  neighborOffsets?: number[];
  neighborIndices?: number[];
  neighborWeights?: number[];
};

async function createRunner(layout: Layout, localized = false) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let next = 0;
  const input = <Format extends GPUVectorFormat>(
    length: number,
    format: Format
  ): {view: GraphDataView<Format>; buffer: Buffer} => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? new Uint32Array(length) : new Float32Array(length)
    );
    buffers.push(buffer);
    return {view: importGraphBuffer(graph, `input-${next++}`, buffer, format, length), buffer};
  };
  const output = <Format extends GPUVectorFormat>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      view: importGraphBuffer(graph, `output-${next++}`, buffer, format, length),
      read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
    };
  };
  const foldCount = layout.foldCount ?? 1;
  const observations = input(layout.rows, 'float32');
  const predictions = input(foldCount * layout.rows, 'float32');
  const folds = layout.folds ? input(layout.rows, 'uint32') : undefined;
  const scales = layout.scales ? input(layout.rows, 'float32') : undefined;
  const mask = layout.mask ? input(layout.rows, 'uint32') : undefined;
  const queryPredictions = input(foldCount * layout.queryRows, 'float32');
  const queryScales = layout.queryScales ? input(layout.queryRows, 'float32') : undefined;
  const parameters = input(4, 'float32');
  const intervals = output(layout.queryRows * 2, 'float32');
  const status = output(layout.queryRows, 'uint32');
  const calibration = {
    observations: observations.view,
    predictions: predictions.view,
    folds: folds?.view,
    scales: scales?.view,
    mask: mask?.view
  };
  const query = {predictions: queryPredictions.view, scales: queryScales?.view};
  const count = output(1, 'uint32');
  const neighborOffsets = localized ? input(layout.queryRows + 1, 'uint32') : undefined;
  const neighborIndices = localized ? input(layout.neighbors ?? 1, 'uint32') : undefined;
  const neighborWeights = localized ? input(layout.neighbors ?? 1, 'float32') : undefined;
  if (localized) {
    graph.add(
      new GPUGeoConformalPredictionIntervals({
        calibration,
        query,
        neighbors: {
          offsets: neighborOffsets!.view,
          indices: neighborIndices!.view,
          weights: neighborWeights!.view
        },
        parameters: parameters.view,
        output: {intervals: intervals.view, status: status.view}
      })
    );
  } else {
    graph.add(
      new GPUConformalPredictionIntervals({
        calibration,
        query,
        foldCount,
        parameters: parameters.view,
        output: {intervals: intervals.view, status: status.view, calibrationCount: count.view}
      })
    );
  }
  const compiled = graph.compile();
  return {
    async execute(inputs: Inputs) {
      observations.buffer.write(new Float32Array(inputs.observations));
      predictions.buffer.write(new Float32Array(inputs.predictions));
      folds?.buffer.write(new Uint32Array(inputs.folds!));
      scales?.buffer.write(new Float32Array(inputs.scales!));
      mask?.buffer.write(new Uint32Array(inputs.mask!));
      queryPredictions.buffer.write(new Float32Array(inputs.queryPredictions));
      queryScales?.buffer.write(new Float32Array(inputs.queryScales!));
      parameters.buffer.write(
        getGPUConformalPredictionParameterValues({
          miscoverage: inputs.alpha,
          selfWeight: inputs.selfWeight
        })
      );
      neighborOffsets?.buffer.write(new Uint32Array(inputs.neighborOffsets!));
      neighborIndices?.buffer.write(new Uint32Array(inputs.neighborIndices!));
      neighborWeights?.buffer.write(new Float32Array(inputs.neighborWeights!));
      submitGraph(device, compiled, undefined);
      return {
        intervals: await intervals.read(),
        status: await status.read(),
        count: (await count.read())[0]
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

// ---- Independent binary64 references. ----
const f = Math.fround;

/**
 * Brute force: every candidate sorted, with independent binary64 rank arithmetic.
 * A float32 significand times these small fixture counts (at most 1000) is exact in binary64.
 */
function referenceIntervals(
  layout: Layout,
  inputs: Inputs
): {intervals: number[]; status: number[]; count: number} {
  const foldCount = layout.foldCount ?? 1;
  const used: {score: number; fold: number}[] = [];
  for (let row = 0; row < layout.rows; row++) {
    const fold = inputs.folds ? inputs.folds[row] : 0;
    if (fold >= foldCount || (inputs.mask && inputs.mask[row] === 0)) continue;
    const prediction = inputs.predictions[fold * layout.rows + row];
    const scale = inputs.scales ? inputs.scales[row] : 1;
    const observation = inputs.observations[row];
    if (![observation, prediction, scale].every(Number.isFinite) || !(scale > 0)) continue;
    used.push({score: f(f(Math.abs(f(observation - prediction))) / scale), fold});
  }
  const n = used.length;
  const lowerRank = Math.floor(Math.fround(inputs.alpha) * (n + 1));
  const upperRank = n + 1 - lowerRank;
  const intervals: number[] = [];
  const status: number[] = [];
  for (let query = 0; query < layout.queryRows; query++) {
    const scale = inputs.queryScales ? inputs.queryScales[query] : 1;
    const predictionsOfQuery = Array.from(
      {length: foldCount},
      (_, fold) => inputs.queryPredictions[fold * layout.queryRows + query]
    );
    if (!predictionsOfQuery.every(Number.isFinite) || !(scale > 0)) {
      intervals.push(NaN, NaN);
      status.push(4);
      continue;
    }
    if (n === 0) {
      intervals.push(NaN, NaN);
      status.push(8);
      continue;
    }
    const uppers = used
      .map(({score, fold}) => f(predictionsOfQuery[fold] + f(scale * score)))
      .sort((left, right) => left - right);
    const lowers = used
      .map(({score, fold}) => f(predictionsOfQuery[fold] - f(scale * score)))
      .sort((left, right) => left - right);
    let code = 0;
    let upper = Infinity;
    let lower = -Infinity;
    if (upperRank > n) code |= GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED;
    else upper = uppers[upperRank - 1];
    if (lowerRank < 1) code |= GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED;
    else lower = lowers[lowerRank - 1];
    intervals.push(lower, upper);
    status.push(code);
  }
  return {intervals, status, count: n};
}

function buildRandomInputs(
  layout: Layout,
  random: ReturnType<typeof createRandom>,
  alpha: number
): Inputs {
  const foldCount = layout.foldCount ?? 1;
  const powersOfTwo = [0.5, 1, 2, 4];
  const rows = layout.rows;
  return {
    alpha,
    observations: Array.from({length: rows}, () => f(random.normal() * 3)),
    predictions: Array.from({length: foldCount * rows}, () => f(random.normal())),
    folds: layout.folds
      ? Array.from({length: rows}, () => Math.floor(random.uniform() * (foldCount + 1)))
      : undefined,
    scales: layout.scales
      ? Array.from({length: rows}, () => powersOfTwo[Math.floor(random.uniform() * 4)])
      : undefined,
    mask: layout.mask
      ? Array.from({length: rows}, () => (random.uniform() < 0.8 ? 1 : 0))
      : undefined,
    queryPredictions: Array.from({length: foldCount * layout.queryRows}, () =>
      f(random.normal() * 2)
    ),
    queryScales: layout.queryScales
      ? Array.from({length: layout.queryRows}, () => powersOfTwo[Math.floor(random.uniform() * 4)])
      : undefined
  };
}

it('split conformal equals the sorted-score reference, including exact integer ranks', async () => {
  // The packed alpha 2/5 with 24 or 44 scores gives exact ranks 15 or 27 without rounding 1 - alpha;
  // alpha 1/10 with 99 scores gives rank 90 exactly; alpha 1/20 with 19 gives 19; alpha 1/4 with 7 gives 6.
  for (const [rows, alphaRational] of [
    [99, [1, 10]],
    [19, [1, 20]],
    [7, [1, 4]],
    [24, [2, 5]],
    [44, [2, 5]],
    [40, [1, 10]]
  ] as const) {
    const layout: Layout = {rows, queryRows: 64, queryScales: true};
    const runner = await createRunner(layout);
    try {
      const random = createRandom(rows);
      const inputs = buildRandomInputs(layout, random, alphaRational[0] / alphaRational[1]);
      const actual = await runner.execute(inputs);
      const expected = referenceIntervals(layout, inputs);
      expect(actual.count).toBe(expected.count);
      expect(actual.status).toEqual(expected.status);
      expect(actual.intervals).toEqual(expected.intervals);
    } finally {
      runner.destroy();
    }
  }
});

it('selects interval endpoints from exact packed-alpha ranks at decimal boundaries', async () => {
  const runner = await createRunner({rows: 9, queryRows: 1});
  try {
    for (const [alpha, endpoint] of [
      [0.7, 4],
      [0.9, 2]
    ]) {
      const actual = await runner.execute({
        alpha,
        observations: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        predictions: new Array(9).fill(0),
        queryPredictions: [0]
      });
      expect(actual.count).toBe(9);
      expect(actual.status).toEqual([0]);
      expect(actual.intervals).toEqual([-endpoint, endpoint]);
    }
  } finally {
    runner.destroy();
  }
});

it('CV+ equals the brute-force order statistics with folds, scales and masks', async () => {
  for (const [foldCount, rows, alphaRational] of [
    [3, 60, [1, 10]],
    [5, 100, [1, 5]],
    [64, 200, [1, 10]]
  ] as const) {
    const layout: Layout = {
      rows,
      queryRows: 48,
      foldCount,
      folds: true,
      scales: true,
      mask: true,
      queryScales: true
    };
    const runner = await createRunner(layout);
    try {
      const random = createRandom(foldCount);
      const inputs = buildRandomInputs(layout, random, alphaRational[0] / alphaRational[1]);
      inputs.observations[3] = NaN;
      inputs.queryPredictions[5] = NaN;
      const actual = await runner.execute(inputs);
      const expected = referenceIntervals(layout, inputs);
      expect(actual.count).toBe(expected.count);
      expect(actual.status).toEqual(expected.status);
      expect(actual.intervals).toEqual(expected.intervals);
      expect(actual.status[5]).toBe(4);
    } finally {
      runner.destroy();
    }
  }
});

it('keeps both sides unbounded at the packed 0.02 boundary with 49 scores (R4)', async () => {
  const layout: Layout = {rows: 49, queryRows: 1};
  const inputs: Inputs = {
    alpha: 0.02,
    observations: Array.from({length: 49}, (_, row) => row + 1),
    predictions: new Array(49).fill(0),
    queryPredictions: [0]
  };
  const expected = referenceIntervals(layout, inputs);
  expect(expected).toEqual({intervals: [-Infinity, Infinity], status: [3], count: 49});
  const runner = await createRunner(layout);
  try {
    const actual = await runner.execute(inputs);
    expect(actual.intervals).toEqual([-Infinity, Infinity]);
    expect(actual.status).toEqual([3]);
    expect(actual.count).toBe(49);
  } finally {
    runner.destroy();
  }
});

it('retains tied scores even when coverage exceeds the distinct-score upper bound (R5)', async () => {
  const runner = await createRunner({rows: 7, queryRows: 1});
  try {
    const actual = await runner.execute({
      alpha: 0.25,
      observations: new Array(7).fill(1),
      predictions: new Array(7).fill(0),
      queryPredictions: [0]
    });
    expect(actual.intervals).toEqual([-1, 1]);
    expect(actual.status).toEqual([0]);
    expect(actual.count).toBe(7);
    // A population always equal to one is exchangeable and is covered with probability one.
    const coverage = Number(actual.intervals[0] <= 1 && actual.intervals[1] >= 1);
    expect(coverage).toBe(1);
    expect(coverage).toBeGreaterThan(1 - 0.25 + 1 / 8);
  } finally {
    runner.destroy();
  }
});

it('reports unbounded sides, empty calibration and invalid parameters', async () => {
  // Five scores with alpha 0.1: upper rank ceil(0.9 * 6) = 6 > 5 and lower rank floor(0.6) = 0.
  const layout: Layout = {rows: 5, queryRows: 2};
  const runner = await createRunner(layout);
  try {
    const result = await runner.execute({
      alpha: 0.1,
      observations: [1, 2, 3, 4, 5],
      predictions: [0, 0, 0, 0, 0],
      queryPredictions: [10, NaN]
    });
    expect(result.intervals.slice(0, 2)).toEqual([-Infinity, Infinity]);
    expect(result.status[0]).toBe(
      GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED | GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED
    );
    expect(result.status[1]).toBe(4);
    expect(result.intervals[2]).toBeNaN();
  } finally {
    runner.destroy();
  }
  const emptyLayout: Layout = {rows: 2, queryRows: 1};
  const empty = await createRunner(emptyLayout);
  try {
    const result = await empty.execute({
      alpha: 0.1,
      observations: [NaN, NaN],
      predictions: [0, 0],
      queryPredictions: [1]
    });
    expect(result.status).toEqual([8]);
    expect(result.count).toBe(0);
  } finally {
    empty.destroy();
  }
  expect(() => getGPUConformalPredictionParameterValues({miscoverage: 0})).toThrow('(0, 1)');
  expect(() => getGPUConformalPredictionParameterValues({miscoverage: 1})).toThrow('(0, 1)');
  expect(() =>
    getGPUConformalPredictionParameterValues({miscoverage: 0.1, selfWeight: -1})
  ).toThrow('selfWeight');
});

it('rejects CV+ without folds, a query shape mismatch and aliased outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const view = <Format extends GPUVectorFormat>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return importGraphBuffer(graph, `view-${buffers.length}`, buffer, format, length);
  };
  try {
    const base = {
      calibration: {observations: view(4, 'float32'), predictions: view(8, 'float32')},
      query: {predictions: view(4, 'float32')},
      foldCount: 2,
      parameters: view(4, 'float32'),
      output: {
        intervals: view(4, 'float32'),
        status: view(2, 'uint32'),
        calibrationCount: view(1, 'uint32')
      }
    };
    expect(() => new GPUConformalPredictionIntervals(base)).toThrow(
      'calibration.folds is required'
    );
    const withFolds = {...base, calibration: {...base.calibration, folds: view(4, 'uint32')}};
    expect(() => new GPUConformalPredictionIntervals(withFolds)).not.toThrow();
    expect(
      () =>
        new GPUConformalPredictionIntervals({
          ...withFolds,
          output: {...withFolds.output, calibrationCount: withFolds.output.status as never}
        })
    ).toThrow();
    expect(
      () =>
        new GPUConformalPredictionIntervals({
          ...withFolds,
          calibration: {...withFolds.calibration, predictions: view(7, 'float32')}
        })
    ).toThrow('calibration.predictions');
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

// ---- Localized (GeoConformal) intervals. ----
it('spatial-4:F4 equal-weight localized and global intervals share both unbounded status bits', async () => {
  const rows = 49;
  const layout: Layout = {rows, queryRows: 1};
  const global = await createRunner(layout);
  const localized = await createRunner({...layout, neighbors: rows}, true);
  try {
    const inputs: Inputs = {
      alpha: 0.02,
      observations: Array.from({length: rows}, (_, row) => row + 1),
      predictions: new Array(rows).fill(0),
      queryPredictions: [0]
    };
    const globalResult = await global.execute(inputs);
    for (const selfWeight of [1, 0.1, 1e37]) {
      const localizedResult = await localized.execute({
        ...inputs,
        selfWeight,
        neighborOffsets: [0, rows],
        neighborIndices: Array.from({length: rows}, (_, row) => row),
        neighborWeights: new Array(rows).fill(selfWeight)
      });
      expect(globalResult.intervals).toEqual([-Infinity, Infinity]);
      expect(globalResult.status).toEqual([
        GPU_CONFORMAL_STATUS_LOWER_UNBOUNDED | GPU_CONFORMAL_STATUS_UPPER_UNBOUNDED
      ]);
      expect(localizedResult.intervals).toEqual(globalResult.intervals);
      expect(localizedResult.status).toEqual(globalResult.status);
    }
  } finally {
    global.destroy();
    localized.destroy();
  }
});

it('localized intervals equal the weighted-quantile reference and, with equal weights, the global result', async () => {
  const rows = 39;
  const layout: Layout = {rows, queryRows: 16, scales: true, queryScales: true};
  const random = createRandom(7);
  const inputs = buildRandomInputs(layout, random, 0.25);
  // Every query sees every calibration row at weight 1 and selfWeight 1: the global split result.
  const offsets = Array.from({length: 17}, (_, index) => index * rows);
  const indices = Array.from({length: 16 * rows}, (_, index) => index % rows);
  const global = await createRunner(layout);
  const localized = await createRunner({...layout, neighbors: indices.length}, true);
  try {
    const uniform = {
      ...inputs,
      neighborOffsets: offsets,
      neighborIndices: indices,
      neighborWeights: indices.map(() => 1),
      selfWeight: 1
    };
    const globalResult = await global.execute(inputs);
    const localizedResult = await localized.execute(uniform);
    // 39 scores, alpha 1/4: rank ceil(0.75 * 40) = 30 exactly, so the quantile is a hit, not a tie-break.
    expect(localizedResult.intervals).toEqual(globalResult.intervals);
    expect(localizedResult.status).toEqual(globalResult.status);
  } finally {
    global.destroy();
    localized.destroy();
  }

  // Dyadic weights make the float32 cumulative sums exact, so the binary64 reference must match.
  const weighted = await createRunner({...layout, neighbors: 16 * 12}, true);
  try {
    const weightRandom = createRandom(11);
    const neighborIndices: number[] = [];
    const neighborWeights: number[] = [];
    const neighborOffsets = [0];
    for (let query = 0; query < 16; query++) {
      for (let entry = 0; entry < 12; entry++) {
        neighborIndices.push(Math.floor(weightRandom.uniform() * rows));
        neighborWeights.push(Math.floor(weightRandom.uniform() * 9) / 8);
      }
      neighborOffsets.push(neighborIndices.length);
    }
    const selfWeight = 0.5;
    const result = await weighted.execute({
      ...inputs,
      neighborOffsets,
      neighborIndices,
      neighborWeights,
      selfWeight
    });
    const scores = inputs.observations.map((observation, row) =>
      f(f(Math.abs(f(observation - inputs.predictions[row]))) / inputs.scales![row])
    );
    for (let query = 0; query < 16; query++) {
      const entries = neighborIndices
        .slice(neighborOffsets[query], neighborOffsets[query + 1])
        .map((row, position) => ({
          score: scores[row],
          weight: neighborWeights[neighborOffsets[query] + position]
        }))
        .sort((left, right) => left.score - right.score);
      const total = selfWeight + entries.reduce((sum, entry) => sum + entry.weight, 0);
      let cumulative = 0;
      let quantile = Infinity;
      for (const [position, entry] of entries.entries()) {
        cumulative += entry.weight;
        const tied = entries[position + 1]?.score === entry.score;
        if (!tied && cumulative >= 0.75 * total) {
          quantile = entry.score;
          break;
        }
      }
      const scale = inputs.queryScales![query];
      const prediction = inputs.queryPredictions[query];
      const width = f(scale * quantile);
      expect(result.intervals[2 * query]).toBe(f(prediction - width));
      expect(result.intervals[2 * query + 1]).toBe(f(prediction + width));
    }
  } finally {
    weighted.destroy();
  }
});

it('reports truncated neighbour lists', async () => {
  const rows = 300;
  const layout: Layout = {rows, queryRows: 1, neighbors: rows};
  const runner = await createRunner(layout, true);
  try {
    const result = await runner.execute({
      alpha: 0.25,
      observations: Array.from({length: rows}, (_, row) => row),
      predictions: new Array(rows).fill(0),
      queryPredictions: [0],
      neighborOffsets: [0, rows],
      neighborIndices: Array.from({length: rows}, (_, row) => row),
      neighborWeights: new Array(rows).fill(1)
    });
    expect(result.status[0] & GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED).toBe(
      GPU_CONFORMAL_STATUS_NEIGHBORS_TRUNCATED
    );
    // The first 256 entries are scores 0 to 255; rank ceil(0.75 * 257) = 193 gives score 192.
    expect(result.intervals).toEqual([-192, 192]);
  } finally {
    runner.destroy();
  }
});

// ---- Coverage by simulation. The graph compiles once; each replicate rewrites the buffers. ----
function getSplitCoverageTolerance(
  rows: number,
  rank: number,
  queries: number,
  replicates: number
) {
  // Calibration coverage C is Beta(rank, rows + 1 - rank); a replicate covers a fraction of
  // `queries` test points with mean p and variance p (1 - p) / m + (1 - 1 / m) Var(C).
  const p = rank / (rows + 1);
  const varianceOfC = (rank * (rows + 1 - rank)) / ((rows + 1) ** 2 * (rows + 2));
  const variance = (p * (1 - p)) / queries + (1 - 1 / queries) * varianceOfC;
  return {p, standardError: Math.sqrt(variance / replicates)};
}

for (const [rows, alpha, rank] of [
  [99, 0.1, 90],
  [19, 0.05, 19]
] as const) {
  it(`split coverage over replicates matches Beta(${rank}, ${rows + 1 - rank}) within 4.5 standard errors (n ${rows}, alpha ${alpha})`, async () => {
    const queries = 1000;
    const replicates = 1000;
    const layout: Layout = {rows, queryRows: queries};
    const runner = await createRunner(layout);
    const random = createRandom(2024 + rows);
    try {
      let total = 0;
      for (let replicate = 0; replicate < replicates; replicate++) {
        const result = await runner.execute({
          alpha,
          observations: Array.from({length: rows}, () => f(random.normal())),
          predictions: new Array(rows).fill(0),
          queryPredictions: new Array(queries).fill(0)
        });
        const lower = result.intervals[0];
        const upper = result.intervals[1];
        let covered = 0;
        for (let query = 0; query < queries; query++) {
          const outcome = f(random.normal());
          if (outcome >= lower && outcome <= upper) covered++;
        }
        total += covered / queries;
      }
      const {p, standardError} = getSplitCoverageTolerance(rows, rank, queries, replicates);
      expect(Math.abs(total / replicates - p)).toBeLessThan(4.5 * standardError);
      expect(p).toBeGreaterThanOrEqual(1 - alpha - 1e-12);
    } finally {
      runner.destroy();
    }
  }, 240000);
}

it('CV+ coverage is at least 1 - 2 alpha over replicates', async () => {
  const foldCount = 5;
  const rows = 100;
  const queries = 400;
  const replicates = 200;
  const alpha = 0.1;
  const layout: Layout = {rows, queryRows: queries, foldCount, folds: true};
  const runner = await createRunner(layout);
  const random = createRandom(99);
  try {
    const coverages: number[] = [];
    const slopes = [0.9, 0.95, 1, 1.05, 1.1];
    for (let replicate = 0; replicate < replicates; replicate++) {
      const x = Array.from({length: rows}, () => random.normal());
      const folds = x.map((_, row) => row % foldCount);
      const queryX = Array.from({length: queries}, () => random.normal());
      const result = await runner.execute({
        alpha,
        folds,
        observations: x.map(value => f(value + random.normal())),
        predictions: slopes.flatMap(slope => x.map(value => f(slope * value))),
        queryPredictions: slopes.flatMap(slope => queryX.map(value => f(slope * value)))
      });
      let covered = 0;
      for (let query = 0; query < queries; query++) {
        const outcome = f(queryX[query] + random.normal());
        if (outcome >= result.intervals[2 * query] && outcome <= result.intervals[2 * query + 1]) {
          covered++;
        }
      }
      coverages.push(covered / queries);
    }
    const mean = coverages.reduce((sum, value) => sum + value, 0) / replicates;
    const variance =
      coverages.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (replicates - 1);
    const standardError = Math.sqrt(variance / replicates);
    expect(mean).toBeGreaterThan(1 - 2 * alpha - 4.5 * standardError);
  } finally {
    runner.destroy();
  }
}, 240000);

it('localized intervals cover each region where global intervals do not', async () => {
  const rows = 1000;
  const replicates = 200;
  const alpha = 0.1;
  const random = createRandom(31);
  const calibrationX = Array.from({length: rows}, () => random.uniform());
  const standardDeviation = (x: number) => (x < 0.5 ? 0.2 : 2);
  // Test locations stay at least 0.2 from the change at 0.5, beyond the neighbour radius of 0.1.
  const queryX = [
    ...Array.from({length: 200}, (_, index) => 0.05 + (0.25 * index) / 200),
    ...Array.from({length: 200}, (_, index) => 0.7 + (0.25 * index) / 200)
  ];
  const queryRows = queryX.length;
  const bandwidth = 0.05;
  const neighborOffsets = [0];
  const neighborIndices: number[] = [];
  const neighborWeights: number[] = [];
  for (const x of queryX) {
    calibrationX.forEach((other, row) => {
      const distance = Math.abs(other - x);
      if (distance <= 0.1) {
        neighborIndices.push(row);
        neighborWeights.push(f(Math.exp(-0.5 * (distance / bandwidth) ** 2)));
      }
    });
    neighborOffsets.push(neighborIndices.length);
  }
  const global = await createRunner({rows, queryRows});
  const localized = await createRunner({rows, queryRows, neighbors: neighborIndices.length}, true);
  try {
    const regionCoverage = {global: [[], []] as number[][], localized: [[], []] as number[][]};
    for (let replicate = 0; replicate < replicates; replicate++) {
      const inputs: Inputs = {
        alpha,
        observations: calibrationX.map(x => f(standardDeviation(x) * random.normal())),
        predictions: new Array(rows).fill(0),
        queryPredictions: new Array(queryRows).fill(0),
        neighborOffsets,
        neighborIndices,
        neighborWeights,
        selfWeight: 1
      };
      const outcomes = queryX.map(x => f(standardDeviation(x) * random.normal()));
      for (const [name, runner] of [
        ['global', global],
        ['localized', localized]
      ] as const) {
        const result = await runner.execute(inputs);
        expect(result.status.every(code => code === 0)).toBe(true);
        for (const region of [0, 1]) {
          let covered = 0;
          for (let query = region * 200; query < region * 200 + 200; query++) {
            if (
              outcomes[query] >= result.intervals[2 * query] &&
              outcomes[query] <= result.intervals[2 * query + 1]
            ) {
              covered++;
            }
          }
          regionCoverage[name][region].push(covered / 200);
        }
      }
    }
    const mean = (values: number[]) =>
      values.reduce((sum, value) => sum + value, 0) / values.length;
    const standardError = (values: number[]) => {
      const center = mean(values);
      return Math.sqrt(
        values.reduce((sum, value) => sum + (value - center) ** 2, 0) /
          (values.length - 1) /
          values.length
      );
    };
    // The global quantile mixes sigma 0.2 and 2: it over-covers the quiet region and under-covers the noisy one.
    expect(mean(regionCoverage.global[0])).toBeGreaterThan(0.97);
    expect(mean(regionCoverage.global[1])).toBeLessThan(0.85);
    for (const region of [0, 1]) {
      const values = regionCoverage.localized[region];
      // Weighted conformal coverage is at least 1 - alpha and at most 1 - alpha + 1 / (sum w + selfWeight),
      // about 0.008 here; allow 4.5 standard errors of the replicate mean around that band.
      expect(mean(values)).toBeGreaterThan(1 - alpha - 4.5 * standardError(values));
      expect(mean(values)).toBeLessThan(1 - alpha + 0.01 + 4.5 * standardError(values));
    }
  } finally {
    global.destroy();
    localized.destroy();
  }
}, 600000);
