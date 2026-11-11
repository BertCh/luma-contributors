// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The exact rank of `GPUColumnQuantiles` with `rationalProbabilities` (math review 2026-11-10,
 * dataframe-08), against `test/oracles/fixtures/dataframe-quantile-rational-rank.json`: NumPy
 * `quantile` and the definition on Python integers and `fractions.Fraction` (see the generator
 * `test/oracles/dataframe-quantile-rational-rank.py`). No expectation here repeats the WGSL.
 */

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {UINT64_WGSL} from '../../../src/analysis-core/kernels/uint64-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesRationalProbabilityLength,
  getGPUColumnQuantilesRationalProbabilityValues,
  type GPUColumnQuantileInterpolation,
  type GPUColumnQuantileRationalProbability
} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {GPUColumnQuantiles} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import {RATIONAL_ORDER_STATISTIC_RANK_WGSL} from '../../../src/dataframe-analysis/column-classification/order-statistic-wgsl';
import fixtureJson from '../../oracles/fixtures/dataframe-quantile-rational-rank.json';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type References = {
  count: number;
  exact: Record<GPUColumnQuantileInterpolation, number[]>;
  numpyDifferences: Record<GPUColumnQuantileInterpolation, [number, number][]>;
};

type Fixture = {
  versions: {python: string; numpy: string};
  methods: GPUColumnQuantileInterpolation[];
  small: {
    values: number[];
    probabilities: GPUColumnQuantileRationalProbability[];
    cases: References[];
    caseCount: number;
    numpyDifferenceCount: number;
  };
  float32Path: {
    comparedCount: number;
    floorDifferenceCount: number;
    pinned: {
      count: number;
      probability: GPUColumnQuantileRationalProbability;
      float32Floor: number;
      numpyFloor: number;
      numpyLower: number;
      numpyLinear: number;
    }[];
  };
  large: {
    probabilities: GPUColumnQuantileRationalProbability[];
    cases: (References & {
      thresholds: number[];
      float32PathLinearAtFirstProbability: number | null;
    })[];
  };
  rank: {cases: number[][]};
};

const fixture = fixtureJson as unknown as Fixture;
const QUANTILE_COUNT = 64;

type Runner = {
  /** Runs one frame and returns the quantile slots and the valid count. */
  run(frame: {
    interpolation: GPUColumnQuantileInterpolation;
    rational?: readonly GPUColumnQuantileRationalProbability[];
    quantiles?: number[];
    mask?: Uint32Array;
  }): Promise<{quantiles: number[]; validCount: number}>;
  destroy(): void;
};

/** One compiled `GPUColumnQuantiles` graph; probabilities, interpolation and mask are per frame. */
function createRunner(
  device: Device,
  scene: {values: Float32Array; withMask: boolean; withRational: boolean}
): Runner {
  const rows = scene.values.length;
  const graph = new GPUCommandGraph(device, {id: 'rational-quantiles-graph'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const valuesBuffer = track(createInputBuffer(device, scene.values));
  const maskBuffer = scene.withMask
    ? track(createInputBuffer(device, new Uint32Array(rows)))
    : undefined;
  const quantilesBuffer = track(createOutputBuffer(device, QUANTILE_COUNT));
  const validCountBuffer = track(createOutputBuffer(device, 1));
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'rational-quantile-parameters',
    format: 'float32',
    length: getGPUColumnQuantilesParameterLength(QUANTILE_COUNT)
  });
  const rationalBuffer = new GPUParameterBuffer(device, {
    id: 'rational-quantile-probabilities',
    format: 'uint32',
    length: getGPUColumnQuantilesRationalProbabilityLength(QUANTILE_COUNT)
  });
  graph.add(
    new GPUColumnQuantiles({
      id: 'rational-quantiles',
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
      mask: maskBuffer ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows) : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      rationalProbabilities: scene.withRational ? rationalBuffer.importToGraph(graph) : undefined,
      quantileCount: QUANTILE_COUNT,
      output: {
        quantiles: importGraphBuffer(
          graph,
          'o-quantiles',
          quantilesBuffer,
          'float32',
          QUANTILE_COUNT
        ),
        validCount: importGraphBuffer(graph, 'o-valid', validCountBuffer, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(frame) {
      // With rational probabilities the float32 slots are ignored: fill them with a value that
      // would give another rank, so a kernel that still read them would fail.
      parameterBuffer.write(
        getGPUColumnQuantilesParameterValues({
          quantiles: frame.quantiles ?? new Array<number>(QUANTILE_COUNT).fill(0.123),
          interpolation: frame.interpolation,
          quantileCount: QUANTILE_COUNT
        })
      );
      if (scene.withRational) {
        rationalBuffer.write(
          getGPUColumnQuantilesRationalProbabilityValues(frame.rational ?? [], QUANTILE_COUNT)
        );
      }
      if (maskBuffer && frame.mask) {
        maskBuffer.write(frame.mask);
      }
      submitGraph(device, compiled, undefined);
      return {
        quantiles: await readFloat32(quantilesBuffer, QUANTILE_COUNT),
        validCount: (await readUint32(validCountBuffer, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      rationalBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

it('the rational rank WGSL equals exact integer arithmetic for counts up to 2^32 - 1', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = fixture.rank.cases;
  expect(cases.length).toBe(480);
  expect(Math.max(...cases.map(([count]) => count))).toBe(2 ** 32 - 1);
  const graph = new GPUCommandGraph<undefined>(device, {id: 'rational-rank-graph'});
  const inputBuffer = createInputBuffer(
    device,
    Uint32Array.from(
      cases.flatMap(([count, numerator, denominator]) => [count, numerator, denominator])
    )
  );
  const outputBuffer = createOutputBuffer(device, cases.length * 5);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'rational-rank',
      operation: 'RationalRankSpec',
      bindings: [
        {
          name: 'triples',
          view: importGraphBuffer(graph, 'triples', inputBuffer, 'uint32', cases.length * 3),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'ranks',
          view: importGraphBuffer(graph, 'ranks', outputBuffer, 'uint32', cases.length * 5),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: cases.length,
      declarations: `${UINT64_WGSL}
${RATIONAL_ORDER_STATISTIC_RANK_WGSL}`,
      body: `let count = triples[triplesOffset + 3u * index];
  let numerator = triples[triplesOffset + 3u * index + 1u];
  let denominator = triples[triplesOffset + 3u * index + 2u];
  let rank = getRationalOrderStatisticRank(count, numerator, denominator);
  ranks[ranksOffset + 5u * index] = rank.lower;
  ranks[ranksOffset + 5u * index + 1u] = rank.remainder;
  ranks[ranksOffset + 5u * index + 2u] = bitcast<u32>(getRationalOrderStatisticFraction(rank.remainder, denominator));
  ranks[ranksOffset + 5u * index + 3u] = getRationalOrderStatisticUpperRank(rank);
  ranks[ranksOffset + 5u * index + 4u] = getRationalOrderStatisticNearestRank(rank, denominator);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readUint32(outputBuffer, cases.length * 5);
    cases.forEach(([count, numerator, denominator, lower, remainder, fractionBits], index) => {
      const label = `count ${count}, p ${numerator}/${denominator}`;
      expect(actual[5 * index], `${label} lower`).toBe(lower);
      expect(actual[5 * index + 1], `${label} remainder`).toBe(remainder);
      expect(actual[5 * index + 2], `${label} fraction bits`).toBe(fractionBits);
      expect(actual[5 * index + 3], `${label} upper`).toBe(lower + (remainder === 0 ? 0 : 1));
      // Round half to even on exact integers: 2 * remainder against the denominator.
      const doubled = 2n * BigInt(remainder);
      const nearest =
        doubled < BigInt(denominator)
          ? lower
          : doubled > BigInt(denominator)
            ? lower + 1
            : lower + (lower % 2);
      expect(actual[5 * index + 4], `${label} nearest`).toBe(nearest);
    });
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});

/** Runs every probability of `references` in frames of 64 and compares with the exact values. */
async function expectMatchesReferences(
  runner: Runner,
  references: References,
  probabilities: readonly GPUColumnQuantileRationalProbability[],
  options: {mask?: Uint32Array; linearTolerance: number}
): Promise<void> {
  for (const interpolation of fixture.methods) {
    for (let start = 0; start < probabilities.length; start += QUANTILE_COUNT) {
      const chunk = probabilities.slice(start, start + QUANTILE_COUNT);
      const actual = await runner.run({interpolation, rational: chunk, mask: options.mask});
      expect(actual.validCount).toBe(references.count);
      chunk.forEach(([numerator, denominator], offset) => {
        const expected = references.exact[interpolation][start + offset];
        const label = `n ${references.count} ${interpolation} p ${numerator}/${denominator}`;
        if (interpolation === 'linear') {
          expect(Math.abs(actual.quantiles[offset] - expected), label).toBeLessThanOrEqual(
            options.linearTolerance
          );
        } else {
          // An order statistic, or the mean of two values with spare float32 bits: exact.
          expect(actual.quantiles[offset], label).toBe(expected);
        }
      });
      // Unused slots hold the ratio 0 / 0, which is not a probability.
      for (let slot = chunk.length; slot < QUANTILE_COUNT; slot++) {
        expect(actual.quantiles[slot]).toBeNaN();
      }
    }
  }
}

it('rationalProbabilities matches NumPy and exact rationals at small n over 256 fractions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  expect(fixture.versions.numpy).toBe('2.5.4');
  const {values, probabilities, cases} = fixture.small;
  expect(probabilities.length).toBe(256);
  // NumPy's float64 virtual index crosses an integer in these cases (lower, higher, midpoint);
  // everywhere else the generator asserted that numpy equals the exact value compared here.
  expect(fixture.small.caseCount).toBe(34560);
  expect(fixture.small.numpyDifferenceCount).toBe(22);
  expect(cases.every(entry => entry.numpyDifferences.linear.length === 0)).toBe(true);
  const runner = createRunner(device, {
    values: Float32Array.from(values),
    withMask: true,
    withRational: true
  });
  try {
    for (const references of cases) {
      const mask = new Uint32Array(values.length).fill(1, 0, references.count);
      // Values are multiples of 1/8 below 1,001: the fraction and two float32 roundings.
      await expectMatchesReferences(runner, references, probabilities, {
        mask,
        linearTolerance: 4e-4
      });
    }
  } finally {
    runner.destroy();
  }
}, 240_000);

it('a ratio that is not a probability gives NaN, and the float32 slots are ignored', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Float32Array.from({length: 11}, (_, index) => 10 * index);
  const runner = createRunner(device, {values, withMask: false, withRational: true});
  try {
    const actual = await runner.run({
      interpolation: 'linear',
      rational: [
        [1, 2],
        [3, 2],
        [1, 4],
        [19, 20],
        [0, 7],
        [7, 7]
      ],
      // Deliberately different float32 probabilities: they must not be read.
      quantiles: new Array<number>(QUANTILE_COUNT).fill(0.9)
    });
    // Hand computation on x = 0, 10, ..., 100: h = 10 p.
    expect(actual.quantiles.slice(0, 6)).toEqual([50, NaN, 25, 95, 0, 100]);
    expect(actual.quantiles.slice(6).every(Number.isNaN)).toBe(true);
    expect(actual.validCount).toBe(11);
  } finally {
    runner.destroy();
  }
});

it('the default float32 rank keeps its results where it differs from NumPy; the rational rank matches NumPy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {values} = fixture.small;
  expect(fixture.float32Path.floorDifferenceCount).toBe(40);
  expect(fixture.float32Path.comparedCount).toBe(6912);
  expect(fixture.float32Path.pinned.length).toBeGreaterThan(0);
  const column = Float32Array.from(values);
  const float32Runner = createRunner(device, {values: column, withMask: true, withRational: false});
  const rationalRunner = createRunner(device, {values: column, withMask: true, withRational: true});
  try {
    for (const pinned of fixture.float32Path.pinned) {
      const [numerator, denominator] = pinned.probability;
      const mask = new Uint32Array(values.length).fill(1, 0, pinned.count);
      const sorted = values.slice(0, pinned.count).sort((first, second) => first - second);
      const label = `n ${pinned.count} p ${numerator}/${denominator}`;
      expect(pinned.float32Floor, label).not.toBe(pinned.numpyFloor);
      const float32Result = await float32Runner.run({
        interpolation: 'lower',
        quantiles: [numerator / denominator],
        mask
      });
      // Today's result of the default path: the order statistic one rank from numpy's.
      expect(float32Result.quantiles[0], label).toBe(sorted[pinned.float32Floor]);
      expect(float32Result.quantiles[0], label).not.toBe(pinned.numpyLower);
      for (const [interpolation, expected] of [
        ['lower', pinned.numpyLower],
        ['linear', pinned.numpyLinear]
      ] as const) {
        const rationalResult = await rationalRunner.run({
          interpolation,
          rational: [pinned.probability],
          mask
        });
        expect(
          Math.abs(rationalResult.quantiles[0] - expected),
          `${label} ${interpolation}`
        ).toBeLessThanOrEqual(interpolation === 'lower' ? 0 : 4e-4);
      }
    }
  } finally {
    float32Runner.destroy();
    rationalRunner.destroy();
  }
});

/** The step column of a large case: row `i` holds the number of thresholds below `i`. */
function createStepColumn(count: number, thresholds: number[]): Float32Array {
  const values = new Float32Array(count);
  let level = 0;
  for (let row = 0; row < count; row++) {
    while (level < thresholds.length && thresholds[level] < row) {
      level++;
    }
    values[row] = level;
  }
  return values;
}

for (const references of fixture.large.cases) {
  it(`rationalProbabilities matches NumPy at ${references.count} rows, where the float32 rank has no fraction`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {count} = references;
    expect(fixture.methods.every(method => references.numpyDifferences[method].length === 0)).toBe(
      true
    );
    // The column must fit one storage binding; the default limit holds 2^25 rows.
    expect(count * Float32Array.BYTES_PER_ELEMENT).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
    const values = createStepColumn(count, references.thresholds);
    const runner = createRunner(device, {values, withMask: false, withRational: true});
    try {
      // Results are level + fraction with level at most 20: two float32 roundings near 16.
      await expectMatchesReferences(runner, references, fixture.large.probabilities, {
        linearTolerance: 4e-6
      });
    } finally {
      runner.destroy();
    }
    if (references.float32PathLinearAtFirstProbability !== null) {
      // The defect of dataframe-08, kept on the default path: at 2^24 rows and p = 0.95 the
      // float32 rank has no fraction, so `linear` returns x[lo]; numpy interpolates by 0.25.
      const [numerator, denominator] = fixture.large.probabilities[0];
      const float32Runner = createRunner(device, {values, withMask: false, withRational: false});
      try {
        const actual = await float32Runner.run({
          interpolation: 'linear',
          quantiles: [numerator / denominator]
        });
        expect(actual.quantiles[0]).toBe(references.float32PathLinearAtFirstProbability);
        expect(references.exact.linear[0] - actual.quantiles[0]).toBe(0.25);
      } finally {
        float32Runner.destroy();
      }
    }
  }, 240_000);
}
