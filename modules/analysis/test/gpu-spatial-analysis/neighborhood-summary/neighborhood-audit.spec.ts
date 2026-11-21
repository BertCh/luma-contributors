// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  GPUNeighborhoodSummary
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectOracleRows} from '../spatial-interaction/spatial-interaction-audit-fixtures';
import {
  STATISTICS,
  INPUT,
  RAW_EXPECTED,
  STANDARDIZED_FOCAL_EXPECTED,
  INCOMPLETE_EXPECTED
} from './neighborhood-audit-fixtures';

async function run(
  input = INPUT,
  includeFocal = false,
  maximumNeighbors = 32,
  status: Partial<
    Record<'overflow' | 'candidateOverflow' | 'iterationLimitReached' | 'converged', number>
  > = {},
  statistics = STATISTICS,
  focalWeight = 2,
  standardized = includeFocal
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const weightsStatus = Object.fromEntries(
    Object.keys(status).map(name => [name, `weights-${name}`])
  );
  const props = {
    rowCount: 8,
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: weightsStatus,
      information: {
        rowCount: 8,
        transform: standardized ? ('row' as const) : ('raw' as const),
        symmetric: 'unknown' as const
      }
    },
    values: 'values',
    categories: 'categories',
    mask: 'mask',
    includeFocal,
    focalWeight,
    maximumNeighbors,
    statistics: [...statistics],
    output: {table: 'table', modes: 'modes', entropy: 'entropy', status: {overflow: 'overflow'}}
  };
  const operation = new GPUNeighborhoodSummaryOperation(props);
  props.includeFocal = !includeFocal;
  return runOperationGraph(device, {
    inputs: {
      offsets: {format: 'uint32', values: input.offsets},
      neighbors: {format: 'uint32', values: input.neighbors},
      weights: {format: 'float32', values: input.weights},
      values: {format: 'float32', values: input.values},
      categories: {format: 'uint32', values: input.categories},
      mask: {format: 'uint32', values: input.mask},
      ...Object.fromEntries(
        Object.entries(status).map(([name, value]) => [
          `weights-${name}`,
          {format: 'uint32' as const, values: Uint32Array.of(value!)}
        ])
      )
    },
    operations: [operation],
    outputs: ['table', 'modes', 'entropy', 'overflow']
  });
}

for (const maximumNeighbors of [3, 8]) {
  it(`retains dyadic category residuals with capacity ${maximumNeighbors}`, async () => {
    const actual = await run(
      {
        offsets: Uint32Array.of(0, 5, 5, 5, 5, 5, 5, 5, 5),
        neighbors: Uint32Array.of(1, 2, 3, 4, 5),
        weights: Float32Array.of(
          1 / 8,
          1 / 8 + 2 ** -26,
          1 / 8 + 2 ** -26,
          1 / 8 - 2 ** -25,
          1 / 2
        ),
        categories: Uint32Array.of(0, 1, 1, 1, 1, 2, 0, 0),
        values: new Float32Array(8),
        mask: Uint32Array.of(1, 1, 1, 1, 1, 1, 0, 0)
      },
      false,
      maximumNeighbors,
      {},
      ['count', 'mean'],
      0.5,
      true
    );
    // Exact binary inputs give two equal masses 1/2, hence mode 1 and entropy ln(2).
    expectOracleRows(
      actual['modes'],
      [1, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295],
      0
    );
    expectOracleRows(actual['entropy'], [Math.LN2, NaN, NaN, NaN, NaN, NaN, NaN, NaN], 1e-6);
  });
  it(`preserves an exact standardized category tie with capacity ${maximumNeighbors}`, async () => {
    // Equal exact raw masses 31/64; masked last slot must not enter the divisor.
    for (const includeFocal of [false, true]) {
      const actual = await run(
        {
          offsets: Uint32Array.of(0, 5, 5, 5, 5, 5, 5, 5, 5),
          neighbors: Uint32Array.of(1, 2, 3, 4, 5),
          weights: Float32Array.of(1 / 64, 9 / 64, 21 / 64, 31 / 64, 2 / 64),
          categories: Uint32Array.of(2, 1, 1, 1, 2, 3, 0, 0),
          values: new Float32Array(8),
          mask: Uint32Array.of(1, 1, 1, 1, 1, 0, 0, 0)
        },
        includeFocal,
        maximumNeighbors,
        {},
        ['count', 'mean'],
        0.5,
        true
      );
      expectOracleRows(
        actual['modes'],
        includeFocal
          ? [2, 1, 1, 1, 2, 4294967295, 4294967295, 4294967295]
          : [1, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295, 4294967295],
        0
      );
      expectOracleRows(
        actual['entropy'],
        includeFocal
          ? [0.6365141682948128, 0, 0, 0, 0, NaN, NaN, NaN]
          : [Math.LN2, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
        1e-6
      );
    }
  });
}

const NO_MODE = GPU_NEIGHBORHOOD_SUMMARY_NO_MODE;

/** Every row of a single-column result: exact values, NaN where NaN is expected. */
function expectRows(actual: ArrayLike<number>, expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, row) => {
    if (Number.isNaN(value)) expect(Number.isNaN(actual[row]), `row ${row}`).toBe(true);
    else {
      expect(Number.isFinite(actual[row]), `row ${row}`).toBe(true);
      // Adding zero folds -0 into +0: an entropy of -0 (from -1 * ln 1) is still zero.
      expect(actual[row] + 0, `row ${row}`).toBe(value + 0);
    }
  });
}

for (const maximumNeighbors of [1, 8]) {
  it(`W5-18 retains standardized unit focal mass with capacity ${maximumNeighbors}`, async () => {
    // Each row-standardized fixture lists masked neighbors carrying the rest of the unit
    // row mass, so the supplied row sums to one. Masked neighbors are skipped and the retained
    // slots are re-standardized over the active members, so the category-2 mass is exactly one,
    // tying the unit focal mass of category 1.
    for (const retained of [
      [0.5, 2 ** -27],
      [1 / 8, 1 / 8 + 2 ** -26, 1 / 8 + 2 ** -26, 1 / 8 - 2 ** -25]
    ]) {
      // 1/2 - 2^-27 is not one float32. Three masked terms represent that remainder exactly.
      const maskedWeights =
        retained.length === 2 ? [1 / 4, 1 / 8, 1 / 8 - 2 ** -27] : [1 / 2, 0, 0];
      const allWeights = [...retained, ...maskedWeights];
      expect(allWeights.reduce((total, weight) => total + weight, 0)).toBe(1);
      const slotCount = allWeights.length;
      const actual = await run(
        {
          offsets: Uint32Array.of(0, ...new Array(8).fill(slotCount)),
          neighbors: Uint32Array.of(...retained.map((_, slot) => slot + 1), 5, 6, 7),
          weights: Float32Array.from(allWeights),
          categories: Uint32Array.of(1, 2, 2, 2, 2, 0, 0, 0),
          values: new Float32Array(8),
          mask: Uint32Array.of(1, 1, 1, 1, 1, 0, 0, 0)
        },
        true,
        maximumNeighbors,
        {},
        ['count'],
        1,
        true
      );
      // Row 0: focal plus the retained slots; the tie goes to the lowest category, 1, with
      // entropy ln 2. Other unmasked rows hold only their own focal member (entropy 0); the
      // masked rows 5-7 are not summarized (count 0, no mode, NaN entropy).
      expectRows(actual['table'], [1 + retained.length, 1, 1, 1, 1, 0, 0, 0]);
      expect(Array.from(actual['modes'])).toEqual([1, 2, 2, 2, 2, NO_MODE, NO_MODE, NO_MODE]);
      expect(Number.isFinite(actual['entropy'][0])).toBe(true);
      expect(Math.abs(actual['entropy'][0] - Math.LN2)).toBeLessThan(1e-7);
      expectRows(Array.from(actual['entropy']).slice(1), [0, 0, 0, 0, NaN, NaN, NaN]);
    }
  });
  it(`W5-18 uses one mass denominator for single-category entropy with capacity ${maximumNeighbors}`, async () => {
    const actual = await run(
      {
        offsets: Uint32Array.of(0, 3, 3, 3, 3, 3, 3, 3, 3),
        neighbors: Uint32Array.of(1, 2, 3),
        weights: Float32Array.of(1, 2 ** -24, 2 ** -24),
        categories: Uint32Array.of(0, 7, 7, 7, 0, 0, 0, 0),
        values: new Float32Array(8),
        mask: new Uint32Array(8).fill(1)
      },
      false,
      maximumNeighbors,
      {},
      ['count']
    );
    // A single category has p=1 exactly, hence -p*log(p)=0. Rows 1-7 have no members: count 0,
    // no mode and NaN entropy (zero mass).
    expectRows(actual['table'], [3, 0, 0, 0, 0, 0, 0, 0]);
    expect(Array.from(actual['modes'])).toEqual([7, ...new Array(7).fill(NO_MODE)]);
    expectRows(actual['entropy'], [0, NaN, NaN, NaN, NaN, NaN, NaN, NaN]);
  });
  it(`W5-18 keeps finite large-weight variance with capacity ${maximumNeighbors}`, async () => {
    for (const [weights, values, deviation] of [
      // Ratio 1:2 gives mean -1/3 and variance 1 - (1/3)^2 = 8/9.
      [[1e38, 2e38], [1, -1], Math.sqrt(8 / 9)],
      // Ratio 1:2 gives mean (-4 + 4) / 3 = 0 and variance (16 + 2 * 4) / 3 = 8; the raw
      // weighted squares (2^126 * 4 and more) overflow float32 before the division.
      [[2 ** 125, 2 ** 126], [-4, 2], Math.sqrt(8)]
    ] as const) {
      const actual = await run(
        {
          offsets: Uint32Array.of(0, 2, 2, 2, 2, 2, 2, 2, 2),
          neighbors: Uint32Array.of(1, 2),
          weights: Float32Array.from(weights),
          values: Float32Array.of(0, ...values, 0, 0, 0, 0, 0),
          categories: new Uint32Array(8),
          mask: new Uint32Array(8).fill(1)
        },
        false,
        maximumNeighbors,
        {},
        ['standardDeviation']
      );
      // Rows 1-7 have W = 0 (NaN).
      expect(actual['table'].length).toBe(8);
      expect(Number.isFinite(actual['table'][0]), `weights ${weights}`).toBe(true);
      expect(Math.abs(actual['table'][0] - deviation)).toBeLessThan(2e-7 * deviation);
      expectRows(Array.from(actual['table']).slice(1), new Array(7).fill(NaN));
    }
  });
}

it('keeps population spread after an exactly representable large translation', async () => {
  for (const origin of [0, 8192]) {
    const actual = await run(
      {
        ...INPUT,
        offsets: Uint32Array.of(0, 2, 2, 2, 2, 2, 2, 2, 2),
        neighbors: Uint32Array.of(1, 2),
        weights: Float32Array.of(1, 1),
        values: Float32Array.of(0, origin, origin + 1 / 1024, 0, 0, 0, 0, 0),
        mask: new Uint32Array(8).fill(1)
      },
      false,
      32,
      {},
      ['standardDeviation']
    );
    // Two equal masses separated by 2^-10 have population deviation 2^-11.
    expectOracleRows(actual['table'], [1 / 2048, NaN, NaN, NaN, NaN, NaN, NaN, NaN], 1e-9);
  }
});

it('publishes a nonzero mean when relative-coordinate reconstruction cancels it', async () => {
  const actual = await run(
    {
      ...INPUT,
      offsets: Uint32Array.of(0, 4, 4, 4, 4, 4, 4, 4, 4),
      neighbors: Uint32Array.of(1, 2, 3, 4),
      weights: Float32Array.of(1, 1, 1, 1),
      values: Float32Array.of(0, 1, 1, -1, -1 + 2 ** -24, 0, 0, 0),
      mask: new Uint32Array(8).fill(1)
    },
    false,
    32,
    {},
    ['count', 'sum', 'weightSum', 'mean']
  );
  // Four exact f32 values sum to 2^-24. Their public mean is sum/W=2^-26.
  expectOracleRows(actual['table'].slice(0, 4), [4, 2 ** -24, 4, 2 ** -26], 0);
});

it('chooses a positive-weight variance anchor independently of a zero-weight focal member', async () => {
  const large = Math.fround(3.2e38);
  const actual = await run(
    {
      ...INPUT,
      offsets: Uint32Array.of(0, 1, 1, 1, 1, 1, 1, 1, 1),
      neighbors: Uint32Array.of(1),
      weights: Float32Array.of(1),
      values: Float32Array.of(large, -large, 0, 0, 0, 0, 0, 0),
      mask: Uint32Array.of(1, 1, 0, 0, 0, 0, 0, 0)
    },
    true,
    32,
    {},
    ['mean', 'standardDeviation'],
    0
  );
  // Only the neighbor has positive mass: its exact weighted mean is -large and spread is zero.
  // The zero-weight focal member must neither anchor the deviations nor evaluate 0*Infinity.
  expectOracleRows(
    actual['table'],
    [-large, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
    0
  );
});

it('excludes reserved missing categories in the GPU operation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const includeFocal of [false, true]) {
    const actual = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 1)},
        neighbors: {format: 'uint32', values: Uint32Array.of(1)},
        weights: {format: 'float32', values: Float32Array.of(1)},
        categories: {format: 'uint32', values: Uint32Array.of(7, 0xffffffff)}
      },
      operations: [
        new GPUNeighborhoodSummaryOperation({
          rowCount: 2,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          categories: 'categories',
          includeFocal,
          output: {modes: 'modes', entropy: 'entropy'}
        })
      ],
      outputs: ['modes', 'entropy']
    });
    expectOracleRows(actual['modes'], [includeFocal ? 7 : 0xffffffff, 0xffffffff], 0);
    expectOracleRows(actual['entropy'], [includeFocal ? 0 : NaN, NaN], 0);
  }
});

for (const includeFocal of [false, true]) {
  it(`checks every neighborhood GPU port against NumPy moments/frequencies with focal=${includeFocal}`, async () => {
    const expected = includeFocal ? STANDARDIZED_FOCAL_EXPECTED : RAW_EXPECTED;
    for (const transformed of [false, true]) {
      const actual = await run(
        {...INPUT, values: INPUT.values.map(value => (transformed ? value * 3 + 32 : value))},
        includeFocal
      );
      const table = expected.table.map((value, index) => {
        if (!transformed || index % 8 < 2) return value;
        if (index % 8 === 2) return 3 * value + 32 * expected.table[index - 1];
        if (index % 8 === 6) return 3 * value;
        return 3 * value + 32;
      });
      // Output magnitudes <=200. Two-pass moments and row-normalization need
      // 3e-5 absolute for the transformed fixture; counts/categories/status are exact.
      for (const port of ['table', 'modes', 'entropy', 'overflow'] as const)
        expectOracleRows(
          actual[port],
          port === 'table' ? table : expected[port],
          port === 'table' || port === 'entropy' ? 3e-5 : 0
        );
    }
  });
}

for (const field of [
  'overflow',
  'candidateOverflow',
  'iterationLimitReached',
  'converged'
] as const) {
  for (const complete of [false, true]) {
    it(`checks every neighborhood port with ${field} reporting complete=${complete}`, async () => {
      const actual = await run(INPUT, false, 32, {
        [field]: field === 'converged' ? Number(complete) : Number(!complete)
      });
      const expected = complete ? RAW_EXPECTED : INCOMPLETE_EXPECTED;
      for (const port of ['table', 'modes', 'entropy', 'overflow'] as const)
        expectOracleRows(
          actual[port],
          expected[port],
          port === 'table' || port === 'entropy' ? 1e-6 : 0
        );
    });
  }
}

it('suppresses only over-capacity medians while categorical fallback and other numeric columns remain exact', async () => {
  const actual = await run(INPUT, false, 1);
  const expectedTable = RAW_EXPECTED.table.map((value, index) =>
    index % 8 === 7 && RAW_EXPECTED.table[index - 7] > 1 ? NaN : value
  );
  expectOracleRows(actual['table'], expectedTable, 1e-6);
  expectOracleRows(actual['modes'], RAW_EXPECTED.modes, 0);
  expectOracleRows(actual['entropy'], RAW_EXPECTED.entropy, 1e-6);
  expectOracleRows(actual['overflow'], [1], 0);
});

it('publishes finite extrema and even medians near both float32 bounds', async () => {
  const value = Math.fround(3.2e38);
  const actual = await run(
    {
      ...INPUT,
      offsets: Uint32Array.of(0, 2, 2, 2, 4, 4, 4, 4, 4),
      neighbors: Uint32Array.of(1, 2, 4, 5),
      weights: Float32Array.of(1, 1, 1, 1),
      values: Float32Array.of(0, value, value, 0, -value, -value, 0, 0),
      mask: new Uint32Array(8).fill(1)
    },
    false,
    32,
    {},
    ['min', 'max', 'median']
  );
  const table = new Array<number>(24).fill(NaN);
  table.splice(0, 3, value, value, value);
  table.splice(9, 3, -value, -value, -value);
  expectOracleRows(actual['table'], table, 0);
  expectOracleRows(
    actual['modes'],
    [3, 4294967295, 4294967295, 7, 4294967295, 4294967295, 4294967295, 4294967295],
    0
  );
  expectOracleRows(actual['entropy'], [0.6931471805599453, NaN, NaN, 0, NaN, NaN, NaN, NaN], 1e-6);
  expectOracleRows(actual['overflow'], [0], 0);
});

it('writes optional overflow status when median is not requested', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const table = rig.output('float32', 16);
    const overflow = rig.output('uint32', 1);
    // Direct rig pre-fills outputs with garbage: omitting the clear must fail.
    expect(await rig.readUint(overflow)).toEqual([0x7f7f7f7f]);
    rig.run(
      new GPUNeighborhoodSummary({
        weights: {
          offsets: rig.input(INPUT.offsets, 'uint32'),
          neighbors: rig.input(INPUT.neighbors, 'uint32'),
          weights: rig.input(INPUT.weights, 'float32')
        },
        values: rig.input(INPUT.values, 'float32'),
        mask: rig.input(INPUT.mask, 'uint32'),
        statistics: ['count', 'mean'],
        output: {table, status: {overflow}}
      })
    );
    expectOracleRows(
      await rig.readFloat(table),
      [2, 5, 2, 7, 0, NaN, 2, 6, 2, 4, 1, -2, 0, NaN, 0, NaN],
      0
    );
    expectOracleRows(await rig.readUint(overflow), [0], 0);
  } finally {
    rig.destroy();
  }
});

it('uses a zero-weight focal member when its validated float32 prop underflows to zero', async () => {
  const values = Float32Array.of(2, 3, 4, 5, 6, 7, 8, 9);
  const actual = await run(
    {
      offsets: new Uint32Array(9),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(0),
      values,
      categories: Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 8),
      mask: new Uint32Array(8).fill(1)
    },
    true,
    32,
    {},
    STATISTICS,
    1e-50
  );
  expectOracleRows(
    actual['table'],
    Array.from(values, value => [1, 0, 0, NaN, value, value, NaN, value]).flat(),
    0
  );
  expectOracleRows(actual['modes'], [1, 2, 3, 4, 5, 6, 7, 8], 0);
  expectOracleRows(actual['entropy'], new Array<number>(8).fill(NaN), 0);
  expectOracleRows(actual['overflow'], [0], 0);
});
