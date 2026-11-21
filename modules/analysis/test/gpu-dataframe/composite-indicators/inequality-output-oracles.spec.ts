// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUInequalityOperation} from '../../../src/dataframe-analysis/composite-indicators/gpu-inequality-operation';
import {
  expectLiteralRows,
  poisonOperation
} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {INEQUALITY_INPUT, INEQUALITY_EXPECTED} from './inequality-output-oracles-fixture';
import {getGPUInequalityParameterValues} from '../../../src/dataframe-analysis/composite-indicators/inequality-parameters';
import {getInequalityOperationTestOutput} from './inequality-operation-test-output';

it.each([
  'parallel',
  'reference'
] as const)('B3 %s retains Gini and the Atkinson boundary when weighted products underflow', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scale = 2 ** -80;
  const rows = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: Float32Array.of(0, scale, 0, 0)},
      weights: {format: 'float32', values: new Float32Array(4).fill(scale)},
      zones: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 1)},
      parameters: {format: 'float32', values: getGPUInequalityParameterValues({epsilon: 3})}
    },
    operations: [
      poisonOperation(
        new GPUInequalityOperation({
          mode,
          values: 'values',
          weights: 'weights',
          zoneIds: 'zones',
          zoneCount: 2,
          parameters: 'parameters',
          output: {gini: 'gini', atkinson: 'atkinson', globalSummary: 'summary'}
        })
      )
    ],
    outputs: ['gini', 'atkinson', 'summary']
  });
  // Equal-weight [0,s] has Gini 1/2. Pooled [0,0,0,s] has Gini 3/4.
  // The included zero has equivalent income zero at epsilon 3; the all-zero zone is undefined.
  expectLiteralRows(rows['gini'], [0.5, NaN]);
  expectLiteralRows(rows['atkinson'], [1, NaN]);
  expect(rows['summary']).toHaveLength(8);
  expectLiteralRows(rows['summary'].slice(3, 4), [0.75]);
});

it('inequality output ports independently match NumPy with other optional ports absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = Object.keys(INEQUALITY_EXPECTED) as (keyof typeof INEQUALITY_EXPECTED)[];
  // All present, each alone, and each omitted: every optional output is exercised independently.
  for (const selected of [
    names,
    ...names.map(name => [name]),
    ...names.map(name => names.filter(other => other !== name))
  ]) {
    const output = getInequalityOperationTestOutput(selected);
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: INEQUALITY_INPUT.values},
        zoneIds: {format: 'uint32', values: INEQUALITY_INPUT.zoneIds},
        parameters: {format: 'float32', values: INEQUALITY_INPUT.parameters}
      },
      operations: [
        poisonOperation(
          new GPUInequalityOperation({
            values: 'values',
            zoneIds: 'zoneIds',
            parameters: 'parameters',
            zoneCount: 3,
            lorenzKnotCount: 5,
            output
          })
        )
      ],
      outputs: selected
    });
    // 2e-5 relative/absolute allows float32 log and cumulative reduction rounding.
    for (const name of selected) expectLiteralRows(rows[name], INEQUALITY_EXPECTED[name]);
  }
});

it.each([
  'parallel',
  'reference'
] as const)('Atkinson %s preserves inequality at epsilon one plus one float32 ulp', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: new Float32Array([1, 1, 3, 3])},
      zones: {format: 'uint32', values: new Uint32Array(4)},
      parameters: {
        format: 'float32',
        values: getGPUInequalityParameterValues({epsilon: 1 + 2 ** -23})
      }
    },
    operations: [
      poisonOperation(
        new GPUInequalityOperation({
          mode,
          values: 'values',
          zoneIds: 'zones',
          zoneCount: 1,
          parameters: 'parameters',
          output: {atkinson: 'atkinson'}
        })
      )
    ],
    outputs: ['atkinson']
  });
  // Decimal oracle; 5e-4 is the documented float32 Atkinson tolerance.
  expectLiteralRows(rows['atkinson'], [0.133974611790977], 5e-4);
});

it.each([
  'parallel',
  'reference'
] as const)('Atkinson %s keeps the zero-income correction and continuous boundary', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const epsilon of [0.999, 1, 1 + 2 ** -23]) {
    const values = new Float32Array([0, ...new Array(512).fill(1), ...new Array(512).fill(3)]);
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values},
        zones: {format: 'uint32', values: new Uint32Array(values.length)},
        parameters: {format: 'float32', values: getGPUInequalityParameterValues({epsilon})}
      },
      operations: [
        poisonOperation(
          new GPUInequalityOperation({
            mode,
            values: 'values',
            zoneIds: 'zones',
            zoneCount: 1,
            parameters: 'parameters',
            output: {atkinson: 'atkinson'}
          })
        )
      ],
      outputs: ['atkinson']
    });
    // Decimal at packed float32 epsilon: inequality-output-oracles.oracle.py; absolute float32 tolerance.
    expectLiteralRows(rows['atkinson'], [epsilon < 1 ? 0.673332558 : 1], 5e-4);
  }
});

it.each([
  'parallel',
  'reference'
] as const)('Atkinson %s handles large centered weights (R1)', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: Float32Array.of(0, 1 / 256, 1, 0, 1 / 256, 1)},
      weights: {
        format: 'float32',
        values: Float32Array.of(2 ** 110, 2 ** 126, 2 ** 126, 2 ** -16, 1, 1)
      },
      zones: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 1, 1, 1)},
      parameters: {format: 'float32', values: getGPUInequalityParameterValues({epsilon: 0.999})}
    },
    operations: [
      poisonOperation(
        new GPUInequalityOperation({
          mode,
          values: 'values',
          weights: 'weights',
          zoneIds: 'zones',
          zoneCount: 2,
          parameters: 'parameters',
          output: {atkinson: 'atkinson'}
        })
      )
    ],
    outputs: ['atkinson']
  });
  // Decimal at packed epsilon; scaling every weight by 2^-126 leaves the definition unchanged.
  expectLiteralRows(rows['atkinson'], [0.8759559391035893, 0.8759559391035893], 5e-4);
});

it.each([
  'parallel',
  'reference'
] as const)('Atkinson %s handles large weights at epsilon one', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: Float32Array.of(1 / 1024, 1, 1 / 1024, 1)},
      weights: {format: 'float32', values: Float32Array.of(2 ** 126, 2 ** 126, 1, 1)},
      zones: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 1)},
      parameters: {format: 'float32', values: getGPUInequalityParameterValues({epsilon: 1})}
    },
    operations: [
      poisonOperation(
        new GPUInequalityOperation({
          mode,
          values: 'values',
          weights: 'weights',
          zoneIds: 'zones',
          zoneCount: 2,
          parameters: 'parameters',
          output: {atkinson: 'atkinson'}
        })
      )
    ],
    outputs: ['atkinson']
  });
  // Equal relative weights give one minus the geometric mean divided by the arithmetic mean.
  // Both zones have the same Decimal result from inequality-output-oracles.oracle.py.
  expectLiteralRows(rows['atkinson'], [0.9375609756097561, 0.9375609756097561], 5e-4);
});

it.each([
  'parallel',
  'reference'
] as const)('NUM-09 %s distinguishes positive-weight zeros, all-zero, zero-weight and empty zones', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const epsilon of [1, 1 + 2 ** -23, 2, 32]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.of(0, 2, 0, 0, 2)},
        zones: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 2, 2)},
        weights: {format: 'float32', values: Float32Array.of(1, 1, 1, 0, 1)},
        parameters: {format: 'float32', values: getGPUInequalityParameterValues({epsilon})}
      },
      operations: [
        poisonOperation(
          new GPUInequalityOperation({
            mode,
            values: 'values',
            zoneIds: 'zones',
            weights: 'weights',
            parameters: 'parameters',
            zoneCount: 4,
            output: {atkinson: 'atkinson', theilL: 'theilL', counts: 'counts'}
          })
        )
      ],
      outputs: ['atkinson', 'theilL', 'counts']
    });
    // Positive-weight zero forces geometric/harmonic/negative-power mean to zero. A zero-weight
    // zero is excluded before that test; normalization remains undefined at zero/empty mean.
    expectLiteralRows(rows['atkinson'], [1, NaN, 0, NaN]);
    expectLiteralRows(rows['theilL'], [NaN, NaN, 0, NaN]);
    expectLiteralRows(rows['counts'], [2, 1, 1, 0]);
  }
});

it.each([
  'parallel',
  'reference'
] as const)('Atkinson %s remains finite when a positive ratio underflows', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: new Float32Array([1e-30, 1e30])},
      zones: {format: 'uint32', values: new Uint32Array(2)},
      parameters: {
        format: 'float32',
        values: getGPUInequalityParameterValues({epsilon: 1 - 2 ** -8})
      }
    },
    operations: [
      poisonOperation(
        new GPUInequalityOperation({
          mode,
          values: 'values',
          zoneIds: 'zones',
          zoneCount: 1,
          parameters: 'parameters',
          output: {atkinson: 'atkinson'}
        })
      )
    ],
    outputs: ['atkinson']
  });
  // Decimal using the packed float32 incomes gives a result that rounds to float32 1.
  expectLiteralRows(rows['atkinson'], [1], 5e-4);
});
