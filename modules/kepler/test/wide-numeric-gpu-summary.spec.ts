// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {describeWideNumericNarrowingOnGpu} from '../../../examples/deck/kepler-compute/src/wide-numeric-gpu-summary';
import {describeWideNumericNarrowing} from '../../../examples/deck/kepler-compute/src/analysis2-narrowing-summary';

it('GPU narrowing summary scans every row and agrees with the CPU verdicts beyond 200,000 rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 300_000;
  // The only lossy Int64 value and the only overflowing Float64 value sit past row 200,000.
  const integers = new BigInt64Array(rowCount).map((_, row) => BigInt(row));
  integers[rowCount - 1] = 2n ** 40n + 1n;
  const doubles = new Float64Array(rowCount).map((_, row) => row * 0.5);
  doubles[rowCount - 2] = 1e300;
  const table = new arrow.Table({
    big: arrow.makeVector(integers),
    wide: arrow.makeVector(doubles)
  });
  const gpuLines = await describeWideNumericNarrowingOnGpu(device, table);
  const cpuLines = describeWideNumericNarrowing(table);
  expect(gpuLines).toHaveLength(2);
  expect(gpuLines[0]).toContain('Narrowing big');
  expect(gpuLines[0]).toContain('rounded in 1 of 300000');
  expect(gpuLines[1]).toContain('overflows Float32 in 1 rows');
  expect(gpuLines.map(line => line.replace(' (GPU scan)', ''))).toEqual(cpuLines);
});
