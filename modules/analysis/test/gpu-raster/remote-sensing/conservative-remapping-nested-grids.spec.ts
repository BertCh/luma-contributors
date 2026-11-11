// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterConservativeRemapping} from '../../../src/raster-analysis/remote-sensing/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';

// Finding raster-imagery-03. Expected values come from the geometry of the grids written out by
// hand (which source cells a destination cell covers, and by how much), not from the kernel.

type Axis = {sourceSpacing: number; destinationSpacing: number; destinationOrigin: number};

async function remapRow(
  axis: Axis,
  sourceValues: number[],
  sourceValidity: number[],
  destinationWidth: number,
  normalization: 'destination-area' | 'covered-area' | 'extensive'
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  return harness.run(
    new GPURasterConservativeRemapping({
      source: {
        width: sourceValues.length,
        height: 1,
        origin: [0, 0],
        spacing: [axis.sourceSpacing, 1]
      },
      destination: {
        width: destinationWidth,
        height: 1,
        origin: [axis.destinationOrigin, 0],
        spacing: [axis.destinationSpacing, 1]
      },
      values: harness.input(sourceValues),
      mask: harness.mask(sourceValidity),
      normalization,
      output: {
        values: harness.output('values', 'float32', destinationWidth),
        coveredArea: harness.output('area', 'float32', destinationWidth),
        status: harness.output('status', 'uint32', destinationWidth)
      }
    })
  );
}

it('RS03 exactly nested grids (0.1 into 0.3) never take a value from a cell they do not touch', async () => {
  const destinationWidth = 4000;
  const sourceWidth = 3 * destinationWidth + 8;
  const sourceValues = Array.from({length: sourceWidth}, (_, column) => 1 + (column % 11));
  // Every odd destination column has its three real source cells invalid. Before the fix 912 of
  // these columns had a float32 sliver overlap with the next source cell (for example column 3414
  // with source column 10245) and published that neighbour's value with status 0.
  const sourceValidity = Array.from({length: sourceWidth}, (_, column) =>
    Math.floor(column / 3) % 2 === 1 ? 0 : 1
  );
  const result = await remapRow(
    {sourceSpacing: 0.1, destinationSpacing: 0.3, destinationOrigin: 0},
    sourceValues,
    sourceValidity,
    destinationWidth,
    'covered-area'
  );
  let leakedCount = 0;
  let largestError = 0;
  let largestAreaError = 0;
  for (let column = 0; column < destinationWidth; column++) {
    if (column % 2 === 1) {
      if (
        result['status'][column] !== 1 ||
        result['area'][column] !== 0 ||
        !Number.isNaN(result['values'][column])
      ) {
        leakedCount++;
      }
    } else {
      const expected =
        (sourceValues[3 * column] + sourceValues[3 * column + 1] + sourceValues[3 * column + 2]) /
        3;
      largestError = Math.max(largestError, Math.abs(result['values'][column] - expected));
      largestAreaError = Math.max(largestAreaError, Math.abs(result['area'][column] - 0.3));
      expect(result['status'][column]).toBe(0);
    }
  }
  expect(leakedCount).toBe(0);
  expect(largestError).toBeLessThan(2e-6);
  expect(largestAreaError).toBeLessThan(1e-6);
});

it('RS03 a finer destination (0.3 into 0.1) copies the one source cell that contains it', async () => {
  const sourceWidth = 3000;
  const destinationWidth = 3 * sourceWidth;
  const sourceValues = Array.from({length: sourceWidth}, (_, column) => 1 + (column % 13));
  const sourceValidity = Array.from({length: sourceWidth}, (_, column) =>
    column % 2 === 1 ? 0 : 1
  );
  const result = await remapRow(
    {sourceSpacing: 0.3, destinationSpacing: 0.1, destinationOrigin: 0},
    sourceValues,
    sourceValidity,
    destinationWidth,
    'covered-area'
  );
  let wrongCount = 0;
  for (let column = 0; column < destinationWidth; column++) {
    const sourceColumn = Math.floor(column / 3);
    const isValid = sourceColumn % 2 === 0;
    const value = result['values'][column];
    const isRight = isValid
      ? // value * area / area is one float32 step from the value at most.
        result['status'][column] === 0 && Math.abs(value - sourceValues[sourceColumn]) < 2e-6
      : result['status'][column] === 1 && Number.isNaN(value) && result['area'][column] === 0;
    if (!isRight) wrongCount++;
  }
  expect(wrongCount).toBe(0);
});

it('RS03 a rational ratio with an offset origin (0.1 into 0.25, shifted by 0.05) has exact weights', async () => {
  // Destination column j covers [0.5 + 2.5 j, 3 + 2.5 j] in source cells.
  // Even j: half of cell 2.5 j, cells 2.5 j + 1 and 2.5 j + 2 whole.
  // Odd j: cells floor(2.5 j) + 1 and + 2 whole, half of + 3 (cell floor(2.5 j) is not touched).
  const destinationWidth = 3000;
  const sourceWidth = Math.ceil(2.5 * destinationWidth) + 8;
  const sourceValues = Array.from({length: sourceWidth}, (_, column) => 2 + (column % 7));
  const result = await remapRow(
    {sourceSpacing: 0.1, destinationSpacing: 0.25, destinationOrigin: 0.05},
    sourceValues,
    new Array(sourceWidth).fill(1),
    destinationWidth,
    'destination-area'
  );
  let largestError = 0;
  for (let column = 0; column < destinationWidth; column++) {
    const first = Math.floor(2.5 * column);
    const expected =
      column % 2 === 0
        ? (0.5 * sourceValues[first] + sourceValues[first + 1] + sourceValues[first + 2]) / 2.5
        : (sourceValues[first + 1] + sourceValues[first + 2] + 0.5 * sourceValues[first + 3]) / 2.5;
    largestError = Math.max(largestError, Math.abs(result['values'][column] - expected));
    expect(result['status'][column]).toBe(0);
  }
  expect(largestError).toBeLessThan(3e-6);
});

for (const destinationOrigin of [0.1 * Math.E, -0.1 * Math.PI]) {
  it(`RS03 incommensurate grids from origin ${destinationOrigin.toFixed(3)} match binary64 interval overlaps far from the origin`, async () => {
    const sourceSpacing = 0.1;
    const destinationSpacing = 0.1 * Math.SQRT2;
    const destinationWidth = 6000;
    const sourceWidth =
      Math.ceil((destinationOrigin + destinationWidth * destinationSpacing) / 0.1) + 4;
    const sourceValues = Array.from({length: sourceWidth}, (_, column) => 1 + (column % 5));
    const result = await remapRow(
      {sourceSpacing, destinationSpacing, destinationOrigin},
      sourceValues,
      new Array(sourceWidth).fill(1),
      destinationWidth,
      'destination-area'
    );
    let largestError = 0;
    for (let column = 0; column < destinationWidth; column++) {
      const lower = destinationOrigin + column * destinationSpacing;
      const upper = lower + destinationSpacing;
      if (upper <= 0) {
        // Entirely left of the source grid.
        expect(result['status'][column]).toBe(1);
        expect(result['values'][column]).toBeNaN();
        continue;
      }
      let numerator = 0;
      for (
        let cell = Math.max(0, Math.floor(lower / sourceSpacing));
        cell <= Math.floor(upper / sourceSpacing);
        cell++
      ) {
        const overlap =
          Math.min(upper, (cell + 1) * sourceSpacing) - Math.max(lower, cell * sourceSpacing);
        numerator += Math.max(0, overlap) * sourceValues[cell];
      }
      largestError = Math.max(
        largestError,
        Math.abs(result['values'][column] - numerator / destinationSpacing)
      );
    }
    // Values are 1 to 5; the old float32 edges gave errors of 3e-3 at this distance.
    expect(largestError).toBeLessThan(2e-5);
  });
}
