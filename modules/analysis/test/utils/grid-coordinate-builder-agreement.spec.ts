// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createGridBuilderAgreementFixture} from './grid-coordinate-builder-agreement-fixture';
import {GRID_COORDINATE_CASES, WIDE_GRID_COORDINATE_CASES} from './grid-coordinate-bump-fixtures';
import {readFloat32, readUint32, submitGraph} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

it.each([
  ...GRID_COORDINATE_CASES,
  ...WIDE_GRID_COORDINATE_CASES
])('LUMA-112 grid coordinate readbacks: $name', async testCase => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGridBuilderAgreementFixture(device, testCase);
  const {graph, countsBuffer, sumsBuffer, densityBuffer, flowBuffer, cellCount} = fixture;
  let compiled: CompiledGPUCommandGraph | undefined;
  try {
    compiled = await graph.compileAsync();
    submitGraph(device, compiled, undefined);
    // The literal x cells are independent of all four kernels.
    // y=.25/.75 in [0,1] with two rows means row 0/1; weights are exactly 7/13.
    const expectedCells = [testCase.cell, testCase.size + testCase.cell];
    expect.soft(await readUint32(densityBuffer, 2), 'density keys').toEqual(expectedCells);
    expect.soft(await readUint32(flowBuffer, 2), 'flow zones').toEqual(expectedCells);

    // The additional wide cases pin the local readers. The original cases also pin the builders.
    if (!('oldBinningCell' in testCase)) return;
    const counts = await readUint32(countsBuffer, cellCount);
    const sums = await readFloat32(sumsBuffer, cellCount);
    const expectedCounts = new Array<number>(cellCount).fill(0);
    const expectedSums = new Array<number>(cellCount).fill(0);
    expectedCounts[expectedCells[0]] = 1;
    expectedCounts[expectedCells[1]] = 1;
    expectedSums[expectedCells[0]] = 7;
    expectedSums[expectedCells[1]] = 13;
    expect.soft(sums.every(Number.isFinite), 'published aggregation sums are finite').toBe(true);
    if (testCase.name === 'overflowing domain width') {
      // Alpha.16 places both rows in column 0 with resident bounds on the tested adapter.
      const alpha16Cells = [testCase.oldBinningCell, testCase.size + testCase.oldBinningCell];
      const alpha16Counts = new Array<number>(cellCount).fill(0);
      const alpha16Sums = new Array<number>(cellCount).fill(0);
      alpha16Counts[alpha16Cells[0]] = 1;
      alpha16Counts[alpha16Cells[1]] = 1;
      alpha16Sums[alpha16Cells[0]] = 7;
      alpha16Sums[alpha16Cells[1]] = 13;
      expect.soft(counts, 'alpha.16 resident-bounds binning defect').toEqual(alpha16Counts);
      expect.soft(counts).not.toEqual(expectedCounts);
      expect.soft(sums, 'alpha.16 resident-bounds aggregation defect').toEqual(alpha16Sums);
      expect.soft(sums).not.toEqual(expectedSums);
    } else {
      expect.soft(counts, 'published binning counts').toEqual(expectedCounts);
      expect.soft(sums, 'published aggregation sums').toEqual(expectedSums);
    }
  } finally {
    compiled?.destroy();
    fixture.destroy();
  }
});
