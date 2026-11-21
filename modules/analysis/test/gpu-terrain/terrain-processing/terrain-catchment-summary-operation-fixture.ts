// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {TerrainCatchmentSummaryOperationProps} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary-operation';

export function makeCatchmentInput() {
  return {
    elevation: Float32Array.of(10, 8, 6, 4, 20, 16, 12, 8),
    basinIds: Uint32Array.of(0, 0, 1, 1, 0, 0, 1, 1),
    streams: Uint32Array.of(1, 2, 3, 1, 0, 1, 1, 0),
    flowDirections: Uint32Array.of(1, 1, 1, 0, 0, 1, 1, 0),
    validity: new Uint32Array(8).fill(1)
  };
}

export function makeCatchmentProps(): TerrainCatchmentSummaryOperationProps {
  return {
    id: 'catchment-test',
    grid: {width: 4, height: 2, affine: [2, 0, 0, 0, -3, 0], pixelInterpretation: 'area'},
    basinCount: 3,
    histogramBins: 4,
    elevation: 'elevation',
    basinIds: 'basinIds',
    streams: 'streams',
    flowDirections: 'flowDirections',
    validity: 'validity',
    output: {
      statistics: 'statistics',
      hypsometry: 'hypsometry',
      profiles: 'profiles',
      profileStatus: 'profileStatus'
    }
  };
}

/** Four cells with hand-counted profile statuses, independent of either implementation. */
export function makeCatchmentStatusCases() {
  return [
    // 0 <-> 1 cycles, 2 excluded, 3 terminal: [2,2,0,1].
    {
      name: 'cycle/exclusion',
      directions: [1, 16, 0, 0],
      basins: [0, 0, 0xffffffff, 0],
      streams: [1, 1, 1, 1],
      maximumSteps: 4,
      expected: [0, 1, 1]
    },
    // Two unknown memberships and two nonstreams: [2,2,0,0].
    {
      name: 'unknown input',
      directions: [0, 0, 0, 0],
      basins: [0, 0, 0, 0],
      streams: [0xfffffffe, 0xffffffff, 0, 0],
      maximumSteps: 4,
      expected: [0, 2, 0]
    },
    // A three-edge path with exclusive maximumSteps=2: [2,2,1,1].
    {
      name: 'exhausted budget',
      directions: [1, 1, 1, 0],
      basins: [0, 0, 0, 0],
      streams: [1, 1, 1, 1],
      maximumSteps: 2,
      expected: [0, 0, 1]
    },
    {
      name: 'complete',
      directions: [1, 1, 1, 0],
      basins: [0, 0, 0, 0],
      streams: [1, 1, 1, 1],
      maximumSteps: 4,
      expected: [1, 0, 0]
    }
  ];
}

export function expectCatchmentStatusReadbacks(
  result: Record<string, ArrayLike<number>>,
  expected: readonly number[]
): void {
  for (const [index, port] of ['done', 'invalid', 'limited'].entries()) {
    expect(Array.from(result[port])).toEqual([expected[index]]);
  }
}
