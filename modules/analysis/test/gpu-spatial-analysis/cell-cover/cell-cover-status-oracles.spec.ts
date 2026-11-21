// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';

// Independent closed-form oracle at resolution 2 (four Mercator rows/columns):
// lon [-170,-100] lies wholly in column 0, whose center is -135 degrees.
// lat [10,85] spans rows 0..1; their centers are about 79.1713 and 40.9799
// degrees (atan(sinh(pi*(1-2*(row+1/2)/4)))). Both centers are strictly inside.
// Thus bounds enumeration has exactly two candidates and accepts both.
// Raising the northern bound to 90 sets uncertainCount to 1 and complete to 0 because
// it exceeds the documented Mercator latitude limit, 85.051129 degrees.
// The seam ring has non-pole edges with longitude jumps of 340 degrees, so
// invalidCount=1 and all candidate/result counts are 0. Every status scalar is
// poisoned on each submit, including zero-only flags for some capacities.
const NORMAL = [-170, 10, -100, 10, -100, 85, -170, 85];
const POLAR = [-170, 10, -100, 10, -100, 90, -170, 90];
const UNSPLIT = [170, 10, -170, 10, -170, 85, 170, 85];

// Status order: count, requiredCount, overflow, candidateOverflow,
// candidateCount, invalidCount, uncertainCount. Candidate truncation limits
// requiredCount to the accepted retained candidates; it is then a lower bound.
for (const scenario of [
  {candidateCapacity: 1, outputCapacity: 1, status: [1, 1, 0, 1, 2, 0, 0]},
  {candidateCapacity: 1, outputCapacity: 4, status: [1, 1, 0, 1, 2, 0, 0]},
  {candidateCapacity: 4, outputCapacity: 1, status: [1, 2, 1, 0, 2, 0, 0]},
  {candidateCapacity: 4, outputCapacity: 4, status: [2, 2, 0, 0, 2, 0, 0]}
]) {
  it(`cover status follows two analytic tile centers and clears diagnostics (candidate ${scenario.candidateCapacity}, output ${scenario.outputCapacity})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createOutputOracleFixture(device);
    const {graph, input, output} = fixture;
    const positions = input('positions', 'float32x2', Float32Array.from(NORMAL));
    const status = {
      count: output('count', 'uint32', 1),
      requiredCount: output('required', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1),
      candidateOverflow: output('candidate-overflow', 'uint32', 1),
      candidateCount: output('candidate-count', 'uint32', 1),
      invalidCount: output('invalid', 'uint32', 1),
      uncertainCount: output('uncertain', 'uint32', 1),
      complete: output('complete', 'uint32', 1)
    };
    graph.add(
      new GPUCellCover({
        family: 'quadbin',
        resolution: 2,
        containment: 'center',
        positions,
        featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)),
        polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)),
        ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)),
        candidateEnumeration: 'bounds',
        candidateCapacity: scenario.candidateCapacity,
        output: {
          featureIds: output('feature-ids', 'uint32', scenario.outputCapacity),
          cells: output('cells', 'uint32x2', scenario.outputCapacity),
          status
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const frame of [
        {positions: NORMAL, status: scenario.status},
        {
          positions: POLAR,
          status: [...scenario.status.slice(0, 6), 1]
        },
        {positions: UNSPLIT, status: [0, 0, 0, 0, 0, 1, 0]},
        {positions: NORMAL, status: scenario.status}
      ]) {
        fixture.writeInput(positions, Float32Array.from(frame.positions));
        const actual = await fixture.submitTwice(compiled, () =>
          fixture.readRows([
            status.count,
            status.requiredCount,
            status.overflow,
            status.candidateOverflow,
            status.candidateCount,
            status.invalidCount,
            status.uncertainCount
          ])
        );
        expect(actual).toEqual(frame.status);
        expect(await fixture.read(status.complete)).toEqual([
          frame.status[2] || frame.status[3] || frame.status[5] || frame.status[6] ? 0 : 1
        ]);
      }
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  });
}
