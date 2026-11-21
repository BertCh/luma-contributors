// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRasterReconstructionOnCPU} from '../../src/raster-analysis/raster-morphology/cpu-raster-reconstruction-operation';

it('reconstruction independently counts propagation and fixed-point verification', () => {
  for (const frame of [
    {marker: [1, 0, 0, 0], values: [1, 1, 0, 0], status: [0, 1, 2]},
    {marker: [1, 1, 1, 1], values: [1, 1, 1, 1], status: [1, 0, 1]},
    {marker: [1, 0, 0, 0], values: [1, 1, 0, 0], status: [0, 1, 2]}
  ]) {
    // The donor at cell 0 reaches cell 1 in one round; cell 2 is an unresolved next-round
    // change. A uniform marker instead needs one unchanged pass. These are hand derivations,
    // not expected values obtained from another reconstruction run.
    const result = computeRasterReconstructionOnCPU(
      Float32Array.from(frame.marker),
      Float32Array.of(1, 1, 1, 1),
      {
        grid: {
          width: 4,
          height: 1,
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        maximumIterations: 1
      }
    );
    expect(Array.from(result.values)).toEqual(frame.values);
    expect(Array.from(result.validity)).toEqual([1, 1, 1, 1]);
    expect([
      result.converged[0],
      result.iterationLimitReached[0],
      result.iterationCount[0]
    ]).toEqual(frame.status);
  }
});
