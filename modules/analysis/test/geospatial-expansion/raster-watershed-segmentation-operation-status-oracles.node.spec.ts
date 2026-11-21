// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeRasterWatershedSegmentationOnCPU} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';

for (const maximumSteps of [1, 2]) {
  it(`honors the watershed CPU FIFO budget ${maximumSteps} with independently derived status words`, () => {
    const result = computeRasterWatershedSegmentationOnCPU(
      {id: 'band', format: 'float32', values: Float32Array.of(1, 1)},
      Uint32Array.of(7, 0),
      {
        grid: {width: 2, height: 1, affine: [1, 0, 0, 0, 1, 0], pixelInterpretation: 'area'},
        maximumSteps
      }
    );
    // One pop discovers cell 1 but leaves it in the queue; two pops empty it.
    expect({
      converged: Array.from(result.converged),
      iterationLimitReached: Array.from(result.iterationLimitReached)
    }).toEqual({
      converged: [maximumSteps === 1 ? 0 : 1],
      iterationLimitReached: [maximumSteps === 1 ? 1 : 0]
    });
  });
}
