// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import {addDriveTimeCatchmentRecipe} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  createOracleRingOutput,
  type IsochroneOracleHarness
} from './isochrones-output-oracle-utils';

/** A zero-cost road and radius 2 cover the four sample centers of a unit square. */
export function createRecipeSquareOracle(harness: IsochroneOracleHarness) {
  const output = createOracleRingOutput(harness, 'minimal', 2);
  const isochroneCostBreaks = harness.input('float32', [1]);
  const edgeOverflow = harness.output('ringEdgeOverflow', 'uint32', 1);
  const recipe = addDriveTimeCatchmentRecipe(harness.graph, {
    topology: {
      offsets: harness.input('uint32', [0, 1, 2]),
      neighbors: harness.input('uint32', [1, 0]),
      weights: harness.input('float32', [0, 0]),
      nodePositions: harness.input('float32x2', [0.5, 1, 1.5, 1])
    },
    facilities: harness.input('float32x2', [1, 1]),
    demand: harness.input('float32x2', [1, 1]),
    costBreaks: harness.input('float32', [1]),
    isochrones: {
      costBreaks: isochroneCostBreaks,
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 1,
            extent: [0, 0, 2, 2],
            bufferRadius: 2
          })
        )
      ),
      raster: {
        width: 2,
        height: 2,
        output: {bandClasses: harness.output('classes', 'uint32', 4)},
        rings: {edgeCapacity: 2, output: {...output, status: {...output.status, edgeOverflow}}}
      }
    }
  });
  return {recipe, output, edgeOverflow, isochroneCostBreaks};
}
