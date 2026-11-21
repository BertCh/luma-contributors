// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUNetworkIsochronesParameterValues} from '../../../src/gpu-network/network-isochrones/index';
import type {GPUDriveTimeCatchmentRecipeProps} from '../../../src/gpu-network/recipes/drive-time-catchment-recipe';
import {
  createOracleRingOutput,
  type IsochroneOracleHarness
} from './isochrones-output-oracle-utils';

/** Two reached z2 cells overflow a one-row table. A zero cell-cost limit retains the first.
 * Demand (-60,30) lies strictly within the first tile [-90,0] x [0,66.51326...]. */
export function createRecipeCellReviewProps(
  harness: IsochroneOracleHarness
): GPUDriveTimeCatchmentRecipeProps {
  const rings = createOracleRingOutput(harness, 'all', 8, 10);
  delete rings.ringGroups;
  delete rings.polygons!.sourceIds;
  return {
    id: 'review-cell-recipe',
    topology: {
      offsets: harness.input('uint32', [0, 1, 2]),
      neighbors: harness.input('uint32', [1, 0]),
      weights: harness.input('float32', [1, 1]),
      nodePositions: harness.input('float32x2', [-70, 20, 20, 20])
    },
    facilities: harness.input('float32x2', [-70, 20]),
    demand: harness.input('float32x2', [-60, 30]),
    costBreaks: harness.input('float32', [2]),
    isochrones: {
      costBreaks: harness.input('float32', [2]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 1,
            extent: [-180, -80, 180, 80],
            cellCostLimit: 2
          })
        )
      ),
      cellOutline: {
        family: 'quadbin',
        resolution: 2,
        table: {
          cells: harness.output('tableCells', 'uint32x2', 1),
          counts: harness.output('tableCounts', 'uint32', 1),
          status: {
            count: harness.output('tableCount', 'uint32', 1),
            overflow: harness.output('tableOverflow', 'uint32', 1),
            candidateOverflow: harness.output('tableCandidateOverflow', 'uint32', 1),
            sumOverflow: harness.output('tableSumOverflow', 'uint32', 1)
          }
        },
        output: {
          rows: harness.output('outlineRows', 'uint32', 8),
          cells: harness.output('outlineCells', 'uint32x2', 8),
          edgeIndices: harness.output('outlineEdges', 'uint32', 8),
          endpoints: harness.output('outlineEndpoints', 'float32x4', 8),
          status: {
            count: harness.output('outlineCount', 'uint32', 1),
            overflow: harness.output('outlineOverflow', 'uint32', 1),
            candidateOverflow: harness.output('outlineCandidateOverflow', 'uint32', 1)
          }
        },
        rings: {
          output: {
            ...rings,
            status: {
              ...rings.status,
              candidateOverflow: harness.output('ringCandidateOverflow', 'uint32', 1),
              complete: harness.output('ringComplete', 'uint32', 1),
              polygonCount: harness.output('polygonCount', 'uint32', 1)
            }
          }
        }
      }
    }
  };
}
