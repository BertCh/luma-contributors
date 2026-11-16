// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getS2TopologyCell} from '../cell-topology/s2-topology-oracle';
import {s2CellToCenter} from '../cell-indexing/cell-indexing-oracle';
import {isInsideRingsExact, type Point} from '../segment-intersection/exact-geometry-reference';
import type {CoverFeature, CoverResult} from './cell-cover-oracle';

/** Exhaustive low-level center oracle; independent of the GPU's cap pruning and frontier. */
export function coverS2OnCPU(features: CoverFeature[], level: number): CoverResult {
  const result: CoverResult = {cells: [], featureRows: []};
  const size = 2 ** level;
  const cells: bigint[] = [];
  for (let face = 0; face < 6; face++)
    for (let column = 0; column < size; column++)
      for (let row = 0; row < size; row++)
        cells.push(getS2TopologyCell({face, ij: [column, row], level}));
  cells.sort((first, second) => (first < second ? -1 : first > second ? 1 : 0));
  for (const [featureRow, feature] of features.entries())
    for (const cell of cells) {
      const center = s2CellToCenter(cell);
      if (
        feature.some(polygon =>
          isInsideRingsExact(
            polygon.map(ring => ring.map(point => [point[0], point[1]] as Point)),
            center
          )
        )
      ) {
        result.cells.push(cell);
        result.featureRows.push(featureRow);
      }
    }
  return result;
}
