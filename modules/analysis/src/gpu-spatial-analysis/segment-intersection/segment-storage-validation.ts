// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getGeometrySortScratchByteLengths,
  validateGeometryScratchByteLengths
} from '../geometry-edit/geometry-storage-validation';

/** Checks every BVH column, Morton-sort column, radix histogram and bounds-reduction partial. */
export function validateSegmentBVHScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number,
  leafCapacity = 2 ** Math.ceil(Math.log2(Math.max(1, rows))),
  spatialSort = true
): void {
  validateGeometryScratchByteLengths(graph, id, {
    bvhNodes: (2 * leafCapacity - 1) * 8,
    bvhLeaves: leafCapacity * 4,
    ...(spatialSort && rows >= 2
      ? {
          sortedBounds: rows * 8,
          boundsPartials: Math.min(256, Math.max(1, Math.ceil(rows / 4096))) * 16,
          ...getGeometrySortScratchByteLengths(rows)
        }
      : {})
  });
}

/** Shared by segment intersections and the predicates/validity contributors that nest them. */
export function validateSegmentIntersectionScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  leftRows: number,
  rightRows: number,
  pairCapacity: number,
  leafCapacity?: number,
  spatialSort = true
): void {
  validateGeometryScratchByteLengths(graph, id, {
    segmentTables: Math.max(leftRows, rightRows) * 32,
    segmentBounds: rightRows * 8,
    pairColumns: pairCapacity * 4,
    countScan: leftRows * 4,
    state: 16
  });
  validateSegmentBVHScratch(graph, id, rightRows, leafCapacity, spatialSort);
}
