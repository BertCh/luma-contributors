// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUSegmentIntersection} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPUSpatialContext} from '@luma-contributors/analysis/analysis-core';

/** Inputs the lens scenario uploads from `s2-chicago-north-roads.json`. */
export type SegmentIntersectionExampleInput = {
  /** Street-edge vertices, projected to local-frame metres. */
  positions: GraphDataView<'float32x2'>;
  /** Start of each street edge in `positions`, plus the final end. */
  lineOffsets: GraphDataView<'uint32'>;
  /** Intersection points, filled with NaN so that unwritten rows are recognisable. */
  points: GraphDataView<'float32x2'>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Every intersecting pair of street segments, with the kind and point of each. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {positions, lineOffsets, points, spatialContext}: SegmentIntersectionExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const capacity = points.length;
  const leftRows = createTransientView(graph, 'leftRows', 'uint32', capacity, usage);
  const rightRows = createTransientView(graph, 'rightRows', 'uint32', capacity, usage);
  const kinds = createTransientView(graph, 'kinds', 'uint32', capacity, usage);
  const count = createTransientView(graph, 'count', 'uint32', 1, usage);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  const requiredCount = createTransientView(graph, 'requiredCount', 'uint32', 1, usage);
  const uncertainCount = createTransientView(graph, 'uncertainCount', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPUSegmentIntersection({
      spatialContext,
      left: {kind: 'lines', positions, lineOffsets}, // no `right`: self mode
      output: {
        pairs: {leftRows, rightRows, status: {count, overflow, requiredCount}},
        status: {uncertainCount},
        kinds,
        points
      }
    })
  );
  // points, coloured by kinds, are where street segments meet or cross
  // #endregion example
  return {leftRows, rightRows, kinds, count, overflow, requiredCount, uncertainCount};
}
