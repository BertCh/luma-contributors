// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUAnalysisSpacing} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the public spacing composition; inputs and summary belong to the caller's graph. */
export async function runExample(
  device: Device,
  input: {
    graph: GPUCommandGraph;
    positions: GraphDataView<'float32x2'>;
    summary: GraphDataView<'uint32'>;
  }
) {
  const {graph, positions, summary} = input;
  if (graph.device !== device) throw new Error('analysis-spacing: graph device must match device');
  // #region example
  graph.add(
    new GPUAnalysisSpacing({
      positions,
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      neighborRank: 3,
      gridSize: [4, 4],
      output: {summary}
    })
  );
  // Word 21 is the float32 maximum nearest-neighbor distance in the public summary layout.
  const nearestNeighborMaximum = graph.createDataView(summary.buffer, {
    format: 'float32',
    byteOffset: summary.byteOffset + 21 * 4,
    length: 1
  });
  // #endregion example
  return {nearestNeighborMaximum};
}
