// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUAnalysisContiguityStatistics,
  type GPUAnalysisContiguityStatisticsProps
} from '@luma-contributors/analysis/gpu-spatial-analysis';

/** Adds the public all-feature contiguity profile with caller-owned outputs. */
export async function runExample(
  device: Device,
  input: {graph: GPUCommandGraph} & Pick<
    GPUAnalysisContiguityStatisticsProps,
    'polygons' | 'output'
  >
) {
  const {graph, polygons, output} = input;
  if (graph.device !== device)
    throw new Error('analysis-contiguity-statistics: graph device must match device');
  // #region example
  graph.add(
    new GPUAnalysisContiguityStatistics({
      criterion: 'queen',
      polygons,
      output
    })
  );
  // Counts and median include zero-neighbor islands. Inspect validity and both overflow words
  // before interpreting them; decodeGPUAnalysisContiguityStatistics refuses truncated profiles.
  // #endregion example
  return output;
}
