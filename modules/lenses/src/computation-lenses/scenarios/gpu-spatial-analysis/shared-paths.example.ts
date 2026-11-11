// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSharedPaths} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPULineGeometryFields} from '@luma-contributors/analysis/analysis-core';

/** Uploaded Chicago geometry and parameters for GPUSharedPaths. */
export type SharedPathsExampleInput = {
  left: GPULineGeometryFields;
  right: GPULineGeometryFields;
  intersectionCapacity: number;
};

/** Runs GPUSharedPaths and returns the views drawn or inspected by its card. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: SharedPathsExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const {left, right, intersectionCapacity} = input;
  const runs = {
    positions: createTransientView(
      graph,
      'sharedPaths',
      'float32x2',
      intersectionCapacity * 2,
      usage
    ),
    offsets: createTransientView(graph, 'runOffsets', 'uint32', intersectionCapacity + 1, usage),
    leftSourceIds: createTransientView(
      graph,
      'leftSourceIds',
      'uint32',
      intersectionCapacity,
      usage
    ),
    rightSourceIds: createTransientView(
      graph,
      'rightSourceIds',
      'uint32',
      intersectionCapacity,
      usage
    ),
    forward: createTransientView(graph, 'forward', 'uint32', intersectionCapacity, usage)
  };
  const status = {
    count: createTransientView(graph, 'count', 'uint32', 1, usage),
    overflow: createTransientView(graph, 'overflow', 'uint32', 1, usage),
    uncertainCount: createTransientView(graph, 'uncertainCount', 'uint32', 1, usage)
  };
  // #region example
  // Compare disjoint sets of community-area boundary rings in UTM 16N metres.
  graph.add(
    new GPUSharedPaths({
      left: {kind: 'lines', ...left},
      right: {kind: 'lines', ...right},
      intersectionCapacity,
      output: {runs, status}
    })
  );
  const {positions: sharedPaths, offsets: runOffsets} = runs;
  // #endregion example
  return {runs, status, sharedPaths, runOffsets};
}
