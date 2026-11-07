// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPUTerrainDepressionHierarchy,
  type GPUTerrainDepressionHierarchyProps
} from '@luma-contributors/graph/gpu-terrain';

/** Adds the public contributor with explicit caller-owned views. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPUTerrainDepressionHierarchyProps
): Promise<void> {
  // #region example
  graph.add(new GPUTerrainDepressionHierarchy(input));
  // #endregion example
}
