// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  GPURasterRegistration,
  type GPURasterRegistrationProps
} from '@luma-contributors/graph/gpu-raster';

/** Adds the public contributor with explicit caller-owned buffers and model options. */
export async function runExample(
  graph: GPUCommandGraph,
  input: GPURasterRegistrationProps
): Promise<void> {
  // #region example
  graph.add(new GPURasterRegistration(input));
  // #endregion example
}
