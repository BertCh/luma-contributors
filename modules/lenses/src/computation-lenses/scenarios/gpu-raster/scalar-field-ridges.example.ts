// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUScalarFieldRidges,
  type GPUScalarFieldRidgesOutput,
  GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH,
  getGPUScalarFieldRidgesParameterValues
} from '@luma-contributors/analysis/raster-analysis';

/** Adds the ridge detector; the caller owns the band and outputs and destroys the returned buffer. */
export async function runExample(
  device: Device,
  graph: GPUCommandGraph,
  grid: GPURasterMetadata,
  input: GPURasterBand,
  output: GPUScalarFieldRidgesOutput
) {
  const parameters = new GPUParameterBuffer(device, {
    id: 'scalar-field-ridges-parameters',
    format: 'float32',
    length: GPU_SCALAR_FIELD_RIDGES_PARAMETER_LENGTH
  });
  // #region example
  parameters.write(getGPUScalarFieldRidgesParameterValues({minimumCurvature: 1}, grid));
  graph.add(
    new GPUScalarFieldRidges({
      input,
      grid, // orthogonal metre grid; uniform spacing
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      cellSizeMode: 'uniform',
      parameters: parameters.importToGraph(graph),
      output // ridge mask, stencil validity and downward curvature, one row per cell
    })
  );
  // #endregion example
  return parameters;
}
