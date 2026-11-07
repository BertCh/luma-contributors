// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUParameterBuffer,
  GPUTerrainSummits,
  getGPUTerrainSummitsParameterValues,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-terrain';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainSummitsExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  summitMask: GraphDataView<'uint32'>;
  drop: GraphDataView<'float32'>;
};

/** Adds summit detection on `dem` to `graph`; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainSummitsExampleInput) {
  const {graph, grid, dem, summitMask, drop} = input;
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'summits-parameters',
    format: 'float32',
    length: GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
    values: getGPUTerrainSummitsParameterValues({radius: 600, minimumDrop: 5}, grid) // m
  });
  graph.add(
    new GPUTerrainSummits({
      grid, // 64 × 64 cells of 120 m, UTM 10N
      elevation: makeGPURasterBand(dem, grid),
      parameters: parameters.importToGraph(graph),
      maximumRadiusPixels: 8,
      output: {summitMask, drop} // 1 on a summit, drop in m (NaN elsewhere)
    })
  );
  // #endregion example
  return {parameters};
}
