// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  GPUTerrainFlow,
  GPUTerrainHeightAboveDrainage,
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-terrain';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

/** Graph, grid and caller-owned views the example wires together. */
export type TerrainHeightAboveDrainageExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  dem: GraphDataView<'float32'>;
  filledElevation: GraphDataView<'float32'>;
  flowDirections: GraphDataView<'uint32'>;
  streams: GraphDataView<'uint32'>;
  heightAboveDrainage: GraphDataView<'float32'>;
  drainageCells: GraphDataView<'uint32'>;
  converged: GraphDataView<'uint32'>;
  iterationCount: GraphDataView<'uint32'>;
};

/** Adds HAND of `dem` to `graph` on top of a GPUTerrainFlow; returns the parameter buffer to destroy. */
export async function runExample(device: Device, input: TerrainHeightAboveDrainageExampleInput) {
  const {
    graph,
    grid,
    dem,
    filledElevation,
    flowDirections,
    streams,
    heightAboveDrainage,
    drainageCells,
    converged,
    iterationCount
  } = input;
  // Prerequisite: filled D8 routing of the DEM (see GPUTerrainFlow).
  const flowParameters = new GPUParameterBuffer(device, {
    id: 'hand-flow-parameters',
    format: 'float32',
    length: GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    values: getGPUTerrainFlowParameterValues({fillEpsilon: 0.01, streamThreshold: 40}, grid)
  });
  graph.add(
    new GPUTerrainFlow({
      id: 'hand-flow',
      grid,
      elevation: makeGPURasterBand(dem, grid),
      parameters: flowParameters.importToGraph(graph),
      fillDepressions: true,
      maximumFillIterations: 64,
      output: {filledElevation, flowDirections, streams}
    })
  );
  // #region example
  graph.add(
    new GPUTerrainHeightAboveDrainage({
      grid,
      elevation: makeGPURasterBand(filledElevation, grid, {id: 'filled'}), // m
      flowDirections, // ESRI D8 codes
      streams, // 1 on stream cells
      output: {heightAboveDrainage, drainageCells, status: {converged, iterationCount}}
    })
  );
  // #endregion example
  return {parameters: flowParameters};
}
