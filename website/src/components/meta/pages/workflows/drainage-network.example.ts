// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// example:hide-start
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {getGPUTerrainFlowParameterValues} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterBand} from '@luma-contributors/analysis/raster-analysis';

declare const graph: GPUCommandGraph<void>;
/** The elevation raster's grid and its values, already a view in `graph`. */
declare const grid: GPURasterMetadata;
declare const elevation: GraphDataView<'float32'>;
// example:hide-end
// A cell is a stream where at least 200 cells drain through it.
const values = getGPUTerrainFlowParameterValues({streamThreshold: 200}, grid);
const flowParameters = new GPUParameterBuffer(graph.device, {
  id: 'flow-parameters',
  format: 'float32',
  length: values.length,
  values
});
const drainage = addDrainageNetworkRecipe(graph, {
  flow: {
    grid,
    elevation: makeGPURasterBand(elevation, grid),
    parameters: flowParameters.importToGraph(graph),
    fillDepressions: true,
    resolveFlats: true,
    accumulationUnits: 'cells'
  },
  maximumSteps: grid.width * grid.height
});
const {streamOrder, catchments, network} = drainage.outputs;
// example:hide-start
export {catchments, network, streamOrder};
// example:hide-end
