// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';
import {
  GPUPolygonRasterization,
  GPURasterJoin,
  GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
  getGPUPolygonRasterizationParameterValues
} from '@luma-contributors/analysis/raster-analysis';

/** Graph, grid and caller-owned views for raster-join. */
export type RasterJoinExampleInput = {
  graph: GPUCommandGraph;
  grid: GPURasterMetadata;
  /** GeoArrow layout of the perimeter polygons. */
  perimeter: {
    positions: GraphDataView<'float32x2'>;
    featureOffsets: GraphDataView<'uint32'>;
    polygonOffsets: GraphDataView<'uint32'>;
    ringOffsets: GraphDataView<'uint32'>;
  };
  burnedCells: GraphDataView<'float32x2'>;
  zones: GraphDataView<'uint32'>;
  overflow: GraphDataView<'uint32'>;
  /** Receives 1 on rasterization loss; otherwise a graph-owned status view is returned. */
  incomplete?: GraphDataView<'uint32'>;
  counts: GraphDataView<'uint32'>;
  outsideCount: GraphDataView<'uint32'>;
  pointZones: GraphDataView<'uint32'>;
};

/** Adds the recorded raster-join computation; returns its owned parameter buffer. */
export async function runExample(device: Device, input: RasterJoinExampleInput) {
  const {graph, grid, perimeter, burnedCells, zones, overflow} = input;
  const {counts, outsideCount, pointZones} = input;
  const incomplete = input.incomplete ?? createTransientView(graph, 'join-incomplete', 'uint32', 1);
  // #region example
  const parameters = new GPUParameterBuffer(device, {
    id: 'join-parameters',
    format: 'float32',
    length: GPU_POLYGON_RASTERIZATION_PARAMETER_LENGTH,
    values: getGPUPolygonRasterizationParameterValues(grid)
  });
  const raster = {grid, parameters: parameters.importToGraph(graph)}; // 64 × 64 cells of 120 m
  graph.add(
    new GPUPolygonRasterization({
      ...raster,
      ...perimeter, // positions and offsets of the Dixie perimeter, UTM metres
      crossingCapacity: 256,
      output: {zones, status: {overflow}}
    })
  );
  graph.add(
    new GPURasterJoin({
      ...raster,
      zones,
      producerOverflow: overflow,
      zoneCount: 1, // one feature: the perimeter
      points: burnedCells, // centres of cells with dNBR above 0.27
      output: {counts, outsideCount, pointZones, status: {incomplete}}
    })
  );
  // #endregion example
  return {parameters, incomplete};
}
