// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  addResistanceConnectivityRecipe,
  addSparseResistanceConnectivityRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';

export type CircuitConnectivity = 4 | 8;

export type CircuitEdgeInputs = {
  edges: Uint32Array;
  conductances: Float32Array;
  edgeLengthMeters: Float32Array;
};

/**
 * Independent CPU assembly of the declared raster edge contract. The returned arrays are uploaded
 * to the GPU solvers; this function never computes circuit potentials or currents.
 */
export function makeCircuitEdgeInputs(
  resistance: ArrayLike<number>,
  width: number,
  height: number,
  cellSize: readonly [number, number],
  connectivity: CircuitConnectivity
): CircuitEdgeInputs {
  const directions = connectivity === 8 ? 4 : 2;
  const offsets: readonly (readonly [number, number])[] =
    connectivity === 8
      ? [
          [1, 0],
          [0, 1],
          [1, 1],
          [-1, 1]
        ]
      : [
          [1, 0],
          [0, 1]
        ];
  const edgeCount = width * height * directions;
  const edges = new Uint32Array(edgeCount * 2);
  const conductances = new Float32Array(edgeCount);
  const edgeLengthMeters = new Float32Array(edgeCount);
  for (let cell = 0; cell < width * height; cell++) {
    const column = cell % width;
    const row = Math.floor(cell / width);
    for (let direction = 0; direction < directions; direction++) {
      const edgeIndex = cell * directions + direction;
      const [columnOffset, rowOffset] = offsets[direction];
      const nextColumn = column + columnOffset;
      const nextRow = row + rowOffset;
      const neighbor =
        nextColumn >= 0 && nextColumn < width && nextRow >= 0 && nextRow < height
          ? nextRow * width + nextColumn
          : cell;
      const edgeLength = Math.hypot(columnOffset * cellSize[0], rowOffset * cellSize[1]);
      edges[edgeIndex * 2] = cell;
      edges[edgeIndex * 2 + 1] = neighbor;
      edgeLengthMeters[edgeIndex] = edgeLength;
      const first = resistance[cell];
      const second = resistance[neighbor];
      if (
        neighbor !== cell &&
        Number.isFinite(first) &&
        Number.isFinite(second) &&
        first > 0 &&
        second > 0
      ) {
        conductances[edgeIndex] = 2 / (first + second) / edgeLength;
      }
    }
  }
  return {edges, conductances, edgeLengthMeters};
}

/** Calls both independent GPU solver recipes over the exact same uploaded edge arrays. */
export function addCircuitOracleRecipes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  inputs: {
    edges: GraphDataView<'uint32x2'>;
    conductances: GraphDataView<'float32'>;
    vertexCount: number;
    source: number;
    sink: number;
    maximumIterations: number;
    tolerance: number;
  }
) {
  const recipeProps = {
    edges: inputs.edges,
    conductances: inputs.conductances,
    vertexCount: inputs.vertexCount,
    source: inputs.source,
    sink: inputs.sink,
    maximumIterations: inputs.maximumIterations,
    tolerance: inputs.tolerance
  };
  const dense = addResistanceConnectivityRecipe(graph, {
    ...recipeProps,
    id: 'habitat-circuit-dense-reference'
  });
  const sparse = addSparseResistanceConnectivityRecipe(graph, {
    ...recipeProps,
    id: 'habitat-circuit-sparse-reference'
  });
  return {dense, sparse};
}
