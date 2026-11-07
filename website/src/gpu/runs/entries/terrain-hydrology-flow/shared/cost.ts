// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Shared inputs for the cost-surface runs: a friction raster with a wall (two gaps) across the map,
// source cells on the left, and the real GPUCostDistance with its per-frame settings.

import {
  getGPUCostDistanceParameterValues,
  GPUCostDistance,
  makeGPURasterBand,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import type {ParameterValues, RunBuffer, RunBuilder, RunInputContext} from '../../../define-run';
import {createFractalRaster} from '../../../inputs';

type RasterGrid = ReturnType<typeof makeGPURasterMetadata>;

export const SOURCE_CAPACITY = 4;

export type CostInputs = {
  columns: number;
  rows: number;
  cellCount: number;
  cellSize: number;
  grid: RasterGrid;
  /** Cost per world unit for a contrast (1 plus contrast times roughness); NaN is the wall. */
  getFriction(contrast: number): Float32Array;
  /** Source cell indices, the first one in the middle of the left edge. */
  sources: Uint32Array;
  /** Drawing-world position of every cell center. */
  cellCenters: Float32Array;
};

export function createCostInputs(context: RunInputContext, rows = 48): CostInputs {
  const {random, world} = context;
  const field = createFractalRaster(random, world, rows, 4);
  const {columns, cellSize} = field;
  const cellCount = columns * rows;
  const roughness = new Float32Array(cellCount);
  const isWallCell = new Uint8Array(cellCount);
  const cellCenters = new Float32Array(cellCount * 2);
  const wallColumn = Math.round(columns * 0.55);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      const rowFraction = row / rows;
      const isWall =
        Math.abs(column - wallColumn) <= 1 &&
        rowFraction > 0.04 &&
        rowFraction < 0.96 &&
        !(rowFraction > 0.2 && rowFraction < 0.3) &&
        !(rowFraction > 0.72 && rowFraction < 0.8);
      isWallCell[cell] = isWall ? 1 : 0;
      roughness[cell] = field.values[cell] ** 2;
      cellCenters[cell * 2] = (column + 0.5) * cellSize;
      cellCenters[cell * 2 + 1] = world.height - (row + 0.5) * cellSize;
    }
  }
  const sources = Uint32Array.from(
    [0.5, 0.2, 0.8, 0.35]
      .slice(0, SOURCE_CAPACITY)
      .map(
        (rowFraction, index) =>
          Math.floor(rowFraction * rows) * columns +
          Math.floor(columns * (index === 0 ? 0.12 : 0.2))
      )
  );
  return {
    columns,
    rows,
    cellCount,
    cellSize,
    grid: makeGPURasterMetadata({
      width: columns,
      height: rows,
      cellSize,
      origin: [0, rows * cellSize]
    }),
    getFriction: contrast =>
      Float32Array.from(roughness, (value, cell) =>
        isWallCell[cell] ? Number.NaN : 1 + contrast * value
      ),
    sources,
    cellCenters
  };
}

export type CostDistanceBuffers = {
  friction: RunBuffer<'float32'>;
  costs: RunBuffer<'float32'>;
  backLinks: RunBuffer<'uint32'>;
  sources: RunBuffer<'float32x2'>;
  /** Writes the per-frame settings. */
  writeSettings(values: ParameterValues): void;
};

/** Uploads the inputs and adds the real GPUCostDistance with costs and back-links. */
export function addCostDistance(builder: RunBuilder, inputs: CostInputs): CostDistanceBuffers {
  const {grid, cellCount} = inputs;
  const friction = builder.upload('friction', 'float32', inputs.getFriction(7));
  const frictionParameters = builder.allocate('friction-parameters', 'float32', 2);
  const parameters = builder.allocate('parameters', 'float32', 8);
  const sources = builder.upload('sources', 'uint32', inputs.sources);
  const sourceCount = builder.allocate('source-count', 'uint32', 1);
  const costs = builder.allocate('costs', 'float32', cellCount);
  const backLinks = builder.allocate('back-links', 'uint32', cellCount);
  builder.graph.add(
    new GPUCostDistance({
      grid,
      friction: makeGPURasterBand(friction.view, grid, {id: 'friction'}),
      frictionParameters: frictionParameters.view,
      parameters: parameters.view,
      sources: sources.view,
      sourceCount: sourceCount.view,
      output: {
        costs: costs.view,
        backLinks: backLinks.view,
        status: {converged: builder.allocate('converged', 'uint32', 1).view}
      }
    })
  );
  const sourcePositions = new Float32Array(SOURCE_CAPACITY * 2);
  inputs.sources.forEach((cell, index) => {
    sourcePositions[index * 2] = inputs.cellCenters[cell * 2];
    sourcePositions[index * 2 + 1] = inputs.cellCenters[cell * 2 + 1];
  });
  return {
    friction,
    costs,
    backLinks,
    sources: builder.upload('source-positions', 'float32x2', sourcePositions),
    writeSettings: values => {
      if (values.contrast !== undefined) friction.write(inputs.getFriction(values.contrast));
      frictionParameters.write(Float32Array.of(values.frictionScale ?? 1, 0));
      parameters.write(
        getGPUCostDistanceParameterValues({costLimit: values.costLimit ?? Infinity}, grid)
      );
      sourceCount.write(Uint32Array.of(Math.round(values.sourceCount ?? 1)));
    }
  };
}
