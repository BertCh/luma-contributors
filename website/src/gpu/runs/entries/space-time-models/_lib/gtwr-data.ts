// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Synthetic space-time data on Chicago tracts for the two space-time regression runs: every tract is
// observed in every period, the response is intercept + slope * poverty share + noise, and the
// coefficients are known surfaces that drift in space and time. The tract map of each (component,
// period) pair is one tile of a grid; all tiles are one mesh so a single polygon layer draws them.

import {type GPUTemporalContext} from '@luma-contributors/analysis/analysis-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import {createTracts, type Random, type TractsInput, type World} from '../../../inputs';

/** Time stamps are period numbers, so one tick is one period. */
export const PERIOD_TEMPORAL_CONTEXT: GPUTemporalContext = {
  unit: 'second',
  epoch: 0n,
  encoding: 'float32-relative',
  axis: {kind: 'irregular'}
};

export const PERIOD_COUNT = 4;
const TRACT_ROWS = 10;
const NOISE = 0.5;

export type SpaceTimeMeshData = {
  vertices: Float32Array;
  indices: Uint32Array;
  owners: Uint32Array;
  triangleCount: number;
  outline: Float32Array;
  outlineCount: number;
};

export type SpaceTimeScene = {
  tracts: TractsInput;
  tractCount: number;
  rowCount: number;
  tileColumns: number;
  tileRows: number;
  tileWidth: number;
  tileHeight: number;
  /** `(x, y)` of every observation row (`period * tractCount + tract`) inside one tile. */
  positions: Float32Array;
  /** Period number of every row. */
  times: Float32Array;
  /** Poverty share of every row, the one predictor. */
  predictors: Float32Array;
  response: Float32Array;
  trueIntercept: Float32Array;
  trueSlope: Float32Array;
  /** One mesh per tile row, its tiles being the periods; a triangle's owner is the row of its tract in its period. */
  meshes: SpaceTimeMeshData[];
  /** World center of each tile. */
  tileCenters: [number, number][];
};

function getSigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

/** The slope front sweeps west to east: high slopes behind it, low ahead of it. */
export function getTrueSlope(unitX: number, period: number): number {
  return 0.3 + 2.8 * getSigmoid(-(unitX - (0.2 + 0.2 * period)) / 0.1);
}

/** The intercept rises gently to the north and with time. */
export function getTrueIntercept(unitY: number, period: number): number {
  return 0.8 + 0.8 * unitY + 0.15 * period;
}

function getNormal(random: Random): number {
  let sum = 0;
  for (let draw = 0; draw < 6; draw++) sum += random();
  return (sum - 3) / Math.sqrt(0.5);
}

/**
 * `tileColumns x tileRows` tiles fill the picture. With `periodsFillGrid` the tiles in reading order
 * are the periods and form one mesh; otherwise each tile row is one component (the top one first),
 * its tiles are the periods, and every tile row is its own mesh.
 */
export function createSpaceTimeScene(
  random: Random,
  world: World,
  layout: {
    tileColumns: number;
    tileRows: number;
    periodsFillGrid: boolean;
    /** Height at the bottom of the world left free for a panel. */
    bottomMargin?: number;
  }
): SpaceTimeScene {
  const {tileColumns, tileRows, periodsFillGrid} = layout;
  const tileWidth = world.width / tileColumns;
  const bottomMargin = layout.bottomMargin ?? 0;
  const tileHeight = (world.height - bottomMargin) / tileRows;
  const tracts = createTracts(random, {width: tileWidth, height: tileHeight}, TRACT_ROWS);
  const tractCount = tracts.count;
  const rowCount = tractCount * PERIOD_COUNT;
  const positions = new Float32Array(rowCount * 2);
  const times = new Float32Array(rowCount);
  const predictors = new Float32Array(rowCount);
  const response = new Float32Array(rowCount);
  const trueIntercept = new Float32Array(rowCount);
  const trueSlope = new Float32Array(rowCount);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    for (let tract = 0; tract < tractCount; tract++) {
      const row = period * tractCount + tract;
      const x = tracts.centers[tract * 2];
      const y = tracts.centers[tract * 2 + 1];
      positions.set([x, y], row * 2);
      times[row] = period;
      predictors[row] = tracts.values[tract];
      trueIntercept[row] = getTrueIntercept(y / tileHeight, period);
      trueSlope[row] = getTrueSlope(x / tileWidth, period);
      response[row] =
        trueIntercept[row] + trueSlope[row] * predictors[row] + NOISE * getNormal(random);
    }
  }
  const meshes: SpaceTimeMeshData[] = [];
  const tileCenters: [number, number][] = [];
  const groupCount = periodsFillGrid ? 1 : tileRows;
  const tilesPerGroup = periodsFillGrid ? tileColumns * tileRows : tileColumns;
  for (let group = 0; group < groupCount; group++) {
    const vertices = new Float32Array(tilesPerGroup * tracts.vertices.length);
    const indices = new Uint32Array(tilesPerGroup * tracts.triangleIndices.length);
    const owners = new Uint32Array(tilesPerGroup * tracts.triangleOwners.length);
    const outline = new Float32Array(tilesPerGroup * tracts.outline.length);
    const vertexCount = tracts.vertices.length / 2;
    for (let column = 0; column < tilesPerGroup; column++) {
      const period = column;
      const tileRow = periodsFillGrid ? Math.floor(column / tileColumns) : group;
      const offsetX = (column % tileColumns) * tileWidth;
      const offsetY = world.height - (tileRow + 1) * tileHeight;
      tileCenters.push([offsetX + tileWidth / 2, offsetY + tileHeight / 2]);
      for (let index = 0; index < tracts.vertices.length; index += 2) {
        vertices[column * tracts.vertices.length + index] = tracts.vertices[index] + offsetX;
        vertices[column * tracts.vertices.length + index + 1] =
          tracts.vertices[index + 1] + offsetY;
      }
      for (let index = 0; index < tracts.triangleIndices.length; index++) {
        indices[column * tracts.triangleIndices.length + index] =
          tracts.triangleIndices[index] + column * vertexCount;
      }
      for (let index = 0; index < tracts.triangleOwners.length; index++) {
        owners[column * tracts.triangleOwners.length + index] =
          tracts.triangleOwners[index] + period * tractCount;
      }
      for (let index = 0; index < tracts.outline.length; index += 4) {
        const target = column * tracts.outline.length + index;
        outline[target] = tracts.outline[index] + offsetX;
        outline[target + 1] = tracts.outline[index + 1] + offsetY;
        outline[target + 2] = tracts.outline[index + 2] + offsetX;
        outline[target + 3] = tracts.outline[index + 3] + offsetY;
      }
    }
    meshes.push({
      vertices,
      indices,
      owners,
      triangleCount: tilesPerGroup * tracts.triangleOwners.length,
      outline,
      outlineCount: tilesPerGroup * tracts.outlineCount
    });
  }
  return {
    tracts,
    tractCount,
    rowCount,
    tileColumns,
    tileRows,
    tileWidth,
    tileHeight,
    positions,
    times,
    predictors,
    response,
    trueIntercept,
    trueSlope,
    meshes,
    tileCenters
  };
}

/** The buffers of the observations every space-time regression reads. */
export function uploadSpaceTimeObservations(builder: RunBuilder, scene: SpaceTimeScene) {
  return {
    positions: builder.upload('positions', 'float32x2', scene.positions),
    times: builder.upload('times', 'float32', scene.times),
    predictors: builder.upload('predictors', 'float32', scene.predictors),
    response: builder.upload('response', 'float32', scene.response)
  };
}

/** The buffers of one mesh group (the only one, or one tile row). */
export function uploadSpaceTimeMesh(builder: RunBuilder, scene: SpaceTimeScene, group: number) {
  const mesh = scene.meshes[group];
  return {
    vertices: builder.upload(`tile-vertices-${group}`, 'float32x2', mesh.vertices),
    indices: builder.upload(`tile-indices-${group}`, 'uint32', mesh.indices),
    owners: builder.upload(`tile-owners-${group}`, 'uint32', mesh.owners),
    outline: builder.upload(`tile-outline-${group}`, 'float32x4', mesh.outline)
  };
}

export type SpaceTimeMesh = ReturnType<typeof uploadSpaceTimeMesh>;

/** Polygon layer of one per-row value buffer over the tiles of one mesh group, plus the tract outlines. */
export function drawSpaceTimeTiles(
  scene: SpaceTimeScene,
  group: number,
  mesh: SpaceTimeMesh,
  values: RunBuffer,
  range: readonly [number, number]
) {
  return [
    {
      layer: 'polygons' as const,
      positions: mesh.vertices,
      indices: mesh.indices,
      owners: mesh.owners,
      values,
      triangleCount: scene.meshes[group].triangleCount,
      ramp: 'sequential' as const,
      range,
      bands: 10
    },
    {
      layer: 'lines' as const,
      context: true as const,
      segments: mesh.outline,
      count: scene.meshes[group].outlineCount,
      alpha: 0.5
    }
  ];
}
