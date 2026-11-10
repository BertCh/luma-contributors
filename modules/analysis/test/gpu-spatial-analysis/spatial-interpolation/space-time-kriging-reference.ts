// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {SpaceTimeVariogramModel} from '../../../src/dataframe-analysis/pair-statistics/space-time-variogram-model';
import {
  MATERN_3_2_RANGE_FACTOR,
  MATERN_5_2_RANGE_FACTOR
} from '../../../src/dataframe-analysis/pair-statistics/variogram-model';

type Structure = {model: string; nugget: number; sill: number; range: number};
export type Point = readonly [number, number];

function correlation(structure: Structure, lag: number): number {
  const scaled = lag / structure.range;
  switch (structure.model) {
    case 'spherical':
      return scaled >= 1 ? 0 : 1 - 1.5 * scaled + 0.5 * scaled ** 3;
    case 'exponential':
      return Math.exp(-3 * scaled);
    case 'gaussian':
      return Math.exp(-3 * scaled * scaled);
    case 'matern-3/2': {
      const argument = MATERN_3_2_RANGE_FACTOR * scaled;
      return (1 + argument) * Math.exp(-argument);
    }
    case 'matern-5/2': {
      const argument = MATERN_5_2_RANGE_FACTOR * scaled;
      return (1 + argument + (argument * argument) / 3) * Math.exp(-argument);
    }
    default:
      throw new Error(structure.model);
  }
}

function structureCovariance(structure: Structure, lag: number, coincident: boolean): number {
  return structure.sill * correlation(structure, lag) + (coincident ? structure.nugget : 0);
}

/** Binary64 covariance written from the gstat and Gneiting definitions. */
export function getReferenceSpaceTimeCovariance(
  model: SpaceTimeVariogramModel,
  spatialLag: number,
  timeLag: number
): number {
  const samePlace = spatialLag === 0;
  const sameTime = timeLag === 0;
  switch (model.type) {
    case 'separable':
      return (
        model.sill *
        structureCovariance(model.spatial, spatialLag, samePlace) *
        structureCovariance(model.temporal, timeLag, sameTime)
      );
    case 'product-sum': {
      const spatial = structureCovariance(model.spatial, spatialLag, samePlace);
      const temporal = structureCovariance(model.temporal, timeLag, sameTime);
      return model.k * spatial * temporal + spatial + temporal;
    }
    case 'metric':
      return structureCovariance(
        model.joint,
        Math.sqrt(spatialLag ** 2 + (model.anisotropy * timeLag) ** 2),
        samePlace && sameTime
      );
    case 'sum-metric':
      return (
        structureCovariance(model.spatial, spatialLag, samePlace) +
        structureCovariance(model.temporal, timeLag, sameTime) +
        structureCovariance(
          model.joint,
          Math.sqrt(spatialLag ** 2 + (model.anisotropy * timeLag) ** 2),
          samePlace && sameTime
        )
      );
    case 'gneiting': {
      const c = 3 / model.spatialRange ** (2 * model.gamma);
      const a = (Math.exp(3 / model.tau) - 1) / model.temporalRange ** (2 * model.alpha);
      const psi = a * timeLag ** (2 * model.alpha) + 1;
      return (
        (model.sill / psi ** model.tau) *
          Math.exp((-c * spatialLag ** (2 * model.gamma)) / psi ** (model.beta * model.gamma)) +
        (samePlace && sameTime ? model.nugget : 0)
      );
    }
    default:
      throw new Error('Unknown model');
  }
}

function solve(matrix: Float64Array[], right: Float64Array): Float64Array {
  const size = right.length;
  for (let column = 0; column < size; column++) {
    let pivotRow = column;
    for (let row = column + 1; row < size; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivotRow][column])) pivotRow = row;
    }
    [matrix[column], matrix[pivotRow]] = [matrix[pivotRow], matrix[column]];
    [right[column], right[pivotRow]] = [right[pivotRow], right[column]];
    for (let row = column + 1; row < size; row++) {
      const factor = matrix[row][column] / matrix[column][column];
      for (let entry = column; entry < size; entry++)
        matrix[row][entry] -= factor * matrix[column][entry];
      right[row] -= factor * right[column];
    }
  }
  const solution = new Float64Array(size);
  for (let row = size - 1; row >= 0; row--) {
    let value = right[row];
    for (let column = row + 1; column < size; column++)
      value -= matrix[row][column] * solution[column];
    solution[row] = value / matrix[row][row];
  }
  return solution;
}

export type ReferenceScene = {
  positions: readonly Point[];
  times: readonly number[];
  observations: readonly (readonly [number, number])[];
};

/** Independent binary64 selection with the documented search rules. */
export function selectReferenceNeighbors(
  scene: ReferenceScene,
  query: Point,
  queryTime: number,
  covariance: SpaceTimeVariogramModel,
  options: {
    search: 'metric' | 'octant' | 'staged';
    neighborCount: number;
    searchAnisotropy: number;
    neighborsPerOctant: number;
    stagedCandidateCount: number;
  }
): number[] {
  const candidates = scene.positions
    .map((position, row) => {
      const spatialLag = Math.hypot(position[0] - query[0], position[1] - query[1]);
      const timeLag = Math.abs(scene.times[row] - queryTime);
      return {
        row,
        spatialLag,
        timeLag,
        metric: Math.sqrt(spatialLag ** 2 + (options.searchAnisotropy * timeLag) ** 2)
      };
    })
    .sort((first, second) => first.metric - second.metric || first.row - second.row);
  if (options.search === 'metric') {
    return candidates.slice(0, options.neighborCount).map(candidate => candidate.row);
  }
  if (options.search === 'octant') {
    const perOctant = new Map<number, typeof candidates>();
    for (const candidate of candidates) {
      const position = scene.positions[candidate.row];
      const octant =
        (position[0] - query[0] < 0 ? 1 : 0) +
        (position[1] - query[1] < 0 ? 2 : 0) +
        (scene.times[candidate.row] < queryTime ? 4 : 0);
      const list = perOctant.get(octant) ?? [];
      if (list.length < options.neighborsPerOctant) {
        list.push(candidate);
        perOctant.set(octant, list);
      }
    }
    return [...perOctant.values()]
      .flat()
      .sort((first, second) => first.metric - second.metric || first.row - second.row)
      .slice(0, options.neighborCount)
      .map(candidate => candidate.row);
  }
  return candidates
    .slice(0, options.stagedCandidateCount)
    .map((candidate, order) => ({
      ...candidate,
      order,
      covarianceValue: getReferenceSpaceTimeCovariance(
        covariance,
        candidate.spatialLag,
        candidate.timeLag
      )
    }))
    .sort(
      (first, second) =>
        second.covarianceValue - first.covarianceValue || first.order - second.order
    )
    .slice(0, options.neighborCount)
    .map(candidate => candidate.row);
}

/** Independent binary64 solve on an explicit neighbor set. */
export function solveReferenceSpaceTimeKriging(props: {
  scene: ReferenceScene;
  query: Point;
  queryTime: number;
  covariance: SpaceTimeVariogramModel;
  rows: readonly number[];
  mode: 'simple' | 'ordinary';
  mean?: number;
}): {prediction: number; variance: number} {
  const {scene, query, queryTime, covariance, rows, mode} = props;
  const mean = props.mean ?? 0;
  const count = rows.length;
  const size = count + (mode === 'ordinary' ? 1 : 0);
  const matrix = Array.from({length: size}, () => new Float64Array(size));
  const right = new Float64Array(size);
  const cross = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    const first = scene.positions[rows[row]];
    for (let column = 0; column < count; column++) {
      const second = scene.positions[rows[column]];
      matrix[row][column] = getReferenceSpaceTimeCovariance(
        covariance,
        Math.hypot(first[0] - second[0], first[1] - second[1]),
        Math.abs(scene.times[rows[row]] - scene.times[rows[column]])
      );
    }
    matrix[row][row] += scene.observations[rows[row]][1];
    cross[row] = getReferenceSpaceTimeCovariance(
      covariance,
      Math.hypot(first[0] - query[0], first[1] - query[1]),
      Math.abs(scene.times[rows[row]] - queryTime)
    );
    right[row] = cross[row];
    if (mode === 'ordinary') {
      matrix[row][count] = 1;
      matrix[count][row] = 1;
    }
  }
  if (mode === 'ordinary') {
    right[count] = 1;
  }
  const solution = solve(matrix, right);
  let prediction = mode === 'ordinary' ? 0 : mean;
  let variance = getReferenceSpaceTimeCovariance(covariance, 0, 0);
  for (let row = 0; row < count; row++) {
    prediction +=
      solution[row] * (scene.observations[rows[row]][0] - (mode === 'ordinary' ? 0 : mean));
    variance -= solution[row] * cross[row];
  }
  if (mode === 'ordinary') {
    variance -= solution[count];
  }
  return {prediction, variance};
}
