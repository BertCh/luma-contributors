// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/**
 * Independent binary-rational fixtures, query (0,0), application IDs [900,100].
 * At squared distance 1 the f32 spacing is 2^-23, midpoint 1+2^-24.
 * 1+(2^-13)^2 rounds to 1, so the farther row 0 still wins a rank tie.
 * (2^-12-2^-36)^2 < 2^-24 < (2^-12+2^-35)^2: the last pair
 * straddles that midpoint and row 1 wins. Neither test orders by application ID.
 */
export const NEAREST_RANK_BOUNDARY_CASES = [
  {
    name: 'exact tie keeps the lowest feature row',
    positions: [1, 0, -1, 0],
    order: [900, 100],
    row: 0
  },
  {
    name: 'near pair sharing a rank keeps the farther lowest row',
    positions: [1, 2 ** -13, 1, 0],
    order: [900, 100],
    row: 0
  },
  {
    name: 'opposite sides of a rank boundary order by rank',
    positions: [1, 2 ** -12 + 2 ** -35, 1, 2 ** -12 - 2 ** -36],
    order: [100, 900],
    row: 1
  }
];

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import type {JoinReviewCase} from './join-review-response-fixture';

/** Exact geometry: the direction is (3,4), hence distance |4*2r-3*r|/5 = r. */
export function createNearestFinalWaveCases(): JoinReviewCase[] {
  const cases: JoinReviewCase[] = [];
  const radius = 2 - 2 ** -23;
  for (const neighbors of [false, true])
    for (const reverse of [false, true])
      for (const boundary of [true, false])
        cases.push(
          createCase(
            `KI-rr-joins-R1 neighbors=${neighbors} reverse=${reverse} boundary=${boundary}`,
            neighbors,
            [2 * radius, radius],
            reverse ? [10068795, 13425060] : [0, 0],
            reverse ? [0, 0] : [10068795, 13425060],
            boundary ? radius : radius - 2 ** -23,
            boundary,
            radius
          )
        );
  for (const reverse of [false, true]) {
    // Dot((2^25,-2^25), (1,1)) = 0, so the foot is exactly the origin.
    // Python struct.unpack('f', struct.pack('f', sqrt(2) * 2**25))[0] = 47453132.
    cases.push(
      createCase(
        `KI-rr-joins-R4 symmetric reverse=${reverse}`,
        true,
        [2 ** 25, -(2 ** 25)],
        reverse ? [3, 3] : [-3, -3],
        reverse ? [-3, -3] : [3, 3],
        2 ** 26,
        true,
        47453132,
        [0, 0]
      )
    );
    // The entire segment lies on y=x and straddles the origin, so the same perpendicular
    // query has exact foot (0,0); endpoint fractions 6/19 and -13/19 are not binary-exact.
    cases.push(
      createCase(
        `KI-rr-joins-R4 asymmetric long diagonal reverse=${reverse}`,
        true,
        [33554432, -33554432],
        reverse ? [13000000, 13000000] : [-6000000, -6000000],
        reverse ? [-6000000, -6000000] : [13000000, 13000000],
        67108864,
        true,
        47453132,
        [0, 0]
      )
    );
    // A tangential query shift of four moves the exact foot to (4,4), preserving its distance.
    cases.push(
      createCase(
        `KI-rr-joins-R4 asymmetric nonzero foot reverse=${reverse}`,
        true,
        [33554436, -33554428],
        reverse ? [13000000, 13000000] : [-6000000, -6000000],
        reverse ? [-6000000, -6000000] : [13000000, 13000000],
        67108864,
        true,
        47453132,
        [4, 4]
      )
    );
    cases.push(
      createCase(
        `KI-rr-joins-R4 horizontal control reverse=${reverse}`,
        true,
        [1, 8192],
        reverse ? [2, 2 ** -13] : [0, 2 ** -13],
        reverse ? [0, 2 ** -13] : [2, 2 ** -13],
        10000,
        true,
        8192,
        [1, 2 ** -13]
      )
    );
  }
  return cases;
}

function createCase(
  name: string,
  neighbors: boolean,
  query: number[],
  start: number[],
  end: number[],
  radius: number,
  hit: boolean,
  distance: number,
  foot?: number[]
): JoinReviewCase {
  return {
    name,
    inputs: {
      query: {format: 'float32x2', values: Float32Array.from(query)},
      starts: {format: 'float32x2', values: Float32Array.from(start)},
      ends: {format: 'float32x2', values: Float32Array.from(end)},
      sourceIds: {format: 'uint32', values: Uint32Array.of(33)},
      radius: {format: 'float32', values: Float32Array.of(radius)}
    },
    outputs: {
      ids: {format: 'uint32', length: 1},
      rows: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      distances: {format: 'float32', length: 1},
      counts: {format: 'uint32', length: 1},
      ...(!neighbors
        ? {
            matchIds: {format: 'uint32' as const, length: 1},
            matchOverflow: {format: 'uint32' as const, length: 1}
          }
        : {}),
      ...(foot
        ? {
            foot: {format: 'float32x2' as const, length: 1},
            queryPoint: {format: 'float32x2' as const, length: 1}
          }
        : {})
    },
    expected: {
      ids: [hit ? 33 : 0xffffffff],
      rows: [hit ? 0 : 0xffffffff],
      overflow: [0],
      distances: [hit ? distance : -1],
      counts: [Number(hit)],
      ...(!neighbors ? {matchOverflow: [0], ...(hit ? {matchIds: [0]} : {})} : {}),
      ...(foot ? {foot, queryPoint: query} : {})
    },
    tolerance: foot ? 4 : 2 ** -22,
    declare(graph, input, output) {
      const integer = (key: string) => output[key] as GraphDataView<'uint32'>;
      return new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: input['query'] as GraphDataView<'float32x2'>},
        features: {
          kind: 'segments',
          starts: input['starts'] as GraphDataView<'float32x2'>,
          ends: input['ends'] as GraphDataView<'float32x2'>,
          sourceIds: input['sourceIds'] as GraphDataView<'uint32'>
        },
        radius: input['radius'] as GraphDataView<'float32'>,
        spatialSort: false,
        querySort: false,
        candidateCapacity: 8,
        output: {
          status: {overflow: integer('overflow')},
          ...(!neighbors
            ? {
                matches: {
                  ids: integer('matchIds'),
                  status: {count: integer('counts'), overflow: integer('matchOverflow')}
                }
              }
            : {}),
          ...(neighbors
            ? {
                neighborIds: integer('ids'),
                neighborRows: integer('rows'),
                neighborCounts: integer('counts'),
                neighborDistances: output['distances'] as GraphDataView<'float32'>
              }
            : {
                nearestFeatureIds: integer('ids'),
                nearestFeatureRows: integer('rows'),
                nearestDistances: output['distances'] as GraphDataView<'float32'>
              }),
          ...(foot
            ? {
                neighborFootPoints: output['foot'] as GraphDataView<'float32x2'>,
                neighborQueryPoints: output['queryPoint'] as GraphDataView<'float32x2'>
              }
            : {})
        }
      }).getCommandNodes(graph);
    }
  };
}

/** Shared Node/WebGPU output assertion, including the complete readable array length. */
export function expectNearestResultValues(
  actual: readonly number[],
  expected: readonly number[],
  exact: boolean,
  tolerance: number
): void {
  expect(actual.length).toBe(expected.length);
  if (exact) expect(actual).toEqual(expected);
  else
    expected.forEach((value, row) => {
      expect(Number.isFinite(actual[row])).toBe(true);
      expect(Math.abs(actual[row] - value)).toBeLessThanOrEqual(tolerance);
    });
}
