// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export type GeographicGroupReadback = {
  bounds: number[];
  groupBounds: number[];
  counts: number[];
  vertices: number[];
};
export function expectGeographicGroupBounds(
  result: GeographicGroupReadback,
  singletonLongitude: number
): void {
  expect(result.bounds).toEqual([179, 0, 181, 1, singletonLongitude, 0.5, singletonLongitude, 0.5]);
  // Both longitudes are exact binary halves; align the singleton by adding one turn.
  expect(result.groupBounds).toEqual([179, 0, Math.max(181, singletonLongitude + 360), 1]);
  expect(result.counts).toEqual([2]);
  expect(result.vertices).toEqual([3]);
}

export type SharedSourceReadback = {
  featureOffsets: number[];
  polygons: number[][][][];
  count: number;
  sourceIds: number[];
  overflow: number;
  candidateOverflow: number;
};
export function expectSharedSourceFeatures(result: SharedSourceReadback): void {
  expect(result.count).toBe(2);
  expect(result.polygons).toHaveLength(2);
  expect(result.sourceIds).toEqual([77, 77]);
  expect(result.overflow).toBe(0);
  expect(result.candidateOverflow).toBe(0);
  expect(result.featureOffsets).toEqual([0, 1, 2]);
  // Traverse each feature's published polygon range rather than treating polygons as features.
  const areas = Array.from({length: result.count}, (_, feature) => {
    const polygons = result.polygons.slice(
      result.featureOffsets[feature],
      result.featureOffsets[feature + 1]
    );
    expect(polygons).toHaveLength(1);
    return polygons.reduce((area, polygon) => {
      expect(polygon).toHaveLength(1);
      const ring = polygon[0];
      let twiceArea = 0;
      for (let vertex = 0; vertex < ring.length; vertex++) {
        const next = ring[(vertex + 1) % ring.length];
        expect(Number.isFinite(ring[vertex][0])).toBe(true);
        expect(Number.isFinite(ring[vertex][1])).toBe(true);
        twiceArea += ring[vertex][0] * next[1] - next[0] * ring[vertex][1];
      }
      return area + Math.abs(twiceArea) / 2;
    }, 0);
  }).sort((first, second) => first - second);
  expect(areas).toEqual([9, 16]);
}

export function expectTruncatedPairPrefix(result: {leftRows: number[]; rightRows: number[]}): void {
  expect(result.leftRows).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  expect(result.rightRows).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
}

const GROUP_COLUMN_LENGTHS = {
  counts: 2,
  bounds: 8,
  meanCenters: 4,
  weightedCenters: 4,
  weightSums: 2,
  medoids: 2,
  ellipses: 6,
  standardDistances: 2,
  overflow: 1
} as const;
export function expectGroupColumnShapes(
  result: {[Column in keyof typeof GROUP_COLUMN_LENGTHS]: readonly number[]}
): void {
  for (const [column, length] of Object.entries(GROUP_COLUMN_LENGTHS)) {
    expect(result[column as keyof typeof GROUP_COLUMN_LENGTHS], column).toHaveLength(length);
  }
}

export function expectNearestDistance(distances: ArrayLike<number>): void {
  expect(distances.length).toBe(1);
  expect(Number.isFinite(distances[0])).toBe(true);
  expect(distances[0]).toBe(1);
}

export function expectProjectedUsability(validity: number[]): void {
  expect(validity).toHaveLength(2);
  expect(validity.map(value => value !== 0)).toEqual([true, false]);
}

export const PACKED_PADDING_SENTINEL = 0xa5a5a5a5;
export function prefillPackedOutputs(
  outputs: readonly {buffer: {write(values: Uint32Array): unknown}; wordCount: number}[]
): void {
  for (const {buffer, wordCount} of outputs)
    buffer.write(new Uint32Array(wordCount).fill(PACKED_PADDING_SENTINEL));
}

export async function withTestResourceCleanup<Result>(
  resources: {destroy(): void}[],
  run: () => Promise<Result>
): Promise<Result> {
  try {
    return await run();
  } finally {
    resources
      .slice()
      .reverse()
      .forEach(resource => resource.destroy());
  }
}
