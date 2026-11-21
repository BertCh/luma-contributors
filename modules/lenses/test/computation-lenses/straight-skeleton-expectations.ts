// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export function expectStraightSkeleton(values: (name: string) => number[]) {
  // Square/hole wavefronts meet at distance 1.5: 8 corner trajectories,
  // then 4 axis-aligned meeting arcs. Final provenance follows the live edges.
  expect(values('status')).toEqual([12, 12, 8, 0]);
  expect(values('segments').slice(0, 48)).toEqual([
    3, 3, 1.5, 1.5, 0, 0, 1.5, 1.5, 3, 7, 1.5, 8.5, 7, 7, 8.5, 8.5, 7, 3, 8.5, 1.5, 10, 0, 8.5, 1.5,
    10, 10, 8.5, 8.5, 0, 10, 1.5, 8.5, 8.5, 1.5, 1.5, 1.5, 8.5, 1.5, 8.5, 8.5, 8.5, 8.5, 1.5, 8.5,
    1.5, 8.5, 1.5, 1.5
  ]);
  expect(values('source-edges').slice(0, 24)).toEqual([
    7, 4, 3, 0, 4, 5, 5, 6, 6, 7, 0, 1, 1, 2, 2, 3, 7, 0, 1, 6, 2, 5, 3, 4
  ]);
}
