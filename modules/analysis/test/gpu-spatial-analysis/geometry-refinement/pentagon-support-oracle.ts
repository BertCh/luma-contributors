// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export const PENTAGON_POSITIONS = [0, 0, 8, 0, 10, 3, 6, 8, 1, 6];

/** Independent supporting-line distances, without running either wavefront implementation. */
export function expectPentagonSupports(result: {
  segments: ArrayLike<number>;
  sourceEdges: ArrayLike<number>;
  status: ArrayLike<number>;
}): void {
  // A strictly convex nondegenerate n-gon has n leaf arcs and n-3 interior arcs: 7 for n=5.
  expect(Array.from(result.status).slice(0, 2)).toEqual([7, 7]);
  expect(result.status[3]).toBe(0);
  let hasObliqueTerminal = false;
  for (let row = 0; row < 7; row++) {
    const arc = Array.from({length: 4}, (_, component) => result.segments[row * 4 + component]);
    expect(arc.every(Number.isFinite)).toBe(true);
    const midpointX = (arc[0] + arc[2]) / 2;
    const midpointY = (arc[1] + arc[3]) / 2;
    const distances = Array.from({length: 5}, (_, edge) => {
      const next = (edge + 1) % 5;
      const directionX = PENTAGON_POSITIONS[next * 2] - PENTAGON_POSITIONS[edge * 2];
      const directionY = PENTAGON_POSITIONS[next * 2 + 1] - PENTAGON_POSITIONS[edge * 2 + 1];
      return (
        (directionX * (midpointY - PENTAGON_POSITIONS[edge * 2 + 1]) -
          directionY * (midpointX - PENTAGON_POSITIONS[edge * 2])) /
        Math.hypot(directionX, directionY)
      );
    });
    // Every convex-wavefront arc is equidistant from its two nearest inward supporting lines.
    const minimum = Math.min(...distances);
    const supports = distances.flatMap((distance, edge) =>
      Math.abs(distance - minimum) < 2e-5 ? [edge] : []
    );
    expect(supports).toHaveLength(2);
    const sources = [result.sourceEdges[row * 2], result.sourceEdges[row * 2 + 1]].sort(
      (left, right) => left - right
    );
    expect(sources).toEqual(supports);
    if (supports[0] === 0 && supports[1] === 2) {
      hasObliqueTerminal = true;
      // Edges y=0 and 5*x+4*y=62 have bisector 5*x+(4+sqrt(41))*y=62.
      expect(Math.abs(5 * midpointX + (4 + Math.sqrt(41)) * midpointY - 62)).toBeLessThan(2e-4);
    }
  }
  expect(hasObliqueTerminal).toBe(true);
}
