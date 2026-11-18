// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import fixture from '../../oracles/fixtures/parity-wp2-buffer-selection.json';

for (const entry of fixture.cases) {
  it(`buffer selection oracle records winning segment rows separately from IDs: ${entry.name}`, () => {
    let distinctIdentityCount = 0;
    for (const frame of entry.frames) {
      for (const [queryRow, neighbors] of frame.neighbors.entries()) {
        for (const recordedNeighbor of neighbors) {
          const neighbor: {row?: number; id: number; distance: number} = recordedNeighbor;
          expect(Number.isInteger(neighbor.row), 'the oracle must retain the pre-remap row').toBe(
            true
          );
          const featureRow = neighbor.row!;
          expect(featureRow).toBeGreaterThanOrEqual(0);
          expect(featureRow).toBeLessThan(entry.sourceIds.length);
          expect(entry.sourceIds[featureRow]).toBe(neighbor.id);
          if (featureRow !== neighbor.id) distinctIdentityCount++;

          // Independent closed-form point/segment distances check the stored winning row.
          // These are CPU oracle checks, not the production nearest traversal or ID lookup.
          const queryX = frame.queries[queryRow * 2];
          const queryY = frame.queries[queryRow * 2 + 1];
          const distances = entry.sourceIds.map((_, segmentRow) => {
            const startX = frame.starts[segmentRow * 2];
            const startY = frame.starts[segmentRow * 2 + 1];
            const deltaX = frame.ends[segmentRow * 2] - startX;
            const deltaY = frame.ends[segmentRow * 2 + 1] - startY;
            const squaredLength = deltaX * deltaX + deltaY * deltaY;
            const fraction =
              squaredLength === 0
                ? 0
                : Math.max(
                    0,
                    Math.min(
                      1,
                      ((queryX - startX) * deltaX + (queryY - startY) * deltaY) / squaredLength
                    )
                  );
            return Math.hypot(
              queryX - startX - fraction * deltaX,
              queryY - startY - fraction * deltaY
            );
          });
          const expectedDistance = Math.min(...distances);
          expect(Number.isFinite(neighbor.distance)).toBe(true);
          expect(Number.isFinite(distances[featureRow])).toBe(true);
          const tolerance = 1e-8 * (1 + expectedDistance);
          expect(Math.abs(neighbor.distance - expectedDistance)).toBeLessThanOrEqual(tolerance);
          expect(Math.abs(distances[featureRow] - expectedDistance)).toBeLessThanOrEqual(tolerance);
        }
      }
    }
    expect(distinctIdentityCount).toBeGreaterThan(0);
  });
}
