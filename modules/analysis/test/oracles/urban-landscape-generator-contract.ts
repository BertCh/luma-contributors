// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export const LANDSCAPE_VALIDATION_MARKERS = [
  'stopifnot(length(proximity_values) == 2, all(is.finite(proximity_values)))',
  'stopifnot(identical(tile_roots, c(0L,1L,2L,1L,0L,2L,2L,2L,0L)))',
  'stopifnot(length(roots) == 1, length(classes) == 1)',
  'stopifnot(identical(as.integer(tile_metrics$root), 0:2))',
  'stopifnot(isTRUE(all.equal(tile_metrics$area*10000, c(18,12,24), tolerance=1e-10)))'
] as const;

export const LANDSCAPE_PUBLICATION_MARKERS = [
  'result$proximity <-',
  'result$tileReconciliation <-',
  'stopifnot('
] as const;

export function expectLandscapeSectionsBeforePublication(source: string): void {
  const writes = [...source.matchAll(/^\s*write_json\s*\(/gm)];
  expect(writes).toHaveLength(1);
  for (const marker of [...LANDSCAPE_PUBLICATION_MARKERS, ...LANDSCAPE_VALIDATION_MARKERS]) {
    const position = source.lastIndexOf(marker);
    expect(position, `missing ${marker}`).toBeGreaterThanOrEqual(0);
    expect(writes[0].index).toBeGreaterThan(position);
  }
}
