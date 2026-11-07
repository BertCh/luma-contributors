// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerComputeFilter} from './kepler-table-bridge';
import {validateDrawnPolygon} from './resident-drawing-geometry';

/**
 * Human-readable reasons a native polygon filter cannot become a resident mask, or undefined when
 * the shape is usable. Warnings (antimeridian, out-of-range) are not problems: they document the
 * native planar-degree semantics and never block the mask.
 */
export function describePolygonFilterProblem(filter: KeplerComputeFilter): string | undefined {
  const value = filter.value;
  const feature =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Parameters<typeof validateDrawnPolygon>[0])
      : null;
  const validation = validateDrawnPolygon(feature);
  if (validation.status === 'valid') return undefined;
  return `Polygon filter ${filter.id} cannot be applied: ${validation.reasons
    .map(reason => reason.message)
    .join(' ')}`;
}

/** One entry per enabled polygon filter with an unusable shape, for the native workflow notice. */
export function listPolygonFilterProblems(filters: readonly KeplerComputeFilter[]): string[] {
  return filters
    .filter(filter => filter.type === 'polygon' && filter.enabled !== false)
    .flatMap(filter => describePolygonFilterProblem(filter) ?? []);
}
