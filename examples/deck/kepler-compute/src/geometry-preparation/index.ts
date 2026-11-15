// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Kepler geometry, polygon-contiguity and trip preparation.
 *
 * This is example-local domain code re-homed from the old luma.gl fork because published
 * luma.gl 10.0.0-alpha.14 does not contain it. Sources in the fork (commit e0ed33791):
 * - modules/engine/src/arrow/geometry/ (geometry-gpu-vectors, geometry-native-geoarrow,
 *   geometry-globe, geometry-properties, geometry-wkb, gpu-upload-utils,
 *   polygon-contiguity-topology)
 * - modules/engine/src/arrow/trips/ (geojson-trips, tabular-trips, trip-gpu-vectors,
 *   tabular-trip-gpu-vectors, trip-time)
 *
 * It uses only published public entry points: `@luma.gl/core`, `@luma.gl/graph/gpu-data`,
 * `@luma.gl/graph/arrow`, `@luma.gl/engine/arrow` (polygon tessellation and upload),
 * `@math.gl/arrow/geoarrow-utils` and `@math.gl/arrow/wkb`. Exported names carry a Kepler prefix so
 * they cannot be mistaken for luma.gl API.
 *
 * Departures from the fork: the fp64 polygon residual (`positionsLow`) that the fork's polygon
 * converter emitted is derived locally in `polygon-positions-low.ts` and uploaded here.
 */

import type {PreparedPolygonTopologySource} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {PreparedKeplerGeometrySource} from './geometry-gpu-vectors';

/** Host adapters also inspect geometry kinds and CRS; the analysis preparer needs only row diagnostics. */
export type PreparedKeplerPolygonTopologySource = Omit<PreparedPolygonTopologySource, 'metadata'> &
  Pick<PreparedKeplerGeometrySource, 'metadata'>;

export * from './geometry-gpu-vectors';
export * from './geometry-globe';
export * from './geometry-properties';
export {
  preparePolygonContiguityTopology,
  POLYGON_CONTIGUITY_MAXIMUM_ROWS,
  POLYGON_CONTIGUITY_MAXIMUM_VERTICES,
  POLYGON_CONTIGUITY_PREPARATION_BYTES
} from '@luma-contributors/analysis/gpu-spatial-analysis';
export type {
  PreparedPolygonTopologySource,
  PreparedPolygonContiguityTopology,
  PolygonContiguityDiagnostic
} from '@luma-contributors/analysis/gpu-spatial-analysis';
export * from './geojson-trips';
export * from './tabular-trips';
export * from './trip-gpu-vectors';
export * from './tabular-trip-gpu-vectors';
export {makeKeplerPolygonPositionsLow} from './polygon-positions-low';
