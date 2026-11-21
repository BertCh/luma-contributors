// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {freezeCatalog} from '../freeze-catalog';
import type {GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';

type WebsiteHelperKinds = Readonly<{
  input: readonly GPUSpatialAnalysisDataKind[];
  output: readonly GPUSpatialAnalysisDataKind[];
}>;

const HELPER_KINDS = {
  // healpix-cells/healpix-dggs-json.ts: packed cells, aligned properties and completeness status.
  createHealpixDGGSJSONFromReadback: {
    input: ['cells', 'attributes', 'summary'],
    output: ['cells', 'attributes']
  },
  // recipes/points-in-polygons-choropleth-operation.ts: aggregate columns and loss counters.
  createPointsInPolygonsChoroplethOperation: {
    input: ['points', 'polygons', 'attributes'],
    output: ['attributes', 'summary']
  },
  // spatial-regression/ordinary-least-squares-composition.ts: pack predictor columns by row.
  createOrdinaryLeastSquaresDesign: {input: ['attributes'], output: ['attributes']},
  // Same file: derive the training mask from match flags and forward completeness status.
  createOrdinaryLeastSquaresJoinedTableMask: {
    input: ['attributes', 'summary'],
    output: ['attributes', 'summary']
  },
  // raster-analysis/raster-time-series: host settings and fixed host plans have no data ports.
  // 'parameters' describes a GPU-resident view with values that can change between runs.
  createGPURasterStreamingAccumulatorEncodingPlan: {input: [], output: []},
  createGPURasterStandardizedIndexCalibrationEncodingPlan: {input: [], output: []},
  createGPURasterClimateRunAccumulatorEncodingPlan: {input: [], output: []},
  // pointcloud-analysis/operations/gpu-pointcloud-coverage-validity.ts: raster flags to a mask.
  createGroundSeedCoverageNode: {input: ['surface'], output: ['attributes']}
} as const satisfies Record<string, WebsiteHelperKinds>;

Object.setPrototypeOf(HELPER_KINDS, null);

/** Source-derived kinds for exported factories that the website inventory lists as helpers. */
export const GPU_ANALYSIS_WEBSITE_HELPER_KINDS: typeof HELPER_KINDS &
  Readonly<Partial<Record<string, WebsiteHelperKinds>>> = freezeCatalog(HELPER_KINDS);
