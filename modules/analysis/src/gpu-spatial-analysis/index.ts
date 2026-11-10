// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Every subdirectory index is re-exported whole, so the entry cannot drift from its parts and a
// name exported by two subdirectories is a compile error.

export * from '../analysis-core/contracts/index';
export * from '../analysis-core/ring-assembly/index';
export * from './areal-interpolation/index';
export * from './bounds-filter/index';
export * from './categorical-association/index';
export * from './catchment-accessibility/index';
export * from './cell-aggregation/index';
export * from './change-points/index';
export * from './cell-cover/index';
export * from './cell-indexing/index';
export * from './cell-set-outline/index';
export * from './cell-table-compare/index';
export * from './cell-topology/index';
export * from './delaunay-tessellation/index';
export * from './distribution-dynamics/index';
export * from './dot-density/index';
export * from './emerging-hot-spots/index';
export * from './geographic-distribution/index';
export * from './geographical-detector/index';
export * from './geometry-edit/index';
export * from './geometry-measures/index';
export * from './geometry-predicates/index';
export * from './geometry-topology/index';
export * from './geometry-validity/index';
export * from './global-spatial-statistics/index';
export * from './grid-generators/index';
export * from './group-geometry/index';
export * from './healpix-cells/index';
export * from './hilbert-keys/index';
export * from './label-point/index';
export * from './line-clip/index';
export * from './line-density/index';
export * from './line-merge/index';
export * from './line-segmentize/index';
export * from './line-simplification/index';
export * from './line-split/index';
export * from './linear-referencing/index';
export * from './location-allocation/index';
export * from './map-coloring/index';
export * from './minimum-bounds/index';
export * from './minimum-clearance/index';
export * from './multiple-testing/index';
export * from './neighbor-search/index';
export * from './neighborhood-summary/index';
export * from './offset-expansion/index';
// @deprecated Import the ontology catalogs from '@luma-contributors/analysis/gpu-spatial-analysis/ontology'.
// This barrel re-export is kept for compatibility and will be removed in a future release.
export * from './ontology/index';
export * from './outlier-detection/index';
export * from './outline-geometry/index';
export * from './pair-gather/index';
export * from './permutation-inference/index';
export * from './point-density/index';
export * from './polygon-buffer/index';
export * from './polygon-coverage-simplification/index';
export * from './polygon-coverage-topology/index';
export * from './polygon-overlay/index';
export * from './polygon-triangulation/index';
export * from './rate-smoothing/index';
export * from './recipes/index';
export * from './rectangle-clip/index';
export * from './region-statistics/index';
export * from './scan-statistics/index';
export * from './scoring-rules/index';
export * from './optimal-transport/index';
export * from './space-time-hawkes/index';
export * from './segment-intersection/index';
export * from './segregation/index';
export * from './shape-descriptors/index';
export * from './similar-locations/index';
export * from './space-time-statistics/index';
export * from './space-time-tests/index';
export * from './spatial-autocorrelation/index';
export * from './spatial-clustering/index';
export * from './spatial-interpolation/index';
export * from './spatial-join/index';
export * from './redcap-regions/index';
export * from './spatial-regionalization/index';
export * from './spatial-regression/index';
export * from './spatial-weights/index';
export * from './study-window/index';
export * from './track-similarity/index';
export * from './trajectory-analysis/index';
export * from './trajectory-cleaning/index';
export * from './trajectory-encounters/index';
export * from './trajectory-interpolation/index';
export * from './trajectory-home-range/index';
export * from './trajectory-mobility/index';
export * from './trajectory-recursion/index';
export * from './trajectory-staypoints/index';
export * from './trajectory-zones/index';
export * from './vertex-snap/index';
export * from './zonal-statistics/index';

// Top-level files and shared contracts.
export {
  getGPUGridIndexMaximumCellCount,
  GPUParameterBuffer
} from '../analysis-core/kernels/gpu-contributor-utils';
export type {
  GPUParameterBufferProps,
  GPUParameterFormat
} from '../analysis-core/kernels/gpu-contributor-utils';
export * from './lon-lat-polygons/index';

export * from './concave-hull/index';
export * from './geometry-refinement/index';
export * from './urban-morphology/index';
export * from './landscape-ecology/index';
export * from './point-process/index';
export * from './trajectory-estimation/index';
export * from './trajectory-restriction/index';
export * from './spatial-validation/index';
export * from './spatial-entropy/index';
export * from './spatial-interaction/index';
export * from './global-coordinates/index';

export * from './dorling-cartogram/index';
