// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility names for existing harnesses. This is the implementation under test, not an
// independent oracle: expectations come from literal signs, Shapely areas and boundary coverage.
export {
  triangulateMonotoneOnCPU as triangulateMonotoneModel,
  MONOTONE_VERTEX_TYPE as MODEL_TYPE,
  type MonotonePoint as ModelPoint,
  type MonotoneDiagnostics as MonotoneModelDiagnostics,
  type MonotoneResult as MonotoneModelResult
} from '../../../src/analysis-core/polygon-triangulation/cpu-monotone-triangulation';
export {computeTriangulationOrientation as orientExactSign} from '../../../src/analysis-core/polygon-triangulation/cpu-triangulation-predicates';
