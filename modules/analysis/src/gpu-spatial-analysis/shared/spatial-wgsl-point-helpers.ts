// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility path: existing spatial-family callers use the core implementation through here.
export {
  getReadPointWGSL,
  getReadPoint3WGSL,
  IS_SPATIAL_FINITE_WGSL,
  IS_SPATIAL_FINITE_POINT_WGSL,
  GET_SPATIAL_NAN_WGSL,
  IS_SAME_SPATIAL_POINT_WGSL
} from '../../analysis-core/kernels/spatial-wgsl-point-helpers';
