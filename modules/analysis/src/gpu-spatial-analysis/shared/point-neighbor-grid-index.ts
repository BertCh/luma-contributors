// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Compatibility path for point-family callers. The implementation lives below every analysis
// entry so dataframe pair statistics can share it without importing the spatial-analysis entry.
export {
  createPointNeighborGridIndexNodes,
  POINT_NEIGHBOR_GRID_UNIT_POSITION_WGSL,
  type PointNeighborGridIndex
} from '../../analysis-core/kernels/point-neighbor-grid-index';
