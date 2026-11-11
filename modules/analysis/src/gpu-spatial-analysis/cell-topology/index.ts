// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
export {
  GPUS2CellHierarchy,
  GPU_S2_CELL_HIERARCHY_STATUS_VALID,
  GPU_S2_CELL_HIERARCHY_STATUS_INVALID,
  GPU_S2_CELL_HIERARCHY_STATUS_UNAVAILABLE_LEVEL
} from './gpu-s2-cell-hierarchy';
export type {GPUS2CellHierarchyProps} from './gpu-s2-cell-hierarchy';
export {
  GPUS2CellMeasures,
  GPU_S2_CELL_MEASURES_STATUS_VALID,
  GPU_S2_CELL_MEASURES_STATUS_INVALID
} from './gpu-s2-cell-measures';
export type {GPUS2CellMeasuresProps} from './gpu-s2-cell-measures';
export {
  GPUS2CapCovering,
  GPU_S2_CAP_COVERING_MAXIMUM_UNIFORM_LEVEL,
  GPU_S2_CAP_COVERING_MAXIMUM_REFINED_LEVEL,
  GPU_S2_CAP_COVERING_MINIMUM_TOLERANCE,
  GPU_S2_CAP_COVERING_MAXIMUM_TOLERANCE,
  GPU_S2_CAP_COVERING_STATUS_COMPLETE,
  GPU_S2_CAP_COVERING_STATUS_INVALID,
  GPU_S2_CAP_COVERING_STATUS_CAPACITY,
  GPU_S2_CAP_COVERING_STATUS_FRONTIER
} from './gpu-s2-cap-covering';
export type {GPUS2CapCoveringProps, GPUS2CapCoveringRefinement} from './gpu-s2-cap-covering';

export {
  GPUCellTopology,
  getCellTopologyStride,
  GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS,
  GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE
} from './gpu-cell-topology';
export type {
  GPUCellTopologyProps,
  GPUCellTopologyOperation,
  GPUCellTopologyOutput
} from './gpu-cell-topology';
export {GPUCellCompaction, GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH} from './gpu-cell-compaction';
export type {
  GPUCellCompactionProps,
  GPUCellCompactionOutput,
  GPUCellCompactOperation,
  GPUCellUncompactOperation
} from './gpu-cell-compaction';
export {
  GPUCellGridPath,
  GPU_CELL_GRID_DISTANCE_UNDEFINED,
  GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH
} from './gpu-cell-grid-path';
export type {GPUCellGridPathOutput, GPUCellGridPathProps} from './gpu-cell-grid-path';

export {GPUH3CellIdentities} from './gpu-h3-cell-identities';
export type {GPUH3CellIdentitiesProps} from './gpu-h3-cell-identities';
export {
  GPUS2PolygonCovering,
  GPU_S2_POLYGON_COVERING_MAXIMUM_LEVEL,
  GPU_S2_POLYGON_COVERING_MINIMUM_VISITED_CELLS,
  GPU_S2_POLYGON_COVERING_MAXIMUM_VISITED_CELLS,
  GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS,
  GPU_S2_POLYGON_COVERING_STATUS_INVALID,
  GPU_S2_POLYGON_COVERING_STATUS_CAPACITY,
  GPU_S2_POLYGON_COVERING_STATUS_BUDGET
} from './gpu-s2-polygon-covering';
export type {
  GPUS2PolygonCoveringProps,
  GPUS2PolygonCoveringRefinement
} from './gpu-s2-polygon-covering';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeS2CapCoveringOnCPU,
  CPUS2CapCoveringOperation
} from './cpu-s2-cap-covering-operation';
export type {
  CPUS2CapCoveringOptions,
  CPUS2CapCoveringResult
} from './cpu-s2-cap-covering-operation';
export {
  computeS2PolygonCoveringOnCPU,
  CPUS2PolygonCoveringOperation
} from './cpu-s2-polygon-covering-operation';
export type {
  CPUS2PolygonCoveringInput,
  CPUS2PolygonCoveringOptions,
  CPUS2PolygonCoveringResult
} from './cpu-s2-polygon-covering-operation';
export {GPUS2CapCoveringOperation} from './gpu-s2-cap-covering-operation';
export type {S2CapCoveringOperationProps} from './gpu-s2-cap-covering-operation';
export {GPUS2PolygonCoveringOperation} from './gpu-s2-polygon-covering-operation';
export type {S2PolygonCoveringOperationProps} from './gpu-s2-polygon-covering-operation';
