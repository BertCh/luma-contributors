// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MAXIMUM_WORK,
  GPUPolygonTriangulation,
  getGPUPolygonTriangulationMaximumNodeCount,
  getPolygonTriangulationIndexCount,
  getPolygonTriangulationNodeCount,
  validatePolygonTriangulationScratch
} from './gpu-polygon-triangulation';
export {
  GPU_POLYGON_TRIANGULATION_DEFAULT_MONOTONE_MINIMUM_VERTICES,
  GPU_POLYGON_TRIANGULATION_MONOTONE_BYTES_PER_VERTEX,
  getGPUPolygonTriangulationMonotoneMaximumVertexCount,
  validatePolygonTriangulationMonotoneCapacity
} from './polygon-triangulation-monotone';
export type {
  GPUPolygonTriangulationOutput,
  GPUPolygonTriangulationProps
} from './gpu-polygon-triangulation';
export {GPUConstrainedDelaunayTriangulation} from './gpu-constrained-delaunay-triangulation';
export type {GPUConstrainedDelaunayTriangulationProps} from './gpu-constrained-delaunay-triangulation';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeConstrainedDelaunayTriangulationOnCPU,
  CPUConstrainedDelaunayTriangulationOperation
} from './cpu-constrained-delaunay-triangulation-operation';
export type {
  CPUConstrainedDelaunayTriangulationInput,
  CPUConstrainedDelaunayTriangulationOptions,
  CPUConstrainedDelaunayTriangulationOperationProps,
  CPUConstrainedDelaunayTriangulationResult
} from './cpu-constrained-delaunay-triangulation-operation';
export {GPUConstrainedDelaunayTriangulationOperation} from './gpu-constrained-delaunay-triangulation-operation';
export type {ConstrainedDelaunayTriangulationOperationProps} from './gpu-constrained-delaunay-triangulation-operation';
export {
  computePolygonTriangulationOnCPU,
  CPUPolygonTriangulationOperation
} from '../../analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
export type {
  CPUPolygonTriangulationInput,
  CPUPolygonTriangulationOptions,
  CPUPolygonTriangulationOperationProps,
  CPUPolygonTriangulationResult
} from '../../analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
export {GPUPolygonTriangulationOperation} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
export type {PolygonTriangulationOperationProps} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
