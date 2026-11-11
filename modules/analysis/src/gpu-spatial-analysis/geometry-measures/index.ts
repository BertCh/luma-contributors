// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPU_GEODESIC_MEAN_EARTH_RADIUS,
  GPU_GEODESIC_WGS84_FLATTENING,
  GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS
} from './geodesic-wgsl';
export {GPUGeometryMeasures} from './gpu-geometry-measures';
export type {
  GPUGeometryMeasuresGroupOutput,
  GPUGeometryMeasuresOutput,
  GPUGeometryMeasuresProps
} from './gpu-geometry-measures';
export type {GPUGeometryCoordinateSystem} from './geometry-measures-kernels';
export {GPUGeodesicPairs} from './gpu-geodesic-pairs';
export type {GPUGeodesicPairsOutput, GPUGeodesicPairsProps} from './gpu-geodesic-pairs';
export {GPUGeodesicDestination} from './gpu-geodesic-destination';
export type {
  GPUGeodesicDestinationOutput,
  GPUGeodesicDestinationProps
} from './gpu-geodesic-destination';
export {GPU_GEODESIC_DEFAULT_ITERATIONS} from './geodesic-kernels';
export type {GPUGeodesicModel} from './geodesic-kernels';
export {GPULocalFrameProjection} from './gpu-local-frame-projection';
export type {GPULocalFrameProjectionProps} from './gpu-local-frame-projection';
export {GEODESIC_WGSL} from './geodesic-wgsl';

// GPUGraph operation forms and CPU twins (dev-docs/contributor-contract.md, section 14).
export {
  computeLocalFrameProjectionOnCPU,
  CPULocalFrameProjectionOperation
} from './cpu-local-frame-projection-operation';
export type {CPULocalFrameProjectionResult} from './cpu-local-frame-projection-operation';
export {GPULocalFrameProjectionOperation} from './gpu-local-frame-projection-operation';
export type {LocalFrameProjectionOperationProps} from './gpu-local-frame-projection-operation';
export {computeGeodesicPairsOnCPU, CPUGeodesicPairsOperation} from './cpu-geodesic-pairs-operation';
export type {CPUGeodesicPairsInput, CPUGeodesicPairsResult} from './cpu-geodesic-pairs-operation';
export {GPUGeodesicPairsOperation} from './gpu-geodesic-pairs-operation';
export type {GeodesicPairsOperationProps} from './gpu-geodesic-pairs-operation';
