// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {captureGraphCommandNodes} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUFloat32Positions, GPUGeospatialPositions} from '@luma.gl/graph/geospatial';
import {GPUProjection, type GPUProjectionValidity} from '@luma.gl/graph/gpu-project';
import {compileProjectionPlan} from '@luma.gl/graph/gpu-project';
import type {ProjectionBounds, ProjectionPlan} from '@luma.gl/graph/gpu-project';
import type {GPULocalFrame} from '../../analysis-core/contracts/local-frame';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';

/** Props of {@link GPULocalFrameProjection}. */
export type GPULocalFrameProjectionProps = {
  /** Prefix of generated graph nodes. Defaults to `'gpu-local-frame-projection'`. */
  id?: string;
  /** Frame created with `createGPULocalFrame`. */
  frame: GPULocalFrame;
  /**
   * Longitude/latitude positions in degrees: `float32x2` rows, or raw binary64 rows stored as
   * `uint32x4` (`Float64Array` word order) to keep the full input precision.
   */
  positions: GPUGeospatialPositions;
  /** Caller-owned `float32x2` frame coordinates in meters, one row per position. */
  output: GPUFloat32Positions;
  /**
   * `[minLongitude, minLatitude, maxLongitude, maxLatitude]` covering every position, in the
   * positions' own longitude convention (data crossing the antimeridian must be stored with
   * continuous longitudes, for example 179 to 181). Rows outside the bounds are rejected.
   */
  bounds: ProjectionBounds;
  /**
   * Maximum error in meters of the polynomial patches accepted by the patch compiler, measured
   * relative to the center of `bounds`. Defaults to `0.01`. It does not bound the float32 rounding
   * of the output rows, which are absolute frame meters: the patch origin and the sum are each
   * rounded, so add up to one float32 spacing of the largest frame coordinate inside `bounds`
   * (0.004 m within 65 km of the frame origin, 0.016 m within 262 km, 0.03 m within 524 km).
   */
  tolerance?: number;
  /** Optional `uint32` rows: 1 for projected rows, 0 for rows outside `bounds` (output 0). */
  validity?: GPUProjectionValidity;
};

/**
 * Projects GPU-resident longitude/latitude positions into a `createGPULocalFrame` frame on the GPU,
 * so planar operators can consume them without a CPU readback ("project first").
 *
 * The frame's binary64 projection is compiled on the CPU into error-bounded local polynomial
 * patches over `bounds` and evaluated by `GPUProjection` (`gpu-project`). Output rows are absolute
 * frame meters: the plan is referenced to the frame origin, so float32 rounding is relative to the
 * frame extent. Pass {@link spatialContext} (the frame's planar context) to downstream operators.
 *
 * **Error budget.** `compileProjectionPlan` validates `tolerance` and reports
 * `plan.float32MaxError` for output relative to the center of `bounds`. This class then moves the
 * plan's destination origin to the frame origin, which the published compiler has no option for,
 * so those two numbers describe the patch fit and not the rows written here. The rows carry in
 * addition the float32 rounding of an absolute frame coordinate: up to one spacing per axis, which
 * is 0.016 m for coordinates between 131 km and 262 km from the frame origin and passes the
 * default tolerance beyond that. Keep `bounds` near the frame origin when centimeter output
 * matters.
 */
export class GPULocalFrameProjection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly frame: GPULocalFrame;
  /** Planar context of the output rows, carrying `units: 'meters'` and the frame identity/kind. */
  readonly spatialContext: GPUSpatialContext;
  /**
   * Compiled plan; its destination origin is the frame origin `[0, 0]`. Its `tolerance`,
   * `maxError` and `float32MaxError` were computed before that change, for output relative to the
   * center of `bounds` (see the class documentation).
   */
  readonly plan: ProjectionPlan;
  private readonly projection: GPUProjection;

  constructor(props: GPULocalFrameProjectionProps) {
    this.id = props.id ?? 'gpu-local-frame-projection';
    this.frame = props.frame;
    this.spatialContext = props.frame.spatialContext;
    const compiled = compileProjectionPlan({
      projection: (coordinates: number[]) => props.frame.project(coordinates[0], coordinates[1]),
      bounds: props.bounds,
      tolerance: props.tolerance ?? 0.01
    });
    // Reference the float32 output to the frame origin instead of the bounds center: packed
    // patch offsets become frame coordinates, so the output rows are absolute frame meters.
    this.plan = {...compiled, destinationOrigin: [0, 0]};
    this.projection = new GPUProjection({
      id: this.id,
      positions: props.positions,
      output: props.output,
      validity: props.validity,
      plan: this.plan
    });
  }

  /** Returns the projection passes; add the contributor with `graph.add(projection)`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return captureGraphCommandNodes(graph, () => this.projection.addToGraph(graph));
  }

  /** Releases the owned plan buffer. */
  destroy(): void {
    this.projection.destroy();
  }
}
