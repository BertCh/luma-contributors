// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.

import {assert, type Device} from '@luma.gl/core';
import {CPUVector, GPUVector} from '@luma-contributors/graph/gpu-data';
import {GPUSimpleGraph} from '../gpu-simple-graph/gpu-simple-graph';
import {GPUGraph} from '../gpu-core/gpu-graph';
import type {CompiledGPUGraph} from '../gpu-core/compiled-gpu-graph';
import type {CompiledGPUSimpleExpressionGraph} from '../gpu-simple-graph/compiled-gpu-expression-graph';
import {gpuGraph} from '../gpu-core/gpu-graph-value';
import type {GPUSpatialContext, GPUSpatialFrame} from '../analysis-core/contracts/spatial-context';
import {getGPUProjectionSpatialContext} from './projection-spatial-context';
import {CPUProjectionOperation, type CPUProjectionOperationProps} from './cpu-projection-operation';

export type CoordinateProjectionGraphProps = {
  format: 'float32x2' | 'float32x3';
  projection: CPUProjectionOperationProps['projection'];
  /** Source frame of the input coordinates. */
  sourceFrame?: GPUSpatialFrame;
  /** Target frame of the output coordinates; tags results. Takes precedence over `targetCrs`. */
  targetFrame?: GPUSpatialFrame;
  /** Target CRS of the output coordinates, for example `'EPSG:3857'`. */
  targetCrs?: string;
};
export type ProjectedCoordinateResult = {
  positions: GPUVector;
  /** Planar context of `positions` when a target frame or CRS is known. */
  spatialContext?: GPUSpatialContext;
  /** Release this invocation's GPU result after downstream consumers finish. */
  destroy(): void;
};

/** Reusable layer-facing point-coordinate pipeline. Projection loading remains caller-owned. */
export class CoordinateProjectionGraph {
  private readonly graph: CompiledGPUGraph | CompiledGPUSimpleExpressionGraph;
  readonly format: CoordinateProjectionGraphProps['format'];
  readonly sourceFrame?: GPUSpatialFrame;
  readonly targetFrame?: GPUSpatialFrame;
  /** Planar context stamped on every result, when a target frame or CRS is known. */
  readonly spatialContext?: GPUSpatialContext;

  constructor(device: Device, props: CoordinateProjectionGraphProps) {
    this.format = props.format;
    this.sourceFrame = props.sourceFrame;
    this.targetFrame = props.targetFrame;
    this.spatialContext = getGPUProjectionSpatialContext(
      props.targetFrame?.crs ?? props.targetCrs,
      props.targetFrame
    );
    const graph = device.type === 'webgpu' ? new GPUGraph() : new GPUSimpleGraph();
    graph.addInput('source', gpuGraph.vector(props.format));
    graph.add(
      new CPUProjectionOperation({
        id: 'project-coordinates',
        input: 'source',
        output: 'positions',
        format: props.format,
        projection: props.projection
      })
    );
    this.graph = graph.compile(device, {outputs: ['positions']});
  }

  /** Borrow input storage and preserve row/chunk boundaries; each result owns independent storage. */
  async run(source: CPUVector | GPUVector): Promise<ProjectedCoordinateResult> {
    assert(source.format === this.format);
    // Point layers must handle null filtering before this numeric projection boundary.
    assert(source.data.every(chunk => !chunk.nullBitmap && !chunk.valueOffsets));
    const {positions} = await this.graph.run({inputs: {source}});
    return {positions, spatialContext: this.spatialContext, destroy: () => positions.destroy()};
  }

  /** Destroy the compiled pipeline. Previously returned results remain caller-owned. */
  destroy(): void {
    this.graph.destroy();
  }
}
