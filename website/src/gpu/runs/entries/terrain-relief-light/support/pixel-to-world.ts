// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A small graph kernel that maps pixel-space positions an operation wrote (x right, y down, in cells)
// into the run's drawing world, so the layers can draw the operation's own buffers unchanged.

import {createWGSLKernelNode} from '@luma-contributors/analysis/analysis-core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {World} from '../../../inputs';

type PixelToWorldProps = {
  id: string;
  /** Interleaved x, y pairs in pixels. */
  input: GraphDataView<'float32'>;
  /** Interleaved x, y pairs in world units. */
  output: GraphDataView<'float32'>;
  pairCount: number;
  columns: number;
  rows: number;
  world: World;
  /** Added to both pixel coordinates first: 0.5 for pixel-centre index space. Default 0. */
  pixelOffset?: number;
};

class PixelToWorld implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: PixelToWorldProps;

  constructor(props: PixelToWorldProps) {
    this.id = props.id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: props.id,
        operation: 'PixelToWorld',
        bindings: [
          {name: 'input', view: props.input, type: 'f32', access: 'read'},
          {name: 'output', view: props.output, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.pairCount,
        declarations: `const SCALE = vec2<f32>(${props.world.width / props.columns}, ${-props.world.height / props.rows});
const OFFSET = vec2<f32>(${((props.pixelOffset ?? 0) * props.world.width) / props.columns}, ${
          props.world.height - ((props.pixelOffset ?? 0) * props.world.height) / props.rows
        });`,
        body: `let pair = vec2<f32>(input[inputOffset + 2u * index], input[inputOffset + 2u * index + 1u]);
  let world = pair * SCALE + OFFSET;
  output[outputOffset + 2u * index] = world.x;
  output[outputOffset + 2u * index + 1u] = world.y;`
      })
    ];
  }
}

/**
 * Adds a kernel that converts `pixels` (float32x2 or float32x4 rows in pixel space) into a new
 * buffer of world positions with the same layout, and returns the new buffer.
 */
export function addPixelToWorld<Format extends 'float32x2' | 'float32x4'>(
  builder: RunBuilder,
  id: string,
  pixels: RunBuffer<Format>,
  grid: {columns: number; rows: number},
  world: World,
  pixelOffset = 0
): RunBuffer<Format> {
  const floatCount = pixels.length * (pixels.format === 'float32x2' ? 2 : 4);
  const worldBuffer = builder.allocate(id, pixels.format, pixels.length);
  const floatView = (buffer: RunBuffer) =>
    builder.graph.createDataView(buffer.view.buffer, {format: 'float32', length: floatCount});
  builder.graph.add(
    new PixelToWorld({
      id: `${id}-kernel`,
      input: floatView(pixels),
      output: floatView(worldBuffer),
      pairCount: floatCount / 2,
      columns: grid.columns,
      rows: grid.rows,
      world,
      pixelOffset
    })
  );
  return worldBuffer;
}
