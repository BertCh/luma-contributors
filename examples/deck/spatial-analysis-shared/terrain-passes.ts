// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {addKernelPass} from './mode-kernels';

/**
 * Adds `depth = filled - elevation` (0 where either is NaN), the thickness of the depression
 * fill, so a raster layer can show which cells were raised and by how much.
 */
export function addFillDepthPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    elevation: GraphDataView<'float32'>;
    filled: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: props.id,
    bindings: [
      {name: 'elevation', view: props.elevation, type: 'f32', access: 'read'},
      {name: 'filled', view: props.filled, type: 'f32', access: 'read'},
      {name: 'depth', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let value = filled[filledOffset + index] - elevation[elevationOffset + index];
  depth[depthOffset + index] = select(value, 0.0, isNaNValue(value));`
  });
}

/**
 * Adds `class = hash(label) % 8` for drainage basin labels so neighboring basins differ in color;
 * the no-data label stays as is.
 */
export function addBasinClassPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    labels: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'basin-class',
    bindings: [
      {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
      {name: 'classes', view: props.output, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let label = labels[labelsOffset + index];
  classes[classesOffset + index] =
    select(((label * 2654435761u) >> 13u) % 8u, 0xffffffffu, label == 0xffffffffu);`
  });
}

export const NAN_DECLARATIONS = /* wgsl */ `
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`;
