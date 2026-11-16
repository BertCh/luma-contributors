// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPULineGeometryOutput} from './line-segmentize-types';

/** Initializes exact integer source rows and source chainage; fractions are blended separately. */
export function createSmoothMetadataPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    rowMeasures: GraphDataView<'float32'>;
    destination: GraphDataView<'uint32x4'>;
    closed: boolean;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: `${props.id}-metadata-prepare`,
    operation: 'GPULineSmooth',
    variant: 'metadata-prepare',
    invocationCount: props.positions.length,
    bindings: [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'rowMeasures', view: props.rowMeasures, type: 'f32', access: 'read'},
      {name: 'destination', view: props.destination, type: 'u32', access: 'read_write'}
    ],
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const CLOSED: bool = ${props.closed};
fn point(row: u32) -> vec2f { return vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]); }`,
    body: `var low = 0u; var high = LINE_COUNT;
while (high - low > 1u) { let middle = low + (high - low) / 2u; if (lineOffsets[lineOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; } }
let first = lineOffsets[lineOffsetsOffset + low];
let last = min(lineOffsets[lineOffsetsOffset + low + 1u], (arrayLength(&positions) - positionsOffset) / 2u);
var total = 0.0;
if (last > first) {
  total = rowMeasures[rowMeasuresOffset + last - 1u];
  if (CLOSED) { total += length(point(first) - point(last - 1u)); }
}
let base = destinationOffset + index * 4u;
destination[base] = index - first;
destination[base + 1u] = 0u;
destination[base + 2u] = bitcast<u32>(rowMeasures[rowMeasuresOffset + index]);
destination[base + 3u] = bitcast<u32>(total);`
  });
}

/** Publishes the interpolated source parameter; measures unwrap at the closed ring's seam. */
export function createSmoothMetadataPublishNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    metadata: GraphDataView<'uint32x4'>;
    lineOffsets: GraphDataView<'uint32'>;
    smoothStarts: GraphDataView<'uint32'>;
    output: GPULineGeometryOutput;
    closed: boolean;
    iterationCount: number;
  }
): GPUCommandNode<Parameters> {
  const {output} = props;
  return createWGSLKernelNode(graph, {
    id: `${props.id}-metadata-publish`,
    operation: 'GPULineSmooth',
    variant: 'metadata-publish',
    invocationCount: output.positions.length,
    bindings: [
      {name: 'metadata', view: props.metadata, type: 'u32', access: 'read'},
      {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
      {name: 'outputOffsets', view: output.lineOffsets, type: 'u32', access: 'read'},
      {name: 'smoothStarts', view: props.smoothStarts, type: 'u32', access: 'read'},
      ...(output.sourceRows
        ? [
            {
              name: 'sourceRows',
              view: output.sourceRows,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : []),
      ...(output.measures
        ? [
            {
              name: 'measures',
              view: output.measures,
              type: 'f32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    declarations: `const LINE_COUNT: u32 = ${props.lineOffsets.length - 1}u;
const CLOSED: bool = ${props.closed};
const SCALE: u32 = ${2 ** props.iterationCount}u;`,
    body: `let lineCount = LINE_COUNT;
var sourceRow = 0xffffffffu; var measure = 0.0;
if (index < outputOffsets[outputOffsetsOffset + lineCount]) {
  var low = 0u; var high = lineCount;
  while (high - low > 1u) { let middle = low + (high - low) / 2u; if (outputOffsets[outputOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; } }
  let base = metadataOffset + index * 4u;
  sourceRow = lineOffsets[lineOffsetsOffset + low] + metadata[base];
  measure = bitcast<f32>(metadata[base + 2u]);
  let smoothCount = smoothStarts[smoothStartsOffset + low + 1u] - smoothStarts[smoothStartsOffset + low];
  let first = outputOffsets[outputOffsetsOffset + low];
  let firstMeasure = bitcast<f32>(metadata[metadataOffset + first * 4u + 2u]);
  if (CLOSED && smoothCount > 0u && (measure < firstMeasure || index - first == smoothCount * SCALE)) { measure += bitcast<f32>(metadata[base + 3u]); }
}
${output.sourceRows ? 'sourceRows[sourceRowsOffset + index] = sourceRow;' : ''}
${output.measures ? 'measures[measuresOffset + index] = measure;' : ''}`
  });
}
