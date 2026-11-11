// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPULineGeometryOutput} from './line-segmentize-types';

/** Returns every view of a line output, for aliasing and graph-membership checks. @internal */
export function getLineGeometryOutputViews(
  output: GPULineGeometryOutput
): (GraphDataView | undefined)[] {
  return [
    output.positions,
    output.lineOffsets,
    output.status.count,
    output.status.overflow,
    output.status.requiredCount,
    output.status.approximation,
    output.status.vertexCount,
    output.status.requiredVertexCount,
    output.sourceLineIndices,
    output.sourceRows,
    output.measures
  ];
}

/**
 * Validates the formats and lengths of a {@link GPULineGeometryOutput}.
 *
 * @param id Contributor ID for error messages.
 * @param output Output to validate.
 * @param lineCapacity Required number of output lines (`lineOffsets.length - 1`).
 * @param writesApproximation Whether the contributor writes `status.approximation`; when false a
 * view for it is rejected, because it would be left unwritten.
 * @internal
 */
export function validateLineGeometryOutput(
  id: string,
  output: GPULineGeometryOutput,
  lineCapacity: number,
  writesApproximation = false
): void {
  // An unwritten status view would read as stale data, so contributors that never flag an
  // approximation refuse one.
  validateRequirement(
    id,
    writesApproximation || output.status.approximation === undefined,
    'output.status.approximation is written only by GPULineSegmentize and GPUGreatCircleArcs'
  );
  validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
  if (output.positions.length < 1) {
    throw new Error(`${id} output.positions must hold at least one row`);
  }
  const capacity = output.positions.length;
  validateSinglePackedView(output.lineOffsets, ['uint32'], `${id} output.lineOffsets`);
  if (output.lineOffsets.length !== lineCapacity + 1) {
    throw new Error(`${id} output.lineOffsets must hold ${lineCapacity + 1} rows`);
  }
  for (const [name, scalar] of [
    ['count', output.status.count],
    ['overflow', output.status.overflow],
    ['requiredCount', output.status.requiredCount],
    ['approximation', output.status.approximation],
    ['vertexCount', output.status.vertexCount],
    ['requiredVertexCount', output.status.requiredVertexCount]
  ] as const) {
    if (scalar) {
      validateSinglePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
      if (scalar.length < 1) {
        throw new Error(`${id} output.status.${name} must contain one uint32 row`);
      }
    }
  }
  if (output.sourceLineIndices) {
    validateSinglePackedView(
      output.sourceLineIndices,
      ['uint32'],
      `${id} output.sourceLineIndices`
    );
    if (output.sourceLineIndices.length !== lineCapacity) {
      throw new Error(`${id} output.sourceLineIndices must hold ${lineCapacity} rows`);
    }
  }
  if (output.sourceRows) {
    validateSinglePackedView(output.sourceRows, ['uint32'], `${id} output.sourceRows`);
    if (output.sourceRows.length !== capacity) {
      throw new Error(`${id} output.sourceRows length must equal output.positions length`);
    }
  }
  if (output.measures) {
    validateSinglePackedView(output.measures, ['float32'], `${id} output.measures`);
    if (output.measures.length !== capacity) {
      throw new Error(`${id} output.measures length must equal output.positions length`);
    }
  }
}

/**
 * Writes the line counts of a {@link GPULineGeometryOutput}: `status.count` and the optional
 * `status.requiredCount`. Without `lineTotal` both are the compile-time `lineCapacity` (one output
 * line per input line or pair); with it, `count` is `min(lineTotal, lineCapacity)` and
 * `requiredCount` is `lineTotal`.
 * @internal
 */
export function createLineGeometryCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    output: GPULineGeometryOutput;
    lineCapacity: number;
    /** One-row unclamped line total written by an earlier node. */
    lineTotal?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const {count, requiredCount} = props.output.status;
  const total = props.lineTotal ? 'lineTotal[lineTotalOffset]' : 'LINE_CAPACITY';
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-line-count`,
    operation: props.operation,
    variant: `line-count-${props.lineTotal ? 1 : 0}${requiredCount ? 1 : 0}`,
    bindings: [
      ...(props.lineTotal
        ? [
            {
              name: 'lineTotal',
              view: props.lineTotal,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'count', view: count, type: 'u32', access: 'read_write'},
      ...(requiredCount
        ? [
            {
              name: 'requiredCount',
              view: requiredCount,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    invocationCount: 1,
    declarations: `const LINE_CAPACITY: u32 = ${props.lineCapacity}u;`,
    body: [
      `let total = ${total};`,
      'count[countOffset] = min(total, LINE_CAPACITY);',
      requiredCount ? 'requiredCount[requiredCountOffset] = total;' : ''
    ].join('\n  ')
  });
}
