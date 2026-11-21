// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPULineMerge} from '../../../src/gpu-spatial-analysis/line-merge/index';

export type OracleView = <Format extends 'uint32' | 'float32' | 'float32x2'>(
  format: Format,
  length: number,
  values?: number[]
) => GraphDataView<Format>;

export const MERGE_OPTIONALS = [
  'requiredCount',
  'vertexCount',
  'requiredVertexCount',
  'sourceIds',
  'lineOrders',
  'lineReversed',
  'absent'
] as const;

export function createMergeOracleFixture(
  view: OracleView,
  optional: (typeof MERGE_OPTIONALS)[number]
) {
  // Line 1 is reversed and joins line 0: (0,0)-(1,0)-(2,0).
  // Line 2 is a singleton and yields no chain; line 3 is independent.
  const chainOffsets = view('uint32', 6);
  const positions = view('float32x2', 7);
  const count = view('uint32', 1);
  const scalar = ['requiredCount', 'vertexCount', 'requiredVertexCount'].includes(optional)
    ? view('uint32', 1)
    : undefined;
  const sourceIds = optional === 'sourceIds' ? view('uint32', 4) : undefined;
  const lineOrders = optional === 'lineOrders' ? view('uint32', 4) : undefined;
  const lineReversed = optional === 'lineReversed' ? view('uint32', 4) : undefined;
  const contributor = new GPULineMerge({
    positions: view('float32x2', 7, [0, 0, 1, 0, 2, 0, 1, 0, 5, 5, 7, 7, 8, 8]),
    lineOffsets: view('uint32', 5, [0, 2, 4, 5, 7]),
    output: {
      positions,
      chainOffsets,
      sourceIds,
      lineOrders,
      lineReversed,
      status: {count, ...(scalar ? {[optional]: scalar} : {})}
    }
  });
  return {contributor, chainOffsets, positions, count, scalar, sourceIds, lineOrders, lineReversed};
}

export function createMergeReuseFixture(view: OracleView) {
  const inputOffsets = view('uint32', 3, [0, 2, 4]);
  const lineReversed = view('uint32', 2);
  const chainOffsets = view('uint32', 4);
  const count = view('uint32', 1);
  const props = {
    positions: view('float32x2', 4, [0, 0, 1, 0, 2, 0, 1, 0]),
    lineOffsets: inputOffsets,
    output: {positions: view('float32x2', 4), chainOffsets, lineReversed, status: {count}}
  };
  const contributor = new GPULineMerge(props);
  return {contributor, props, inputOffsets, lineReversed, chainOffsets, count};
}
