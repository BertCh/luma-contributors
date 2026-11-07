// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';

/** Element alignment that keeps batch blocks on 256-byte storage boundaries. @internal */
export const NETWORK_ALIGNMENT_ELEMENTS = 64;

/** Returns a packed range of `length` rows starting at row `firstRow` of a packed view. @internal */
export function getSubView<Format extends 'uint32' | 'float32', Parameters>(
  graph: GPUCommandGraph<Parameters>,
  view: GraphDataView<Format>,
  firstRow: number,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(view.buffer, {
    format: view.format as Format,
    length,
    byteOffset: view.byteOffset + firstRow * view.byteStride
  });
}

/** Rounds an element count up to a multiple of {@link NETWORK_ALIGNMENT_ELEMENTS}. @internal */
export function alignElementCount(count: number): number {
  return Math.ceil(count / NETWORK_ALIGNMENT_ELEMENTS) * NETWORK_ALIGNMENT_ELEMENTS;
}
