// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode} from '@luma.gl/graph/gpu-core';
import type {GPUChunkedSpatialWeights} from './gpu-chunked-spatial-weights';
import {
  GPUSpatialWeightsValidation,
  type GPUSpatialWeightsValidationOutput,
  type GPUSpatialWeightsValidationProps
} from './gpu-spatial-weights-validation';

/** Properties for {@link GPUChunkedSpatialWeightsValidation}. */
export type GPUChunkedSpatialWeightsValidationProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'chunked-spatial-weights-validation'`. */
  id?: string;
  /** Square chunked weights to check. Read only. */
  weights: GPUChunkedSpatialWeights;
  /** Caller-owned outputs; `flags` covers all rows in global row order. */
  output: GPUSpatialWeightsValidationOutput;
};

/**
 * {@link GPUSpatialWeightsValidation} over a chunked CSR. The validation is one implementation
 * that accepts whole or chunked weights (whole weights are a single chunk); this class is the
 * chunked-typed form kept for existing callers, and `new GPUSpatialWeightsValidation({weights})`
 * with chunked weights is equivalent. One rows dispatch per chunk (chunk `c` checks its own offsets
 * and capacity, and neighbor IDs against the total row count), per-row flags and the invalid count
 * in global row order. Dispatch count is linear in the chunk count and results do not depend on
 * the chunking.
 */
export class GPUChunkedSpatialWeightsValidation extends GPUSpatialWeightsValidation<GPUChunkedSpatialWeights> {
  constructor(props: GPUChunkedSpatialWeightsValidationProps) {
    super(props satisfies GPUSpatialWeightsValidationProps<GPUChunkedSpatialWeights>);
  }

  /** Same nodes as {@link GPUSpatialWeightsValidation}; declared so the class is a public contributor. */
  override getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return super.getCommandNodes(graph);
  }
}
