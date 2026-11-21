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
import type {SpatialWeightsValidationOptions} from './spatial-weights-validation-options';

/** Properties for {@link GPUChunkedSpatialWeightsValidation}. */
export type GPUChunkedSpatialWeightsValidationProps = SpatialWeightsValidationOptions & {
  /** Prefix for generated node and transient IDs. Defaults to `'chunked-spatial-weights-validation'`. */
  id?: string;
  /** Chunked weights to check. Source-column count defaults to the total target-row count. Read only. */
  weights: GPUChunkedSpatialWeights;
  /** Caller-owned outputs; `flags` covers all rows in global row order. */
  output: GPUSpatialWeightsValidationOutput;
};

/**
 * {@link GPUSpatialWeightsValidation} over a chunked CSR. The validation is one implementation
 * that accepts whole or chunked weights (whole weights are a single chunk); this class is the
 * chunked-typed form kept for existing callers, and `new GPUSpatialWeightsValidation({weights})`
 * with chunked weights is equivalent. One rows dispatch per chunk (chunk `c` checks its own offsets
 * and capacity, and neighbor rows against the declared source domain `columnCount`), per-row flags and the invalid count
 * in global row order. Dispatch count is linear in the chunk count and results do not depend on
 * the chunking.
 *
 * After upload, submit validation and read `output.status.invalidCount` on the host before
 * submitting any downstream consumer, including `GPUSpatialLag`. The attached status is not
 * an automatic GPU gate. Under the default policy (`columnCount` is the total target-row count),
 * for example, a two-row chunked matrix whose first chunk has offsets
 * `[0, 1]`, neighbor `[2]`, weight `[1]`, and whose second chunk has offsets `[0, 0]` has one
 * invalid row: neighbor 2 is outside the default two-row source domain. Declaring `columnCount: 3`
 * admits that neighbor in the three-source domain. It is not a second island.
 * Repair and revalidate the CSR before use. This validation count is distinct from `overflow`
 * (final-edge capacity loss), `candidateOverflow` (candidate capacity loss), and coordinates
 * deliberately excluded by a producer. Loss status remains optional and does not certify CSR.
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
