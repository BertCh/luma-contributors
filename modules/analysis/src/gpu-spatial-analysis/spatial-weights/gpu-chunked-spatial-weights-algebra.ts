// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsOrChunksViews,
  validateGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights,
  type GPUSpatialWeightsChunk
} from './gpu-chunked-spatial-weights';
import {
  COMBINE_RULES,
  combineAlgebraTransforms,
  type GPUSpatialWeightsCombineRule
} from './gpu-spatial-weights-algebra';
import type {SpatialWeightsMergeOperation} from './spatial-weights-algebra-wgsl';
import {
  getBinaryMergeStages,
  getSelfWeightStages,
  getSubgraphStages,
  type RowLocalStageBase
} from './spatial-weights-row-local-stages';
import {
  getGPUSpatialWeightsStatusViews,
  getGPUSpatialWeightsViews,
  validateGPUSpatialWeightsOutput,
  type GPUSpatialWeightsInformation,
  type GPUSpatialWeightsOutput
} from './spatial-weights';

const OPERATION = 'GPUChunkedSpatialWeightsAlgebra';

/**
 * Caller-owned chunked output: chunk `c` has the row range of the operand chunk `c` and its own
 * CSR, `status.overflow` and optional `status.requiredCount` (capacity and overflow are per chunk).
 */
export type GPUChunkedSpatialWeightsOutput = {
  chunks: readonly {rowStart: number; rowCount: number; weights: GPUSpatialWeightsOutput}[];
};

type ChunkStageBase = RowLocalStageBase;

type ChunkedAlgebraBaseProps = {
  /** Prefix for generated node IDs. Defaults to `'chunked-spatial-weights-algebra'`. */
  id?: string;
  /** Chunked destination whose chunks have the same row ranges as the operand chunks. */
  output: GPUChunkedSpatialWeightsOutput;
};

/** Properties for {@link GPUChunkedSpatialWeightsAlgebra}. */
export type GPUChunkedSpatialWeightsAlgebraProps =
  | (ChunkedAlgebraBaseProps & {
      operation: SpatialWeightsMergeOperation;
      left: GPUChunkedSpatialWeights;
      right: GPUChunkedSpatialWeights;
      weightRule?: GPUSpatialWeightsCombineRule;
    })
  | (ChunkedAlgebraBaseProps & {
      operation: 'selfWeight';
      weights: GPUChunkedSpatialWeights;
      selfWeight: number | GraphDataView<'float32'>;
    })
  | (ChunkedAlgebraBaseProps & {
      operation: 'subgraph';
      weights: GPUChunkedSpatialWeights;
      mask: GraphDataView<'uint32'>;
    });

/**
 * The row-local operations of {@link GPUSpatialWeightsAlgebra} over chunked weights: `union`,
 * `intersection`, `difference`, `symmetricDifference`, `selfWeight` and `subgraph`. One
 * count/scan/emit pipeline per chunk (dispatch count linear in the chunk count), each writing the
 * matching output chunk, so output row order and content do not depend on the chunking; there is no
 * repacking and overflow is reported per chunk.
 *
 * Not chunked (use the whole-CSR class): `higherOrder`, whose rows read arbitrary other rows
 * through the frontier, and `block`, which has no weights input.
 *
 * Binary operands must be chunked identically (same row ranges); the output must match too.
 */
export class GPUChunkedSpatialWeightsAlgebra implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUChunkedSpatialWeightsAlgebraProps;
  /** The output chunks with `information` attached. */
  readonly outputWeights: GPUChunkedSpatialWeights;

  constructor(props: GPUChunkedSpatialWeightsAlgebraProps) {
    const id = props.id ?? 'chunked-spatial-weights-algebra';
    this.id = id;
    this.props = props;
    const operands = this.getOperands();
    const rows = validateGPUChunkedSpatialWeights(id, operands[0]);
    for (const operand of operands.slice(1)) {
      if (validateGPUChunkedSpatialWeights(id, operand) !== rows) {
        throw new Error(`${id} operands must have the same row count`);
      }
    }
    if (props.output.chunks.length !== operands[0].chunks.length) {
      throw new Error(`${id} output must have one chunk per operand chunk`);
    }
    props.output.chunks.forEach((outputChunk, index) => {
      for (const operand of operands) {
        const chunk = operand.chunks[index];
        if (chunk.rowStart !== outputChunk.rowStart || chunk.rowCount !== outputChunk.rowCount) {
          throw new Error(`${id} output chunk ${index} must match the operand chunk row range`);
        }
      }
      if (validateGPUSpatialWeightsOutput(id, outputChunk.weights) !== outputChunk.rowCount) {
        throw new Error(`${id} output chunk ${index} offsets must hold rowCount + 1 entries`);
      }
    });
    if (props.operation === 'selfWeight' && typeof props.selfWeight !== 'number') {
      validateSinglePackedView(props.selfWeight, ['float32'], `${id} selfWeight`);
      if (props.selfWeight.length !== rows) {
        throw new Error(`${id} selfWeight view length must equal the row count`);
      }
    } else if (props.operation === 'selfWeight') {
      const selfWeight = props.selfWeight as number;
      if (!(Number.isFinite(selfWeight) && selfWeight >= 0)) {
        throw new Error(`${id} selfWeight must be a non-negative finite number`);
      }
    }
    if (props.operation === 'subgraph') {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    if ('weightRule' in props && props.weightRule && !(props.weightRule in COMBINE_RULES)) {
      throw new Error(`${id} unknown weightRule ${props.weightRule}`);
    }
    const inputViews = [
      ...operands.flatMap(getGPUSpatialWeightsOrChunksViews),
      props.operation === 'subgraph' ? props.mask : undefined,
      props.operation === 'selfWeight' && typeof props.selfWeight !== 'number'
        ? props.selfWeight
        : undefined
    ];
    validateGraphOutputsDisjointFromInputs(
      id,
      props.output.chunks.flatMap(chunk => [
        ...getGPUSpatialWeightsViews(chunk.weights),
        ...getGPUSpatialWeightsStatusViews(chunk.weights)
      ]),
      inputViews
    );
    const information = this.getOutputInformation(rows);
    this.outputWeights = {
      rowCount: rows,
      spatialContext: operands[0].spatialContext,
      information,
      chunks: props.output.chunks.map(chunk => ({
        rowStart: chunk.rowStart,
        rowCount: chunk.rowCount,
        weights: {
          ...chunk.weights,
          information: {...information, rowCount: chunk.rowCount},
          status: chunk.weights.status
        }
      }))
    };
  }

  private getOperands(): GPUChunkedSpatialWeights[] {
    const {props} = this;
    return props.operation === 'selfWeight' || props.operation === 'subgraph'
      ? [props.weights]
      : [props.left, props.right];
  }

  private getOutputInformation(rowCount: number): GPUSpatialWeightsInformation {
    const {props} = this;
    const provenance = {source: OPERATION, criterion: props.operation};
    const symmetric = this.getOperands().every(operand => operand.information?.symmetric === true)
      ? true
      : 'unknown';
    if (props.operation === 'selfWeight') {
      return {
        rowCount,
        transform: props.weights.information?.transform === 'raw' ? 'raw' : 'unknown',
        symmetric,
        hasSelfWeights: true,
        provenance
      };
    }
    if (props.operation === 'subgraph') {
      return {
        rowCount,
        transform: combineAlgebraTransforms(props.weights.information?.transform),
        symmetric,
        hasSelfWeights: props.weights.information?.hasSelfWeights,
        provenance
      };
    }
    const operands = props.operation === 'difference' ? [props.left] : [props.left, props.right];
    return {
      rowCount,
      transform:
        props.weightRule === 'binary'
          ? 'binary'
          : combineAlgebraTransforms(
              ...operands.map(operand =>
                props.weightRule === 'sum' && operand.information?.transform === 'binary'
                  ? 'unknown'
                  : operand.information?.transform
              )
            ),
      symmetric,
      hasSelfWeights: false,
      provenance
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.getOperands().flatMap(getGPUSpatialWeightsOrChunksViews),
      ...props.output.chunks.flatMap(chunk => [
        ...getGPUSpatialWeightsViews(chunk.weights),
        ...getGPUSpatialWeightsStatusViews(chunk.weights)
      ]),
      props.operation === 'subgraph' ? props.mask : undefined,
      props.operation === 'selfWeight' && typeof props.selfWeight !== 'number'
        ? props.selfWeight
        : undefined
    ]);
    return props.output.chunks.flatMap((outputChunk, chunkIndex) => {
      const chunkId = `${id}-chunk${chunkIndex}`;
      const chunk = {rowStart: outputChunk.rowStart, totalRows: this.outputWeights.rowCount};
      const output = outputChunk.weights;
      const base = {
        id: chunkId,
        operation: OPERATION,
        rows: outputChunk.rowCount,
        targetOffsets: output.offsets,
        capacity: output.neighbors.length,
        overflow: output.status.overflow,
        initializesOverflow: true,
        totalNeighbors: output.status.requiredCount,
        chunk
      };
      if (props.operation === 'selfWeight' || props.operation === 'subgraph') {
        return props.operation === 'selfWeight'
          ? this.getSelfWeightNodes(graph, base, props, props.weights.chunks[chunkIndex], output)
          : this.getSubgraphNodes(graph, base, props, props.weights.chunks[chunkIndex], output);
      }
      return this.getBinaryNodes(
        graph,
        base,
        props,
        props.left.chunks[chunkIndex],
        props.right.chunks[chunkIndex],
        output
      );
    });
  }

  private getBinaryNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    base: ChunkStageBase,
    props: Extract<GPUChunkedSpatialWeightsAlgebraProps, {left: unknown}>,
    leftChunk: GPUSpatialWeightsChunk,
    rightChunk: GPUSpatialWeightsChunk,
    output: GPUSpatialWeightsOutput
  ): GPUCommandNode<Parameters>[] {
    const rule = props.weightRule ?? 'left';
    return getBinaryMergeStages(graph, base, {
      operation: props.operation,
      left: leftChunk.weights,
      right: rightChunk.weights,
      output,
      isBinaryRule: rule === 'binary',
      combineExpression: COMBINE_RULES[rule]
    });
  }

  private getSelfWeightNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    base: ChunkStageBase,
    props: Extract<GPUChunkedSpatialWeightsAlgebraProps, {operation: 'selfWeight'}>,
    sourceChunk: GPUSpatialWeightsChunk,
    output: GPUSpatialWeightsOutput
  ): GPUCommandNode<Parameters>[] {
    return getSelfWeightStages(graph, base, {
      weights: sourceChunk.weights,
      output,
      selfWeight: props.selfWeight
    });
  }

  private getSubgraphNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    base: ChunkStageBase,
    props: Extract<GPUChunkedSpatialWeightsAlgebraProps, {operation: 'subgraph'}>,
    sourceChunk: GPUSpatialWeightsChunk,
    output: GPUSpatialWeightsOutput
  ): GPUCommandNode<Parameters>[] {
    return getSubgraphStages(graph, base, {
      weights: sourceChunk.weights,
      output,
      mask: props.mask
    });
  }
}
