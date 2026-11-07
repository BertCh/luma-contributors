// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getGPUSpatialWeightsOrChunksViews,
  validateGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';
import {GPUChunkedSpatialWeightsTranspose} from './gpu-chunked-spatial-weights-transpose';
import {
  GPUSpatialWeightsTransform,
  type GPUSpatialWeightsKernel,
  type GPUSpatialWeightsTransformOperation
} from './gpu-spatial-weights-transform';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import type {GPUSpatialWeightsInformation} from './spatial-weights';

const OPERATION = 'GPUChunkedSpatialWeightsTransform';

/** Properties for {@link GPUChunkedSpatialWeightsTransform}. */
export type GPUChunkedSpatialWeightsTransformProps = {
  /** Prefix for generated node IDs. Defaults to `'chunked-spatial-weights-transform'`. */
  id?: string;
  /** Operation, with the semantics of {@link GPUSpatialWeightsTransform}. */
  operation: GPUSpatialWeightsTransformOperation;
  /** Chunked weights to transform. Only each chunk's `weights.weights` is written. */
  weights: GPUChunkedSpatialWeights;
  /**
   * Destination per chunk (same order, each as long as that chunk's `neighbors`). Omitted entries
   * (or the whole array) write in place. `'symmetrize'` needs separate outputs.
   */
  output?: readonly (GraphDataView<'float32'> | undefined)[];
  doubleSum?: 'one' | 'rows';
  kernel?: GPUSpatialWeightsKernel;
  bandwidth?: number | 'adaptive';
  spatialContext?: GPUSpatialContext;
};

/**
 * {@link GPUSpatialWeightsTransform} of chunked weights. `row`, `binary` and `kernel` are row-local
 * and run the whole-CSR transform once per chunk. `double` and `variance` write per-row partials
 * into one global array (chunks in source order), take one fixed-order sum and apply per chunk, so
 * the totals do not depend on the chunking. `symmetrize` first transposes the chunks into one
 * whole scratch CSR (linear in the chunk count, see {@link GPUChunkedSpatialWeightsTranspose}) and
 * looks the reverse weight up there. Results equal the whole transform at any chunking; `double` and
 * `variance` sums can differ from the whole sum only by float32 association of the partials.
 *
 * `symmetrize` allocates a scratch CSR sized by the summed chunk capacity and so requires that to
 * fit one storage binding.
 */
export class GPUChunkedSpatialWeightsTransform implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUChunkedSpatialWeightsTransformProps;
  /** The transformed chunks with `information` attached. */
  readonly outputWeights: GPUChunkedSpatialWeights;
  private readonly chunkTransforms: GPUSpatialWeightsTransform[];

  constructor(props: GPUChunkedSpatialWeightsTransformProps) {
    const id = props.id ?? 'chunked-spatial-weights-transform';
    this.id = id;
    this.props = props;
    const rows = validateGPUChunkedSpatialWeights(id, props.weights);
    const {weights, output} = props;
    if (output && output.length !== weights.chunks.length) {
      throw new Error(`${id} output must hold one entry per chunk`);
    }
    for (const view of output ?? []) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} output`);
      }
    }
    this.chunkTransforms = weights.chunks.map(
      (chunk, chunkIndex) =>
        new GPUSpatialWeightsTransform({
          ...props,
          id: `${id}-chunk${chunkIndex}`,
          weights: chunk.weights,
          output: output?.[chunkIndex],
          // The chunk transform validates the shape; symmetrize and the global sums run here.
          operation:
            props.operation === 'double' ||
            props.operation === 'variance' ||
            props.operation === 'symmetrize'
              ? 'binary'
              : props.operation
        })
    );
    if (props.operation === 'symmetrize' && this.chunkTransforms.some(t => t.isInPlace())) {
      throw new Error(`${id} 'symmetrize' needs a separate output per chunk`);
    }
    const input = weights.information;
    const preserved =
      props.operation === 'binary' ||
      props.operation === 'double' ||
      props.operation === 'symmetrize' ||
      (props.operation === 'kernel' && typeof props.bandwidth === 'number');
    const information: GPUSpatialWeightsInformation = {
      rowCount: rows,
      transform: props.operation === 'symmetrize' ? 'symmetric' : props.operation,
      symmetric: input?.symmetric === true && preserved ? true : 'unknown',
      hasSelfWeights: input?.hasSelfWeights,
      provenance: {
        ...input?.provenance,
        source: OPERATION,
        ...(props.operation === 'kernel' && typeof props.bandwidth === 'number'
          ? {radius: props.bandwidth}
          : {})
      }
    };
    this.outputWeights = {
      ...weights,
      information,
      chunks: weights.chunks.map((chunk, chunkIndex) => ({
        ...chunk,
        weights: {
          ...chunk.weights,
          weights: output?.[chunkIndex] ?? chunk.weights.weights,
          information: {...information, rowCount: chunk.rowCount}
        }
      }))
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, operation} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsOrChunksViews(weights),
      ...(props.output ?? [])
    ]);
    if (operation === 'row' || operation === 'binary' || operation === 'kernel') {
      return this.chunkTransforms.flatMap(transform => transform.getCommandNodes(graph));
    }
    const rows = weights.rowCount;
    const isInPlace = (chunkIndex: number) => this.chunkTransforms[chunkIndex].isInPlace();
    const getBindings = (chunkIndex: number) => {
      const chunk = weights.chunks[chunkIndex];
      const inPlace = isInPlace(chunkIndex);
      const source = inPlace ? 'values' : 'source';
      return {
        source,
        bindings: [
          {
            name: 'offsets',
            view: chunk.weights.offsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          ...(inPlace
            ? [
                {
                  name: 'values',
                  view: chunk.weights.weights,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [
                {
                  name: 'source',
                  view: chunk.weights.weights,
                  type: 'f32' as const,
                  access: 'read' as const
                },
                {
                  name: 'values',
                  view: props.output![chunkIndex]!,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ])
        ]
      };
    };
    if (operation === 'symmetrize') {
      return this.getSymmetrizeNodes(graph);
    }
    // double / variance
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows);
    const total = createTransientView(graph, `${id}-total`, 'float32', 1);
    const variance = operation === 'variance';
    const nodes: GPUCommandNode<Parameters>[] = [];
    weights.chunks.forEach((chunk, chunkIndex) => {
      const {source, bindings} = getBindings(chunkIndex);
      const read = (slot: string) => `${source}[${source}Offset + ${slot}]`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-partials-chunk${chunkIndex}`,
          operation: OPERATION,
          variant: `${operation}-partials`,
          bindings: [
            ...bindings,
            {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
          ],
          invocationCount: chunk.rowCount,
          declarations: `${IS_FINITE_FLOAT_WGSL}\nconst WEIGHTS_ROW_START: u32 = ${chunk.rowStart}u;`,
          body: `let row = index + WEIGHTS_ROW_START;
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${read('slot')};
    sum += value;
    squares += value * value;
  }
  ${
    variance
      ? `let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  partials[partialsOffset + row] = select(0.0, sum / norm, valid);`
      : 'partials[partialsOffset + row] = select(0.0, sum, isFiniteFloat(sum));'
  }`
        })
      );
    });
    nodes.push(
      ...new GPUReduction({
        id: `${id}-reduce`,
        input: partials,
        output: total,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
    const factor = variance
      ? `let totalValue = total[totalOffset];
  let scale = select(0.0, f32(ROWS) / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`
      : `let totalValue = total[totalOffset];
  let scale = select(0.0, ${
    props.doubleSum === 'rows' ? 'f32(ROWS)' : '1.0'
  } / totalValue, totalValue > 0.0 && isFiniteFloat(totalValue));`;
    weights.chunks.forEach((chunk, chunkIndex) => {
      const {source, bindings} = getBindings(chunkIndex);
      const read = (slot: string) => `${source}[${source}Offset + ${slot}]`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-apply-chunk${chunkIndex}`,
          operation: OPERATION,
          variant: `${operation}-apply`,
          bindings: [...bindings, {name: 'total', view: total, type: 'f32', access: 'read'}],
          invocationCount: chunk.rowCount,
          declarations: `${IS_FINITE_FLOAT_WGSL}\nconst ROWS: u32 = ${rows}u;`,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${factor}
  ${
    variance
      ? `var squares = 0.0;
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${read('slot')};
    sum += value;
    squares += value * value;
  }
  let norm = sqrt(squares);
  let valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  let rowScale = select(0.0, scale / norm, valid);`
      : 'let rowScale = scale;'
  }
  for (var slot = begin; slot < end; slot++) {
    let result = ${read('slot')} * rowScale;
    values[valuesOffset + slot] = select(0.0, result, isFiniteFloat(result) && result >= 0.0);
  }`
        })
      );
    });
    return nodes;
  }

  private getSymmetrizeNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights} = props;
    const rows = weights.rowCount;
    const capacity = weights.chunks.reduce((sum, chunk) => sum + chunk.weights.neighbors.length, 0);
    const scratch = {
      offsets: createTransientView(graph, `${id}-reverse-offsets`, 'uint32', rows + 1),
      neighbors: createTransientView(graph, `${id}-reverse-neighbors`, 'uint32', capacity),
      weights: createTransientView(graph, `${id}-reverse-weights`, 'float32', capacity)
    };
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUChunkedSpatialWeightsTranspose({
        id: `${id}-reverse`,
        weights,
        output: scratch
      }).getCommandNodes(graph)
    ];
    weights.chunks.forEach((chunk, chunkIndex) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-symmetrize-chunk${chunkIndex}`,
          operation: OPERATION,
          variant: 'symmetrize',
          bindings: [
            {name: 'offsets', view: chunk.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: chunk.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'source', view: chunk.weights.weights, type: 'f32', access: 'read'},
            {name: 'values', view: props.output![chunkIndex]!, type: 'f32', access: 'read_write'},
            {name: 'reverseOffsets', view: scratch.offsets, type: 'u32', access: 'read'},
            {name: 'reverseNeighbors', view: scratch.neighbors, type: 'u32', access: 'read'},
            {name: 'reverseWeights', view: scratch.weights, type: 'f32', access: 'read'}
          ],
          invocationCount: chunk.weights.neighbors.length,
          declarations: `const ROWS: u32 = ${rows}u;
const CHUNK_ROWS: u32 = ${chunk.rowCount}u;
const WEIGHTS_ROW_START: u32 = ${chunk.rowStart}u;`,
          body: `if (index >= offsets[offsetsOffset + CHUNK_ROWS]) {
    return;
  }
  var localRow = 0u;
  var high = CHUNK_ROWS;
  while (localRow + 1u < high) {
    let middle = (localRow + high) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      localRow = middle;
    } else {
      high = middle;
    }
  }
  let row = localRow + WEIGHTS_ROW_START;
  let neighbor = neighbors[neighborsOffset + index];
  var reverse = 0.0;
  if (neighbor < ROWS) {
    // w_ji is entry j of row i of the transpose.
    var low = reverseOffsets[reverseOffsetsOffset + row];
    var upper = reverseOffsets[reverseOffsetsOffset + row + 1u];
    while (low < upper) {
      let middle = (low + upper) / 2u;
      let candidate = reverseNeighbors[reverseNeighborsOffset + middle];
      if (candidate == neighbor) {
        reverse = reverseWeights[reverseWeightsOffset + middle];
        break;
      }
      if (candidate < neighbor) {
        low = middle + 1u;
      } else {
        upper = middle;
      }
    }
  }
  values[valuesOffset + index] = 0.5 * (source[sourceOffset + index] + reverse);`
        })
      );
    });
    return nodes;
  }
}
