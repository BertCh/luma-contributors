// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialStatisticsBindings} from './spatial-statistics-bindings';
import {getSpatialWeightsTransformBindingViews} from './spatial-weights-transform-bindings';
import {resolveKernelShapeName} from '../shared/kernel-shape-name';
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
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL} from './spatial-weights-transform-arithmetic';
import {validateWeightsLogicalAliases} from './spatial-weights-aliases';
import {
  getGPUSpatialWeightsOrChunksViews,
  validateGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';
import {GPUChunkedSpatialWeightsTranspose} from './gpu-chunked-spatial-weights-transpose';
import {
  GPUSpatialWeightsTransform,
  validateSpatialWeightsTransformOperation,
  type GPUSpatialWeightsKernel,
  type GPUSpatialWeightsTransformOperation
} from './gpu-spatial-weights-transform';
import {
  assertCompatibleGPUSpatialContexts,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUSpatialWeightsInformation} from './spatial-weights';
import {
  snapshotWeightsFields,
  snapshotWeightsPort,
  snapshotWeightsSpatialContext
} from './weights-construction-snapshot';

import {validateWeightsLossSources} from './spatial-weights-loss';

const OPERATION = 'GPUChunkedSpatialWeightsTransform';

/** The fields of a metadata object whose values are defined; absent and undefined are alike. */
function getDefinedFields<T extends object>(value: T | undefined): Partial<T> {
  return Object.fromEntries(
    Object.entries(value ?? {}).filter(([, field]) => field !== undefined)
  ) as Partial<T>;
}

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
   * (or the whole array) write in place. `'average-reciprocal'` needs separate outputs.
   */
  output?: readonly (GraphDataView<'float32'> | undefined)[];
  doubleSum?: 'one' | 'rows';
  kernel?: GPUSpatialWeightsKernel;
  /** Numeric default, adaptive mode, or a shared per-frame fixed-bandwidth parameter view. */
  bandwidth?: number | 'adaptive' | GraphDataView<'float32'>;
  spatialContext?: GPUSpatialContext;
};

/**
 * {@link GPUSpatialWeightsTransform} of chunked weights. `row`, `binary` and `kernel` are row-local
 * and run the whole-CSR transform once per chunk. `double` and `variance` write per-row partials
 * into one global array (chunks in source order), take one fixed-order sum and apply per chunk, so
 * the totals do not depend on the chunking. `average-reciprocal` first transposes the chunks into one
 * whole scratch CSR (linear in the chunk count, see {@link GPUChunkedSpatialWeightsTranspose}) and
 * looks the reverse weight up there. Results equal the whole transform at any chunking; `double` and
 * `variance` sums can differ from the whole sum only by float32 association of the partials.
 *
 * Per-frame: weight/output contents and a shared bandwidth parameter view. Compile-time:
 * operation, kernel, adaptive versus fixed bandwidth, chunk topology and optional views.
 *
 * `row`, `binary` and `kernel` require only fitting chunk bindings and chunk-local scratch;
 * no whole-matrix row scratch is allocated. `double` and `variance` require rowCount float32
 * partials in one storage binding. `average-reciprocal` requires rowCount + 1 uint32 scratch offsets and
 * summed chunk capacity uint32 neighbors and float32 weights, each in one storage binding.
 */
export class GPUChunkedSpatialWeightsTransform implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUChunkedSpatialWeightsTransformProps;
  /** The transformed chunks with `information` attached. */
  readonly outputWeights: GPUChunkedSpatialWeights;
  private readonly chunkTransforms: GPUSpatialWeightsTransform[];

  constructor(props: GPUChunkedSpatialWeightsTransformProps) {
    props = snapshotWeightsFields(props, [
      'id',
      'operation',
      'weights',
      'output',
      'doubleSum',
      'kernel',
      'bandwidth',
      'spatialContext'
    ]);
    props = Object.freeze({
      ...props,
      weights: snapshotWeightsPort(props.weights),
      ...(props.output ? {output: Object.freeze([...props.output])} : {}),
      ...(props.spatialContext
        ? {spatialContext: snapshotWeightsSpatialContext(props.spatialContext)}
        : {})
    });
    const id = props.id ?? 'chunked-spatial-weights-transform';
    this.id = id;
    this.props = props;
    validateSpatialWeightsTransformOperation(id, props.operation);
    if (props.kernel !== undefined) {
      resolveKernelShapeName(id, props.kernel, [
        'gaussian',
        'triangular',
        'epanechnikov',
        'quartic',
        'uniform'
      ] as const);
    }
    const rows = validateGPUChunkedSpatialWeights(id, props.weights);
    const {weights, output} = props;
    validateWeightsLossSources(id, [weights], true);
    if (output && output.length !== weights.chunks.length) {
      throw new Error(`${id} output must hold one entry per chunk`);
    }
    for (const view of output ?? []) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output`);
      }
    }
    const destinations = Object.fromEntries(
      weights.chunks.map((chunk, index) => [
        `chunk${index}.weights`,
        output?.[index] ?? chunk.weights.weights
      ])
    );
    const preserved = Object.fromEntries([
      ...Object.entries(weights.status ?? {}).map(
        ([name, view]) => [`status.${name}`, view] as const
      ),
      ...weights.chunks.flatMap((chunk, index) => [
        [`chunk${index}.offsets`, chunk.weights.offsets] as const,
        [`chunk${index}.neighbors`, chunk.weights.neighbors] as const,
        [`chunk${index}.distances`, chunk.weights.distances] as const,
        ...Object.entries(chunk.weights.status ?? {}).map(
          ([name, view]) => [`chunk${index}.status.${name}`, view] as const
        ),
        ...(props.operation === 'kernel'
          ? []
          : [[`chunk${index}.weights`, chunk.weights.weights] as const])
      ])
    ]);
    validateWeightsLogicalAliases(
      id,
      destinations,
      preserved,
      (destination, input) => destination === input
    );
    if (props.operation === 'average-reciprocal') {
      for (const [index, chunk] of weights.chunks.entries())
        validateGraphOutputsDisjointFromInputs(id, [output?.[index]], [chunk.weights.neighbors]);
    }
    this.chunkTransforms = weights.chunks.map((chunk, chunkIndex) => {
      if (weights.spatialContext && chunk.weights.spatialContext) {
        assertCompatibleGPUSpatialContexts(
          id,
          weights.spatialContext,
          chunk.weights.spatialContext,
          {purpose: props.operation === 'kernel' ? 'metric-results' : 'coordinates'}
        );
      }
      return new GPUSpatialWeightsTransform({
        ...props,
        id: `${id}-chunk${chunkIndex}`,
        weights: {
          ...chunk.weights,
          spatialContext: chunk.weights.spatialContext ?? weights.spatialContext,
          // A chunk field that is present but undefined does not erase the matrix declaration
          // (for example criterion: 'knn', which an adaptive kernel needs); a defined chunk
          // field still takes precedence.
          information:
            weights.information || chunk.weights.information
              ? ({
                  ...weights.information,
                  ...getDefinedFields(chunk.weights.information),
                  rowCount: chunk.rowCount,
                  provenance: {
                    ...weights.information?.provenance,
                    ...getDefinedFields(chunk.weights.information?.provenance)
                  }
                } as GPUSpatialWeightsInformation)
              : undefined
        },
        output: output?.[chunkIndex],
        // The chunk transform validates the shape; average-reciprocal and the global sums run here.
        operation:
          props.operation === 'double' ||
          props.operation === 'variance' ||
          props.operation === 'average-reciprocal'
            ? 'binary'
            : props.operation
      });
    });
    if (props.operation === 'average-reciprocal' && this.chunkTransforms.some(t => t.isInPlace())) {
      throw new Error(`${id} 'average-reciprocal' needs a separate output per chunk`);
    }
    const input = weights.information;
    const preservesSymmetry =
      props.operation === 'binary' ||
      props.operation === 'double' ||
      props.operation === 'average-reciprocal';
    const information: GPUSpatialWeightsInformation = {
      rowCount: rows,
      transform: props.operation === 'average-reciprocal' ? 'symmetric' : props.operation,
      symmetric: input?.symmetric === true && preservesSymmetry ? true : 'unknown',
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
          spatialContext: this.chunkTransforms[chunkIndex].outputWeights.spatialContext,
          weights: output?.[chunkIndex] ?? chunk.weights.weights,
          information: {
            ...this.chunkTransforms[chunkIndex].outputWeights.information,
            transform: information.transform,
            symmetric:
              this.chunkTransforms[chunkIndex].outputWeights.information?.symmetric === true &&
              preservesSymmetry
                ? true
                : 'unknown',
            rowCount: chunk.rowCount,
            provenance: {
              ...this.chunkTransforms[chunkIndex].outputWeights.information?.provenance,
              source: OPERATION
            }
          }
        }
      }))
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpatialStatisticsBindings(
      graph,
      this.id,
      {
        chunks: this.props.weights.chunks.map((chunk, chunkIndex) =>
          getSpatialWeightsTransformBindingViews({
            ...this.props,
            weights: chunk.weights,
            output: this.props.output?.[chunkIndex]
          })
        )
      },
      {
        'row scratch':
          this.props.operation === 'average-reciprocal'
            ? (this.props.weights.rowCount + 1) * 4
            : this.props.operation === 'double' || this.props.operation === 'variance'
              ? this.props.weights.rowCount * 4
              : 0,
        'transpose scratch':
          this.props.operation === 'average-reciprocal'
            ? this.props.weights.chunks.reduce(
                (total, chunk) => total + chunk.weights.neighbors.length,
                0
              ) * 4
            : 0
      }
    );
    const {id, props} = this;
    const {weights, operation} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsOrChunksViews(weights, true),
      ...Object.values(weights.status ?? {}),
      ...weights.chunks.flatMap(chunk => Object.values(chunk.weights.status ?? {})),
      ...(props.output ?? []),
      typeof props.bandwidth === 'object' ? props.bandwidth : undefined
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
    if (operation === 'average-reciprocal') {
      return this.getAverageReciprocalNodes(graph);
    }
    // double / variance
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows);
    const total = createTransientView(graph, `${id}-total`, 'float32', 1);
    const variance = operation === 'variance';
    const getVarianceRow = (source: string) => `var squares = 0.0;
  var sum = 0.0;
  var maximum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let value = ${source}[${source}Offset + slot];
    sum += value;
    squares += value * value;
    maximum = max(maximum, value);
  }
  var norm = sqrt(squares);
  var valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  var normalizationScale = 1.0;
  if ((!valid || squares < 1.1754943508222875e-38) && maximum > 0.0 && isFiniteFloat(maximum)) {
    normalizationScale = maximum;
    squares = 0.0;
    sum = 0.0;
    for (var slot = begin; slot < end; slot++) {
      let value = ${source}[${source}Offset + slot] / maximum;
      sum += value;
      squares += value * value;
    }
    norm = sqrt(squares);
    valid = norm > 0.0 && isFiniteFloat(norm) && isFiniteFloat(sum);
  }`;
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
  ${
    variance
      ? `${getVarianceRow(source)}
  partials[partialsOffset + row] = select(0.0, sum / norm, valid);`
      : `var sum = 0.0;
  for (var slot = begin; slot < end; slot++) { sum += ${read('slot')}; }
  partials[partialsOffset + row] = sum;`
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
  let normalization = getSpatialWeightsDoubleScale(totalValue, ${
    props.doubleSum === 'rows' ? 'f32(ROWS)' : '1.0'
  });
  let scale = normalization.value;`;
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
          declarations: `${IS_FINITE_FLOAT_WGSL}\n${variance ? '' : SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL}\nconst ROWS: u32 = ${rows}u;`,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${factor}
  ${
    variance
      ? `${getVarianceRow(source)}
  let rowScale = select(0.0, scale / norm, valid);`
      : 'let rowScale = scale;'
  }
  for (var slot = begin; slot < end; slot++) {
    var value = ${read('slot')};
    ${variance ? 'if (normalizationScale != 1.0) { value = value / normalizationScale; }' : ''}
    var result = value * rowScale;
    ${
      variance
        ? ''
        : `if (normalization.useScaledQuotient != 0u) {
      result = getScaledQuotient(value, totalValue, 1.0, 0i) * ${props.doubleSum === 'rows' ? 'f32(ROWS)' : '1.0'};
    }`
    }
    values[valuesOffset + slot] = select(0.0, result, isFiniteFloat(result) && result >= 0.0);
  }`
        })
      );
    });
    return nodes;
  }

  private getAverageReciprocalNodes<Parameters>(
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
        // Scratch needs only CSR data. Public outputWeights keeps every original loss view;
        // those flags describe completeness, not a reason to refuse the private transpose.
        weights: {
          rowCount: rows,
          chunks: weights.chunks.map(chunk => ({
            rowStart: chunk.rowStart,
            rowCount: chunk.rowCount,
            weights: {
              offsets: chunk.weights.offsets,
              neighbors: chunk.weights.neighbors,
              weights: chunk.weights.weights
            }
          }))
        },
        output: scratch
      }).getCommandNodes(graph)
    ];
    weights.chunks.forEach((chunk, chunkIndex) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-average-reciprocal-chunk${chunkIndex}`,
          operation: OPERATION,
          variant: 'average-reciprocal',
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
          declarations: `${IS_FINITE_FLOAT_WGSL}\n${SPATIAL_WEIGHTS_TRANSFORM_ARITHMETIC_WGSL}\nconst ROWS: u32 = ${rows}u;
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
  values[valuesOffset + index] = averageSpatialWeights(source[sourceOffset + index], reverse);`
        })
      );
    });
    return nodes;
  }
}
