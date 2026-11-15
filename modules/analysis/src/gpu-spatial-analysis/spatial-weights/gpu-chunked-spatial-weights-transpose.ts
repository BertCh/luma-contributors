// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/status';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getGPUSpatialWeightsOrChunksViews,
  validateGPUChunkedSpatialWeights,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';
import {
  getGPUSpatialWeightsViews,
  validateGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput
} from './spatial-weights';
import {getChunkedSpatialWeightsSymmetryNodes} from './chunked-spatial-weights-symmetry';
import {
  getSpatialWeightsTransposeNodes,
  planSpatialWeightsTranspose,
  type GPUSpatialWeightsTransposeOutput
} from './spatial-weights-transpose-pipeline';

import {
  getGPUSpatialWeightsTransposeParameterValues,
  GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH
} from './spatial-weights-parameters';

const OPERATION = 'GPUChunkedSpatialWeightsTranspose';
// CSR counts and the sentinel key are uint32; symmetry may count two positions per slot.
const MAXIMUM_SLOT_COUNT = 0xffffffff;
// Published GPUSort accepts at most 2^31 logical rows in each packed sort.
const MAXIMUM_SORT_SLOT_COUNT = 0x80000000;
// At most eight radix digits, each with a hierarchical scan of at most four levels,
// plus histogram/scatter passes. This also bounds the one-pass small bitonic sort.
const SORT_COMMAND_NODE_ALLOWANCE = 128;
// A scan or reduction over uint32-indexed rows has at most four 256-row levels.
const SCAN_OR_REDUCTION_COMMAND_NODE_ALLOWANCE = 16;

/** Independently partitioned, caller-owned output row ranges of the transpose. */
export type GPUChunkedSpatialWeightsTransposeOutput = {
  /** Consecutive ranges covering [0, columnCount); offsets start at zero in every chunk. */
  chunks: readonly {
    rowStart: number;
    rowCount: number;
    weights: GPUSpatialWeightsOutput & {
      /** count is clamped to capacity; requiredCount is exact; overflow reports dropped slots. */
      status: Required<GPUCardinalityPort & GPUOverflowStatusPort>;
    };
  }[];
  /** Same symmetry diagnostic as packed output, computed from the input even on overflow. */
  asymmetricSlots?: GraphDataView<'uint32'>;
};

type TransposeOutput = GPUSpatialWeightsTransposeOutput | GPUChunkedSpatialWeightsTransposeOutput;
type DescribedOutput<Output extends TransposeOutput> =
  Output extends GPUChunkedSpatialWeightsTransposeOutput
    ? GPUChunkedSpatialWeights
    : GPUSpatialWeights;

/** Properties for {@link GPUChunkedSpatialWeightsTranspose}. */
export type GPUChunkedSpatialWeightsTransposeProps<
  Output extends TransposeOutput = TransposeOutput
> = {
  /** Prefix for generated node and transient IDs. Defaults to 'chunked-spatial-weights-transpose'. */
  id?: string;
  /** Chunked weights W. Neighbor IDs outside [0, columnCount) are dropped. */
  weights: GPUChunkedSpatialWeights;
  /** Number of columns of W. Defaults to the total input row count. */
  columnCount?: number;
  /** Packed output (capacity >= summed input capacities), or independently sized row chunks. */
  output: Output;
  /** Per-frame absolute symmetry tolerance: numeric default or one-float parameter view. Defaults to 0. */
  symmetryTolerance?: number | GraphDataView<'float32'>;
};

/**
 * Transposes chunked spatial weights into packed or chunked CSR.
 *
 * Chunked output follows GPUChunkedSpatialWeights: consecutive row ranges, local offsets and
 * global neighbor IDs. Output boundaries need not match input boundaries. For each output range,
 * the shared transpose pipeline stable-sorts each input chunk, counts and scans only that range,
 * then scatters in input row order. No scratch buffer holds the summed input slots or all output
 * rows. Work and node count grow with inputChunks * outputChunks; input sorts are repeated for
 * each output range. estimatedCommandNodeCount bounds this work and enforces the shared budget.
 *
 * Each output chunk retains the prefix of its complete CSR that fits its own capacity. Offsets
 * and status.count are clamped, status.requiredCount is exact and status.overflow reports dropped
 * slots. Unused slots (including distances) are zeroed. Output distances require distances on
 * every input chunk. The optional symmetry count describes W, independently of output overflow;
 * it uses chunk-local reductions and input-chunk pairs, without packing the full matrix.
 *
 * Compile-time: view shapes, input/output chunk ranges, columnCount and optional outputs.
 * Per-frame: input CSR contents and symmetryTolerance, packed with
 * getGPUSpatialWeightsTransposeParameterValues (numeric defaults use the same GPUParameterBuffer layout). The output preserves spatial context and
 * input status at matrix level; each chunk has its own output capacity status. Row, variance and
 * double transforms become unknown (chunk capacity can break global normalization). Chunked output symmetry metadata is unknown because it can truncate.
 */
export class GPUChunkedSpatialWeightsTranspose<
  Output extends TransposeOutput = GPUSpatialWeightsTransposeOutput
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPUChunkedSpatialWeightsTransposeProps<Output>;
  readonly rowCount: number;
  readonly columnCount: number;
  private readonly toleranceValues: Float32Array | GraphDataView<'float32'>;
  readonly estimatedCommandNodeCount: number;
  /** Stable description whose CSR views follow the caller's output, including corrected views on retry. */
  readonly outputWeights: DescribedOutput<Output>;

  constructor(props: GPUChunkedSpatialWeightsTransposeProps<Output>) {
    this.id = props.id ?? 'chunked-spatial-weights-transpose';
    this.props = props;
    const tolerance = props.symmetryTolerance ?? 0;
    if (typeof tolerance === 'number') {
      validateFiniteNumber(this.id, 'symmetryTolerance', tolerance, 0);
      this.toleranceValues = getGPUSpatialWeightsTransposeParameterValues({
        symmetryTolerance: tolerance
      });
    } else {
      validateSinglePackedView(tolerance, ['float32'], `${this.id} symmetryTolerance`);
      validateCapacity(
        this.id,
        'symmetryTolerance.length',
        tolerance.length,
        GPU_SPATIAL_WEIGHTS_TRANSPOSE_PARAMETER_LENGTH
      );
      this.toleranceValues = tolerance;
    }
    const {id} = this;
    const {weights, output} = props;
    this.rowCount = validateGPUChunkedSpatialWeights(id, weights);
    this.columnCount = props.columnCount ?? this.rowCount;
    validateCapacity(id, 'rowCount', this.rowCount);
    validateCapacity(id, 'columnCount', this.columnCount, 1, MAXIMUM_SLOT_COUNT - 1);
    const capacity = weights.chunks.reduce((sum, chunk) => sum + chunk.weights.neighbors.length, 0);
    validateCapacity(
      id,
      'summed input capacity',
      capacity,
      1,
      output.asymmetricSlots ? Math.floor(MAXIMUM_SLOT_COUNT / 2) : MAXIMUM_SLOT_COUNT
    );
    for (const chunk of weights.chunks) {
      validateCapacity(
        id,
        'input chunk capacity',
        chunk.weights.neighbors.length,
        1,
        MAXIMUM_SORT_SLOT_COUNT
      );
    }
    if (output.asymmetricSlots) {
      validateSinglePackedView(output.asymmetricSlots, ['uint32'], `${id} output.asymmetricSlots`);
      validateRequirement(
        id,
        output.asymmetricSlots.length === 1,
        'output.asymmetricSlots must hold one uint32 row'
      );
      validateRequirement(
        id,
        this.rowCount === this.columnCount,
        'output.asymmetricSlots requires square weights'
      );
    }
    const inputChunkCount = weights.chunks.length;
    const outputChunkCount = 'chunks' in output ? output.chunks.length : 1;
    // Per output: clear, scan, offsets, status; per input: keys, sort, count, destinations,
    // scatter, fill. Symmetry includes chunk-pair checks, clears and hierarchical reductions.
    this.estimatedCommandNodeCount =
      outputChunkCount *
        (4 +
          SCAN_OR_REDUCTION_COMMAND_NODE_ALLOWANCE +
          inputChunkCount * (5 + SORT_COMMAND_NODE_ALLOWANCE)) +
      (output.asymmetricSlots
        ? inputChunkCount * inputChunkCount +
          inputChunkCount * (1 + SCAN_OR_REDUCTION_COMMAND_NODE_ALLOWANCE) +
          SCAN_OR_REDUCTION_COMMAND_NODE_ALLOWANCE
        : 0) +
      (output.asymmetricSlots && this.toleranceValues instanceof Float32Array ? 1 : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    if (!('chunks' in output)) {
      const description = planSpatialWeightsTranspose(id, OPERATION, {
        ...props,
        output
      }).outputWeights;
      this.outputWeights = this.describeOutputViews(
        description,
        () => props.output as GPUSpatialWeightsTransposeOutput,
        () => {
          const currentOutput = props.output as GPUSpatialWeightsTransposeOutput;
          return currentOutput.status || props.weights.status
            ? {...currentOutput.status, ...props.weights.status}
            : undefined;
        }
      ) as DescribedOutput<Output>;
      return;
    }
    validateCapacity(id, 'output chunk count', output.chunks.length);
    let nextRow = 0;
    for (const [chunkIndex, chunk] of output.chunks.entries()) {
      const name = `output.chunks[${chunkIndex}]`;
      validateRequirement(id, chunk.rowStart === nextRow, `${name}.rowStart must equal ${nextRow}`);
      validateCapacity(id, `${name}.rowCount`, chunk.rowCount, 1, this.columnCount - nextRow);
      validateRequirement(
        id,
        validateGPUSpatialWeights(id, chunk.weights, name) === chunk.rowCount,
        `${name}.offsets must hold rowCount + 1 entries`
      );
      nextRow += chunk.rowCount;
      for (const field of ['count', 'requiredCount', 'overflow'] as const) {
        const view = chunk.weights.status?.[field];
        validateRequirement(id, Boolean(view), `${name}.status.${field} is required`);
        validateSinglePackedView(view!, ['uint32'], `${id} ${name}.status.${field}`);
        validateRequirement(
          id,
          view!.length === 1,
          `${name}.status.${field} must hold one uint32 row`
        );
      }
      validateRequirement(
        id,
        !chunk.weights.distances || weights.chunks.every(input => Boolean(input.weights.distances)),
        `${name}.distances requires distances on every input chunk`
      );
    }
    validateRequirement(
      id,
      nextRow === this.columnCount,
      'output chunks must cover columnCount rows'
    );
    const outputViews = this.getChunkedOutputViews(output);
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      ...getGPUSpatialWeightsOrChunksViews(weights),
      typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined,
      ...Object.values(weights.status ?? {}),
      ...weights.chunks.flatMap(chunk => Object.values(chunk.weights.status ?? {}))
    ]);
    // Separate nodes write chunks and statuses, so check output-output aliases at construction too.
    validateRequirement(
      id,
      new Set(outputViews.filter(view => view).map(view => view!.buffer)).size ===
        outputViews.filter(view => view).length,
      'output views must use separate buffers'
    );
    const inputTransform = weights.information?.transform ?? 'unknown';
    const information: NonNullable<GPUSpatialWeights['information']> = {
      rowCount: this.columnCount,
      transform:
        inputTransform === 'row' || inputTransform === 'variance' || inputTransform === 'double'
          ? 'unknown'
          : inputTransform,
      symmetric: 'unknown',
      hasSelfWeights: weights.information?.hasSelfWeights,
      provenance: {...weights.information?.provenance, source: OPERATION, criterion: 'transpose'}
    };
    const described: GPUChunkedSpatialWeights = {
      rowCount: this.columnCount,
      spatialContext: weights.spatialContext,
      information,
      status: weights.status,
      chunks: output.chunks.map((chunk, chunkIndex) => ({
        ...chunk,
        weights: this.describeOutputViews(
          {
            ...chunk.weights,
            spatialContext: weights.spatialContext,
            information: {...information, rowCount: chunk.rowCount}
          },
          () => (props.output as GPUChunkedSpatialWeightsTransposeOutput).chunks[chunkIndex].weights
        )
      }))
    };
    this.outputWeights = described as DescribedOutput<Output>;
  }

  /** Keep saved descriptions connected to same-shaped view corrections before node generation. */
  private describeOutputViews(
    description: GPUSpatialWeights,
    getOutput: () => GPUSpatialWeights,
    getStatus: () => GPUSpatialWeights['status'] = () => getOutput().status
  ): GPUSpatialWeights {
    return {
      ...description,
      get offsets() {
        return getOutput().offsets;
      },
      set offsets(view) {
        getOutput().offsets = view;
      },
      get neighbors() {
        return getOutput().neighbors;
      },
      set neighbors(view) {
        getOutput().neighbors = view;
      },
      get weights() {
        return getOutput().weights;
      },
      set weights(view) {
        getOutput().weights = view;
      },
      get distances() {
        return getOutput().distances;
      },
      set distances(view) {
        getOutput().distances = view;
      },
      get status() {
        return getStatus();
      },
      set status(status) {
        getOutput().status = status;
      }
    };
  }

  private getChunkedOutputViews(output: GPUChunkedSpatialWeightsTransposeOutput) {
    return [
      output.asymmetricSlots,
      ...output.chunks.flatMap(chunk => [
        ...getGPUSpatialWeightsViews(chunk.weights),
        ...Object.values(chunk.weights.status)
      ])
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, weights} = props;
    if (!('chunks' in output)) {
      return getSpatialWeightsTransposeNodes(
        graph,
        id,
        OPERATION,
        {...props, output},
        {
          rowCount: this.rowCount,
          columnCount: this.columnCount,
          outputWeights: this.outputWeights as GPUSpatialWeights
        },
        this.toleranceValues
      );
    }
    const views = [
      ...getGPUSpatialWeightsOrChunksViews(weights),
      ...this.getChunkedOutputViews(output),
      typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined
    ];
    validateGraphViewsBelongToGraph(id, graph, views);
    // Do all device-dependent checks before creating any node, including symmetry scratch.
    for (const view of views) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          'input/output chunk binding',
          getViewBindingRange(view).size,
          'use smaller input or output row chunks'
        );
      }
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'symmetry chunk totals',
      weights.chunks.length * 4,
      'use fewer input chunks'
    );
    const runtime = output.asymmetricSlots
      ? createSpatialStatisticsParameterNodes(
          graph,
          `${id}-symmetry-parameters`,
          this.toleranceValues
        )
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...(runtime?.nodes ?? []),
      ...output.chunks.flatMap((chunk, chunkIndex) =>
        getSpatialWeightsTransposeNodes(
          graph,
          `${id}-output${chunkIndex}`,
          OPERATION,
          {...props, output: chunk.weights},
          {rowCount: this.rowCount, columnCount: chunk.rowCount, outputWeights: chunk.weights},
          this.toleranceValues,
          {rowStart: chunk.rowStart, status: chunk.weights.status}
        )
      )
    ];
    if (output.asymmetricSlots) {
      nodes.push(
        ...getChunkedSpatialWeightsSymmetryNodes(
          graph,
          id,
          OPERATION,
          weights,
          output.asymmetricSlots,
          runtime!.parameters
        )
      );
    }
    return nodes;
  }
}
