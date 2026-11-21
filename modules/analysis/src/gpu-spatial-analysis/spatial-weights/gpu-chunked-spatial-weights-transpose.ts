// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
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
  createWeightsLossNodes,
  getWeightsLossSources,
  validateWeightsLossSources,
  validateWeightsOverflowEvidence
} from './spatial-weights-loss';
import {snapshotWeightsFields, snapshotWeightsPort} from './weights-construction-snapshot';
import {validateWeightsLogicalAliases} from './spatial-weights-aliases';
import {
  getSpatialWeightsTransposeNodes,
  planSpatialWeightsTranspose,
  getSpatialWeightsTransposeInputStatus,
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
type TransposeAggregateStatus = {
  overflow: GraphDataView<'uint32'>;
  candidateOverflow: GraphDataView<'uint32'>;
};
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
  /** Chunked W. Root and chunk loss/certificates are aggregated on the output. Out-of-range neighbors are dropped. */
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
 * A requiredCount needs the producer's final overflow flag covering those slots. Verify
 * stopping evidence before supplying a complete CSR without those ports.
 * Per-frame: input CSR contents and symmetryTolerance, packed with
 * getGPUSpatialWeightsTransposeParameterValues (numeric defaults use the same GPUParameterBuffer layout). Packed output preserves spatial context and
 * input status except count and requiredCount at matrix level; read offsets[columnCount] for
 * the retained slot count. Chunked `outputWeights` aggregates loss at matrix level, as
 * chunked algebra does: `status.overflow` is one when any output chunk dropped slots and
 * `status.candidateOverflow` is one when the input reported `overflow` or `candidateOverflow`.
 * Its chunk descriptions keep `count` and `requiredCount` but carry no loss flags, so local
 * Moran, hot spot and the global statistics accept it. The caller's output chunk ports still
 * receive their own `overflow`; a supplied chunk `candidateOverflow` is reset to zero each
 * submission. Input `count` and `requiredCount` are omitted from the matrix description because
 * they do not describe the retained output. Row, variance and
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
  private readonly portLayout: string;
  private aggregateStatus?: TransposeAggregateStatus;

  constructor(props: GPUChunkedSpatialWeightsTransposeProps<Output>) {
    props = snapshotWeightsFields(props, [
      'id',
      'weights',
      'columnCount',
      'output',
      'symmetryTolerance'
    ]);
    this.id = props.id ?? 'chunked-spatial-weights-transpose';
    this.props = props;
    this.portLayout = this.getPortLayout(props);
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
    // Validate declared fields by name, including prototype-backed ports. Keep this.props'
    // live descriptors for the documented same-shaped correction path.
    const weights = snapshotWeightsPort(props.weights);
    const output = snapshotWeightsPort(props.output);
    this.rowCount = validateGPUChunkedSpatialWeights(id, weights);
    validateWeightsLossSources(id, [weights], true);
    validateWeightsOverflowEvidence(id, weights);
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
    const inheritedLossBatchCount = Math.max(
      1,
      Math.ceil(Object.values(getWeightsLossSources([weights], id, true)).flat().length / 7)
    );
    const hasChildLoss = Object.values(
      getWeightsLossSources(
        weights.chunks.map(chunk => chunk.weights),
        id,
        true
      )
    ).some(views => views.length > 0);
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
      (output.asymmetricSlots && this.toleranceValues instanceof Float32Array ? 1 : 0) +
      // Chunked output: OR of the chunk overflows (batches of seven) and of the inherited loss.
      ('chunks' in output
        ? Math.max(1, Math.ceil(outputChunkCount / 7)) +
          inheritedLossBatchCount +
          output.chunks.filter(chunk => chunk.weights.status?.candidateOverflow).length
        : hasChildLoss
          ? inheritedLossBatchCount
          : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    if (!('chunks' in output)) {
      validateRequirement(
        id,
        !output.status || Object.values(output.status).every(view => !view),
        'output.status is unsupported for packed transpose; input status is forwarded except count and requiredCount; read offsets[columnCount] for the retained count; omit output.status or use chunked output for destination diagnostics'
      );
      validateRequirement(
        id,
        !output.distances || weights.chunks.every(chunk => Boolean(chunk.weights.distances)),
        'output.distances requires distances on every input chunk; omit output.distances otherwise'
      );
      const description = planSpatialWeightsTranspose(id, OPERATION, {
        ...props,
        output
      }).outputWeights;
      this.outputWeights = this.describeOutputViews(
        description,
        () => props.output as GPUSpatialWeightsTransposeOutput,
        () => {
          const sources = getWeightsLossSources(
            props.weights.chunks.map(chunk => chunk.weights),
            id,
            true
          );
          const inherited = getSpatialWeightsTransposeInputStatus(props.weights.status);
          if (![...sources.overflow, ...sources.candidateOverflow, ...sources.complete].length)
            return inherited;
          const graph = (props.output as GPUSpatialWeightsTransposeOutput).offsets.buffer
            .graph as GPUCommandGraph;
          return {
            ...inherited,
            candidateOverflow: this.getAggregateStatus(graph).candidateOverflow
          };
        }
      ) as DescribedOutput<Output>;
      return;
    }
    validateCapacity(id, 'output chunk count', output.chunks.length);
    let nextRow = 0;
    for (const [chunkIndex, chunk] of output.chunks.entries()) {
      const name = `output.chunks[${chunkIndex}]`;
      for (const [field, view] of Object.entries(chunk.weights.status ?? {})) {
        validateRequirement(
          id,
          !view || ['count', 'requiredCount', 'overflow', 'candidateOverflow'].includes(field),
          `${name}.status.${field} is unsupported because transpose does not compute it; omit it and run the matching diagnostic contributor afterward`
        );
      }
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
      ...getGPUSpatialWeightsOrChunksViews(weights, true),
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
    validateWeightsLogicalAliases(
      id,
      Object.fromEntries(outputViews.map((view, index) => [`output ${index}`, view]))
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
    // Loss is aggregated on the outer port, as for chunked algebra, so the statistics that refuse
    // chunk-local loss flags accept this output: overflow when an output chunk dropped slots,
    // candidateOverflow when the input was already incomplete.
    const contributor = this;
    const getChunkOutput = (chunkIndex: number) =>
      (props.output as GPUChunkedSpatialWeightsTransposeOutput).chunks[chunkIndex].weights;
    const described: GPUChunkedSpatialWeights = {
      rowCount: this.columnCount,
      spatialContext: props.weights.spatialContext,
      information,
      get status() {
        const currentWeights = snapshotWeightsPort(props.weights);
        const currentOutput = snapshotWeightsPort(
          props.output as GPUChunkedSpatialWeightsTransposeOutput
        );
        const graph = currentOutput.chunks[0].weights.offsets.buffer.graph as GPUCommandGraph;
        validateGraphViewsBelongToGraph(
          id,
          graph,
          contributor.getWiringViews(currentWeights, currentOutput)
        );
        return {
          ...getSpatialWeightsTransposeInputStatus(currentWeights.status),
          ...contributor.getAggregateStatus(graph)
        };
      },
      chunks: output.chunks.map((chunk, chunkIndex) => ({
        ...chunk,
        weights: this.describeOutputViews(
          {
            ...chunk.weights,
            spatialContext: props.weights.spatialContext,
            information: {...information, rowCount: chunk.rowCount}
          },
          () => getChunkOutput(chunkIndex),
          () => ({
            ...snapshotWeightsPort(getChunkOutput(chunkIndex)).status,
            overflow: undefined,
            candidateOverflow: undefined
          })
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

  private getWiringViews(
    weights: GPUChunkedSpatialWeights,
    output: GPUChunkedSpatialWeightsTransposeOutput
  ): (GraphDataView | undefined)[] {
    return [
      ...getGPUSpatialWeightsOrChunksViews(weights, true),
      ...Object.values(weights.status ?? {}),
      ...weights.chunks.flatMap(chunk => Object.values(chunk.weights.status ?? {})),
      ...this.getChunkedOutputViews(output),
      typeof this.props.symmetryTolerance === 'object' ? this.props.symmetryTolerance : undefined
    ];
  }

  /** Publish on validated current wiring; temporary retry validators allocate no graph resources. */
  private getAggregateStatus<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): TransposeAggregateStatus {
    if (!this.aggregateStatus) {
      const output = this.props.output;
      const offsets = 'chunks' in output ? output.chunks[0].weights.offsets : output.offsets;
      const statusId = `${this.id}-${offsets.buffer.id}-${offsets.byteOffset}`;
      this.aggregateStatus = {
        overflow: createTransientView(graph, `${statusId}-overflow`, 'uint32', 1),
        candidateOverflow: createTransientView(graph, `${statusId}-candidate-overflow`, 'uint32', 1)
      };
    }
    validateGraphViewsBelongToGraph(this.id, graph, Object.values(this.aggregateStatus));
    return this.aggregateStatus;
  }

  /** Corrections may replace views, but cannot change the declared ports or row layout. */
  private getPortLayout(props: GPUChunkedSpatialWeightsTransposeProps<Output>): string {
    const describe = (port: object): unknown => {
      const captured = snapshotWeightsPort(port) as GPUSpatialWeightsTransposeOutput &
        Partial<GPUChunkedSpatialWeights>;
      return {
        rowCount: captured.rowCount,
        views: [
          ...getGPUSpatialWeightsViews(captured),
          captured.asymmetricSlots,
          ...[
            'count',
            'requiredCount',
            'overflow',
            'candidateOverflow',
            'invalidCount',
            'uncertainCount',
            'unresolvedCount',
            'approximation',
            'iterationCount',
            'iterationLimitReached',
            'converged'
          ].map(field => captured.status?.[field as keyof NonNullable<GPUSpatialWeights['status']>])
        ].map(view =>
          view ? [view.format, view.length, view.byteStride, view.rowByteLength] : null
        ),
        chunks: captured.chunks?.map(chunk => [
          chunk.rowStart,
          chunk.rowCount,
          describe(chunk.weights)
        ])
      };
    };
    return JSON.stringify([describe(props.weights), describe(props.output)]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id} = this;
    // The supported retry path changes opaque views. Re-run wiring validation on one captured
    // correction before any scratch allocation, so aliases and missing optional ports cannot slip in.
    const props = Object.freeze({
      ...this.props,
      weights: snapshotWeightsPort(this.props.weights),
      output: snapshotWeightsPort(this.props.output)
    });
    new GPUChunkedSpatialWeightsTranspose<Output>(props);
    validateRequirement(
      id,
      this.getPortLayout(props) === this.portLayout,
      'corrected input/output ports and shapes must match construction; restore the declared views or rebuild the transpose'
    );
    const {output, weights} = props;
    if (!('chunks' in output)) {
      const nodes = getSpatialWeightsTransposeNodes(
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
      const childSources = getWeightsLossSources(
        weights.chunks.map(chunk => chunk.weights),
        id,
        true
      );
      if (
        ![...childSources.overflow, ...childSources.candidateOverflow, ...childSources.complete]
          .length
      )
        return nodes;
      const sources = getWeightsLossSources([weights], id, true);
      return [
        ...nodes,
        ...createWeightsLossNodes(
          graph,
          `${id}-packed-inherited-loss`,
          [...sources.overflow, ...sources.candidateOverflow],
          this.getAggregateStatus(graph).candidateOverflow,
          false,
          sources.complete
        )
      ];
    }
    const views = this.getWiringViews(weights, output);
    const aggregateViews = Object.values(this.aggregateStatus ?? {});
    validateGraphViewsBelongToGraph(id, graph, [...views, ...aggregateViews]);
    validateGraphOutputsDisjointFromInputs(`${id} aggregate status`, aggregateViews, views);
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
    const status = this.getAggregateStatus(graph);
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
        chunk.weights.status.candidateOverflow
          ? createWeightsLossNodes(
              graph,
              `${id}-output${chunkIndex}-candidate-clear`,
              [],
              chunk.weights.status.candidateOverflow
            )
          : []
      ),
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
    const outputLoss = getWeightsLossSources(output.chunks.map(chunk => chunk.weights));
    const inputLoss = getWeightsLossSources([weights], id, true);
    nodes.push(
      ...createWeightsLossNodes(
        graph,
        `${id}-aggregate-overflow`,
        outputLoss.overflow,
        status.overflow!
      ),
      ...createWeightsLossNodes(
        graph,
        `${id}-aggregate-candidate-overflow`,
        [...inputLoss.overflow, ...inputLoss.candidateOverflow],
        status.candidateOverflow!,
        false,
        inputLoss.complete
      )
    );
    return nodes;
  }
}
