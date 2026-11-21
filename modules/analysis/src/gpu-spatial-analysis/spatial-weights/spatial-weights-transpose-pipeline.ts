// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSpatialStatisticsParameterNodes} from '../../analysis-core/kernels/spatial-statistics-parameter-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {validateSpatialStatisticsWritableDispatches} from './spatial-statistics-writable-dispatches';
import {
  getGPUSpatialWeightsChunkDispatches,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from './gpu-chunked-spatial-weights';
import {
  describeGPUSpatialWeights,
  validateGPUSpatialWeights,
  type GPUSpatialWeights
} from './spatial-weights';
import {snapshotWeightsPort} from './weights-construction-snapshot';

/** Caller-owned output of {@link GPUSpatialWeightsTranspose}: the CSR of `W'`. */
export type GPUSpatialWeightsTransposeOutput = GPUSpatialWeights & {
  /**
   * Optional one-uint32 output: the number of positions `(i, j)` of the union of the patterns of
   * `W` and `W'` whose weights differ by more than `symmetryTolerance` or that are present in one
   * only. Zero means `W = W'`. Requires square weights.
   */
  asymmetricSlots?: GraphDataView<'uint32'>;
};

/** Inputs shared by {@link GPUSpatialWeightsTranspose} and {@link GPUChunkedSpatialWeightsTranspose}. @internal */
export type SpatialWeightsTransposeInputs = {
  /** Whole weights (one chunk starting at row 0) or chunked weights. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  columnCount?: number;
  output: GPUSpatialWeightsTransposeOutput;
  symmetryTolerance?: number | GraphDataView<'float32'>;
};

/** Validated shape and described output of a transpose. @internal */
export type SpatialWeightsTransposePlan = {
  rowCount: number;
  columnCount: number;
  outputWeights: GPUSpatialWeights;
};

/** Input loss is inherited; input cardinalities do not describe retained output columns. @internal */
export function getSpatialWeightsTransposeInputStatus(
  status: GPUSpatialWeights['status']
): GPUSpatialWeights['status'] {
  return status
    ? Object.fromEntries(
        Object.entries(snapshotWeightsPort({status}).status!).filter(
          ([field]) => field !== 'count' && field !== 'requiredCount'
        )
      )
    : undefined;
}

/** Validates a transpose request and describes its output; shared by whole and chunked weights. @internal */
export function planSpatialWeightsTranspose(
  id: string,
  operation: string,
  props: SpatialWeightsTransposeInputs
): SpatialWeightsTransposePlan {
  const {weights, output} = props;
  validateRequirement(
    id,
    !output.status || Object.values(output.status).every(view => !view),
    'output.status is unsupported for packed transpose because destination diagnostics are not written; omit output.status or use chunked output'
  );
  const rows = validateGPUSpatialWeightsOrChunks(id, weights);
  const dispatches = getGPUSpatialWeightsChunkDispatches(weights);
  const columns = props.columnCount ?? rows;
  if (!Number.isInteger(columns) || columns < 1) {
    throw new Error(`${id} columnCount must be a positive integer`);
  }
  const outputRows = validateGPUSpatialWeights(id, output, 'output');
  if (outputRows !== columns) {
    throw new Error(`${id} output.offsets must hold columnCount + 1 entries`);
  }
  const capacity = dispatches.reduce((sum, chunk) => sum + chunk.weights.neighbors.length, 0);
  if (output.neighbors.length < capacity) {
    throw new Error(`${id} output capacity must be >= the input capacity`);
  }
  // Chunked weights may omit output distances; whole weights always carry them along.
  if (dispatches[0].isWhole && dispatches[0].weights.distances && !output.distances) {
    throw new Error(`${id} output.distances is required with weights.distances`);
  }
  if (output.asymmetricSlots) {
    validateSinglePackedView(output.asymmetricSlots, ['uint32'], `${id} asymmetricSlots`);
    if (output.asymmetricSlots.length !== 1) {
      throw new Error(`${id} asymmetricSlots must hold one uint32 row`);
    }
    if (rows !== columns) {
      throw new Error(`${id} asymmetricSlots requires square weights`);
    }
  }
  if (output.distances) {
    validateRequirement(
      id,
      dispatches.every(dispatch => Boolean(dispatch.weights.distances)),
      'output.distances requires distances on every input chunk; omit output.distances otherwise'
    );
    validateSinglePackedView(output.distances, ['float32'], `${id} output.distances`);
  }
  const inputTransform = weights.information?.transform ?? 'unknown';
  // Double normalization survives only when no slot can be dropped. The handle declares no source
  // column bound, so `columnCount === rows` alone does not establish it (neighbors [2, 0] with
  // two columns lose half the mass). A declared symmetric W lists only neighbors in [0, rows), so a
  // square transpose of it keeps every slot (W' = W).
  const keepsEverySlot = columns === rows && weights.information?.symmetric === true;
  const outputWeights = describeGPUSpatialWeights(output, {
    spatialContext: weights.spatialContext,
    information: {
      rowCount: columns,
      transform:
        inputTransform === 'row' ||
        inputTransform === 'variance' ||
        (inputTransform === 'double' && !keepsEverySlot)
          ? 'unknown'
          : inputTransform,
      symmetric: rows === columns ? (weights.information?.symmetric ?? 'unknown') : 'unknown',
      hasSelfWeights: weights.information?.hasSelfWeights,
      provenance: {...weights.information?.provenance, source: operation, criterion: 'transpose'}
    },
    status: getSpatialWeightsTransposeInputStatus(weights.status)
  });
  validateGraphOutputsDisjointFromInputs(
    id,
    [output.offsets, output.neighbors, output.weights, output.distances, output.asymmetricSlots],
    [
      typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined,
      ...getGPUSpatialWeightsOrChunksViews(weights, true),
      ...Object.values(weights.status ?? {}),
      ...dispatches.flatMap(dispatch => Object.values(dispatch.weights.status ?? {}))
    ]
  );
  validateSpatialStatisticsWritableDispatches(
    id,
    {
      offsets: output.offsets,
      neighbors: output.neighbors,
      weights: output.weights,
      distances: output.distances,
      asymmetricSlots: output.asymmetricSlots
    },
    [['neighbors', 'weights', 'distances']]
  );
  return {rowCount: rows, columnCount: columns, outputWeights};
}

/**
 * The one transpose algorithm for whole and chunked weights (whole weights are one chunk).
 *
 * Linear in the chunk count: per chunk one stable sort of its slots by column and one pass that
 * adds the chunk's column counts; one scan over the summed counts gives the output offsets; then
 * per chunk one destination pass (`offsets[column] + slots of earlier chunks in that column + rank
 * in this chunk`) and one scatter, and a cursor update. Chunks run in source order and the sort is
 * stable, so every output row lists source rows ascending and the result does not depend on the
 * chunking. Tail slots past the used count are written as zero. @internal
 */
export function getSpatialWeightsTransposeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  operation: string,
  props: SpatialWeightsTransposeInputs,
  plan: SpatialWeightsTransposePlan,
  toleranceValues: Float32Array | GraphDataView<'float32'>,
  // A bounded output row range, with offsets local to that chunk. Omitted for packed output.
  range?: {rowStart: number; status: NonNullable<GPUSpatialWeights['status']>}
): readonly GPUCommandNode<Parameters>[] {
  const {weights, output} = props;
  const {rowCount: rows, columnCount: columns} = plan;
  validateRequirement(
    id,
    !output.distances ||
      getGPUSpatialWeightsChunkDispatches(weights).every(chunk => Boolean(chunk.weights.distances)),
    'output.distances requires distances on every input chunk; restore the source distances or rebuild without the output port'
  );
  const statusViews = range ? Object.values(range.status) : [];
  // Each input sort, output scan and optional symmetry reduction stays within one binding.
  const bindingViews = [
    ...getGPUSpatialWeightsOrChunksViews(weights, true),
    output.offsets,
    output.neighbors,
    output.weights,
    output.distances,
    output.asymmetricSlots,
    ...statusViews,
    typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined
  ];
  for (const view of bindingViews) {
    if (view) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'weights or output binding',
        getViewBindingRange(view).size,
        'use smaller input or output row chunks'
      );
    }
  }
  if (output.asymmetricSlots) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'symmetry rows',
      rows * 4,
      'use chunked output for a chunked symmetry reduction'
    );
  }
  validateGraphViewsBelongToGraph(id, graph, [
    ...getGPUSpatialWeightsOrChunksViews(weights, true),
    output.offsets,
    output.neighbors,
    output.weights,
    output.distances,
    output.asymmetricSlots,
    ...statusViews,
    typeof props.symmetryTolerance === 'object' ? props.symmetryTolerance : undefined
  ]);
  const dispatches = getGPUSpatialWeightsChunkDispatches(weights);
  const hasDistances =
    Boolean(output.distances) && dispatches.every(chunk => chunk.weights.distances);
  const NONE = '0xffffffffu';
  const columnCounts = createTransientView(graph, `${id}-column-counts`, 'uint32', columns);
  const columnStarts = createTransientView(graph, `${id}-column-starts`, 'uint32', columns);
  const columnFill = createTransientView(graph, `${id}-column-fill`, 'uint32', columns);
  const constants = `const COLUMNS: u32 = ${columns}u;
const COLUMN_START: u32 = ${range?.rowStart ?? 0}u;
const OUTPUT_CAPACITY: u32 = ${output.neighbors.length}u;`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-clear-columns`,
      operation,
      variant: 'clear-columns',
      bindings: [
        {name: 'columnCounts', view: columnCounts, type: 'u32', access: 'read_write'},
        {name: 'columnFill', view: columnFill, type: 'u32', access: 'read_write'}
      ],
      invocationCount: columns,
      body: `columnCounts[columnCountsOffset + index] = 0u;
  columnFill[columnFillOffset + index] = 0u;`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-clear-output`,
      operation,
      variant: hasDistances ? 'clear-output-distances' : 'clear-output',
      bindings: [
        {name: 'outputNeighbors', view: output.neighbors, type: 'u32', access: 'read_write'},
        {name: 'outputWeights', view: output.weights, type: 'f32', access: 'read_write'},
        ...(hasDistances
          ? [
              {
                name: 'outputDistances',
                view: output.distances!,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: output.neighbors.length,
      body: `outputNeighbors[outputNeighborsOffset + index] = 0u;
  outputWeights[outputWeightsOffset + index] = 0.0;
  ${hasDistances ? 'outputDistances[outputDistancesOffset + index] = 0.0;' : ''}`
    })
  ];
  const grouped = dispatches.map(dispatch => {
    const chunkId = `${id}${dispatch.idSuffix}`;
    const chunkWeights = dispatch.weights;
    const capacity = chunkWeights.neighbors.length;
    const slotKeys = createTransientView(graph, `${chunkId}-slot-keys`, 'uint32', capacity);
    const slotIndices = createTransientView(graph, `${chunkId}-slot-indices`, 'uint32', capacity);
    const sortedKeys = createTransientView(graph, `${chunkId}-sorted-keys`, 'uint32', capacity);
    const sortedSlots = createTransientView(graph, `${chunkId}-sorted-slots`, 'uint32', capacity);
    const chunkConstants = `${constants}
const CHUNK_ROWS: u32 = ${dispatch.rowCount}u;`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${chunkId}-slot-keys`,
        operation,
        variant: 'slot-keys',
        bindings: [
          {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: chunkWeights.neighbors, type: 'u32', access: 'read'},
          {name: 'slotKeys', view: slotKeys, type: 'u32', access: 'read_write'},
          {name: 'slotIndices', view: slotIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: chunkConstants,
        // Unused and out-of-range slots share the key COLUMNS, which sorts after every column.
        body: `let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + CHUNK_ROWS] && neighbor >= COLUMN_START && neighbor - COLUMN_START < COLUMNS;
  slotKeys[slotKeysOffset + index] = select(COLUMNS, neighbor - COLUMN_START, used);
  slotIndices[slotIndicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${chunkId}-sort`,
        keys: slotKeys,
        values: slotIndices,
        outputKeys: sortedKeys,
        outputValues: sortedSlots,
        keyBits: getSortKeyBits(columns)
      }).getCommandNodes(graph),
      createColumnCountNode(graph, operation, chunkId, 'count', sortedKeys, capacity, columnCounts)
    );
    return {dispatch, chunkId, capacity, sortedKeys, sortedSlots};
  });
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan`,
      input: columnCounts,
      output: columnStarts,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'offsets',
      bindings: [
        {name: 'columnStarts', view: columnStarts, type: 'u32', access: 'read'},
        {name: 'columnCounts', view: columnCounts, type: 'u32', access: 'read'},
        {name: 'outputOffsets', view: output.offsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: columns + 1,
      declarations: constants,
      body: `if (index < COLUMNS) {
    outputOffsets[outputOffsetsOffset + index] = min(columnStarts[columnStartsOffset + index], OUTPUT_CAPACITY);
  } else {
    outputOffsets[outputOffsetsOffset + COLUMNS] =
      min(columnStarts[columnStartsOffset + COLUMNS - 1u] + columnCounts[columnCountsOffset + COLUMNS - 1u], OUTPUT_CAPACITY);
  }`
    })
  );
  if (range) {
    const statusBindings = Object.entries(range.status).filter(
      ([name]) => name === 'count' || name === 'requiredCount' || name === 'overflow'
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-status`,
        operation,
        variant: 'status',
        bindings: [
          {name: 'columnStarts', view: columnStarts, type: 'u32', access: 'read'},
          {name: 'columnCounts', view: columnCounts, type: 'u32', access: 'read'},
          ...statusBindings.map(([name, view]) => ({
            name,
            view: view!,
            type: 'u32' as const,
            access: 'read_write' as const
          }))
        ],
        invocationCount: 1,
        declarations: constants,
        body: `let required = columnStarts[columnStartsOffset + COLUMNS - 1u] + columnCounts[columnCountsOffset + COLUMNS - 1u];
  ${range.status.count ? 'count[countOffset] = min(required, OUTPUT_CAPACITY);' : ''}
  ${range.status.requiredCount ? 'requiredCount[requiredCountOffset] = required;' : ''}
  ${range.status.overflow ? 'overflow[overflowOffset] = u32(required > OUTPUT_CAPACITY);' : ''}`
      })
    );
  }
  for (const {dispatch, chunkId, capacity, sortedKeys, sortedSlots} of grouped) {
    const chunkWeights = dispatch.weights;
    const destinations = createTransientView(graph, `${chunkId}-destinations`, 'uint32', capacity);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${chunkId}-destinations`,
        operation,
        variant: 'destinations',
        bindings: [
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'columnStarts', view: columnStarts, type: 'u32', access: 'read'},
          {name: 'columnFill', view: columnFill, type: 'u32', access: 'read'},
          {name: 'destinations', view: destinations, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `${constants}
${getBinarySearchWGSL('sortedKeys')}`,
        body: `let key = sortedKeys[sortedKeysOffset + index];
  var destination = ${NONE};
  if (key < COLUMNS) {
    let low = lowerBoundSortedKeys(0u, index, key);
    let candidate = columnStarts[columnStartsOffset + key] + columnFill[columnFillOffset + key] + (index - low);
    if (candidate < OUTPUT_CAPACITY) {
      destination = candidate;
    }
  }
  destinations[destinationsOffset + index] = destination;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${chunkId}-scatter`,
        operation,
        variant: hasDistances ? 'scatter-distances' : 'scatter',
        bindings: [
          {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
          {name: 'weights', view: chunkWeights.weights, type: 'f32', access: 'read'},
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'destinations', view: destinations, type: 'u32', access: 'read'},
          {name: 'outputNeighbors', view: output.neighbors, type: 'u32', access: 'read_write'},
          {name: 'outputWeights', view: output.weights, type: 'f32', access: 'read_write'},
          ...(hasDistances
            ? [
                {
                  name: 'distances',
                  view: chunkWeights.distances!,
                  type: 'f32' as const,
                  access: 'read' as const
                },
                {
                  name: 'outputDistances',
                  view: output.distances!,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: capacity,
        declarations: `const CHUNK_ROWS: u32 = ${dispatch.rowCount}u;
const CHUNK_ROW_START: u32 = ${dispatch.rowStart}u;
${getBinarySearchWGSL('offsets')}
fn findRow(slot: u32) -> u32 {
  return upperBoundOffsets(0u, CHUNK_ROWS + 1u, slot) - 1u;
}`,
        body: `let destination = destinations[destinationsOffset + index];
  if (destination != ${NONE}) {
    let slot = sortedSlots[sortedSlotsOffset + index];
    outputNeighbors[outputNeighborsOffset + destination] = findRow(slot) + CHUNK_ROW_START;
    outputWeights[outputWeightsOffset + destination] = weights[weightsOffset + slot];
    ${hasDistances ? 'outputDistances[outputDistancesOffset + destination] = distances[distancesOffset + slot];' : ''}
  }`
      }),
      createColumnCountNode(graph, operation, chunkId, 'fill', sortedKeys, capacity, columnFill)
    );
  }
  if (output.asymmetricSlots) {
    const runtime = createSpatialStatisticsParameterNodes(
      graph,
      `${id}-symmetry-parameters`,
      toleranceValues
    );
    nodes.push(...runtime.nodes);
    const rowMismatches = createTransientView(graph, `${id}-row-mismatches`, 'uint32', rows);
    for (const dispatch of dispatches) {
      const chunkWeights = dispatch.weights;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-symmetry-rows${dispatch.idSuffix}`,
          operation,
          variant: 'symmetry-rows',
          bindings: [
            {name: 'offsets', view: chunkWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: chunkWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: chunkWeights.weights, type: 'f32', access: 'read'},
            {name: 'transposeOffsets', view: output.offsets, type: 'u32', access: 'read'},
            {name: 'transposeNeighbors', view: output.neighbors, type: 'u32', access: 'read'},
            {name: 'transposeWeights', view: output.weights, type: 'f32', access: 'read'},
            {name: 'rowMismatches', view: rowMismatches, type: 'u32', access: 'read_write'},
            {name: 'tolerance', view: runtime.parameters, type: 'f32', access: 'read'}
          ],
          invocationCount: dispatch.rowCount,
          declarations: `${constants}
const CHUNK_ROW_START: u32 = ${dispatch.rowStart}u;
const NONE: u32 = 0xffffffffu;`,
          // Out-of-range neighbors of W are skipped; the transpose never holds them.
          body: `let row = index + CHUNK_ROW_START;
  var first = offsets[offsetsOffset + index];
  let firstEnd = offsets[offsetsOffset + index + 1u];
  var second = transposeOffsets[transposeOffsetsOffset + row];
  let secondEnd = transposeOffsets[transposeOffsetsOffset + row + 1u];
  var mismatches = 0u;
  loop {
    while (first < firstEnd && neighbors[neighborsOffset + first] >= COLUMNS) {
      first++;
    }
    if (first >= firstEnd && second >= secondEnd) {
      break;
    }
    let firstColumn = select(NONE, neighbors[neighborsOffset + first], first < firstEnd);
    let secondColumn = select(
      NONE,
      transposeNeighbors[transposeNeighborsOffset + second],
      second < secondEnd
    );
    if (firstColumn == secondColumn) {
      if (abs(weights[weightsOffset + first] - transposeWeights[transposeWeightsOffset + second]) > tolerance[toleranceOffset]) {
        mismatches++;
      }
      first++;
      second++;
    } else if (firstColumn < secondColumn) {
      mismatches++;
      first++;
    } else {
      mismatches++;
      second++;
    }
  }
  rowMismatches[rowMismatchesOffset + row] = mismatches;`
        })
      );
    }
    nodes.push(
      ...new GPUReduction({
        id: `${id}-symmetry-total`,
        input: rowMismatches,
        output: output.asymmetricSlots,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  }
  return nodes;
}

/** Adds, per column, the number of slots of one chunk's sorted keys equal to the column. */
function createColumnCountNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  operation: string,
  chunkId: string,
  name: string,
  sortedKeys: GraphDataView<'uint32'>,
  capacity: number,
  target: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${chunkId}-${name}-columns`,
    operation,
    variant: 'column-counts',
    bindings: [
      {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
      {name: 'columnTarget', view: target, type: 'u32', access: 'read_write'}
    ],
    invocationCount: target.length,
    declarations: getBinarySearchWGSL('sortedKeys'),
    body: `let low = lowerBoundSortedKeys(0u, ${capacity}u, index);
  let upper = upperBoundSortedKeys(low, ${capacity}u, index);
  columnTarget[columnTargetOffset + index] += upper - low;`
  });
}
