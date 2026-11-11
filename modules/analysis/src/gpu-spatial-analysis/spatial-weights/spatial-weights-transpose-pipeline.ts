// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
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
  symmetryTolerance?: number;
};

/** Validated shape and described output of a transpose. @internal */
export type SpatialWeightsTransposePlan = {
  rowCount: number;
  columnCount: number;
  outputWeights: GPUSpatialWeights;
};

/** Validates a transpose request and describes its output; shared by whole and chunked weights. @internal */
export function planSpatialWeightsTranspose(
  id: string,
  operation: string,
  props: SpatialWeightsTransposeInputs
): SpatialWeightsTransposePlan {
  const {weights, output} = props;
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
  if (
    props.symmetryTolerance !== undefined &&
    !(Number.isFinite(props.symmetryTolerance) && props.symmetryTolerance >= 0)
  ) {
    throw new Error(`${id} symmetryTolerance must be finite and >= 0`);
  }
  if (output.distances) {
    validateSinglePackedView(output.distances, ['float32'], `${id} output.distances`);
  }
  const inputTransform = weights.information?.transform ?? 'unknown';
  const outputWeights = describeGPUSpatialWeights(output, {
    spatialContext: weights.spatialContext,
    information: {
      rowCount: columns,
      transform:
        inputTransform === 'row' || inputTransform === 'variance' ? 'unknown' : inputTransform,
      symmetric: rows === columns ? (weights.information?.symmetric ?? 'unknown') : 'unknown',
      hasSelfWeights: weights.information?.hasSelfWeights,
      provenance: {...weights.information?.provenance, source: operation, criterion: 'transpose'}
    },
    status: weights.status
  });
  validateGraphOutputsDisjointFromInputs(
    id,
    [output.offsets, output.neighbors, output.weights, output.distances, output.asymmetricSlots],
    getGPUSpatialWeightsOrChunksViews(weights)
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
  plan: SpatialWeightsTransposePlan
): readonly GPUCommandNode<Parameters>[] {
  const {weights, output} = props;
  const {rowCount: rows, columnCount: columns} = plan;
  validateGraphViewsBelongToGraph(id, graph, [
    ...getGPUSpatialWeightsOrChunksViews(weights),
    output.offsets,
    output.neighbors,
    output.weights,
    output.distances,
    output.asymmetricSlots
  ]);
  const dispatches = getGPUSpatialWeightsChunkDispatches(weights);
  const hasDistances =
    Boolean(output.distances) && dispatches.every(chunk => chunk.weights.distances);
  const NONE = '0xffffffffu';
  const columnCounts = createTransientView(graph, `${id}-column-counts`, 'uint32', columns);
  const columnStarts = createTransientView(graph, `${id}-column-starts`, 'uint32', columns);
  const columnFill = createTransientView(graph, `${id}-column-fill`, 'uint32', columns);
  const constants = `const COLUMNS: u32 = ${columns}u;`;
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
  let used = index < offsets[offsetsOffset + CHUNK_ROWS] && neighbor < COLUMNS;
  slotKeys[slotKeysOffset + index] = select(COLUMNS, neighbor, used);
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
    outputOffsets[outputOffsetsOffset + index] = columnStarts[columnStartsOffset + index];
  } else {
    outputOffsets[outputOffsetsOffset + COLUMNS] =
      columnStarts[columnStartsOffset + COLUMNS - 1u] + columnCounts[columnCountsOffset + COLUMNS - 1u];
  }`
    })
  );
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
          {name: 'outputOffsets', view: output.offsets, type: 'u32', access: 'read'},
          {name: 'columnFill', view: columnFill, type: 'u32', access: 'read'},
          {name: 'destinations', view: destinations, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: constants,
        body: `let key = sortedKeys[sortedKeysOffset + index];
  var destination = ${NONE};
  if (key < COLUMNS) {
    var low = 0u;
    var high = index;
    while (low < high) {
      let middle = (low + high) >> 1u;
      if (sortedKeys[sortedKeysOffset + middle] < key) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    destination = outputOffsets[outputOffsetsOffset + key] + columnFill[columnFillOffset + key] + (index - low);
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
// Row of a used slot: the last row whose offset is at most the slot.
fn findRow(slot: u32) -> u32 {
  var low = 0u;
  var high = CHUNK_ROWS;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] <= slot) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
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
            {name: 'rowMismatches', view: rowMismatches, type: 'u32', access: 'read_write'}
          ],
          invocationCount: dispatch.rowCount,
          declarations: `${constants}
const CHUNK_ROW_START: u32 = ${dispatch.rowStart}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.symmetryTolerance ?? 0)};
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
      if (abs(weights[weightsOffset + first] - transposeWeights[transposeWeightsOffset + second]) > TOLERANCE) {
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
    body: `var low = 0u;
  var high = ${capacity}u;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (sortedKeys[sortedKeysOffset + middle] < index) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  var upper = low;
  high = ${capacity}u;
  while (upper < high) {
    let middle = (upper + high) >> 1u;
    if (sortedKeys[sortedKeysOffset + middle] <= index) {
      upper = middle + 1u;
    } else {
      high = middle;
    }
  }
  columnTarget[columnTargetOffset + index] += upper - low;`
  });
}
