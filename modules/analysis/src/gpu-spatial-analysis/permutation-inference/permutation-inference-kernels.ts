// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUReduction,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  getGPUSpatialWeightsNeighborLoopWGSL,
  isGPUSpatialWeightsRowStandardized,
  type GPUIslandPolicy,
  type GPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunks,
  getGPUSpatialWeightsOrChunksViews,
  validateGPUSpatialWeightsOrChunks,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {
  GPU_PERMUTATION_PARAMETER_LENGTH,
  MAXIMUM_PERMUTATION_COUNT
} from './permutation-parameters';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFloat32ValuesNodes,
  type GPUFloat32OrUint32Values,
  validateFloat32OrUint32Values
} from '../spatial-weights/float32-values';

import {getOrderedPairMassNodes} from './ordered-pair-mass';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
import {createSpatialStatisticsRowNode} from '../spatial-weights/spatial-statistics-row-dispatch';
import {
  getWeightsLossSources,
  validateWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {getPermutationWeightsLossChecks} from './permutation-weights-loss';

/** Rows reduced by one workgroup in the first level of the deterministic column sums. @internal */
export const BLOCK_ROWS = 4096;

/** uint32 row addresses leave one full reduction block for strided-loop increments and trailing counts. */
export const PERMUTATION_MAXIMUM_ROW_COUNT = 0xffffffff - BLOCK_ROWS;

/** `rowPositions` value of an excluded row. @internal */
export const PERMUTATION_INVALID_POSITION = 0xffffffff;

/**
 * Slots of the `totals` view produced by {@link getPermutationInputNodes}.
 *
 * @internal
 */
export const PERMUTATION_TOTAL = {
  count: 0,
  sumX: 1,
  sumY: 2,
  sumSquares: 3,
  sumSquaresY: 4,
  s0: 5,
  mean: 6,
  meanY: 7,
  /** Sum over analysis rows of the squared active row weight sum, `1^T W^T W 1` (Lee's L denominator). */
  s0Squared: 8,
  /** Scaled ordered off-diagonal raw-value mass for global General G. */
  offDiagonalMass: 9
} as const;

/** Inputs shared by both permutation tests. @internal */
export type PermutationInputProps = {
  id: string;
  operation: string;
  /** Whole weights, or row chunks that are each read by their own dispatch. */
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  values: GPUFloat32OrUint32Values;
  secondValues?: GPUFloat32OrUint32Values;
  mask?: GraphDataView<'uint32'>;
  parameters: GraphDataView<'uint32'>;
  maximumPermutations: number;
  analysisCount?: GraphDataView<'uint32'>;
  /** Island policy; defaults to `'exclude'`. */
  islands?: GPUIslandPolicy;
  /**
   * Whether `compactY` holds the mean-centered second values (default) or the raw ones (binary
   * join-count variables).
   */
  centerY?: boolean;
  /** Normalize both columns before moments and pair products; homogeneous global statistics only. */
  normalizeValues?: boolean;
  /** Accumulate the raw scaled ordered off-diagonal mass needed by global General G. */
  orderedPairMass?: boolean;
  /** Optional caller-owned one-row uint32 output: the number of islands among the included rows. */
  islandCount?: GraphDataView<'uint32'>;
};

/** Views produced by {@link getPermutationInputNodes}. @internal */
export type PermutationInputs<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /**
   * `rows + 1` entries: the position of each analysis row in the compacted order (ascending row
   * order), or `PERMUTATION_INVALID_POSITION`; entry `rows` holds the analysis count `n` as an
   * exact uint32.
   */
  rowPositions: GraphDataView<'uint32'>;
  /** Original float32 source values before normalization; rowPositions gates inclusion. */
  originalX: GraphDataView<'float32'>;
  /** Compacted `x` values (centered when `centerX`), `n` valid entries. */
  compactX: GraphDataView<'float32'>;
  /** Compacted `y` values, centered unless `centerY` is false (zero without `secondValues`). */
  compactY: GraphDataView<'float32'>;
  /** `PERMUTATION_TOTAL` slots. */
  totals: GraphDataView<'float32'>;
  /** One uint32: the number of islands among the included rows (the caller's view when given). */
  islandCount: GraphDataView<'uint32'>;
  /**
   * The weights the statistics must read, one per slot of `weights.neighbors`: the caller's
   * `weights.weights` itself, or, for row-standardized weights, a transient copy in which every
   * row is divided by the sum of its active weights (re-standardization over the analysis rows).
   * For chunked weights this is the first chunk's view; read {@link PermutationInputs.chunks}.
   */
  weights: GraphDataView<'float32'>;
  /** The effective weights of every row chunk (one entry for whole weights). */
  chunks: {
    rowStart: number;
    rowCount: number;
    weights: GPUSpatialWeights;
    effective: GraphDataView<'float32'>;
  }[];
};

/**
 * Validates the props shared by both permutation tests and returns the row count.
 *
 * @internal
 */
export function validatePermutationInputs(props: PermutationInputProps): number {
  const {id} = props;
  const rows = validateGPUSpatialWeightsOrChunks(id, props.weights);
  validateCapacity(id, 'weights row count', rows, 2, PERMUTATION_MAXIMUM_ROW_COUNT);
  if (validateFloat32OrUint32Values(props.values, `${id} values`) !== rows) {
    throw new Error(`${id} values length must equal the weights row count`);
  }
  if (
    props.secondValues &&
    validateFloat32OrUint32Values(props.secondValues, `${id} secondValues`) !== rows
  ) {
    throw new Error(`${id} secondValues length must equal the weights row count`);
  }
  if (props.mask) {
    validatePackedUint32View(props.mask, `${id} mask`);
    if (props.mask.length !== rows) {
      throw new Error(`${id} mask length must equal the weights row count`);
    }
  }
  if (props.analysisCount) {
    validatePackedUint32View(props.analysisCount, `${id} output.analysisCount`);
    validateCapacity(id, 'output.analysisCount length', props.analysisCount.length, 1, 1);
  }
  if (props.islandCount) {
    validatePackedUint32View(props.islandCount, `${id} output.islandCount`);
    validateCapacity(id, 'output.islandCount length', props.islandCount.length);
  }
  validateWeightsLossSources(id, [props.weights], true);
  for (const complete of getWeightsLossSources([props.weights], id, true).complete)
    validateCapacity(id, 'weights.status.complete length', complete.length, 1, 1);
  validatePackedUint32View(props.parameters, `${id} parameters`);
  validateCapacity(
    id,
    'parameters length',
    props.parameters.length,
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  validateCapacity(
    id,
    'maximumPermutations',
    props.maximumPermutations,
    1,
    MAXIMUM_PERMUTATION_COUNT
  );
  return rows;
}

/** Preflights the exact-rank preparation bindings and its split-column scratch. @internal */
export function validatePermutationInputBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PermutationInputProps,
  outputs: readonly (GraphDataView | undefined)[] = []
): void {
  validateGraphChunkGroupKernelSupport(graph, props.id, 8);
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(
      [
        ...getGPUSpatialWeightsOrChunksViews(props.weights, true),
        props.weights.status?.overflow,
        props.values,
        props.secondValues,
        props.mask,
        props.parameters,
        props.islandCount,
        props.analysisCount,
        ...outputs
      ].map((view, index) => [`binding ${index}`, view])
    )
  );

  validateGPUStorageBindingByteLength(
    graph.device,
    props.id,
    'row positions scratch',
    (props.values.length + 1) * 4,
    'reduce packed rows or request larger device bindings'
  );
}

/**
 * WGSL reading the per-frame permutation parameters. Requires a `parameters` binding of `u32`.
 *
 * - `readPermutationCount()` clamps the requested count to `[1, maximum]`; kept for contributors
 *   that clamp by design.
 * - `isPermutationCountValid()` is true when the requested count lies in `[1, maximum]`.
 * - `readValidPermutationCount()` is the requested count, or 0 when it is not valid. The
 *   permutation tests use it: they never clamp, they write NaN and report the problem.
 *
 * @internal
 */
export function getPermutationParameterWGSL(maximumPermutations: number): string {
  return /* wgsl */ `
const MAXIMUM_PERMUTATIONS: u32 = ${maximumPermutations}u;
fn readSeedKey() -> vec2<u32> {
  return vec2<u32>(parameters[parametersOffset], parameters[parametersOffset + 1u]);
}
fn readPermutationCount() -> u32 {
  return clamp(parameters[parametersOffset + 2u], 1u, MAXIMUM_PERMUTATIONS);
}
fn isPermutationCountValid() -> bool {
  let requested = parameters[parametersOffset + 2u];
  return requested >= 1u && requested <= MAXIMUM_PERMUTATIONS;
}
fn readValidPermutationCount() -> u32 {
  return select(0u, parameters[parametersOffset + 2u], isPermutationCountValid());
}
// Positive, finite float32 strictly below one, including a positive subnormal cutoff.
fn isPermutationSignificanceValid() -> bool {
  let bits = parameters[parametersOffset + 3u];
  return bits > 0u && bits < 0x3f800000u;
}
fn readSignificanceLevel() -> f32 {
  return bitcast<f32>(parameters[parametersOffset + 3u]);
}
`;
}

/** Float helpers (finiteness, quiet NaN, two-sided p) shared with the autocorrelation contributors. @internal */
export const PERMUTATION_FLOAT_WGSL = SPATIAL_AUTOCORRELATION_FLOAT_WGSL;

/**
 * Builds the shared front end.
 *
 * Definitions (one rule for both tests):
 * - A row is *included* when its mask is nonzero and its value(s) are finite.
 * - An *island* is an included row without an active slot to another included row (an active
 *   slot has a valid neighbor other than the row, a positive weight and an included neighbor; see
 *   `getGPUSpatialWeightsNeighborLoopWGSL`).
 * - The *analysis rows* are the included rows, minus the islands under `islands: 'exclude'`. They
 *   are compacted in ascending row order; `n` (exact uint32), the mean, the sums of squares and
 *   `S0` are over them. Active slots whose neighbor is not an analysis row are dropped; there is
 *   no cascade, so a row whose only neighbors were islands keeps its place in `n` with no active
 *   slot (it is not tested and contributes 0 to `S0`).
 * - When the weights are row standardized, each analysis row's active weights are re-standardized
 *   to sum to 1 (see the returned `weights`), so masked or missing neighbors drop out of the lag
 *   instead of shrinking it toward 0.
 * - If root or chunk `overflow` / `candidateOverflow` is nonzero, or supplied `complete` is zero,
 *   the weights are incomplete and no row is analyzed (all
 *   statistics are NaN).
 * - The mean uses a fixed-order f32 tree sum followed by a second pass on the deviations from that
 *   first mean, so the centered values sum to zero to f32 accuracy for data with a large offset.
 *
 * @internal
 */
export function getPermutationInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: PermutationInputProps & {centerX: boolean}
): PermutationInputs<Parameters> {
  validatePermutationInputBindings(graph, props);
  const {id, operation, weights, mask} = props;
  const valuesInput = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
  const secondInput = props.secondValues
    ? createFloat32ValuesNodes(graph, `${id}-second-values`, props.secondValues)
    : undefined;
  const values = valuesInput.view;
  const secondValues = secondInput?.view;
  const keepIslands = (props.islands ?? 'exclude') === 'keep';
  const standardize = isGPUSpatialWeightsRowStandardized(weights);
  const weightsOverflow = weights.status?.overflow;
  const rows = values.length;
  const blockCount = Math.ceil(rows / BLOCK_ROWS);
  const columnCount = 8;
  // A separate binding per column keeps preparation at one word per row, even at 30M rows.
  const columns = Array.from({length: columnCount}, (_, column) =>
    createTransientView(graph, `${id}-column-${column}`, 'float32', rows)
  );
  const columnBinding = (column: number): WGSLKernelBinding => ({
    name: `column${column}`,
    view: columns[column],
    type: 'f32',
    access: 'read_write'
  });
  const flags = createTransientView(graph, `${id}-flags`, 'uint32', rows);
  const analysisFlags = createTransientView(graph, `${id}-analysis-flags`, 'uint32', rows);
  const scanned = createTransientView(graph, `${id}-scanned`, 'uint32', rows);
  const rowPositions = createTransientView(graph, `${id}-row-positions`, 'uint32', rows + 1);
  const compactX = createTransientView(graph, `${id}-compact-x`, 'float32', rows);
  const compactY = createTransientView(graph, `${id}-compact-y`, 'float32', rows);
  const islandCount =
    props.islandCount ?? createTransientView(graph, `${id}-island-count`, 'uint32', 1);
  // Chunked weights run each weights-reading kernel once per chunk; whole weights keep one node.
  const weightChunks = getGPUSpatialWeightsChunks(weights);
  const isWholeWeights = weightChunks.length === 1 && weightChunks[0].rowCount === rows;
  const chunkWeights = weightChunks.map((chunk, chunkIndex) => ({
    ...chunk,
    effective: standardize
      ? createTransientView(
          graph,
          `${id}-scaled-weights${isWholeWeights ? '' : `-${chunkIndex}`}`,
          'float32',
          chunk.weights.neighbors.length
        )
      : chunk.weights.weights
  }));
  const effectiveWeights = chunkWeights[0].effective;
  const totalsA = createTransientView(graph, `${id}-totals-a`, 'float32', 2);
  const totalsB = createTransientView(graph, `${id}-totals-b`, 'float32', 2);
  const totalsC = createTransientView(graph, `${id}-totals-c`, 'float32', 4);
  const totals = createTransientView(graph, `${id}-totals`, 'float32', 10);
  const valueScales = props.normalizeValues
    ? createTransientView(graph, `${id}-value-scales`, 'float32', 2)
    : undefined;
  const constantsWGSL = `const ROWS: u32 = ${rows}u;
const INVALID_POSITION: u32 = ${PERMUTATION_INVALID_POSITION}u;
${PERMUTATION_FLOAT_WGSL}`;
  const getChunkKernelProps = (chunk: (typeof chunkWeights)[number], chunkIndex: number) => ({
    idSuffix: isWholeWeights ? '' : `-${chunkIndex}`,
    invocationCount: chunk.rowCount,
    declarations: constantsWGSL,
    rowStart: 'WEIGHTS_ROW_START',
    wrapBody: (body: string) => body
  });
  const getWeightBindings = (
    chunk: (typeof chunkWeights)[number],
    effective = false
  ): WGSLKernelBinding[] => [
    {name: 'offsets', view: chunk.weights.offsets, type: 'u32', access: 'read'},
    {name: 'neighbors', view: chunk.weights.neighbors, type: 'u32', access: 'read'},
    {
      name: 'weights',
      view: effective ? chunk.effective : chunk.weights.weights,
      type: 'f32',
      access: 'read'
    }
  ];
  const secondBinding: WGSLKernelBinding[] = secondValues
    ? [{name: 'secondValues', view: secondValues, type: 'f32', access: 'read'}]
    : [];
  const nodes: GPUCommandNode<Parameters>[] = [
    ...valuesInput.nodes,
    ...(secondInput?.nodes ?? []),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-flags`,
      operation,
      variant: 'flags',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        ...(mask
          ? [{name: 'mask', view: mask, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let x = values[valuesOffset + index];
  let y = ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'};
  let included = ${mask ? 'mask[maskOffset + index] != 0u &&' : ''} isFiniteFloat(x) && isFiniteFloat(y);
  flags[flagsOffset + index] = select(0u, 1u, included);`
    }),
    createFillNode<Parameters>(graph, {
      id: `${id}-island-clear`,
      operation,
      view: islandCount,
      type: 'u32',
      value: '0u',
      componentCount: 1
    }),
    ...chunkWeights.map((chunk, chunkIndex) => {
      const chunkKernel = getChunkKernelProps(chunk, chunkIndex);
      return createSpatialStatisticsRowNode<Parameters>(graph, {
        rowBase: chunk.rowStart,
        id: `${id}-analysis${chunkKernel.idSuffix}`,
        operation,
        variant: keepIslands ? 'analysis-keep' : 'analysis-exclude',
        bindings: [
          ...getWeightBindings(chunk),
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read_write'},
          {name: 'islandCount', view: islandCount, type: 'atomic<u32>', access: 'read_write'},
          ...(weightsOverflow
            ? [
                {
                  name: 'weightsOverflow',
                  view: weightsOverflow,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        invocationCount: chunkKernel.invocationCount,
        declarations: chunkKernel.declarations,
        body: chunkKernel.wrapBody(`let included = flags[flagsOffset + index] != 0u;
  var hasNeighbor = false;
  if (included) {
    ${getGPUSpatialWeightsNeighborLoopWGSL('flags[flagsOffset + neighbor] != 0u', 'hasNeighbor = true; break;', 'index', chunkKernel.rowStart)}
  }
  if (included && !hasNeighbor) {
    atomicAdd(&islandCount[islandCountOffset], 1u);
  }
  let truncated = ${weightsOverflow ? 'weightsOverflow[weightsOverflowOffset] != 0u' : 'false'};
  let analyzed = included && !truncated && ${keepIslands ? 'true' : 'hasNeighbor'};
  analysisFlags[analysisFlagsOffset + index] = select(0u, 1u, analyzed);`)
      });
    }),
    ...getPermutationWeightsLossChecks(weights).map(({view: loss, incompleteWhen}, lossIndex) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-incomplete-analysis-${lossIndex}`,
        operation,
        variant: 'incomplete-input',
        bindings: [
          {name: 'loss', view: loss, type: 'u32', access: 'read'},
          {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `if (loss[lossOffset] ${incompleteWhen} 0u) { analysisFlags[analysisFlagsOffset + index] = 0u; }`
      })
    ),
    ...(valueScales
      ? [0, 1].flatMap(variable => {
          const source = variable === 0 ? values : secondValues;
          const maximum = graph.createDataView(valueScales.buffer, {
            format: 'float32',
            byteOffset: valueScales.byteOffset + variable * 4,
            length: 1
          });
          return [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-magnitudes-${variable}`,
              operation,
              invocationCount: rows,
              bindings: [
                ...(source
                  ? [{name: 'source', view: source, type: 'f32' as const, access: 'read' as const}]
                  : []),
                {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read'},
                {name: 'magnitudes', view: columns[variable], type: 'f32', access: 'read_write'}
              ],
              body: `magnitudes[magnitudesOffset + index] = select(0.0, ${source ? 'abs(source[sourceOffset + index])' : '0.0'}, analysisFlags[analysisFlagsOffset + index] != 0u);`
            }),
            ...new GPUReduction({
              id: `${id}-maximum-${variable}`,
              input: columns[variable],
              output: maximum,
              operation: 'max'
            }).getCommandNodes(graph)
          ];
        })
      : []),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moment-inputs`,
      operation,
      variant: 'moment-inputs',
      bindings: [
        {name: 'values', view: values, type: 'f32', access: 'read'},
        ...secondBinding,
        ...(valueScales
          ? [
              {
                name: 'valueScales',
                view: valueScales,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read'},
        columnBinding(0),
        columnBinding(1)
      ],
      invocationCount: rows,
      declarations: `${constantsWGSL}
${
  valueScales
    ? `// Exact powers of two preserve representable spacings before centering.
fn getPermutationInverseScale(maximum: f32) -> f32 {
  let exponent = select(0i, clamp(i32((bitcast<u32>(maximum) >> 23u) & 255u) - 126i, -126i, 126i), maximum > 0.0);
  return bitcast<f32>(u32(127i - exponent) << 23u);
}`
    : ''
}`,
      body: `let analyzed = analysisFlags[analysisFlagsOffset + index] != 0u;
  column0[column0Offset + index] =
    select(0.0, values[valuesOffset + index]${valueScales ? ' * getPermutationInverseScale(valueScales[valueScalesOffset])' : ''}, analyzed);
  column1[column1Offset + index] =
    select(0.0, ${secondValues ? 'secondValues[secondValuesOffset + index]' : '0.0'}${valueScales ? ' * getPermutationInverseScale(valueScales[valueScalesOffset + 1u])' : ''}, analyzed);`
    }),
    ...getSplitColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-a`,
      operation,
      columns,
      rows,
      blockCount,
      first: 0,
      count: 2,
      totals: totalsA
    }),
    ...new GPUScan({
      id: `${id}-scan`,
      input: analysisFlags,
      output: scanned,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-positions`,
      operation,
      variant: 'positions',
      bindings: [
        {name: 'analysisFlags', view: analysisFlags, type: 'u32', access: 'read'},
        {name: 'scanned', view: scanned, type: 'u32', access: 'read'},
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows + 1,
      declarations: constantsWGSL,
      body: `if (index < ROWS) {
    rowPositions[rowPositionsOffset + index] =
      select(INVALID_POSITION, scanned[scannedOffset + index], analysisFlags[analysisFlagsOffset + index] != 0u);
  } else {
    rowPositions[rowPositionsOffset + ROWS] =
      scanned[scannedOffset + ROWS - 1u] + analysisFlags[analysisFlagsOffset + ROWS - 1u];
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-deviations`,
      operation,
      variant: 'deviations',
      bindings: [
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        columnBinding(0),
        columnBinding(1),
        columnBinding(2),
        columnBinding(3)
      ],
      // First-pass deviations from the tree-sum mean; their sum corrects that mean below.
      invocationCount: rows,
      declarations: constantsWGSL,
      body: `let included = rowPositions[rowPositionsOffset + index] != INVALID_POSITION;
  let count = rowPositions[rowPositionsOffset + ROWS];
  column2[column2Offset + index] =
    select(0.0, column0[column0Offset + index] - divideBySpatialCount(totalsA[totalsAOffset], count), included);
  column3[column3Offset + index] =
    select(0.0, column1[column1Offset + index] - divideBySpatialCount(totalsA[totalsAOffset + 1u], count), included);`
    }),
    ...getSplitColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-b`,
      operation,
      columns,
      rows,
      blockCount,
      first: 2,
      count: 2,
      totals: totalsB
    }),
    ...[0, 1].map(variable =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact-${variable}`,
        operation,
        variant: 'compact',
        bindings: [
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
          {name: 'rawValues', view: columns[variable], type: 'f32', access: 'read'},
          {name: 'deviations', view: columns[variable + 2], type: 'f32', access: 'read'},
          {
            name: 'compact',
            view: variable === 0 ? compactX : compactY,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'squares', view: columns[variable + 4], type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `${constantsWGSL}\nconst VARIABLE: u32 = ${variable}u;`,
        body: `let position = rowPositions[rowPositionsOffset + index];
  let included = position != INVALID_POSITION;
  let count = rowPositions[rowPositionsOffset + ROWS];
  let centered = deviations[deviationsOffset + index] - divideBySpatialCount(totalsB[totalsBOffset + VARIABLE], count);
  if (included) {
    compact[compactOffset + position] = ${(variable === 0 ? props.centerX : props.centerY !== false) ? 'centered' : 'rawValues[rawValuesOffset + index]'};
  }
  squares[squaresOffset + index] = select(0.0, centered * centered, included);`
      })
    ),
    ...chunkWeights.map((chunk, chunkIndex) => {
      const chunkKernel = getChunkKernelProps(chunk, chunkIndex);
      return createSpatialStatisticsRowNode<Parameters>(graph, {
        rowBase: chunk.rowStart,
        id: `${id}-row-weights${chunkKernel.idSuffix}`,
        operation,
        variant: standardize ? 'row-weights-standardize' : 'row-weights',
        bindings: [
          ...getWeightBindings(chunk),
          {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
          columnBinding(6),
          columnBinding(7),
          ...(standardize
            ? [
                {
                  name: 'scaledWeights',
                  view: chunk.effective,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: chunkKernel.invocationCount,
        declarations: chunkKernel.declarations,
        body: chunkKernel.wrapBody(`var rowWeight = 0.0;
  if (rowPositions[rowPositionsOffset + index] != INVALID_POSITION) {
    ${getGPUSpatialWeightsNeighborLoopWGSL('rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION', 'rowWeight += weight;', 'index', chunkKernel.rowStart)}
  }
  // Row-standardized weights are re-standardized over the active slots: a row with any active slot sums to 1.
  column6[column6Offset + index] = ${standardize ? 'select(0.0, 1.0, rowWeight > 0.0)' : 'rowWeight'};
  column7[column7Offset + index] = ${standardize ? 'select(0.0, 1.0, rowWeight > 0.0)' : 'rowWeight * rowWeight'};
  ${
    standardize
      ? `let scale = select(0.0, 1.0 / rowWeight, rowWeight > 0.0);
  let begin = min(offsets[offsetsOffset + index${chunkKernel.rowStart ? ' - WEIGHTS_ROW_START' : ''}], CAPACITY);
  let end = min(offsets[offsetsOffset + index${chunkKernel.rowStart ? ' - WEIGHTS_ROW_START' : ''} + 1u], CAPACITY);
  for (var slot = begin; slot < end; slot++) {
    scaledWeights[scaledWeightsOffset + slot] = weights[weightsOffset + slot] * scale;
  }`
      : ''
  }`)
      });
    }),
    ...getSplitColumnSumNodes<Parameters>(graph, {
      id: `${id}-sum-c`,
      operation,
      columns,
      rows,
      blockCount,
      first: 4,
      count: 4,
      totals: totalsC
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-totals`,
      operation,
      variant: 'totals',
      bindings: [
        {name: 'rowPositions', view: rowPositions, type: 'u32', access: 'read'},
        {name: 'totalsA', view: totalsA, type: 'f32', access: 'read'},
        {name: 'totalsB', view: totalsB, type: 'f32', access: 'read'},
        {name: 'totalsC', view: totalsC, type: 'f32', access: 'read'},
        {name: 'totals', view: totals, type: 'f32', access: 'read_write'}
      ],
      // contract-allow: single-invocation -- count division takes exactly 64 integer steps.
      invocationCount: 1,
      declarations: `const ROWS: u32 = ${rows}u;
${PERMUTATION_FLOAT_WGSL}`,
      // The count is an exact uint32 (the scan total) and is converted to f32 only here.
      body: `let count = rowPositions[rowPositionsOffset + ROWS];
  let meanX = divideBySpatialCount(totalsA[totalsAOffset], count) + divideBySpatialCount(totalsB[totalsBOffset], count);
  let meanY = divideBySpatialCount(totalsA[totalsAOffset + 1u], count) + divideBySpatialCount(totalsB[totalsBOffset + 1u], count);
  totals[totalsOffset] = f32(count);
  totals[totalsOffset + 1u] = multiplyBySpatialCount(meanX, count);
  totals[totalsOffset + 2u] = multiplyBySpatialCount(meanY, count);
  totals[totalsOffset + 3u] = totalsC[totalsCOffset];
  totals[totalsOffset + 4u] = totalsC[totalsCOffset + 1u];
  totals[totalsOffset + 5u] = totalsC[totalsCOffset + 2u];
  totals[totalsOffset + 6u] = meanX;
  totals[totalsOffset + 7u] = meanY;
  totals[totalsOffset + 8u] = totalsC[totalsCOffset + 3u];`
    })
  ];
  if (props.orderedPairMass)
    nodes.push(
      ...getOrderedPairMassNodes<Parameters>(graph, {
        id: `${id}-g-pair-mass`,
        operation,
        rows,
        bindings: [
          {name: 'compactX', view: compactX, type: 'f32', access: 'read'},
          {name: 'positions', view: rowPositions, type: 'u32', access: 'read'}
        ],
        value: `select(0.0, compactX[compactXOffset + index], index < positions[positionsOffset + ${rows}u])`,
        destination: {name: 'totals', view: totals, type: 'f32', access: 'read_write'},
        write: 'totals[totalsOffset + 9u] = pairMass;'
      })
    );
  if (props.analysisCount)
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-analysis-count`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: rowPositions, type: 'u32', access: 'read'},
          {name: 'analysisCount', view: props.analysisCount, type: 'u32', access: 'read_write'}
        ],
        body: `analysisCount[analysisCountOffset] = positions[positionsOffset + ${rows}u];`
      })
    );
  return {
    nodes,
    rowPositions,
    originalX: values,
    compactX,
    compactY,
    totals,
    islandCount,
    weights: effectiveWeights,
    chunks: chunkWeights.map(chunk => ({
      rowStart: chunk.rowStart,
      rowCount: chunk.rowCount,
      weights: chunk.weights,
      effective: chunk.effective
    }))
  };
}

/** Same fixed-order trees as the packed matrix, with independently bound columns. @internal */
function getSplitColumnSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    columns: GraphDataView<'float32'>[];
    rows: number;
    blockCount: number;
    first: number;
    count: number;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  return Array.from({length: props.count}, (_, column) =>
    getColumnSumNodes(graph, {
      ...props,
      id: `${props.id}-${column}`,
      matrix: props.columns[props.first + column],
      first: 0,
      count: 1,
      totals: graph.createDataView(props.totals.buffer, {
        format: 'float32',
        length: 1,
        byteOffset: props.totals.byteOffset + column * 4
      })
    })
  ).flat();
}

/**
 * Sums `count` columns of a column-major `rows`-row matrix, starting at `first`, into `totals` with
 * two fixed-order workgroup tree levels (per column and block of rows, then per column).
 *
 * @internal
 */
export function getColumnSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    matrix: GraphDataView<'float32'>;
    rows: number;
    blockCount: number;
    first: number;
    count: number;
    totals: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, rows, blockCount, first, count} = props;
  const segmentCount = count * blockCount;
  const blockOffsets = createTransientView(
    graph,
    `${id}-block-offsets`,
    'uint32',
    segmentCount + 1
  );
  const columnOffsets = createTransientView(graph, `${id}-column-offsets`, 'uint32', count + 1);
  const partials = createTransientView(graph, `${id}-partials`, 'float32', segmentCount);
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation,
      variant: 'sum-offsets',
      bindings: [
        {name: 'blockOffsets', view: blockOffsets, type: 'u32', access: 'read_write'},
        {name: 'columnOffsets', view: columnOffsets, type: 'u32', access: 'read_write'}
      ],
      invocationCount: segmentCount + 1,
      declarations: `const ROWS: u32 = ${rows}u;
const BLOCK_ROWS: u32 = ${BLOCK_ROWS}u;
const BLOCK_COUNT: u32 = ${blockCount}u;
const FIRST_COLUMN: u32 = ${first}u;
const COLUMN_COUNT: u32 = ${count}u;`,
      body: `let column = index / BLOCK_COUNT;
  let blockIndex = index % BLOCK_COUNT;
  blockOffsets[blockOffsetsOffset + index] = (FIRST_COLUMN + column) * ROWS + min(blockIndex * BLOCK_ROWS, ROWS);
  if (index <= COLUMN_COUNT) {
    columnOffsets[columnOffsetsOffset + index] = index * BLOCK_COUNT;
  }`
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-blocks`,
      operation,
      segmentCount,
      input: props.matrix,
      segmentOffsets: blockOffsets,
      output: partials
    }),
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-columns`,
      operation,
      segmentCount: count,
      input: partials,
      segmentOffsets: columnOffsets,
      output: props.totals
    })
  ];
}

/**
 * Counts the analysis rows whose compacted `x` is negative, for the statistics that are defined
 * only for non-negative data (Getis-Ord G). Exact uint32 diagnostics stay correct above 2^24.
 * Reads the uncentered `compactX` (`centerX: false`). @internal
 */
export function getNegativeValueCountNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    rows: number;
    compactX: GraphDataView<'float32'>;
    /** When supplied, count original source signs instead of potentially underflowed compact values. */
    originalX?: GraphDataView<'float32'>;
    rowPositions: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; negativeCount: GraphDataView<'uint32'>} {
  const {id, operation, rows} = props;
  const indicators = createTransientView(graph, `${id}-negative-indicators`, 'uint32', rows);
  const negativeCount = createTransientView(graph, `${id}-negative-count`, 'uint32', 1);
  return {
    negativeCount,
    nodes: [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-negative-indicators`,
        operation,
        variant: 'negative-indicators',
        bindings: [
          {
            name: props.originalX ? 'sourceValues' : 'compactX',
            view: props.originalX ?? props.compactX,
            type: 'f32',
            access: 'read'
          },
          {name: 'rowPositions', view: props.rowPositions, type: 'u32', access: 'read'},
          {name: 'indicators', view: indicators, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: `const ROWS: u32 = ${rows}u;`,
        // Positions at or past the analysis count hold no value.
        body: `indicators[indicatorsOffset + index] = select(0u, 1u, ${props.originalX ? 'rowPositions[rowPositionsOffset + index] != 0xffffffffu' : 'index < rowPositions[rowPositionsOffset + ROWS]'} && ${props.originalX ? 'sourceValues[sourceValuesOffset + index]' : 'compactX[compactXOffset + index]'} < 0.0);`
      }),
      ...new GPUReduction({
        id: `${id}-negative-sum`,
        input: indicators,
        output: negativeCount,
        operation: 'sum'
      }).getCommandNodes(graph)
    ]
  };
}
