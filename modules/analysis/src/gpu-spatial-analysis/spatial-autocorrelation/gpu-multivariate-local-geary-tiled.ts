// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getLocalVariabilityCenteringNodes} from './local-variability-centering';
import {ROUNDED_DIFFERENCE_WGSL} from '../spatial-regression/rounded-difference-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {IS_FINITE_BITS_WGSL, GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  getGPUSpatialWeightsChunkDispatches,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';

/** Rows folded by one moment-reduction invocation. @internal */
const MOMENT_CHUNK_ROWS = 4096;
/** Default rows per simulation dispatch. @internal */
export const DEFAULT_GEARY_ROW_TILE_SIZE = 262144;
/** Default conditional permutations per simulation dispatch. @internal */
export const DEFAULT_GEARY_PERMUTATION_TILE_SIZE = 128;
/** Default and maximum active neighbors per row held in the private swap table. @internal */
export const DEFAULT_GEARY_MAXIMUM_NEIGHBORS = 64;
export const MAXIMUM_GEARY_MAXIMUM_NEIGHBORS = 128;

/** Inputs of the tiled strategy, a subset of the contributor props. @internal */
export type TiledMultivariateLocalGearyProps = {
  id: string;
  rows: number;
  columns: number;
  permutations: number;
  parameters: GraphDataView<'uint32'>;
  classifyDirections: boolean;
  tail: 'equal-tail' | 'esda-min-tail';
  rowTileSize: number;
  permutationTileSize: number;
  maximumNeighbors: number;
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  output: {
    statistic: GraphDataView<'float32'>;
    pValues: GraphDataView<'float32'>;
    classes: GraphDataView<'uint32'>;
  };
};

/**
 * Scalable strategy of the multivariate local Geary permutation test. It produces the same
 * statistic, p-values and classes as the exact baseline (identical Philox streams and identical
 * partial Fisher-Yates semantics) without a per-thread row-sized candidate array or a
 * rows x permutations simulation buffer:
 * - valid rows are compacted once, and the baseline's candidate list (valid rows except the focal
 *   row) is addressed virtually through the compacted list plus the focal row's rank,
 * - the partial Fisher-Yates swaps live in a small private table of at most `maximumNeighbors`
 *   chosen slots, so rows with a larger active degree are reported as untested,
 * - each invocation owns one row and loops its replicates, accumulating lower/upper/average in a
 *   rows-sized buffer, and the work is tiled over rows and permutations (one dispatch each).
 * - population moments are reduced in 4096-row chunks and merged with Chan's formula.
 *
 * @internal
 */
export function getTiledMultivariateLocalGearyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TiledMultivariateLocalGearyProps
): readonly GPUCommandNode<Parameters>[] {
  const {id, rows, columns, permutations, maximumNeighbors} = props;
  const operation = 'GPUMultivariateLocalGearyPermutationTest';
  const chunkCount = Math.ceil(rows / MOMENT_CHUNK_ROWS);
  const valid = createTransientView(graph, `${id}-valid`, 'uint32', rows);
  const rowIdentifiers = createTransientView(graph, `${id}-row-ids`, 'uint32', rows);
  const validRows = createTransientView(graph, `${id}-valid-rows`, 'uint32', rows);
  // One count plus the three parameter words; simulation keeps its existing eight bindings.
  const inferenceState = createTransientView(graph, `${id}-inference-state`, 'uint32', 4);
  const validCount = graph.createDataView(inferenceState.buffer, {format: 'uint32', length: 1});
  const standardized = createTransientView(
    graph,
    `${id}-standardized`,
    'float32',
    rows * columns + columns + 1
  );
  const partials = createTransientView(
    graph,
    `${id}-moment-partials`,
    'float32',
    3 * columns * chunkCount
  );
  const moments = createTransientView(graph, `${id}-moments`, 'float32', 2 * columns);
  // Layout: [lower | upper | non-finite replicates | sum of replicates], each `rows` long.
  const accumulators = createTransientView(graph, `${id}-accumulators`, 'float32', 4 * rows);

  const binding = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'f32' | 'u32' = 'f32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const node = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = '',
    constants?: Record<string, number>
  ) =>
    createSpatialStatisticsCollectiveNode<Parameters>(graph, {
      id: `${id}-${name}`,
      operation,
      variant: name,
      invocationCount,
      bindings,
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
var<private> TILE_ROW_BASE: u32;
var<private> REPLICATE_BASE: u32;
var<private> REPLICATE_COUNT: u32;
var<private> WEIGHTS_ROW_START: u32;
var<private> CAPACITY: u32;
${declarations}`,
      parameterValues: [
        constants?.['TILE_ROW_BASE'] ?? 0,
        constants?.['REPLICATE_BASE'] ?? 0,
        constants?.['REPLICATE_COUNT'] ?? 0,
        constants?.['WEIGHTS_ROW_START'] ?? 0
      ],
      body: `if (index >= INVOCATION_COUNT) { return; }
TILE_ROW_BASE = getChunkParameter(36u); REPLICATE_BASE = getChunkParameter(37u); REPLICATE_COUNT = getChunkParameter(38u); WEIGHTS_ROW_START = getChunkParameter(39u);
${bindings.some(binding => binding.name === 'neighbors') ? 'CAPACITY = arrayLength(&neighbors) - neighborsOffset;' : ''}
${body}`
    });
  const overflowBindings = props.weights.status?.overflow
    ? [binding('overflow', props.weights.status.overflow, 'read', 'u32')]
    : [];

  const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
  const centered = getLocalVariabilityCenteringNodes(graph, {
    id,
    rows,
    columns,
    values: props.values,
    bindings: [binding('valid', valid, 'read', 'u32')],
    declarations: 'fn included(row: u32) -> bool { return valid[validOffset + row] != 0u; }'
  });
  const localRow = (dispatch: {isWhole: boolean}) => (void dispatch, '(row - WEIGHTS_ROW_START)');
  const nodes: GPUCommandNode<Parameters>[] = [
    node(
      'parameters',
      1,
      [
        binding('parameters', props.parameters, 'read', 'u32'),
        binding('inferenceState', inferenceState, 'read_write', 'u32')
      ],
      `inferenceState[inferenceStateOffset + 1u] = parameters[parametersOffset];
       inferenceState[inferenceStateOffset + 2u] = parameters[parametersOffset + 1u];
       inferenceState[inferenceStateOffset + 3u] = parameters[parametersOffset + 2u];`
    ),
    node(
      'complete',
      rows,
      [
        binding('values', props.values),
        ...(props.mask ? [binding('mask', props.mask, 'read', 'u32')] : []),
        binding('valid', valid, 'read_write', 'u32'),
        binding('rowIdentifiers', rowIdentifiers, 'read_write', 'u32'),
        ...overflowBindings
      ],
      `var included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'} ${props.weights.status?.overflow ? '&& overflow[overflowOffset] == 0u' : ''};
for (var column = 0u; column < ${columns}u; column++) { included = included && isFiniteBits(values[valuesOffset + index * ${columns}u + column]); }
valid[validOffset + index] = select(0u, 1u, included);
rowIdentifiers[rowIdentifiersOffset + index] = index;`
    ),
    ...new GPUCompaction({
      id: `${id}-compaction`,
      input: rowIdentifiers,
      flags: valid,
      output: validRows,
      count: validCount
    }).getCommandNodes(graph),
    // One invocation per (column, chunk) folds its rows with Welford's update.
    ...centered.nodes,
    node(
      'moment-chunks',
      columns * chunkCount,
      [
        binding('values', centered.values),
        binding('valid', valid, 'read', 'u32'),
        binding('partials', partials, 'read_write')
      ],
      `let column = index / ${chunkCount}u; let chunk = index % ${chunkCount}u;
var count = 0.0; var mean = 0.0; var squares = 0.0;
let chunkStart = chunk * ${MOMENT_CHUNK_ROWS}u; let chunkEnd = min(chunkStart + ${MOMENT_CHUNK_ROWS}u, ${rows}u);
for (var row = chunkStart; row < chunkEnd; row++) { if (valid[validOffset + row] != 0u) { count += 1.0; let value = values[valuesOffset + row * ${columns}u + column]; let difference = value - mean; mean += difference / count; squares += difference * (value - mean); } }
let base = (column * ${chunkCount}u + chunk) * 3u;
partials[partialsOffset + base] = count; partials[partialsOffset + base + 1u] = mean; partials[partialsOffset + base + 2u] = squares;`
    ),
    // Chan's pairwise merge; a constant or underdetermined column gets a NaN scale.
    node(
      'moment-merge',
      columns,
      [binding('partials', partials), binding('moments', moments, 'read_write')],
      `var count = 0u; var mean = 0.0; var squares = 0.0;
for (var chunk = 0u; chunk < ${chunkCount}u; chunk++) {
 let base = (index * ${chunkCount}u + chunk) * 3u;
 let chunkCountValue = u32(partials[partialsOffset + base]);
 if (chunkCountValue > 0u) {
  let chunkMean = partials[partialsOffset + base + 1u]; let chunkSquares = partials[partialsOffset + base + 2u];
  if (count == 0u) { count = chunkCountValue; mean = chunkMean; squares = chunkSquares; continue; }
  let combined = count + chunkCountValue; let delta = chunkMean - mean;
  squares += chunkSquares + multiplyBySpatialCount(delta * delta * divideSpatialCounts(chunkCountValue, combined), count);
  mean += delta * divideSpatialCounts(chunkCountValue, combined); count = combined;
 }
}
var scale = getNaN();
if (count > 1u && squares > 0.0 && isFiniteBits(squares) && isFiniteBits(mean)) { scale = sqrt(divideBySpatialCount(squares, count)); }
moments[momentsOffset + index * 2u] = mean; moments[momentsOffset + index * 2u + 1u] = scale;`,
      SPATIAL_STATISTICS_COUNTS_WGSL
    ),
    node(
      'standardize',
      rows * columns,
      [
        binding('values', props.values),
        binding('valid', valid, 'read', 'u32'),
        binding('moments', moments),
        binding('standardized', standardized, 'read_write')
      ],
      `let row = index / ${columns}u; let column = index % ${columns}u;
var value = getNaN();
let deviation = moments[momentsOffset + column * 2u + 1u];
if (valid[validOffset + row] != 0u && deviation > 0.0 && isFiniteBits(deviation)) { value = values[valuesOffset + index]; }
if (row == 0u) { standardized[standardizedOffset + ${rows * columns}u + column] = deviation; }
if (index == 0u) { standardized[standardizedOffset + ${rows * columns + columns}u] = 0.0; }
standardized[standardizedOffset + index] = value;`
    )
  ];

  const sharedDeclarations = `${PERMUTATION_RANDOM_WGSL}\n${ROUNDED_DIFFERENCE_WGSL}\n${SCALED_QUOTIENT_WGSL}
const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columns}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighbors}u;
const PAIR_CAPACITY: u32 = ${2 * maximumNeighbors}u;

fn isRowValid(row: u32) -> bool { return isFiniteBits(standardized[standardizedOffset + row * COLUMNS]); }
fn getVectorDistance(row: u32, candidate: u32) -> f32 {
 var distance = 0.0;
 for (var column = 0u; column < COLUMNS; column++) { let difference = getRoundedDifference(standardized[standardizedOffset + row * COLUMNS + column], standardized[standardizedOffset + candidate * COLUMNS + column], bitcast<u32>(standardized[standardizedOffset + ROWS*COLUMNS+COLUMNS])); let deviation = standardized[standardizedOffset + ROWS*COLUMNS + column]; var normalized = getNaN(); if (deviation > 0.0 && isFiniteBits(deviation)) { normalized = getScaledQuotient(difference, deviation, 1.0, 0); } distance += normalized * normalized; }
 return distance;
}`;

  const getDispatchDeclarations = (dispatch: {isWhole: boolean; constantsWGSL: string}) => (
    void dispatch, sharedDeclarations
  );
  nodes.push(
    // Observed statistic (replicate 0) and accumulator reset, tiled over rows.
    ...dispatches.flatMap(dispatch =>
      getTileStarts(dispatch.rowCount, props.rowTileSize).map(tileOffset => {
        const tileBase = dispatch.rowStart + tileOffset;
        return node(
          `observed-${tileBase}`,
          Math.min(props.rowTileSize, dispatch.rowCount - tileOffset),
          [
            binding('standardized', standardized),
            binding('validCount', inferenceState, 'read', 'u32'),
            binding('offsets', dispatch.weights.offsets, 'read', 'u32'),
            binding('neighbors', dispatch.weights.neighbors, 'read', 'u32'),
            binding('weights', dispatch.weights.weights),
            binding('observed', props.output.statistic, 'read_write'),
            binding('accumulators', accumulators, 'read_write')
          ],
          `{ let row = index + TILE_ROW_BASE;
for (var slotField = 0u; slotField < 4u; slotField++) { accumulators[accumulatorsOffset + slotField * ROWS + row] = 0.0; }
var value = getNaN();
if (isRowValid(row) && validCount[validCountOffset] > 1u) {
 var total = 0.0; var weightSum = 0.0;
 for (var slot = offsets[offsetsOffset + ${localRow(dispatch)}]; slot < min(offsets[offsetsOffset + ${localRow(dispatch)} + 1u], CAPACITY); slot++) {
  let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
  if (neighbor >= ROWS || neighbor == row || !(weight > 0.0)) { continue; }
  if (!isRowValid(neighbor)) { continue; }
  total += weight * getVectorDistance(row, neighbor) / ${columns}.0; weightSum += weight;
 }
 if (weightSum > 0.0) { value = total / weightSum; }
}
observed[observedOffset + row] = value; }`,
          getDispatchDeclarations(dispatch),
          {TILE_ROW_BASE: tileBase, WEIGHTS_ROW_START: dispatch.rowStart}
        );
      })
    )
  );

  const simulateDeclarations = `${sharedDeclarations}

var<private> pairPositions: array<u32, PAIR_CAPACITY>;
var<private> pairValues: array<u32, PAIR_CAPACITY>;
var<private> pairCount: u32;
fn getVirtualCandidate(position: u32, rank: u32) -> u32 {
 for (var pair = 0u; pair < pairCount; pair++) { if (pairPositions[pair] == position) { return pairValues[pair]; } }
 return validRows[validRowsOffset + select(position + 1u, position, position < rank)];
}
fn setVirtualCandidate(position: u32, value: u32) {
 for (var pair = 0u; pair < pairCount; pair++) { if (pairPositions[pair] == position) { pairValues[pair] = value; return; } }
 pairPositions[pairCount] = position; pairValues[pairCount] = value; pairCount++;
}`;
  for (const dispatch of dispatches) {
    for (const tileOffset of getTileStarts(dispatch.rowCount, props.rowTileSize)) {
      const tileBase = dispatch.rowStart + tileOffset;
      for (const replicateBase of getTileStarts(permutations, props.permutationTileSize)) {
        nodes.push(
          node(
            `simulate-${tileBase}-${replicateBase}`,
            Math.min(props.rowTileSize, dispatch.rowCount - tileOffset),
            [
              binding('standardized', standardized),
              binding('validRows', validRows, 'read', 'u32'),
              binding('validCount', inferenceState, 'read', 'u32'),
              binding('offsets', dispatch.weights.offsets, 'read', 'u32'),
              binding('neighbors', dispatch.weights.neighbors, 'read', 'u32'),
              binding('weights', dispatch.weights.weights),
              binding('observed', props.output.statistic),
              binding('accumulators', accumulators, 'read_write')
            ],
            `{ let row = index + TILE_ROW_BASE;
if (isRowValid(row) && validCount[validCountOffset] > 1u) {
 let observedValue = observed[observedOffset + row];
 let othersCount = validCount[validCountOffset] - 1u;
 var rankLow = 0u; var rankHigh = validCount[validCountOffset];
 while (rankLow < rankHigh) { let middle = (rankLow + rankHigh) / 2u; if (validRows[validRowsOffset + middle] < row) { rankLow = middle + 1u; } else { rankHigh = middle; } }
 let rank = rankLow;
 var lower = accumulators[accumulatorsOffset + row]; var upper = accumulators[accumulatorsOffset + ROWS + row];
 var invalid = accumulators[accumulatorsOffset + 2u * ROWS + row]; var differenceAverage = accumulators[accumulatorsOffset + 3u * ROWS + row];
 for (var tileStep = 0u; tileStep < REPLICATE_COUNT; tileStep++) {
  let replicate = REPLICATE_BASE + tileStep + 1u;
  var stream = createPhiloxStream(vec2<u32>(validCount[validCountOffset + 1u], validCount[validCountOffset + 2u]), row, replicate, ${PHILOX_PURPOSES.localPermutationTest.id}u);
  pairCount = 0u;
  var chosen = 0u; var total = 0.0; var weightSum = 0.0; var overflowed = false;
  for (var slot = offsets[offsetsOffset + ${localRow(dispatch)}]; slot < min(offsets[offsetsOffset + ${localRow(dispatch)} + 1u], CAPACITY); slot++) {
   let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
   if (neighbor >= ROWS || neighbor == row || !(weight > 0.0)) { continue; }
   if (!isRowValid(neighbor)) { continue; }
   if (chosen >= MAXIMUM_NEIGHBORS || chosen >= othersCount) { overflowed = true; break; }
   let selection = chosen + nextPhiloxBelow(&stream, othersCount - chosen);
   let candidate = getVirtualCandidate(selection, rank);
   setVirtualCandidate(selection, getVirtualCandidate(chosen, rank));
   setVirtualCandidate(chosen, candidate);
   chosen++;
   total += weight * getVectorDistance(row, candidate) / ${columns}.0; weightSum += weight;
  }
  var value = getNaN();
  if (!overflowed && weightSum > 0.0) { value = total / weightSum; }
  invalid += select(0.0, 1.0, !isFiniteBits(value));
  lower += select(0.0, 1.0, value <= observedValue); upper += select(0.0, 1.0, value >= observedValue);
  if (value != observedValue) { differenceAverage += (value - observedValue) / ${permutations}.0; }
 }
 accumulators[accumulatorsOffset + row] = lower; accumulators[accumulatorsOffset + ROWS + row] = upper;
 accumulators[accumulatorsOffset + 2u * ROWS + row] = invalid; accumulators[accumulatorsOffset + 3u * ROWS + row] = differenceAverage;
}
}`,
            simulateDeclarations,
            {
              TILE_ROW_BASE: tileBase,
              WEIGHTS_ROW_START: dispatch.rowStart,
              REPLICATE_BASE: replicateBase,
              REPLICATE_COUNT: Math.min(props.permutationTileSize, permutations - replicateBase)
            }
          )
        );
      }
    }
  }
  nodes.push(
    node(
      'inference',
      rows,
      [
        binding('observed', props.output.statistic),
        binding('accumulators', accumulators),
        binding('validCount', inferenceState, 'read', 'u32'),
        binding('pValues', props.output.pValues, 'read_write'),
        binding('classes', props.output.classes, 'read_write', 'u32')
      ],
      `let observedValue = observed[observedOffset + index];
let lower = u32(accumulators[accumulatorsOffset + index]); let upper = u32(accumulators[accumulatorsOffset + ${rows}u + index]);
let differenceAverage = accumulators[accumulatorsOffset + 3u * ${rows}u + index];
let finite = isFiniteBits(observedValue) && accumulators[accumulatorsOffset + 2u * ${rows}u + index] == 0.0;
var probability = getNaN(); var classification = 0xffffffffu;
if (finite) { probability = ${props.tail === 'esda-min-tail' ? `f32(min(upper, ${permutations}u - upper) + 1u) / ${permutations + 1}.0` : `min(1.0, 2.0 * f32(min(lower, upper) + 1u) / ${permutations + 1}.0)`};
classification = 0u;
if (probability <= ${props.classifyDirections ? '0.999999' : 'bitcast<f32>(validCount[validCountOffset + 3u])'}) { classification = select(2u, 1u, differenceAverage > 0.0); }
}
pValues[pValuesOffset + index] = probability; classes[classesOffset + index] = classification;`
    )
  );
  return nodes;
}

function getTileStarts(total: number, tileSize: number): number[] {
  const starts: number[] = [];
  for (let start = 0; start < total; start += Math.max(1, tileSize)) starts.push(start);
  return starts;
}
