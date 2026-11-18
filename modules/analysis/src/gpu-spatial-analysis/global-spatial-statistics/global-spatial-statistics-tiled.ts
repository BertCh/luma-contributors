// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUReduction,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  type WGSLKernelBinding,
  type WGSLKernelElementType
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {createSpatialStatisticsCollectiveNode} from '../spatial-weights/spatial-statistics-collective-node';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {createFloat32ValuesNodes} from '../spatial-weights/float32-values';
import {GPUChunkedSpatialWeightsTranspose} from '../spatial-weights/gpu-chunked-spatial-weights-transpose';
import {
  getGPUSpatialWeightsChunkDispatches,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/gpu-chunked-spatial-weights';
import {isGPUSpatialWeightsRowStandardized} from '../spatial-weights/spatial-weights';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import type {GPUGlobalSpatialStatisticsProps} from './gpu-global-spatial-statistics';
import {getGlobalSpatialStatisticsStatusNodes} from './global-spatial-statistics-status';

/** A tile bounds serial sums to sixteen values per lane before a compensated binary tree. */
const TILE_ROWS = 4096;
/** Padded row dispatch arithmetic must fit uint32. Device bindings impose a tighter row bound. */
export const GLOBAL_SPATIAL_STATISTICS_MAXIMUM_ROWS = 0xffffffff - TILE_ROWS;
/** Binary S1 can contribute twice the ordered edge capacity; all integer diagnostics fit uint32. */
export const GLOBAL_SPATIAL_STATISTICS_MAXIMUM_SLOTS = Math.floor(0xffffffff / 2);
/** All 27 legacy summary terms have a high and a correction word, independent of population. */
const TOTAL_WORDS = 54;

const COMMON = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
fn addCompensated(pair: vec2<f32>, term: f32) -> vec2<f32> {
  let next = pair.x + term;
  return vec2<f32>(next, pair.y + getAdditionRoundingError(pair.x, term, next));
}
fn readControlFloat(word: u32) -> f32 { return bitcast<f32>(getStatisticsControl(word)); }
var<private> ROW_START: u32;
var<private> ROW_COUNT: u32;
var<private> CAPACITY: u32;
fn activeWeight(weight: f32) -> bool { return weight > 0.0 && isFiniteFloat(weight); }
`;

/** Fixed-source, fixed-order global reductions with only separate row-sized bindings. @internal */
export function getTiledGlobalSpatialStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUGlobalSpatialStatisticsProps,
  finalizeDeclarations: string,
  finalizeBody: string
): readonly GPUCommandNode<Parameters>[] {
  const id = props.id ?? 'global-spatial-statistics';
  const dispatches = getGPUSpatialWeightsChunkDispatches(props.weights);
  const capacity = dispatches.reduce((sum, dispatch) => sum + dispatch.weights.neighbors.length, 0);
  const useTranspose =
    !dispatches[0].isWhole &&
    (props.symmetryStrategy === 'transpose' ||
      (props.symmetryStrategy !== 'pairwise' &&
        capacity * 4 <= graph.device.limits.maxStorageBufferBindingSize));
  const rows = dispatches.reduce((sum, dispatch) => sum + dispatch.rowCount, 0);
  const maximumCapacity = Math.max(
    ...dispatches.map(dispatch => dispatch.weights.neighbors.length)
  );
  const blocks = Math.ceil(rows / TILE_ROWS);
  // Every check precedes conversion, scratch declarations, sorts and command-node construction.
  validateSpatialStatisticsBindings(graph, id, props, {
    'row scratch': rows * 4,
    'column offsets': (rows + 1) * 4,
    'transpose scratch': (useTranspose ? capacity : maximumCapacity) * 4,
    'tile partials': blocks * 8 * 2 * 4,
    'totals and control': 256
  });
  const weightsStatus = getGlobalSpatialStatisticsStatusNodes(graph, props);
  const nodes: GPUCommandNode<Parameters>[] = [...weightsStatus.nodes];
  const scratch = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length = rows
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const valuesConversion = createFloat32ValuesNodes(graph, `${id}-values`, props.values);
  const secondConversion = props.secondValues
    ? createFloat32ValuesNodes(graph, `${id}-second-values`, props.secondValues)
    : undefined;
  nodes.push(...valuesConversion.nodes, ...(secondConversion?.nodes ?? []));
  const values = valuesConversion.view;
  const secondValues = secondConversion?.view;
  const baseStates = scratch('base-states', 'uint32');
  const analysisStates = scratch('analysis-states', 'uint32');
  const centeredX = scratch('centered-x', 'float32');
  const centeredY = scratch('centered-y', 'float32');
  const rowScale = scratch('row-scale', 'float32');
  const temporary = scratch('temporary', 'float32');
  const incomingWeight = scratch('incoming-weight', 'float32');
  const incomingWeightError = scratch('incoming-weight-error', 'float32');
  const incomingDegree = scratch('incoming-degree', 'uint32');
  const incomingLag = scratch('incoming-lag', 'float32');
  const incomingLagError = scratch('incoming-lag-error', 'float32');
  const partials = scratch('tile-partials', 'float32', blocks * 8 * 2);
  const totals = scratch('compensated-totals', 'float32', TOTAL_WORDS);
  const control = createTransientView(
    graph,
    `${id}-control`,
    'uint32',
    64,
    Buffer.STORAGE | Buffer.UNIFORM
  );
  const bind = (
    name: string,
    view: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: WGSLKernelElementType = view.format === 'uint32' ? 'u32' : 'f32'
  ): WGSLKernelBinding => ({name, view, access, type});
  const clear = (name: string, view: GraphDataView, type: 'f32' | 'u32') =>
    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear-${name}`,
        operation: 'GPUGlobalSpatialStatistics',
        view,
        type,
        value: type === 'f32' ? '0.0' : '0u'
      })
    );
  clear('control', control, 'u32');
  clear('totals', totals, 'f32');
  for (const [name, view] of Object.entries({
    incomingWeight,
    incomingWeightError,
    incomingLag,
    incomingLagError
  }))
    clear(name, view, 'f32');
  clear('incoming-degree', incomingDegree, 'u32');
  const node = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = '',
    parameters: number[] = [],
    guarded = true
  ) => {
    const writableControl = bindings.find(
      binding => binding.view === control && binding.access === 'read_write'
    );
    const controlReader = writableControl
      ? `fn getStatisticsControl(word: u32) -> u32 { return ${writableControl.type === 'atomic<u32>' ? 'atomicLoad(&counters[countersOffset + word])' : 'counters[countersOffset + word]'}; }`
      : '';
    nodes.push(
      createSpatialStatisticsCollectiveNode(graph, {
        id: `${id}-${name}`,
        operation: 'GPUGlobalSpatialStatistics',
        variant: name.replace(/-\d+/g, ''),
        invocationCount: count,
        bindings,
        control: writableControl ? undefined : control,
        parameterValues: parameters,
        declarations: `${controlReader}\n${COMMON}\n${declarations}`,
        body: `${guarded ? 'if (index >= INVOCATION_COUNT) { return; }' : ''}
ROW_START = getChunkParameter(36u); ROW_COUNT = getChunkParameter(37u);
${bindings.some(binding => binding.name === 'neighbors') ? 'CAPACITY = arrayLength(&neighbors) - neighborsOffset;' : ''}
{ ${body} }`
      })
    );
  };
  const weightBindings = (chunkIndex: number) => {
    const weights = dispatches[chunkIndex].weights;
    return [
      bind('offsets', weights.offsets),
      bind('neighbors', weights.neighbors),
      bind('weights', weights.weights)
    ];
  };
  const neighborLoop = (body: string, inclusion = 'rowScale[rowScaleOffset + neighbor] != 0.0') => `
for (var slot = min(offsets[offsetsOffset + index - ROW_START], CAPACITY); slot < min(offsets[offsetsOffset + index - ROW_START + 1u], CAPACITY); slot++) {
 let neighbor = neighbors[neighborsOffset + slot]; let weight = weights[weightsOffset + slot];
 if (neighbor >= ${rows}u || neighbor == index || !activeWeight(weight)) { continue; }
 if (!(${inclusion})) { continue; }
 ${body}
}`;
  const rowNode = (
    name: string,
    chunkIndex: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) => {
    const dispatch = dispatches[chunkIndex];
    node(
      `${name}-${chunkIndex}`,
      dispatch.rowCount,
      bindings,
      `let index = index + ROW_START;\n${body}`,
      declarations,
      [dispatch.rowStart, dispatch.rowCount]
    );
  };
  const collectiveRowNode = (
    name: 'islands' | 'joins',
    chunkIndex: number,
    bindings: WGSLKernelBinding[],
    body: string
  ) => {
    const dispatch = dispatches[chunkIndex];
    const words = name === 'islands' ? [0, 1, 9, 10, 24, 25, 26, 27] : [2, 3, 4];
    const localBody = body
      .replaceAll('&counters[countersOffset]', '&groupCounts[0u]')
      .replace(/&counters\[countersOffset \+ (\d+)u\]/g, '&groupCounts[$1u]');
    node(
      `${name}-${chunkIndex}`,
      dispatch.rowCount,
      bindings,
      `
if (localInvocationIndex < 28u) { atomicStore(&groupCounts[localInvocationIndex], select(0u, 0xffffffffu, localInvocationIndex == 24u || localInvocationIndex == 26u)); }
workgroupBarrier();
if (index < INVOCATION_COUNT) { let index = index + ROW_START; ${localBody} }
workgroupBarrier();
if (localInvocationIndex == 0u) {
 ${words.map(word => `atomic${word === 24 || word === 26 ? 'Min' : word === 25 || word === 27 ? 'Max' : 'Add'}(&counters[countersOffset + ${word}u], atomicLoad(&groupCounts[${word}u]));`).join('\n ')}
}`,
      'var<workgroup> groupCounts: array<atomic<u32>, 28>;',
      [dispatch.rowStart, dispatch.rowCount],
      false
    );
  };
  // A reduction tile has 256 lanes, sixteen values per lane and a compensated binary tree.
  // Both tile boundaries and chunk bounds are runtime words, including unequal final chunks.
  const reduce = (
    name: string,
    start: number,
    count: number,
    fields: number[],
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = '',
    extraParameters: number[] = []
  ) => {
    const blockCount = Math.ceil(count / TILE_ROWS);
    node(
      `${name}-tiles`,
      blockCount * 256,
      [...bindings, bind('partials', partials, 'read_write')],
      `
let tileIndex = index / 256u;
var sums: array<vec2<f32>, ${fields.length}>;
for (var step = 0u; step < 16u; step++) {
 let index = ROW_START + tileIndex * 4096u + localInvocationIndex + step * 256u;
 if (index < ROW_START + ROW_COUNT) {
  var contribution: array<f32, ${fields.length}>;
  ${body}
  for (var field = 0u; field < ${fields.length}u; field++) { sums[field] = addCompensated(sums[field], contribution[field]); }
 }
}
for (var field = 0u; field < ${fields.length}u; field++) {
 reductionPairs[localInvocationIndex] = sums[field]; workgroupBarrier();
 for (var stride = 128u; stride > 0u; stride /= 2u) {
  if (localInvocationIndex < stride) {
   var pair = addCompensated(reductionPairs[localInvocationIndex], reductionPairs[localInvocationIndex + stride].x);
   pair = addCompensated(pair, reductionPairs[localInvocationIndex + stride].y);
   reductionPairs[localInvocationIndex] = pair;
  }
  workgroupBarrier();
 }
 if (localInvocationIndex == 0u) {
  let destination = partialsOffset + (field * getChunkParameter(38u) + tileIndex) * 2u;
  partials[destination] = reductionPairs[0].x; partials[destination + 1u] = reductionPairs[0].y;
 }
 workgroupBarrier();
}`,
      `var<workgroup> reductionPairs: array<vec2<f32>, 256>;\n${declarations}`,
      [start, count, blockCount, ...extraParameters],
      false
    );
    // Only the bounded tile table is walked serially; no invocation walks the population.
    node(
      `${name}-merge`,
      fields.length,
      [bind('partials', partials), bind('totals', totals, 'read_write')],
      `
let columns = array<u32, ${fields.length}>(${fields.map(field => `${field}u`).join(', ')});
let destination = totalsOffset + columns[index] * 2u;
var pair = vec2<f32>(totals[destination], totals[destination + 1u]);
for (var tileIndex = 0u; tileIndex < getChunkParameter(38u); tileIndex++) {
 let source = partialsOffset + (index * getChunkParameter(38u) + tileIndex) * 2u;
 pair = addCompensated(pair, partials[source]); pair = addCompensated(pair, partials[source + 1u]);
}
totals[destination] = pair.x; totals[destination + 1u] = pair.y;`,
      '',
      [0, 0, blockCount]
    );
  };
  node(
    'extrema-initialize',
    1,
    [bind('counters', control, 'read_write')],
    'counters[countersOffset + 24u] = 0xffffffffu; counters[countersOffset + 26u] = 0xffffffffu;'
  );
  node(
    'include',
    rows,
    [
      bind('values', values),
      ...(secondValues ? [bind('secondValues', secondValues)] : []),
      ...(props.mask ? [bind('mask', props.mask)] : []),
      bind('baseStates', baseStates, 'read_write')
    ],
    `baseStates[baseStatesOffset + index] = u32(isFiniteFloat(values[valuesOffset + index]) ${secondValues ? '&& isFiniteFloat(secondValues[secondValuesOffset + index])' : ''} ${props.mask ? '&& mask[maskOffset + index] != 0u' : ''});`
  );
  for (const [chunkIndex] of dispatches.entries())
    collectiveRowNode(
      'islands',
      chunkIndex,
      [
        ...weightBindings(chunkIndex),
        bind('baseStates', baseStates),
        bind('values', values),
        ...(secondValues ? [bind('secondValues', secondValues)] : []),
        bind('analysisStates', analysisStates, 'read_write'),
        bind('counters', control, 'read_write', 'atomic<u32>')
      ],
      `
let includedBase = baseStates[baseStatesOffset + index] != 0u; var hasNeighbor = false;
if (includedBase) { ${neighborLoop('hasNeighbor = true;', 'baseStates[baseStatesOffset + neighbor] != 0u')} }
let island = includedBase && !hasNeighbor;
let included = includedBase && ${props.islands === 'keep' ? 'true' : '!island'};
analysisStates[analysisStatesOffset + index] = u32(included);
if (island) { atomicAdd(&counters[countersOffset + 9u], 1u); }
if (included) {
 atomicAdd(&counters[countersOffset], 1u);
 let value = values[valuesOffset + index];
 let valueBits = bitcast<u32>(value); let ordered = select(valueBits ^ 0x80000000u, ~valueBits, (valueBits & 0x80000000u) != 0u);
 atomicMin(&counters[countersOffset + 24u], ordered); atomicMax(&counters[countersOffset + 25u], ordered);
 ${secondValues ? 'let secondBits = bitcast<u32>(secondValues[secondValuesOffset + index]); let secondOrdered = select(secondBits ^ 0x80000000u, ~secondBits, (secondBits & 0x80000000u) != 0u); atomicMin(&counters[countersOffset + 26u], secondOrdered); atomicMax(&counters[countersOffset + 27u], secondOrdered);' : ''}
 if (value < 0.0) { atomicAdd(&counters[countersOffset + 1u], 1u); }
 if (value != 0.0) { atomicAdd(&counters[countersOffset + 10u], 1u); }
}`
    );
  for (const [variable, view] of [values, secondValues ?? values].entries()) {
    node(
      `magnitude-${variable}`,
      rows,
      [
        bind('values', view),
        bind('analysisStates', analysisStates),
        bind('temporary', temporary, 'read_write')
      ],
      'temporary[temporaryOffset + index] = select(0.0, abs(values[valuesOffset + index]), analysisStates[analysisStatesOffset + index] != 0u);'
    );
    const maximum = graph.createDataView(control.buffer, {
      format: 'float32',
      byteOffset: (17 + variable) * 4,
      length: 1
    });
    nodes.push(
      ...new GPUReduction({
        id: `${id}-maximum-${variable}`,
        input: temporary,
        output: maximum,
        operation: 'max'
      }).getCommandNodes(graph)
    );
  }
  node(
    'scales',
    1,
    [bind('counters', control, 'read_write')],
    `for (var variable = 0u; variable < 2u; variable++) {
 let magnitude = bitcast<f32>(counters[countersOffset + 17u + variable]);
 let exponent = select(0, clamp(i32((bitcast<u32>(magnitude) >> 23u) & 255u) - 126, -126, 126), magnitude > 0.0);
 counters[countersOffset + 5u + variable * 2u] = u32(127 - exponent) << 23u;
 counters[countersOffset + 6u + variable * 2u] = u32(127 + exponent) << 23u;
}`
  );
  reduce(
    'means',
    0,
    rows,
    [1, 2],
    [
      bind('values', values),
      ...(secondValues ? [bind('secondValues', secondValues)] : []),
      bind('analysisStates', analysisStates)
    ],
    `if (analysisStates[analysisStatesOffset + index] != 0u) { contribution[0] = values[valuesOffset + index] * readControlFloat(5u); contribution[1] = ${secondValues ? 'secondValues[secondValuesOffset + index] * readControlFloat(7u)' : '0.0'}; }`
  );
  const totalDeclarations = `fn readTotal(column: u32) -> f32 { return totals[totalsOffset + column * 2u] + totals[totalsOffset + column * 2u + 1u]; }
fn readMean(column: u32, count: u32) -> f32 {
 let high = totals[totalsOffset + column * 2u]; let low = totals[totalsOffset + column * 2u + 1u];
 let quotient = divideBySpatialCount(high, count);
 if (count <= 16777216u) { return quotient + (fma(-quotient, f32(count), high) + low) / f32(count); }
 return quotient + divideBySpatialCount(low, count);
}`;
  node(
    'mean-control',
    1,
    [bind('totals', totals), bind('counters', control, 'read_write')],
    `let count = counters[countersOffset]; counters[countersOffset + 13u] = bitcast<u32>(readMean(1u, count)); counters[countersOffset + 15u] = bitcast<u32>(readMean(2u, count));`,
    totalDeclarations
  );
  node(
    'deviate',
    rows,
    [
      bind('values', values),
      ...(secondValues ? [bind('secondValues', secondValues)] : []),
      bind('analysisStates', analysisStates),
      bind('centeredX', centeredX, 'read_write'),
      bind('centeredY', centeredY, 'read_write')
    ],
    `let included = analysisStates[analysisStatesOffset + index] != 0u;
centeredX[centeredXOffset + index] = select(getQuietNaN(index), values[valuesOffset + index] * readControlFloat(5u) - readControlFloat(13u), included);
centeredY[centeredYOffset + index] = select(getQuietNaN(index), ${secondValues ? 'secondValues[secondValuesOffset + index] * readControlFloat(7u) - readControlFloat(15u)' : '0.0'}, included);`
  );
  reduce(
    'mean-correction',
    0,
    rows,
    [25, 26],
    [bind('centeredX', centeredX), bind('centeredY', centeredY)],
    `let x = centeredX[centeredXOffset + index]; if (isFiniteFloat(x)) { contribution[0] = x; contribution[1] = centeredY[centeredYOffset + index]; }`
  );
  node(
    'correction-control',
    1,
    [bind('totals', totals), bind('counters', control, 'read_write')],
    `let count = counters[countersOffset]; counters[countersOffset + 14u] = bitcast<u32>(readMean(25u, count)); counters[countersOffset + 16u] = bitcast<u32>(readMean(26u, count));`,
    totalDeclarations
  );
  node(
    'center',
    rows,
    [bind('centeredX', centeredX, 'read_write'), bind('centeredY', centeredY, 'read_write')],
    'centeredX[centeredXOffset + index] = select(centeredX[centeredXOffset + index] - readControlFloat(14u), 0.0, getStatisticsControl(24u) == getStatisticsControl(25u) && isFiniteFloat(centeredX[centeredXOffset + index])); centeredY[centeredYOffset + index] = select(centeredY[centeredYOffset + index] - readControlFloat(16u), 0.0, getStatisticsControl(26u) == getStatisticsControl(27u) && isFiniteFloat(centeredY[centeredYOffset + index]));'
  );
  reduce(
    'moments',
    0,
    rows,
    [3, 4, 5, 6],
    [bind('centeredX', centeredX), bind('centeredY', centeredY)],
    `let x = centeredX[centeredXOffset + index]; if (isFiniteFloat(x)) { let square = x * x; let y = centeredY[centeredYOffset + index]; contribution[0] = square; contribution[1] = square * x; contribution[2] = square * square; contribution[3] = y * y; }`
  );
  for (const [chunkIndex] of dispatches.entries())
    rowNode(
      'row-scale',
      chunkIndex,
      [
        ...weightBindings(chunkIndex),
        bind('analysisStates', analysisStates),
        bind('rowScale', rowScale, 'read_write')
      ],
      `var scale = 0.0; if (analysisStates[analysisStatesOffset + index] != 0u) {
 var sum = vec2<f32>(0.0); ${neighborLoop('sum = addCompensated(sum, weight);', 'analysisStates[analysisStatesOffset + neighbor] != 0u')}
 scale = ${isGPUSpatialWeightsRowStandardized(props.weights) ? 'select(-1.0, 1.0 / (sum.x + sum.y), sum.x + sum.y > 0.0)' : '1.0'};
} rowScale[rowScaleOffset + index] = scale;`
    );
  const symmetryTranspose = useTranspose
    ? {
        offsets: scratch('symmetry-offsets', 'uint32', rows + 1),
        neighbors: scratch('symmetry-neighbors', 'uint32', capacity),
        weights: scratch('symmetry-weights', 'float32', capacity)
      }
    : undefined;
  if (symmetryTranspose)
    nodes.push(
      ...new GPUChunkedSpatialWeightsTranspose({
        id: `${id}-symmetry-transpose`,
        weights: props.weights as GPUChunkedSpatialWeights,
        output: symmetryTranspose
      }).getCommandNodes(graph)
    );
  const symmetryTargets = symmetryTranspose
    ? [{rowStart: 0, rowCount: rows, weights: symmetryTranspose}]
    : dispatches;
  for (const [chunkIndex, dispatch] of dispatches.entries()) {
    reduce(
      `pairs-${chunkIndex}`,
      dispatch.rowStart,
      dispatch.rowCount,
      [8, 10, 11, 12, 13, 14],
      [
        ...weightBindings(chunkIndex),
        bind('centeredX', centeredX),
        bind('centeredY', centeredY),
        bind('values', values),
        bind('rowScale', rowScale)
      ],
      `
if (rowScale[rowScaleOffset + index] != 0.0) {
 let x = centeredX[centeredXOffset + index]; let scale = max(0.0, rowScale[rowScaleOffset + index]);
 var sums: array<vec2<f32>, 5>; var degree = 0u;
 ${neighborLoop(`let scaled = weight * scale; let neighborX = centeredX[centeredXOffset + neighbor]; let difference = x - neighborX;
 sums[0] = addCompensated(sums[0], scaled);
 sums[1] = addCompensated(sums[1], scaled * x * neighborX);
 sums[2] = addCompensated(sums[2], scaled * difference * difference);
 sums[3] = addCompensated(sums[3], scaled * (values[valuesOffset + index] * readControlFloat(5u)) * (values[valuesOffset + neighbor] * readControlFloat(5u)));
 sums[4] = addCompensated(sums[4], scaled * x * centeredY[centeredYOffset + neighbor]); degree++;`)}
 for (var field = 0u; field < 5u; field++) { contribution[field] = sums[field].x + sums[field].y; }
 contribution[5] = f32(degree);
}`
    );
    collectiveRowNode(
      'joins',
      chunkIndex,
      [
        ...weightBindings(chunkIndex),
        bind('values', values),
        bind('rowScale', rowScale),
        bind('counters', control, 'read_write', 'atomic<u32>')
      ],
      `var blackBlack = 0u; var blackWhite = 0u; var whiteWhite = 0u;
if (rowScale[rowScaleOffset + index] != 0.0) { let black = values[valuesOffset + index] != 0.0;
 ${neighborLoop('let neighborBlack = values[valuesOffset + neighbor] != 0.0; if (black && neighborBlack) { blackBlack++; } else if (black != neighborBlack) { blackWhite++; } else { whiteWhite++; }')}
} atomicAdd(&counters[countersOffset + 2u], blackBlack); atomicAdd(&counters[countersOffset + 3u], blackWhite); atomicAdd(&counters[countersOffset + 4u], whiteWhite);`
    );
    for (const [targetIndex, target] of symmetryTargets.entries()) {
      reduce(
        `symmetry-${chunkIndex}-${targetIndex}`,
        dispatch.rowStart,
        dispatch.rowCount,
        [9, 15],
        [
          ...weightBindings(chunkIndex),
          bind('targetOffsets', target.weights.offsets),
          bind('targetNeighbors', target.weights.neighbors),
          bind('targetWeights', target.weights.weights),
          bind('rowScale', rowScale)
        ],
        `
if (rowScale[rowScaleOffset + index] != 0.0) {
 var symmetry = vec2<f32>(0.0); var binarySymmetry = 0u;
 ${neighborLoop(`if (neighbor >= getChunkParameter(39u) && neighbor - getChunkParameter(39u) < getChunkParameter(40u)) {
  let targetRow = select(neighbor - getChunkParameter(39u), index, getChunkParameter(41u) != 0u); let targetColumn = select(index, neighbor, getChunkParameter(41u) != 0u); let targetCapacity = arrayLength(&targetNeighbors) - targetNeighborsOffset;
  var low = min(targetOffsets[targetOffsetsOffset + targetRow], targetCapacity); let end = min(targetOffsets[targetOffsetsOffset + targetRow + 1u], targetCapacity); var high = end;
  while (low < high) { let middle = low + (high - low) / 2u; if (targetNeighbors[targetNeighborsOffset + middle] < targetColumn) { low = middle + 1u; } else { high = middle; } }
  var reverse = -1.0; if (low < end && targetNeighbors[targetNeighborsOffset + low] == targetColumn) { let value = targetWeights[targetWeightsOffset + low]; if (activeWeight(value)) { reverse = value * max(0.0, rowScale[rowScaleOffset + neighbor]); } }
  let scaled = weight * max(0.0, rowScale[rowScaleOffset + index]);
  symmetry = addCompensated(symmetry, select(scaled * scaled, 0.5 * (scaled + reverse) * (scaled + reverse), reverse >= 0.0)); binarySymmetry += select(1u, 2u, reverse > 0.0);
 }`)}
 contribution[0] = symmetry.x + symmetry.y; contribution[1] = f32(binarySymmetry);
}`,
        '',
        [target.rowStart, target.rowCount, Number(useTranspose)]
      );
    }
    const chunkCapacity = dispatch.weights.neighbors.length;
    if (chunkCapacity === 0) continue;
    const slotRows = scratch(`slot-rows-${chunkIndex}`, 'uint32', chunkCapacity);
    const keys = scratch(`keys-${chunkIndex}`, 'uint32', chunkCapacity);
    const slotIdentifiers = scratch(`slot-identifiers-${chunkIndex}`, 'uint32', chunkCapacity);
    const columnOffsets = scratch(`column-offsets-${chunkIndex}`, 'uint32', rows + 1);
    rowNode(
      'slot-rows',
      chunkIndex,
      [
        ...weightBindings(chunkIndex).slice(0, 2),
        bind('rowScale', rowScale),
        bind('slotRows', slotRows, 'read_write')
      ],
      `let source = select(${rows}u, index, rowScale[rowScaleOffset + index] != 0.0);
for (var slot = min(offsets[offsetsOffset + index - ROW_START], CAPACITY); slot < min(offsets[offsetsOffset + index - ROW_START + 1u], CAPACITY); slot++) { slotRows[slotRowsOffset + slot] = source; }`
    );
    node(
      `transpose-keys-${chunkIndex}`,
      chunkCapacity,
      [
        ...weightBindings(chunkIndex),
        bind('rowScale', rowScale),
        bind('slotRows', slotRows),
        bind('keys', keys, 'read_write'),
        bind('slotIdentifiers', slotIdentifiers, 'read_write')
      ],
      `let used = index < min(offsets[offsetsOffset + ROW_COUNT], CAPACITY); var source = ${rows}u; if (used) { source = slotRows[slotRowsOffset + index]; } let neighbor = neighbors[neighborsOffset + index]; var key = ${rows}u;
if (source < ${rows}u && neighbor < ${rows}u && neighbor != source && activeWeight(weights[weightsOffset + index])) { if (rowScale[rowScaleOffset + neighbor] != 0.0) { key = neighbor; } }
keys[keysOffset + index] = key; slotIdentifiers[slotIdentifiersOffset + index] = index;`,
      '',
      [dispatch.rowStart, dispatch.rowCount]
    );
    const sortedKeys = scratch(`sorted-keys-${chunkIndex}`, 'uint32', chunkCapacity);
    const sortedSlots = scratch(`sorted-slots-${chunkIndex}`, 'uint32', chunkCapacity);
    nodes.push(
      ...new GPUSort({
        id: `${id}-transpose-${chunkIndex}-sort`,
        keys,
        values: slotIdentifiers,
        outputKeys: sortedKeys,
        outputValues: sortedSlots,
        keyBits: getSortKeyBits(rows)
      }).getCommandNodes(graph)
    );
    node(
      `column-offsets-${chunkIndex}`,
      rows + 1,
      [bind('sortedKeys', sortedKeys), bind('columnOffsets', columnOffsets, 'read_write')],
      `var low = 0u; var high = arrayLength(&sortedKeys) - sortedKeysOffset;
while (low < high) { let middle = low + (high - low) / 2u; if (sortedKeys[sortedKeysOffset + middle] < index) { low = middle + 1u; } else { high = middle; } }
columnOffsets[columnOffsetsOffset + index] = low;`
    );
    const incomingBindings = [
      bind('columnOffsets', columnOffsets),
      bind('sortedSlots', sortedSlots),
      bind('slotRows', slotRows),
      bind('weights', dispatch.weights.weights),
      bind('rowScale', rowScale)
    ];
    const incomingLoop = (body: string) =>
      `for (var position = columnOffsets[columnOffsetsOffset + index]; position < columnOffsets[columnOffsetsOffset + index + 1u]; position++) { let slot = sortedSlots[sortedSlotsOffset + position]; let source = slotRows[slotRowsOffset + slot]; let weight = weights[weightsOffset + slot] * max(0.0, rowScale[rowScaleOffset + source]); ${body} }`;
    node(
      `incoming-weight-${chunkIndex}`,
      rows,
      [
        ...incomingBindings,
        bind('incomingWeight', incomingWeight, 'read_write'),
        bind('incomingWeightError', incomingWeightError, 'read_write'),
        bind('incomingDegree', incomingDegree, 'read_write')
      ],
      `var pair = vec2<f32>(incomingWeight[incomingWeightOffset + index], incomingWeightError[incomingWeightErrorOffset + index]); var degree = incomingDegree[incomingDegreeOffset + index]; ${incomingLoop('pair = addCompensated(pair, weight); degree++;')} incomingWeight[incomingWeightOffset + index] = pair.x; incomingWeightError[incomingWeightErrorOffset + index] = pair.y; incomingDegree[incomingDegreeOffset + index] = degree;`
    );
    node(
      `incoming-lag-${chunkIndex}`,
      rows,
      [
        ...incomingBindings,
        bind('centeredX', centeredX),
        bind('incomingLag', incomingLag, 'read_write'),
        bind('incomingLagError', incomingLagError, 'read_write')
      ],
      `var pair = vec2<f32>(incomingLag[incomingLagOffset + index], incomingLagError[incomingLagErrorOffset + index]); ${incomingLoop('pair = addCompensated(pair, weight * centeredX[centeredXOffset + source]);')} incomingLag[incomingLagOffset + index] = pair.x; incomingLagError[incomingLagErrorOffset + index] = pair.y;`
    );
  }
  node(
    'incoming-round',
    rows,
    [
      bind('incomingWeight', incomingWeight, 'read_write'),
      bind('incomingWeightError', incomingWeightError),
      bind('incomingLag', incomingLag, 'read_write'),
      bind('incomingLagError', incomingLagError)
    ],
    'incomingWeight[incomingWeightOffset + index] += incomingWeightError[incomingWeightErrorOffset + index]; incomingLag[incomingLagOffset + index] += incomingLagError[incomingLagErrorOffset + index];'
  );
  for (const deviationPass of [false, true]) {
    for (const [chunkIndex, dispatch] of dispatches.entries())
      reduce(
        `${deviationPass ? 'degree-deviations' : 'degrees'}-${chunkIndex}`,
        dispatch.rowStart,
        dispatch.rowCount,
        deviationPass ? [22, 23, 24] : [16, 17, 18, 20, 21],
        [
          ...weightBindings(chunkIndex),
          bind('rowScale', rowScale),
          bind('incomingWeight', incomingWeight),
          bind('incomingDegree', incomingDegree),
          bind('incomingLag', incomingLag)
        ],
        `if (rowScale[rowScaleOffset + index] != 0.0) {
 var outgoing = vec2<f32>(0.0); var degree = 0u; ${neighborLoop('outgoing = addCompensated(outgoing, weight * max(0.0, rowScale[rowScaleOffset + index])); degree++;')}
 let weightedDegree = outgoing.x + outgoing.y + incomingWeight[incomingWeightOffset + index]; let binaryDegree = f32(degree + incomingDegree[incomingDegreeOffset + index]); let lag = incomingLag[incomingLagOffset + index];
 ${deviationPass ? 'let deviation = weightedDegree - readControlFloat(19u); let binaryDeviation = binaryDegree - readControlFloat(20u); let lagDeviation = lag - readControlFloat(21u); contribution[0] = deviation * deviation; contribution[1] = binaryDeviation * binaryDeviation; contribution[2] = lagDeviation * lagDeviation;' : 'contribution[0] = weightedDegree; contribution[1] = binaryDegree; contribution[2] = lag; contribution[3] = weightedDegree * weightedDegree; contribution[4] = binaryDegree * binaryDegree;'}
}`
      );
    if (!deviationPass)
      node(
        'degree-control',
        1,
        [bind('totals', totals), bind('counters', control, 'read_write')],
        'let count = counters[countersOffset]; for (var column = 0u; column < 3u; column++) { counters[countersOffset + 19u + column] = bitcast<u32>(readMean(16u + column, count)); }',
        totalDeclarations
      );
  }
  node(
    'publish',
    1,
    [bind('totals', totals, 'read_write'), bind('counters', control)],
    'totals[totalsOffset + 7u * 2u] = f32(counters[countersOffset + 10u]); totals[totalsOffset + 19u * 2u] = f32(counters[countersOffset + 9u]);'
  );
  // Legacy field positions are unchanged; exact counts have their own additive uint32 port.
  const exact = props.output.counts;
  if (exact)
    node(
      'exact-counts',
      1,
      [bind('counters', control), bind('counts', exact, 'read_write')],
      'counts[countsOffset] = counters[countersOffset]; counts[countsOffset + 1u] = counters[countersOffset + 9u]; counts[countsOffset + 2u] = counters[countersOffset + 10u];'
    );
  node(
    'finalize-tiled',
    1,
    [
      bind('totals', totals),
      bind('counters', control),
      bind('results', props.output.summary, 'read_write'),
      ...(props.output.joinCounts
        ? [bind('joinCounts', props.output.joinCounts, 'read_write')]
        : []),
      ...(weightsStatus.incomplete ? [bind('overflow', weightsStatus.incomplete)] : [])
    ],
    finalizeBody,
    `${totalDeclarations}\n${finalizeDeclarations}`
  );
  return nodes;
}
