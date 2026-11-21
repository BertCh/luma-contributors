// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';

const OPERATION = 'GPUNetworkIsochrones';

/** Raster accumulation rule of the splat: smallest or largest cost per pixel. */
export type GPUNetworkIsochronesMode = 'min' | 'max';

/**
 * Largest facility count of the facility-labelled raster: the low 8 bits of each accumulated pixel
 * word carry the facility row (255 is reserved), the high 24 bits the cost. @internal
 */
export const GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES = 255;

/** Encoded value of an untouched pixel for each mode. @internal */
export function getIsochronesEncodedInitialValue(mode: GPUNetworkIsochronesMode): number {
  return mode === 'min' ? 0xffffffff : 0;
}

/**
 * Writes the {@link GPUIsobands} parameter view from the isochrone parameters (cell sizes from the
 * extent and the raster size). @internal
 */
export function createIsochronesBandParametersNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    width: number;
    height: number;
    parameters: GraphDataView<'float32'>;
    bandParameters: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'band-parameters',
    bindings: [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'bandParams', view: props.bandParameters, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    declarations: `const WIDTH: f32 = ${props.width}.0;
const HEIGHT: f32 = ${props.height}.0;`,
    body: `bandParams[bandParamsOffset] = params[paramsOffset];
  bandParams[bandParamsOffset + 1u] = params[paramsOffset + 7u];
  bandParams[bandParamsOffset + 2u] = params[paramsOffset + 8u];
  bandParams[bandParamsOffset + 3u] = 0.0;
  bandParams[bandParamsOffset + 4u] = params[paramsOffset + 1u];
  bandParams[bandParamsOffset + 5u] = params[paramsOffset + 2u];
  bandParams[bandParamsOffset + 6u] = params[paramsOffset + 3u];
  bandParams[bandParamsOffset + 7u] = params[paramsOffset + 4u];
  bandParams[bandParamsOffset + 8u] = (params[paramsOffset + 3u] - params[paramsOffset + 1u]) / WIDTH;
  bandParams[bandParamsOffset + 9u] = (params[paramsOffset + 4u] - params[paramsOffset + 2u]) / HEIGHT;
  bandParams[bandParamsOffset + 10u] = 0.0;
  bandParams[bandParamsOffset + 11u] = 0.0;`
  });
}

/**
 * Writes the breaks {@link GPUIsobands} reads so that the isochrone bands are upper-inclusive.
 * `GPUIsobands` puts a value in band `k` when exactly `k` breaks are `<=` it, which is
 * `[b[k - 1], b[k])`. Each break is replaced by the next float32 above it, and "the next float
 * above `b` is `<=` the value" is the same test as "`b <` the value", so band `k` becomes
 * `(b[k - 1], b[k]]`: a cost exactly on a break is within it, as in `GPUNetworkReachability`
 * (review network-paths-07). A break of zero becomes the smallest normal float32 and NaN or
 * infinite breaks pass through. The contour geometry moves by one unit in the last place. @internal
 */
export function createIsochronesInclusiveBreaksNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    costBreaks: GraphDataView<'float32'>;
    inclusiveBreaks: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'inclusive-breaks',
    bindings: [
      {name: 'costBreaks', view: props.costBreaks, type: 'u32', access: 'read'},
      {name: 'inclusiveBreaks', view: props.inclusiveBreaks, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.costBreaks.length,
    body: `let bits = costBreaks[costBreaksOffset + index];
  let magnitude = bits & 0x7fffffffu;
  var nextBits = bits;
  if (magnitude == 0u) {
    // Subnormal comparisons are not portable, so zero steps to the smallest normal number.
    nextBits = 0x00800000u;
  } else if (magnitude < 0x7f800000u) {
    // Positive floats grow with their bits, negative floats shrink in magnitude toward zero.
    nextBits = select(bits + 1u, bits - 1u, (bits & 0x80000000u) != 0u);
  }
  inclusiveBreaks[inclusiveBreaksOffset + index] = nextBits;`
  });
}

/**
 * Max mode only: finds the reverse direction of the slot's road, a slot `target -> source` with a
 * valid weight and a finite target cost. Among several such slots (parallel roads share the
 * straight segment between their nodes) it keeps the costliest, which is exact when every parallel
 * road lists both directions with one weight. Declares `reverseWeight` (negative when the road is
 * one-way, a self-loop, or its far end is unreached) and `reverseCost`.
 */
const ISOCHRONES_REVERSE_DIRECTION_WGSL = /* wgsl */ `var reverseWeight = -1.0;
  let reverseCost = costs[costsOffset + targetNode];
  if (targetNode != source && isochronesIsFinite(reverseCost) && reverseCost >= 0.0) {
    let edgeCount = offsets[offsetsOffset + NODE_COUNT];
    let firstReverse = min(offsets[offsetsOffset + targetNode], edgeCount);
    let lastReverse = min(offsets[offsetsOffset + targetNode + 1u], edgeCount);
    for (var reverseSlot = firstReverse; reverseSlot < lastReverse; reverseSlot++) {
      if (neighbors[neighborsOffset + reverseSlot] != source) { continue; }
      let candidateWeight = weights[weightsOffset + reverseSlot];
      if (candidateWeight >= 0.0 && isochronesIsFinite(candidateWeight)) {
        reverseWeight = max(reverseWeight, candidateWeight);
      }
    }
  }`;

/**
 * One thread per edge: samples the edge at a bounded number of points, interpolates the cost
 * linearly from the source node cost along the edge weight, and applies `atomicMin` or `atomicMax`
 * to every pixel within the buffer of each sample. In max mode the cost of a sample is the cheaper
 * of the two directions of its road (`min(cost[u] + t w, cost[v] + (1 - t) wReverse)` when the
 * reverse slot exists), so the maximum runs over road samples and roads, never over the two
 * directions of one road (review network-paths-01). Costs are non-negative f32, so their bit
 * patterns order like integers; max mode stores `bits + 1` so zero means untouched. With
 * `assignments` (min mode) the low 8 bits of the word hold the facility row of the edge's source
 * node and the cost keeps its top 24 bits (relative precision 2^-15, rounded down), so the
 * smallest word is the smallest cost with the lowest facility row on ties. @internal
 */
export function createIsochronesSplatNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    width: number;
    height: number;
    mode: GPUNetworkIsochronesMode;
    maximumBufferPixels: number;
    maximumSamplesPerEdge: number;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    weights: GraphDataView<'float32'>;
    costs: GraphDataView<'float32'>;
    nodePositions: GraphDataView<'float32x2'>;
    parameters: GraphDataView<'float32'>;
    encoded: GraphDataView<'uint32'>;
    assignments?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  const labelled = Boolean(props.assignments);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `splat-${props.mode}${labelled ? '-labelled' : ''}`,
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: props.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'positions', view: props.nodePositions, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'encoded', view: props.encoded, type: 'atomic<u32>', access: 'read_write'},
      ...(props.assignments
        ? [{name: 'assignments', view: props.assignments, type: 'u32', access: 'read'} as const]
        : [])
    ],
    invocationCount: props.neighbors.length,
    declarations: `const NODE_COUNT: u32 = ${props.nodeCount}u;
const WIDTH: i32 = ${props.width};
const HEIGHT: i32 = ${props.height};
const MAXIMUM_BUFFER_PIXELS: u32 = ${props.maximumBufferPixels}u;
const MAXIMUM_SAMPLES: u32 = ${props.maximumSamplesPerEdge}u;
const COST_CEILING: f32 = ${getWGSLFloatLiteral(3.0e38)};

${IS_FINITE_BITS_WGSL}
fn isochronesIsFinite(value: f32) -> bool {
  return isFiniteBits(value) && abs(value) < COST_CEILING;
}

fn isochronesFindSource(edge: u32) -> u32 {
  var low = 0u;
  var high = NODE_COUNT;
  while (low + 1u < high) {
    let middle = (low + high) >> 1u;
    if (offsets[offsetsOffset + middle] <= edge) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

fn isochronesGetPosition(node: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * node], positions[positionsOffset + 2u * node + 1u]);
}`,
    body: `if (index < offsets[offsetsOffset] || index >= offsets[offsetsOffset + NODE_COUNT]) { return; }
  let targetNode = neighbors[neighborsOffset + index];
  if (targetNode >= NODE_COUNT) { return; }
  let weight = weights[weightsOffset + index];
  let source = isochronesFindSource(index);
  let sourceCost = costs[costsOffset + source];
  ${
    labelled
      ? `let facility = assignments[assignmentsOffset + source];
  if (facility >= ${GPU_NETWORK_ISOCHRONES_MAXIMUM_RASTER_FACILITIES}u) { return; }`
      : ''
  }
  if (!(weight >= 0.0) || !isochronesIsFinite(weight) || !isochronesIsFinite(sourceCost) || sourceCost < 0.0) { return; }
  let startPoint = isochronesGetPosition(source);
  let endPoint = isochronesGetPosition(targetNode);
  if (!isochronesIsFinite(startPoint.x) || !isochronesIsFinite(startPoint.y) || !isochronesIsFinite(endPoint.x) || !isochronesIsFinite(endPoint.y)) { return; }
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f(
    (params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT)
  );
  let walk = max(params[paramsOffset + 6u], 0.0);
  let radius = max(max(params[paramsOffset + 5u], 0.0), 0.5 * length(cellSize));
  let reachX = i32(min(u32(ceil(radius / cellSize.x)), MAXIMUM_BUFFER_PIXELS));
  let reachY = i32(min(u32(ceil(radius / cellSize.y)), MAXIMUM_BUFFER_PIXELS));
  // Extent culling (review B-20): an edge, or a sample, farther than the splat reach from the
  // raster extent touches no pixel.
  let maximum = vec2f(params[paramsOffset + 3u], params[paramsOffset + 4u]);
  let reach = vec2f(f32(reachX), f32(reachY)) * abs(cellSize) + vec2f(radius);
  let extentLow = min(minimum, maximum) - reach;
  let extentHigh = max(minimum, maximum) + reach;
  if (any(max(startPoint, endPoint) < extentLow) || any(min(startPoint, endPoint) > extentHigh)) {
    return;
  }
  let spacing = 0.5 * radius;
  let intervals = ceil(length(endPoint - startPoint) / spacing);
  // Bound the float before conversion/addition: an overlong edge can exceed uint32.
  var sampleCount = MAXIMUM_SAMPLES;
  if (intervals < f32(MAXIMUM_SAMPLES - 1u)) {
    sampleCount = max(u32(intervals) + 1u, 2u);
  }
  ${isMin ? '' : ISOCHRONES_REVERSE_DIRECTION_WGSL}
  for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
    let t = f32(sampleIndex) / f32(sampleCount - 1u);
    let point = mix(startPoint, endPoint, vec2f(t));
    if (any(point < extentLow) || any(point > extentHigh)) { continue; }
    ${
      isMin
        ? 'let edgeCost = sourceCost + t * weight;'
        : `// The network cost of a road point is the cheaper of the two directions that reach it. The
    // reverse parameter is formed like the reverse slot's own t, so both slots of a road agree.
    var edgeCost = sourceCost + t * weight;
    if (reverseWeight >= 0.0) {
      let reverseT = f32(sampleCount - 1u - sampleIndex) / f32(sampleCount - 1u);
      edgeCost = min(edgeCost, reverseCost + reverseT * reverseWeight);
    }`
    }
    let centerPixel = vec2i(floor((point - minimum) / cellSize));
    for (var dy = -reachY; dy <= reachY; dy++) {
      let row = centerPixel.y + dy;
      if (row < 0 || row >= HEIGHT) { continue; }
      for (var dx = -reachX; dx <= reachX; dx++) {
        let column = centerPixel.x + dx;
        if (column < 0 || column >= WIDTH) { continue; }
        let pixelCenter = minimum + (vec2f(f32(column), f32(row)) + 0.5) * cellSize;
        let pixelDistance = length(pixelCenter - point);
        if (pixelDistance > radius) { continue; }
        let cost = abs(edgeCost + walk * pixelDistance);
        if (!isochronesIsFinite(cost)) { continue; }
        let slot = encodedOffset + u32(row * WIDTH + column);
        // Test-before-RMW: overlapping samples and edges rewrite most pixels with a no-op, and a
        // plain atomic load is far cheaper than a contended read-modify-write.
        ${
          labelled
            ? `let encodedValue = (bitcast<u32>(cost) & 0xffffff00u) | facility;
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`
            : isMin
              ? `let encodedValue = bitcast<u32>(cost);
        if (encodedValue < atomicLoad(&encoded[slot])) { atomicMin(&encoded[slot], encodedValue); }`
              : `let encodedValue = bitcast<u32>(cost) + 1u;
        if (encodedValue > atomicLoad(&encoded[slot])) { atomicMax(&encoded[slot], encodedValue); }`
        }
      }
    }
  }`
  });
}

/** Decodes the accumulated pixels to float32 costs, `unreachedCost` where nothing landed. @internal */
export function createIsochronesDecodeNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    mode: GPUNetworkIsochronesMode;
    unreachedCost: number;
    encoded: GraphDataView<'uint32'>;
    values: GraphDataView<'float32'>;
    pixelCount: number;
    /** Facility-labelled encoding (min mode): also writes the facility row per pixel. */
    pixelFacilities?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const isMin = props.mode === 'min';
  const labelled = Boolean(props.pixelFacilities);
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `decode-${props.mode}${labelled ? '-labelled' : ''}`,
    bindings: [
      {name: 'encoded', view: props.encoded, type: 'u32', access: 'read'},
      {name: 'values', view: props.values, type: 'f32', access: 'read_write'},
      ...(props.pixelFacilities
        ? [
            {
              name: 'facilitiesOut',
              view: props.pixelFacilities,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : [])
    ],
    invocationCount: props.pixelCount,
    declarations: `const UNREACHED_COST: f32 = ${getWGSLFloatLiteral(props.unreachedCost)};`,
    body: `let word = encoded[encodedOffset + index];
  ${
    labelled
      ? `values[valuesOffset + index] = select(bitcast<f32>(word & 0xffffff00u), UNREACHED_COST, word == 0xffffffffu);
  facilitiesOut[facilitiesOutOffset + index] = select(word & 0xffu, 0xffffffffu, word == 0xffffffffu);`
      : isMin
        ? 'values[valuesOffset + index] = select(bitcast<f32>(word), UNREACHED_COST, word == 0xffffffffu);'
        : 'values[valuesOffset + index] = select(bitcast<f32>(word - 1u), UNREACHED_COST, word == 0u);'
  }`
  });
}

/** One thread per node: finite nonnegative costs at or below the cell cost limit. @internal */
export function createIsochronesNodeMaskNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    costs: GraphDataView<'float32'>;
    parameters: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'node-mask',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.costs.length,
    declarations: IS_FINITE_BITS_WGSL,
    body: `let cost = costs[costsOffset + index];
  let isReached = isFiniteBits(cost) && cost >= 0.0 && abs(cost) < ${getWGSLFloatLiteral(3.0e38)} && cost <= params[paramsOffset + 9u];
  mask[maskOffset + index] = select(0u, 1u, isReached);`
  });
}

/**
 * Labels every table cell with the facility of its cheapest reached node. Two atomic passes over
 * the nodes: the minimum cost bits per cell, then the smallest facility row among nodes whose cost
 * equals that minimum, so ties resolve to the lowest facility row independent of thread order.
 * `nodeCells` rows that are zero (masked out) or absent from the table are ignored; table rows
 * past the cell count keep `0xffffffff`. @internal
 */
export function createIsochronesCellFacilityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    nodeCount: number;
    nodeCells: GraphDataView<'uint32x2'>;
    tableCells: GraphDataView<'uint32x2'>;
    tableCount: GraphDataView<'uint32'>;
    costs: GraphDataView<'float32'>;
    assignments: GraphDataView<'uint32'>;
    cellMinimumCosts: GraphDataView<'uint32'>;
    cellFacilities: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const declarations = `const NODE_COUNT: u32 = ${props.nodeCount}u;

// Row of the node's cell in the sorted table, or 0xffffffff.
fn isochronesFindCellRow(node: u32) -> u32 {
  let low = nodeCells[nodeCellsOffset + 2u * node];
  let high = nodeCells[nodeCellsOffset + 2u * node + 1u];
  if (low == 0u && high == 0u) {
    return 0xffffffffu;
  }
  var first = 0u;
  var last = tableCount[tableCountOffset];
  while (first < last) {
    let middle = (first + last) >> 1u;
    let rowLow = tableCells[tableCellsOffset + 2u * middle];
    let rowHigh = tableCells[tableCellsOffset + 2u * middle + 1u];
    if (rowHigh < high || (rowHigh == high && rowLow < low)) {
      first = middle + 1u;
    } else {
      last = middle;
    }
  }
  if (first < tableCount[tableCountOffset]
    && tableCells[tableCellsOffset + 2u * first] == low
    && tableCells[tableCellsOffset + 2u * first + 1u] == high) {
    return first;
  }
  return 0xffffffffu;
}`;
  const common = [
    {name: 'nodeCells', view: props.nodeCells, type: 'u32', access: 'read'},
    {name: 'tableCells', view: props.tableCells, type: 'u32', access: 'read'},
    {name: 'tableCount', view: props.tableCount, type: 'u32', access: 'read'},
    {name: 'costs', view: props.costs, type: 'f32', access: 'read'}
  ] as const;
  return [
    createFillNode<Parameters>(graph, {
      id: `${props.id}-minimum-fill`,
      operation: OPERATION,
      view: props.cellMinimumCosts,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createFillNode<Parameters>(graph, {
      id: `${props.id}-facility-fill`,
      operation: OPERATION,
      view: props.cellFacilities,
      type: 'u32',
      value: '0xffffffffu'
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-minimum`,
      operation: OPERATION,
      variant: 'cell-minimum-cost',
      bindings: [
        ...common,
        {
          name: 'cellMinimum',
          view: props.cellMinimumCosts,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: props.nodeCount,
      declarations,
      // Non-negative f32 bits order like their values; -0 folds to +0 and negative or NaN costs
      // are skipped, the same rule as the splat path and reachability.
      body: `let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0) {
    atomicMin(&cellMinimum[cellMinimumOffset + row], select(bitcast<u32>(cost), 0u, cost == 0.0));
  }`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${props.id}-facility`,
      operation: OPERATION,
      variant: 'cell-facility',
      bindings: [
        ...common,
        {name: 'assignments', view: props.assignments, type: 'u32', access: 'read'},
        {name: 'cellMinimum', view: props.cellMinimumCosts, type: 'u32', access: 'read'},
        {
          name: 'cellFacility',
          view: props.cellFacilities,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: props.nodeCount,
      declarations,
      body: `let row = isochronesFindCellRow(index);
  let cost = costs[costsOffset + index];
  if (row != 0xffffffffu && cost >= 0.0 &&
      select(bitcast<u32>(cost), 0u, cost == 0.0) == cellMinimum[cellMinimumOffset + row]) {
    atomicMin(&cellFacility[cellFacilityOffset + row], assignments[assignmentsOffset + index]);
  }`
    })
  ];
}

type TriangleFacilityProps = {
  id: string;
  width: number;
  height: number;
  triangles: GraphDataView<'float32x2'>;
  triangleCount: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  pixelFacilities: GraphDataView<'uint32'>;
  triangleFacilities: GraphDataView<'uint32'>;
  /** Optional internal search-work diagnostic, one evaluated-leaf count per triangle slot. */
  visitedSamples?: GraphDataView<'uint32'>;
};

function getTriangleFacilityLevels(width: number, height: number) {
  const levels = [{width, height, offset: 0}];
  while (width > 1 || height > 1) {
    const previous = levels[levels.length - 1];
    width = Math.ceil(width / 2);
    height = Math.ceil(height / 2);
    levels.push({
      width,
      height,
      offset: previous.offset + (levels.length === 1 ? 0 : previous.width * previous.height)
    });
  }
  return levels;
}

/** Build a parallel min-label pyramid, then query its occupied spatial regions. @internal */
export function createIsochronesTriangleFacilityNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TriangleFacilityProps
): GPUCommandNode<Parameters>[] {
  const levels = getTriangleFacilityLevels(props.width, props.height);
  const root = levels[levels.length - 1];
  const hierarchy = createTransientView(graph, `${props.id}-hierarchy`, 'uint32', root.offset + 1);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let level = 1; level < levels.length; level++) {
    const previous = levels[level - 1];
    const current = levels[level];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${props.id}-hierarchy-${level}`,
        operation: OPERATION,
        variant: 'facility-hierarchy-parent',
        invocationCount: current.width * current.height,
        bindings: [
          {name: 'labels', view: hierarchy, type: 'u32', access: 'read_write'},
          ...(level === 1
            ? [
                {
                  name: 'pixels',
                  view: props.pixelFacilities,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : [])
        ],
        declarations: `fn childLabel(column: u32, row: u32) -> u32 {
  if (column >= ${previous.width}u || row >= ${previous.height}u) { return 0xffffffffu; }
  return ${level === 1 ? 'pixels[pixelsOffset' : `labels[labelsOffset + ${previous.offset}u`} + row * ${previous.width}u + column];
}`,
        body: `let column = 2u * (index % ${current.width}u);
  let row = 2u * (index / ${current.width}u);
  labels[labelsOffset + ${current.offset}u + index] = min(
    min(childLabel(column, row), childLabel(column + 1u, row)),
    min(childLabel(column, row + 1u), childLabel(column + 1u, row + 1u)));`
      })
    );
  }
  nodes.push(createIsochronesTriangleFacilityNode(graph, {...props, hierarchy}));
  return nodes;
}

/**
 * Exact spatial branch-and-bound, one invocation per triangle. Empty pyramid regions and regions
 * beyond the current nearest distance are pruned. Local vertex displacements are summed before
 * squaring: the common centroid divisor cancels, preserving ordinary fan ties and translations.
 * Without a hierarchy, spatial distance pruning still applies (the standalone internal helper).
 * Writes NONE for unlabelled rasters and inactive slots. @internal
 */
export function createIsochronesTriangleFacilityNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TriangleFacilityProps & {hierarchy?: GraphDataView<'uint32'>}
): GPUCommandNode<Parameters> {
  const levels = getTriangleFacilityLevels(props.width, props.height);
  const levelCount = levels.length;
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'triangle-facility',
    bindings: [
      {name: 'triangles', view: props.triangles, type: 'f32', access: 'read'},
      {name: 'triangleCount', view: props.triangleCount, type: 'u32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'pixels', view: props.pixelFacilities, type: 'u32', access: 'read'},
      ...(props.hierarchy
        ? [{name: 'labels', view: props.hierarchy, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'facilitiesOut', view: props.triangleFacilities, type: 'u32', access: 'read_write'},
      ...(props.visitedSamples
        ? [
            {
              name: 'visited',
              view: props.visitedSamples,
              type: 'u32' as const,
              access: 'read_write' as const
            }
          ]
        : [])
    ],
    invocationCount: props.triangleFacilities.length,
    declarations: `const WIDTH: u32 = ${props.width}u;
const HEIGHT: u32 = ${props.height}u;
const LEVEL_WIDTHS = array<u32, ${levelCount}>(${levels.map(level => `${level.width}u`).join(', ')});
const LEVEL_HEIGHTS = array<u32, ${levelCount}>(${levels.map(level => `${level.height}u`).join(', ')});
const LEVEL_OFFSETS = array<u32, ${levelCount}>(${levels.map(level => `${level.offset}u`).join(', ')});
fn regionLabel(column: u32, row: u32, level: u32) -> u32 {
  if (level == 0u) { return pixels[pixelsOffset + row * WIDTH + column]; }
  ${props.hierarchy ? 'return labels[labelsOffset + LEVEL_OFFSETS[level] + row * LEVEL_WIDTHS[level] + column];' : 'return 0u;'}
}`,
    body: `${props.visitedSamples ? 'visited[visitedOffset + index] = 0u;' : ''}
  if (index >= triangleCount[triangleCountOffset]) {
    facilitiesOut[facilitiesOutOffset + index] = 0xffffffffu;
    return;
  }
  let base = trianglesOffset + 6u * index;
  let minimum = vec2f(params[paramsOffset + 1u], params[paramsOffset + 2u]);
  let cellSize = vec2f((params[paramsOffset + 3u] - minimum.x) / f32(WIDTH),
    (params[paramsOffset + 4u] - minimum.y) / f32(HEIGHT));
  let scale = max(abs(cellSize.x), abs(cellSize.y));
  let relativeCellSize = cellSize / scale;
  let first = (vec2f(triangles[base], triangles[base + 1u]) - minimum) / scale;
  let second = (vec2f(triangles[base + 2u], triangles[base + 3u]) - minimum) / scale;
  let third = (vec2f(triangles[base + 4u], triangles[base + 5u]) - minimum) / scale;
  let vertexSum = first + second + third;
  var stack: array<vec3u, ${3 * levelCount + 1}>;
  stack[0] = vec3u(0u, 0u, ${levelCount - 1}u);
  var stackCount = 1u;
  var best = 0xffffffffu;
  var bestDistance = 0.0;
  var visits = 0u;
  while (stackCount > 0u) {
    stackCount--;
    let region = stack[stackCount];
    let label = regionLabel(region.x, region.y, region.z);
    if (label == 0xffffffffu) { continue; }
    let span = 1u << region.z;
    let lower = (vec2f(region.xy * span) + vec2f(0.5)) * relativeCellSize;
    let upper = (vec2f(min((region.xy + vec2u(1u)) * span, vec2u(WIDTH, HEIGHT))) - vec2f(0.5)) * relativeCellSize;
    // Expand the bound by 32 f32 epsilons of its operands. This can only defer pruning;
    // candidate comparisons below use no tolerance. It covers rounding/reassociation of sums.
    let guard = max(abs(vertexSum), max(abs(3.0 * lower), abs(3.0 * upper))) * 0.000003814697265625;
    let displacement = max(max(3.0 * lower - vertexSum, vertexSum - 3.0 * upper) - guard, vec2f(0.0));
    if (best != 0xffffffffu && dot(displacement, displacement) > bestDistance) { continue; }
    if (region.z == 0u) {
      visits++;
      let offset = (first - lower) + (second - lower) + (third - lower);
      let distance = dot(offset, offset);
      if (best == 0xffffffffu || distance < bestDistance || (distance == bestDistance && label < best)) {
        best = label;
        bestDistance = distance;
      }
    } else {
      let childLevel = region.z - 1u;
      let split = vec2f((region.xy * 2u + vec2u(1u)) * (span / 2u)) * relativeCellSize;
      let preferred = select(0u, 1u, vertexSum.x >= 3.0 * split.x) | select(0u, 2u, vertexSum.y >= 3.0 * split.y);
      // Push far quadrants first so the nearest child is popped first.
      for (var order = 0u; order < 4u; order++) {
        let quadrant = preferred ^ (3u - order);
        let child = region.xy * 2u + vec2u(quadrant & 1u, quadrant >> 1u);
        if (child.x < LEVEL_WIDTHS[childLevel] && child.y < LEVEL_HEIGHTS[childLevel]) {
          stack[stackCount] = vec3u(child, childLevel);
          stackCount++;
        }
      }
    }
  }
  facilitiesOut[facilitiesOutOffset + index] = best;
  ${props.visitedSamples ? 'visited[visitedOffset + index] = visits;' : ''}`
  });
}
