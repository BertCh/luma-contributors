// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createFixedPointClearNode,
  createFixedPointScaleNodes,
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../utils/fixed-point-accumulator';
import type {GPURasterMetadata} from '../types';
import {validateRasterGrid} from '../raster-band-utils';
import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterCount,
  validateRasterViews,
  validateRasterGraph
} from './raster-kernel-utils';

export type GPURasterRegionGraphProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Dense IDs 0..regionCount-1; other labels are background. */
  labels: GraphDataView<'uint32'>;
  values: GraphDataView<'float32'>;
  regionCount: number;
  /** Fixed single linkage (default) or one globally closest mean-updating merge per round. */
  mergeMode?: 'fixed' | 'adaptive';
  /** Merge mean-difference edges at or below this nonnegative threshold. */
  mergeThreshold: number;
  maximumIterations: number;
  output: {
    /** Symmetric dense regionCount squared matrix; four-neighbor shared-edge count. */
    adjacency: GraphDataView<'uint32'>;
    /** Symmetric dense mean-difference matrix; absent/undefined edges are NaN. */
    weights: GraphDataView<'float32'>;
    /** Original region to smallest connected source region after the bounded rounds. */
    mapping: GraphDataView<'uint32'>;
    /** One word, 1 if another fixed propagation or adaptive merge is possible, otherwise 0. */
    status: GraphDataView<'uint32'>;
  };
};
/** Four-neighbor region adjacency with bounded fixed or adaptive mean-difference merging. Explicit rounds use ping-pong graph scratch; no cross-workgroup barrier.
 * Labels remain source IDs; missing observations exclude only mean contributions. Complete
 * domains are required: tile seams must be reconciled before this contributor.
 * All adaptive merges recompute observation-count-weighted component means; the closest
 * eligible pair wins, with lexicographic source-root ties. One adaptive round performs one merge.
 */
export class GPURasterRegionGraph implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPURasterRegionGraphProps) {
    this.id = props.id ?? 'raster-region-graph';
    validateRasterGrid(this.id, props.grid);
    validateRasterCount(props.grid.width);
    validateRasterCount(props.grid.height);
    validateRasterCount(props.grid.width * props.grid.height, 0x3fffffff);
    validateRasterCount(props.regionCount, 256);
    validateRasterCount(props.maximumIterations, 256);
    assert(['fixed', 'adaptive'].includes(props.mergeMode ?? 'fixed'));
    assert(Number.isFinite(Math.fround(props.mergeThreshold)) && props.mergeThreshold >= 0);
    assert(
      props.labels.length >= props.grid.width * props.grid.height &&
        props.values.length >= props.grid.width * props.grid.height
    );
    assert(
      props.output.adjacency.length >= props.regionCount ** 2 &&
        props.output.weights.length >= props.regionCount ** 2 &&
        props.output.mapping.length >= props.regionCount &&
        props.output.status.length >= 1
    );
    validateRasterViews(this.id, [props.labels, props.values], Object.values(props.output));
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const count = props.grid.width * props.grid.height;
    const regions = props.regionCount;
    validateRasterGraph(id, graph, [props.labels, props.values], Object.values(props.output));
    const means = createTransientView(graph, `${id}-means`, 'float32', regions);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', regions);
    const mapping = createTransientView(graph, `${id}-mapping`, 'uint32', regions);
    const other = createTransientView(graph, `${id}-other`, 'uint32', regions);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-clear`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions * regions,
        bindings: [rasterBinding('adjacency', props.output.adjacency, true)],
        body: 'adjacency[adjacencyOffset + index] = 0u;'
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-boundaries`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count,
        bindings: [
          rasterBinding('labels', props.labels),
          {
            name: 'adjacency',
            view: props.output.adjacency,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        body: `let label = labels[labelsOffset + index]; if (label >= ${regions}u) { return; }
let neighbors = array<u32, 2>(index + 1u, index + ${props.grid.width}u);
for (var direction = 0u; direction < 2u; direction++) { if (direction == 0u && index % ${props.grid.width}u == ${props.grid.width - 1}u) { continue; } let neighbor = neighbors[direction]; if (neighbor >= ${count}u) { continue; } let other = labels[labelsOffset + neighbor]; if (other < ${regions}u && other != label) { atomicAdd(&adjacency[adjacencyOffset + label * ${regions}u + other], 1u); atomicAdd(&adjacency[adjacencyOffset + other * ${regions}u + label], 1u); } }`
      })
    );
    // Per-pixel exact fixed-point accumulation: O(N) total work, order independent, no per-region full scans.
    const scale = createTransientView(graph, `${id}-scale`, 'uint32', 1);
    const sums = createTransientView(
      graph,
      `${id}-sums`,
      'uint32',
      regions * FIXED_POINT_WORD_COUNT
    );
    nodes.push(
      ...createFixedPointScaleNodes(graph, {
        id: `${id}-means-scale`,
        operation: 'GPURasterRegionGraph',
        values: props.values,
        scale
      }),
      createFixedPointClearNode(graph, {
        id: `${id}-means-clear-sums`,
        operation: 'GPURasterRegionGraph',
        words: sums
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means-clear-counts`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [rasterBinding('counts', counts, true)],
        body: 'counts[countsOffset + index] = 0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means-accumulate`,
        operation: 'GPURasterRegionGraph',
        invocationCount: count,
        bindings: [
          rasterBinding('labels', props.labels),
          rasterBinding('values', props.values),
          rasterBinding('scale', scale),
          {name: 'sums', view: sums, type: 'atomic<u32>', access: 'read_write'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}${getFixedPointAtomicAddWGSL('sums')}`,
        body: `let label = labels[labelsOffset + index]; let value = values[valuesOffset + index];
if (label >= ${regions}u || !finite(value)) { return; }
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], ${getFixedPointHeadroomBits(count)}u);
addFixedPointToSums(label, quantizeFixedPoint(value, unitExponent));
atomicAdd(&counts[countsOffset + label], 1u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-means`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('scale', scale),
          rasterBinding('sums', sums),
          rasterBinding('means', means, true),
          rasterBinding('counts', counts),
          rasterBinding('mapping', mapping, true)
        ],
        declarations: `${RASTER_FINITE_WGSL}${FIXED_POINT_WGSL}`,
        body: `let count = counts[countsOffset + index];
let unitExponent = getFixedPointUnitExponent(scale[scaleOffset], ${getFixedPointHeadroomBits(count)}u);
let words = vec3<u32>(sums[sumsOffset + index * 3u], sums[sumsOffset + index * 3u + 1u], sums[sumsOffset + index * 3u + 2u]);
var mean = getFixedPointFloat(words, unitExponent) / f32(max(count, 1u));
// A sum beyond the f32 range: decode 2^32 times smaller, divide, and scale back.
if (!finite(mean) && count > 0u) { mean = getFixedPointFloat(words, unitExponent - 32) / f32(count) * 4294967296.0; }
means[meansOffset + index] = select(missing(), mean, count > 0u && finite(mean)); mapping[mappingOffset + index] = index;`
      })
    );
    if (props.mergeMode === 'adaptive') {
      return [...nodes, ...getAdaptiveRegionNodes(graph, props, id, means, counts, mapping, other)];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-weights`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions * regions,
        bindings: [
          rasterBinding('adjacency', props.output.adjacency),
          rasterBinding('means', means),
          rasterBinding('weights', props.output.weights, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `let first = means[meansOffset + index / ${regions}u]; let second = means[meansOffset + index % ${regions}u]; weights[weightsOffset + index] = select(missing(), abs(first - second), adjacency[adjacencyOffset + index] > 0u && finite(first) && finite(second));`
      })
    );
    let source = mapping;
    let destination = other;
    for (let round = 0; round < props.maximumIterations; round++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-merge-${round}`,
          operation: 'GPURasterRegionGraph',
          invocationCount: regions,
          bindings: [
            rasterBinding('weights', props.output.weights),
            rasterBinding('source', source),
            rasterBinding('destination', destination, true)
          ],
          declarations: RASTER_FINITE_WGSL,
          body: `var selected = source[sourceOffset + index]; for (var neighbor = 0u; neighbor < ${regions}u; neighbor++) { let weight = weights[weightsOffset + index * ${regions}u + neighbor]; if (finite(weight) && weight <= ${props.mergeThreshold}) { selected = min(selected, source[sourceOffset + neighbor]); } } destination[destinationOffset + index] = selected;`
        })
      );
      [source, destination] = [destination, source];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPURasterRegionGraph',
        invocationCount: 1,
        bindings: [
          rasterBinding('weights', props.output.weights),
          rasterBinding('source', source),
          rasterBinding('mapping', props.output.mapping, true),
          rasterBinding('status', props.output.status, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var pending = 0u; for (var region = 0u; region < ${regions}u; region++) { let current = source[sourceOffset + region]; mapping[mappingOffset + region] = current; for (var neighbor = 0u; neighbor < ${regions}u; neighbor++) { let weight = weights[weightsOffset + region * ${regions}u + neighbor]; if (finite(weight) && weight <= ${props.mergeThreshold} && source[sourceOffset + neighbor] < current) { pending = 1u; } } } status[statusOffset] = pending;`
      })
    );
    return nodes;
  }
}

function getAdaptiveRegionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterRegionGraphProps,
  id: string,
  originalMeans: GraphDataView<'float32'>,
  originalCounts: GraphDataView<'uint32'>,
  initialMapping: GraphDataView<'uint32'>,
  temporaryMapping: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const regions = props.regionCount;
  const componentMeans = createTransientView(graph, `${id}-component-means`, 'float32', regions);
  const nodes: GPUCommandNode<Parameters>[] = [];
  let source = initialMapping;
  let destination = temporaryMapping;
  // Include a final diagnostic round which checks eligibility but does not apply another merge.
  for (let round = 0; round <= props.maximumIterations; round++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-adaptive-means-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: regions,
        bindings: [
          rasterBinding('source', source),
          rasterBinding('originalMeans', originalMeans),
          rasterBinding('originalCounts', originalCounts),
          rasterBinding('componentMeans', componentMeans, true)
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var count = 0u; var mean = 0.0;
for (var region = 0u; region < ${regions}u; region++) {
  let observations = originalCounts[originalCountsOffset + region];
  if (source[sourceOffset + region] != index || observations == 0u) { continue; }
  let combined = count + observations;
  mean = mean * (f32(count) / f32(combined)) + originalMeans[originalMeansOffset + region] * (f32(observations) / f32(combined)); count = combined;
}
componentMeans[componentMeansOffset + index] = select(missing(), mean, count > 0u);`
      })
    );
    const finalRound = round === props.maximumIterations;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-adaptive-merge-${round}`,
        operation: 'GPURasterRegionGraph',
        invocationCount: 1,
        bindings: [
          rasterBinding('source', source),
          rasterBinding('componentMeans', componentMeans),
          rasterBinding('adjacency', props.output.adjacency),
          rasterBinding('destination', finalRound ? props.output.mapping : destination, true),
          ...(finalRound
            ? [
                rasterBinding('weights', props.output.weights, true),
                rasterBinding('status', props.output.status, true)
              ]
            : [])
        ],
        declarations: RASTER_FINITE_WGSL,
        body: `var selectedFirst = 0xffffffffu; var selectedSecond = 0xffffffffu; var selectedWeight: f32 = ${props.mergeThreshold};
for (var first = 0u; first < ${regions}u; first++) {
  let firstRoot = source[sourceOffset + first];
  for (var second = 0u; second < ${regions}u; second++) {
    let secondRoot = source[sourceOffset + second]; let edge = first * ${regions}u + second;
    var weight = missing();
    if (firstRoot != secondRoot && adjacency[adjacencyOffset + edge] > 0u) { weight = abs(componentMeans[componentMeansOffset + firstRoot] - componentMeans[componentMeansOffset + secondRoot]); }
    ${finalRound ? 'weights[weightsOffset + edge] = weight;' : ''}
    let lower = min(firstRoot, secondRoot); let upper = max(firstRoot, secondRoot);
    if (finite(weight) && weight <= ${props.mergeThreshold} && (weight < selectedWeight || (weight == selectedWeight && (lower < selectedFirst || (lower == selectedFirst && upper < selectedSecond))))) { selectedWeight = weight; selectedFirst = lower; selectedSecond = upper; }
  }
}
for (var region = 0u; region < ${regions}u; region++) { let root = source[sourceOffset + region]; destination[destinationOffset + region] = ${finalRound ? 'root' : 'select(root, selectedFirst, root == selectedSecond)'}; }
${finalRound ? 'status[statusOffset] = select(0u, 1u, selectedFirst != 0xffffffffu);' : ''}`
      })
    );
    [source, destination] = [destination, source];
  }
  return nodes;
}
