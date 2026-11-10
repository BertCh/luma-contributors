// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../raster-analysis/raster-band-utils';
import {
  validateCapacity,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  createTerrainValidityMask,
  getTerrainGroundMeasureWGSL,
  terrainBinding,
  validateTerrainValidityView,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

export type GPUTerrainDepressionHierarchyProps = {
  id?: string;
  /**
   * Grid. Stage areas and volumes are in squared affine units and elevation units times squared
   * affine units by default (geographic degrees are not ground meters), or in square meters under
   * `cellSizeMode`.
   */
  grid: GPURasterMetadata;
  /** Ground model for cell areas. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */
  cellSizeMode?: GPURasterCellSizeMode;
  elevation: GraphDataView<'float32'>;
  /** Optional per-cell validity of `elevation` (uint32, nonzero valid); a zero word counts as NaN, so the cell is a barrier. */
  validity?: GraphDataView<'uint32'>;
  /** Explicit ocean/drain cells. Masked/nodata cells are barriers, never implicit outlets. */
  outlets: GraphDataView<'uint32'>;
  /** Shared query stages in elevation units. */
  stages: GraphDataView<'float32'>;
  nodeCapacity: number;
  output: {
    /** Four uints per node: parent (0xffffffff root), minimum cell, lowest-ID outlet, valid. */
    nodes: GraphDataView<'uint32'>;
    /** Two floats per node: birth elevation, spill/merge elevation (Infinity for closed roots). */
    levels: GraphDataView<'float32'>;
    /** Each valid cell is owned exactly once by its activation node; parent queries include descendants. */
    cellNodes: GraphDataView<'uint32'>;
    /** Node-major, stage-major pairs [inundated area, volume]. */
    curves: GraphDataView<'float32'>;
    /** [stored node count, required node count, overflow]. Any overflow invalidates the whole hierarchy. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Stable GPU-sorted sublevel-set merge tree with compressed component roots and linear stage reductions.
 * The merge-tree build is serial (one GPU invocation walks the sorted cells with union-find), so its
 * time grows linearly with cell count; very large grids can approach GPU watchdog limits.
 */
export class GPUTerrainDepressionHierarchy implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainDepressionHierarchyProps) {
    this.id = props.id ?? 'terrain-depression-hierarchy';
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateViewLength(this.id, 'elevation', props.elevation, count);
    validateViewLength(this.id, 'outlets', props.outlets, count);
    validateViewLength(this.id, 'output.cellNodes', props.output.cellNodes, count);
    validateTerrainValidityView(this.id, props.validity, count);
    validateCapacity(this.id, 'nodeCapacity', props.nodeCapacity);
    validateViewLength(this.id, 'output.nodes', props.output.nodes, props.nodeCapacity * 4);
    validateViewLength(this.id, 'output.levels', props.output.levels, props.nodeCapacity * 2);
    validateViewLength(
      this.id,
      'output.curves',
      props.output.curves,
      props.nodeCapacity * props.stages.length * 2
    );
    validateViewLength(this.id, 'output.status', props.output.status, 3);
    checkTerrainViews(
      this.id,
      [props.elevation, props.outlets, props.stages, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    checkTerrainGraph(id, graph, [
      props.elevation,
      ...(props.validity ? [props.validity] : []),
      props.outlets,
      props.stages,
      ...Object.values(props.output)
    ]);
    const masked = createTerrainValidityMask(graph, {
      id,
      operation: 'GPUTerrainDepressionHierarchy',
      values: props.elevation,
      validity: props.validity
    });
    const elevation = masked.values;
    const isUniform = (props.cellSizeMode ?? 'uniform') === 'uniform';
    // Full internal capacity keeps required counts exact even when caller output is too small.
    const labels = createTransientView(graph, `${id}-labels`, 'uint32', count * 2);
    const tree = createTransientView(graph, `${id}-tree`, 'uint32', count * 4);
    const levels = createTransientView(graph, `${id}-levels`, 'float32', count * 2);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', count);
    const cells = createTransientView(graph, `${id}-cells`, 'uint32', count);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', count);
    const sortedCells = createTransientView(graph, `${id}-sorted-cells`, 'uint32', count);
    const curveCompensation = createTransientView(
      graph,
      `${id}-curve-compensation`,
      'float32',
      Math.max(1, props.nodeCapacity * props.stages.length * 2)
    );
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...masked.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-order-keys`,
        operation: 'GPUTerrainDepressionHierarchy',
        invocationCount: count,
        bindings: [
          terrainBinding('elevation', elevation),
          terrainBinding('keys', keys, 'read_write'),
          terrainBinding('cells', cells, 'read_write')
        ],
        declarations: TERRAIN_FINITE_WGSL,
        body: `let value = elevation[elevationOffset + index];
let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
keys[keysOffset + index] = select(0xffffffffu, select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u), finite(value));
cells[cellsOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-order`,
        keys,
        values: cells,
        outputKeys: sortedKeys,
        outputValues: sortedCells
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-build`,
        operation: 'GPUTerrainDepressionHierarchy',
        invocationCount: 1,
        declarations,
        bindings: [
          terrainBinding('elevation', elevation),
          terrainBinding('outlets', props.outlets),
          terrainBinding('labels', labels, 'read_write'),
          terrainBinding('tree', tree, 'read_write'),
          terrainBinding('levels', levels, 'read_write'),
          terrainBinding('ownership', props.output.cellNodes, 'read_write'),
          terrainBinding('status', props.output.status, 'read_write'),
          terrainBinding('order', sortedCells)
        ],
        body: `for (var cell = 0u; cell < ${count}u; cell++) { labels[labelsOffset + cell] = NO_CELL; ownership[ownershipOffset + cell] = NO_CELL; }
var nodeCount = 0u;
for (var iteration = 0u; iteration < ${count}u; iteration++) {
 let selected = order[orderOffset + iteration];
 let selectedHeight = elevation[elevationOffset + selected];
 if (!finite(selectedHeight)) { break; }
 var neighbors: array<u32, 8>; var neighborCount = 0u;
 for (var direction = 0u; direction < 8u; direction++) {
  let cell = getFlowReceiver(selected, 1u << direction); if (cell == NO_CELL) { continue; }
  var label = labels[labelsOffset + cell]; if (label == NO_CELL) { continue; }
  // Disjoint-set links can be compressed independently of historical hierarchy parents.
  var root = label;
  while (labels[labelsOffset + ${count}u + root] != root) { root = labels[labelsOffset + ${count}u + root]; }
  while (label != root) { let next = labels[labelsOffset + ${count}u + label]; labels[labelsOffset + ${count}u + label] = root; label = next; }
  labels[labelsOffset + cell] = root; label = root;
  var found = false; for (var entry = 0u; entry < neighborCount; entry++) { found = found || neighbors[entry] == label; }
  if (!found) { neighbors[neighborCount] = label; neighborCount++; }
 }
 var destination = NO_CELL;
 if (neighborCount == 1u) { destination = neighbors[0]; } else {
  destination = nodeCount; nodeCount++; labels[labelsOffset + ${count}u + destination] = destination;
  tree[treeOffset + destination * 4u] = NO_CELL;
  tree[treeOffset + destination * 4u + 1u] = selected;
  tree[treeOffset + destination * 4u + 2u] = NO_CELL;
  tree[treeOffset + destination * 4u + 3u] = 1u;
  levels[levelsOffset + destination * 2u] = selectedHeight;
  levels[levelsOffset + destination * 2u + 1u] = infinity();
  for (var entry = 0u; entry < neighborCount; entry++) {
   let child = neighbors[entry]; labels[labelsOffset + ${count}u + child] = destination;
   tree[treeOffset + child * 4u] = destination; levels[levelsOffset + child * 2u + 1u] = selectedHeight;
   let childPit = tree[treeOffset + child * 4u + 1u]; let currentPit = tree[treeOffset + destination * 4u + 1u];
   if (elevation[elevationOffset + childPit] < elevation[elevationOffset + currentPit] || (elevation[elevationOffset + childPit] == elevation[elevationOffset + currentPit] && childPit < currentPit)) { tree[treeOffset + destination * 4u + 1u] = childPit; }
   tree[treeOffset + destination * 4u + 2u] = min(tree[treeOffset + destination * 4u + 2u], tree[treeOffset + child * 4u + 2u]);
  }

 }
 labels[labelsOffset + selected] = destination; ownership[ownershipOffset + selected] = destination;
 if (outlets[outletsOffset + selected] != 0u) { tree[treeOffset + destination * 4u + 2u] = min(tree[treeOffset + destination * 4u + 2u], selected); }
}
status[statusOffset] = min(nodeCount, ${props.nodeCapacity}u); status[statusOffset + 1u] = nodeCount; status[statusOffset + 2u] = u32(nodeCount > ${props.nodeCapacity}u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUTerrainDepressionHierarchy',
        invocationCount: props.nodeCapacity,
        bindings: [
          terrainBinding('tree', tree),
          terrainBinding('sourceLevels', levels),
          terrainBinding('status', props.output.status),
          terrainBinding('nodes', props.output.nodes, 'read_write'),
          terrainBinding('levels', props.output.levels, 'read_write')
        ],
        declarations: TERRAIN_FINITE_WGSL,
        body: `let valid = status[statusOffset + 2u] == 0u && index < status[statusOffset];
for (var component = 0u; component < 4u; component++) { nodes[nodesOffset + index * 4u + component] = 0xffffffffu; }
levels[levelsOffset + index * 2u] = missing(); levels[levelsOffset + index * 2u + 1u] = missing();
nodes[nodesOffset + index * 4u + 3u] = 0u;
if (valid) {
 for (var component = 0u; component < 4u; component++) { nodes[nodesOffset + index * 4u + component] = tree[treeOffset + index * 4u + component]; }
 levels[levelsOffset + index * 2u] = sourceLevels[sourceLevelsOffset + index * 2u]; levels[levelsOffset + index * 2u + 1u] = sourceLevels[sourceLevelsOffset + index * 2u + 1u];
}`
      })
    ];
    if (props.stages.length > 0)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-curves`,
          operation: 'GPUTerrainDepressionHierarchy',
          invocationCount: props.stages.length,
          declarations: `${TERRAIN_FINITE_WGSL}
${isUniform ? '' : `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${getTerrainGroundMeasureWGSL(props.grid, props.cellSizeMode)}`}
// Recover positive-addition rounding error through integer significands. Unlike a floating
// two-sum transform, shader fast-math cannot algebraically eliminate this correction.
fn positiveAdditionError(first: f32, second: f32, sum: f32) -> f32 {
 if (first == 0.0 || second == 0.0 || !finite(sum)) { return 0.0; }
 let firstBits = bitcast<u32>(first); let secondBits = bitcast<u32>(second);
 let larger = max(firstBits, secondBits); let smaller = min(firstBits, secondBits);
 let largerExponent = max((larger >> 23u) & 255u, 1u);
 let smallerExponent = max((smaller >> 23u) & 255u, 1u);
 let largerMantissa = (larger & 0x7fffffu) | select(0u, 0x800000u, (larger & 0x7f800000u) != 0u);
 let smallerMantissa = (smaller & 0x7fffffu) | select(0u, 0x800000u, (smaller & 0x7f800000u) != 0u);
 let gap = largerExponent - smallerExponent;
 var aligned = 0u; var remainder = smallerMantissa;
 if (gap < 32u) { aligned = smallerMantissa >> gap; remainder = smallerMantissa & ((1u << gap) - 1u); }
 let sumBits = bitcast<u32>(sum); let sumExponent = max((sumBits >> 23u) & 255u, 1u);
 let sumMantissa = (sumBits & 0x7fffffu) | select(0u, 0x800000u, (sumBits & 0x7f800000u) != 0u);
 let difference = i32(largerMantissa + aligned) - i32(sumMantissa << (sumExponent - largerExponent));
 return f32(difference) * exp2(f32(i32(largerExponent) - 150)) + f32(remainder) * exp2(f32(i32(smallerExponent) - 150));
}
fn addCurve(offset: u32, area: f32, volume: f32) {
 let additions = vec2<f32>(area, volume);
 for (var component = 0u; component < 2u; component++) {
  let current = curves[curvesOffset + offset + component];
  let next = current + additions[component];
  compensation[compensationOffset + offset + component] += positiveAdditionError(current, additions[component], next);
  curves[curvesOffset + offset + component] = next;
 }
}`,
          bindings: [
            terrainBinding('elevation', elevation),
            terrainBinding('ownership', props.output.cellNodes),
            terrainBinding('nodes', props.output.nodes),
            terrainBinding('stages', props.stages),
            terrainBinding('status', props.output.status),
            terrainBinding('curves', props.output.curves, 'read_write'),
            terrainBinding('compensation', curveCompensation, 'read_write')
          ],
          body: `let stage = stages[stagesOffset + index];
let valid = status[statusOffset + 2u] == 0u && finite(stage);
let nodeCount = status[statusOffset];
for (var node = 0u; node < ${props.nodeCapacity}u; node++) {
 let offset = curvesOffset + (node * ${props.stages.length}u + index) * 2u;
 curves[offset] = select(missing(), 0.0, valid && node < nodeCount);
 curves[offset + 1u] = select(missing(), 0.0, valid && node < nodeCount);
 compensation[compensationOffset + (node * ${props.stages.length}u + index) * 2u] = 0.0;
 compensation[compensationOffset + (node * ${props.stages.length}u + index) * 2u + 1u] = 0.0;
}
if (!valid) { return; }
${isUniform ? `let uniformCellArea = ${getWGSLFloatLiteral(Math.abs(props.grid.affine[0] * props.grid.affine[4] - props.grid.affine[1] * props.grid.affine[3]))};` : ''}
for (var cell = 0u; cell < ${count}u; cell++) {
 ${isUniform ? 'let cellArea = uniformCellArea;' : 'let cellArea = getGroundCellArea(cell);'}
 let owner = ownership[ownershipOffset + cell];
 let depth = stage - elevation[elevationOffset + cell];
 if (owner < nodeCount && depth >= 0.0) {
  let offset = curvesOffset + (owner * ${props.stages.length}u + index) * 2u;
  addCurve(offset - curvesOffset, cellArea, depth * cellArea);
 }
}
// Parents are created after their children. Each exclusive cell contribution moves once per edge.
for (var node = 0u; node < nodeCount; node++) {
 let localOffset = (node * ${props.stages.length}u + index) * 2u;
 curves[curvesOffset + localOffset] += compensation[compensationOffset + localOffset];
 curves[curvesOffset + localOffset + 1u] += compensation[compensationOffset + localOffset + 1u];
 let parent = nodes[nodesOffset + node * 4u];
 if (parent < nodeCount) {
  let source = curvesOffset + (node * ${props.stages.length}u + index) * 2u;
  let destination = curvesOffset + (parent * ${props.stages.length}u + index) * 2u;
  addCurve(destination - curvesOffset, curves[source], curves[source + 1u]);
 }
}`
        })
      );
    return nodes;
  }
}
