// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createRasterCountNodes} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createTerrainBreachSearchNodes} from './terrain-breach-search';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  createTerrainValidityMask,
  terrainBinding,
  validateTerrainValidityView,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

/** Largest search scratch, in uint32 words: the shader indexes it with a u32. */
export const TERRAIN_BREACH_MAXIMUM_SCRATCH_WORDS = 0xffffffff;

export type GPUTerrainBreachProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Elevation in vertical units; the breach cuts vertically, so no horizontal unit or `cellSizeMode` applies. */
  elevation: GraphDataView<'float32'>;
  /**
   * Optional per-cell validity of `elevation` (uint32, nonzero valid). A zero word counts as NaN:
   * the cell cannot carry a breach and its output elevation is NaN.
   */
  validity?: GraphDataView<'uint32'>;
  /** Explicit candidate pits; no implicit filling or burning. Empty lists are supported. */
  pitCells: GraphDataView<'uint32'>;
  maximumSteps: number;
  maximumCutDepth: number;
  /** Positive representable decrease per path step in elevation units. */
  elevationDecrement: number;
  /** Optional explicit burn depth per cell, applied after breach cuts. */
  burnDepth?: GraphDataView<'float32'>;
  output: {
    elevation: GraphDataView<'float32'>;
    cutDepth: GraphDataView<'float32'>;
    /** Bit 1 breach, bit 2 burn, bit 4 invalid burn depth. */
    modificationMask: GraphDataView<'uint32'>;
    /** Pit-major paths, pit first, exactly pitCount * (maximumSteps + 1) capacity. */
    pathCells: GraphDataView<'uint32'>;
    pathLengths: GraphDataView<'uint32'>;
    /** Per pit: 0 invalid, 1 breached, 2 no feasible outlet within bounds. */
    pitStatus: GraphDataView<'uint32'>;
    unresolvedCount: GraphDataView<'uint32'>;
  };
};

/**
 * Bounded least total excavation-depth path on a step-layered D8 graph, with deterministic
 * simultaneous minimum-elevation edits. The search scratch holds `pitCells.length * (maximumSteps
 * + 1) * min(width, 2 * maximumSteps + 1) * min(height, 2 * maximumSteps + 1) * 2` words in one
 * storage binding; `getCommandNodes` rejects a request above the device binding limit (128 MiB by
 * default, about 15 pits at `maximumSteps` 64) with a message that names the scratch.
 */
export class GPUTerrainBreach implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUTerrainBreachProps) {
    this.id = props.id ?? 'terrain-breach';
    const count = validateRasterGrid(this.id, props.grid);
    validateViewLength(this.id, 'elevation', props.elevation, count);
    if (props.burnDepth) validateViewLength(this.id, 'burnDepth', props.burnDepth, count);
    validateTerrainValidityView(this.id, props.validity, count);
    validateCapacity(this.id, 'maximumSteps', props.maximumSteps);
    validateFiniteNumber(this.id, 'maximumCutDepth', props.maximumCutDepth, 0);
    validateRequirement(
      this.id,
      Number.isFinite(props.elevationDecrement) &&
        props.elevationDecrement > 0 &&
        Math.fround(props.elevationDecrement) > 0,
      'elevationDecrement must be a positive finite number that is above zero in float32'
    );
    validateViewLength(this.id, 'output.elevation', props.output.elevation, count);
    validateViewLength(this.id, 'output.cutDepth', props.output.cutDepth, count);
    validateViewLength(this.id, 'output.modificationMask', props.output.modificationMask, count);
    validateViewLength(
      this.id,
      'output.pathCells',
      props.output.pathCells,
      props.pitCells.length * (props.maximumSteps + 1)
    );
    validateViewLength(
      this.id,
      'output.pathLengths',
      props.output.pathLengths,
      props.pitCells.length
    );
    validateViewLength(this.id, 'output.pitStatus', props.output.pitStatus, props.pitCells.length);
    validateViewLength(this.id, 'output.unresolvedCount', props.output.unresolvedCount, 1);
    const searchCount =
      Math.min(props.grid.width, props.maximumSteps * 2 + 1) *
      Math.min(props.grid.height, props.maximumSteps * 2 + 1);
    validateMaximumCount(
      this.id,
      'search scratch words (pitCells x (maximumSteps + 1) x search window cells x 2)',
      props.pitCells.length * (props.maximumSteps + 1) * searchCount * 2,
      TERRAIN_BREACH_MAXIMUM_SCRATCH_WORDS
    );
    // Search uses two nodes per step; all other search, edit, mask and count passes use at most 15.
    this.estimatedCommandNodeCount = props.pitCells.length > 0 ? 15 + 2 * props.maximumSteps : 6;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    checkTerrainViews(
      this.id,
      [
        props.elevation,
        props.pitCells,
        ...(props.burnDepth ? [props.burnDepth] : []),
        ...(props.validity ? [props.validity] : [])
      ],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.elevation.length;
    const pitCount = props.pitCells.length;
    // A path with at most S eight-neighbor steps cannot leave this exact Chebyshev window.
    const searchWidth = Math.min(props.grid.width, props.maximumSteps * 2 + 1);
    const searchHeight = Math.min(props.grid.height, props.maximumSteps * 2 + 1);
    const searchCount = searchWidth * searchHeight;
    checkTerrainGraph(id, graph, [
      props.elevation,
      ...(props.validity ? [props.validity] : []),
      props.pitCells,
      ...(props.burnDepth ? [props.burnDepth] : []),
      ...Object.values(props.output)
    ]);
    const scratchWords = Math.max(1, pitCount * (props.maximumSteps + 1) * searchCount * 2);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'search scratch',
      scratchWords * 4,
      'reduce pitCells.length or maximumSteps'
    );
    const masked = createTerrainValidityMask(graph, {
      id,
      operation: 'GPUTerrainBreach',
      values: props.elevation,
      validity: props.validity
    });
    const elevation = masked.values;
    const edits = createTransientView(graph, `${id}-edits`, 'uint32', count);
    const decrement = getWGSLFloatLiteral(props.elevationDecrement);
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...masked.nodes,
      ...createTerrainBreachSearchNodes(graph, {id, props, elevation, searchWidth, searchHeight})
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-edit-initialize`,
        operation: 'GPUTerrainBreach',
        invocationCount: count,
        bindings: [
          terrainBinding('source', elevation),
          terrainBinding('edits', edits, 'read_write')
        ],
        body: `let bits = bitcast<u32>(source[sourceOffset + index]); edits[editsOffset + index] = select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-edit-proposals`,
        operation: 'GPUTerrainBreach',
        invocationCount: pitCount * (props.maximumSteps + 1),
        bindings: [
          terrainBinding('source', elevation),
          terrainBinding('paths', props.output.pathCells),
          terrainBinding('lengths', props.output.pathLengths),
          {name: 'edits', view: edits, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `let pitIndex = index / ${props.maximumSteps + 1}u; let step = index % ${props.maximumSteps + 1}u;
if (step == 0u || step >= lengths[lengthsOffset + pitIndex]) { return; }
let start = paths[pathsOffset + pitIndex * ${props.maximumSteps + 1}u];
let cell = paths[pathsOffset + index];
let value = source[sourceOffset + start] - f32(step) * ${decrement};
let bits = bitcast<u32>(value);
atomicMin(&edits[editsOffset + cell], select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u));`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-apply`,
        operation: 'GPUTerrainBreach',
        invocationCount: count,
        declarations,
        bindings: [
          terrainBinding('source', elevation),
          terrainBinding('edits', edits),
          terrainBinding('elevation', props.output.elevation, 'read_write'),
          terrainBinding('depth', props.output.cutDepth, 'read_write'),
          terrainBinding('mask', props.output.modificationMask, 'read_write'),
          ...(props.burnDepth ? [terrainBinding('burn', props.burnDepth)] : [])
        ],
        body: `let original = source[sourceOffset + index]; var edited = original; var flags = 0u;
if (isFiniteValue(original)) {
 let key = edits[editsOffset + index];
 edited = bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
 if (edited < original) { flags |= 1u; }
 ${props.burnDepth ? 'let burnDepth = burn[burnOffset + index]; if (!isFiniteValue(burnDepth) || burnDepth < 0.0) { edited = getNaN(); flags |= 4u; } else if (burnDepth > 0.0) { edited -= burnDepth; flags |= 2u; }' : ''}
}
elevation[elevationOffset + index] = edited; depth[depthOffset + index] = original - edited; mask[maskOffset + index] = flags;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-verify-paths`,
        operation: 'GPUTerrainBreach',
        invocationCount: pitCount,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('elevation', props.output.elevation),
          terrainBinding('mask', props.output.modificationMask),
          terrainBinding('paths', props.output.pathCells),
          terrainBinding('lengths', props.output.pathLengths),
          terrainBinding('status', props.output.pitStatus, 'read_write')
        ],
        body: `if (status[statusOffset + index] != 1u) { return; }
let pathLength = lengths[lengthsOffset + index];
for (var step = 1u; step < pathLength; step++) {
 let previous = paths[pathsOffset + index * ${props.maximumSteps + 1}u + step - 1u];
 let current = paths[pathsOffset + index * ${props.maximumSteps + 1}u + step];
 let previousElevation = elevation[elevationOffset + previous];
 let currentElevation = elevation[elevationOffset + current];
 if (((mask[maskOffset + previous] | mask[maskOffset + current]) & 4u) != 0u || !isFiniteValue(previousElevation) || !isFiniteValue(currentElevation) || !(currentElevation < previousElevation)) { status[statusOffset + index] = 2u; }
}`
      }),
      ...createRasterCountNodes(graph, {
        id: `${id}-status`,
        operation: 'GPUTerrainBreach',
        values: props.output.pitStatus,
        count: pitCount,
        condition: 'value != 1u',
        output: props.output.unresolvedCount
      })
    );
    return nodes;
  }
}
