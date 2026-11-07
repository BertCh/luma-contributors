// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {
  createRasterIterationState,
  createRasterIterationResetNode,
  createRasterGatedLoopNodes,
  validateRasterIterations
} from '../../gpu-raster/cost-distance/raster-relaxation';
import {createRasterCountNodes} from '../../gpu-raster/remote-sensing/raster-kernel-utils';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from '../hydrology/terrain-flow-tracing';
import {
  checkTerrainViews,
  checkTerrainGraph,
  terrainBinding,
  TERRAIN_FINITE_WGSL
} from './terrain-processing-utils';

export type GPUTerrainBranchingFlowProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Eight fractions per cell, east first clockwise. Positive rows must sum to one. Zero rows are outlets. */
  fractions: GraphDataView<'float32'>;
  /** Transport injection or outlet membership (0 or 1) for fractional catchments. */
  values: GraphDataView<'float32'>;
  mode:
    | 'transport'
    | 'expected-distance'
    | 'minimum-distance'
    | 'maximum-distance'
    | 'fractional-catchment';
  /** Optional eight edge lengths per cell. Defaults to affine planar distances. */
  edgeLengths?: GraphDataView<'float32'>;
  /** GPU sweeps; default clamps width + height to [128, 1024]. Maximum 1024. A sweep can finalize many dependency levels. */
  maximumIterations?: number;
  /** Bounded successor walk after each publication. Default 32. */
  maximumWalkLength?: number;
  output: {
    values: GraphDataView<'float32'>;
    /** 0 invalid local input, 1 resolved, 2 cycle, downstream failure or exhausted iteration budget. */
    status: GraphDataView<'uint32'>;
    unresolvedCount: GraphDataView<'uint32'>;
  };
};

/** Deterministic DAG pulls with atomic value/readiness publication and GPU-gated successor walks. */
export class GPUTerrainBranchingFlow implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUTerrainBranchingFlowProps) {
    this.id = props.id ?? 'terrain-branching-flow';
    const count = validateRasterGrid(this.id, props.grid);
    assert(props.fractions.length === count * 8 && props.values.length === count);
    assert(!props.edgeLengths || props.edgeLengths.length === count * 8);
    assert(
      props.output.values.length === count &&
        props.output.status.length === count &&
        props.output.unresolvedCount.length === 1
    );
    validateRasterIterations(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024)
    );
    assert(
      Number.isInteger(props.maximumWalkLength ?? 32) &&
        (props.maximumWalkLength ?? 32) > 0 &&
        (props.maximumWalkLength ?? 32) <= 1024
    );
    checkTerrainViews(
      this.id,
      [props.fractions, props.values, ...(props.edgeLengths ? [props.edgeLengths] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const count = props.values.length;
    checkTerrainGraph(id, graph, [
      props.fractions,
      props.values,
      ...(props.edgeLengths ? [props.edgeLengths] : []),
      ...Object.values(props.output)
    ]);
    const state = createTransientView(graph, `${id}-state`, 'uint32', count);
    const iterationState = createRasterIterationState(graph, id, 'GPUTerrainBranchingFlow', count);
    const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUTerrainBranchingFlow',
        invocationCount: count,
        bindings: [
          terrainBinding('fractions', props.fractions),
          terrainBinding('source', props.values),
          terrainBinding('state', state, 'read_write')
        ],
        declarations,
        body: `var valid = finite(source[sourceOffset + index]) && source[sourceOffset + index] >= 0.0;
var total = 0.0;
for (var direction = 0u; direction < 8u; direction++) {
 let fraction = fractions[fractionsOffset + index * 8u + direction];
 valid = valid && finite(fraction) && fraction >= 0.0;
 if (fraction > 0.0 && getFlowReceiver(index, 1u << direction) == NO_CELL) { valid = false; }
 total += fraction;
}
valid = valid && (total == 0.0 || abs(total - 1.0) <= 0.00001);
${props.mode === 'fractional-catchment' ? 'valid = valid && source[sourceOffset + index] <= 1.0;' : ''}
state[stateOffset + index] = select(0xfffffffeu, 0xffffffffu, valid);`
      })
    ];
    const transport = props.mode === 'transport';
    const affine = props.grid.affine.map(getWGSLFloatLiteral);
    const lengths = props.edgeLengths ? [terrainBinding('lengths', props.edgeLengths)] : [];
    const solveDeclarations = `${declarations}
const WAITING: u32 = 0xffffffffu;
const INVALID: u32 = 0xfffffffeu;
fn tryFinalize(cell: u32) -> bool {
 if (atomicLoad(&state[stateOffset + cell]) != WAITING) { return false; }
var value = ${transport ? 'source[sourceOffset + cell]' : '0.0'};
var ready = true;
var hasReceiver = false;
var minimum = infinity();
for (var direction = 0u; direction < 8u; direction++) {
 let neighbor = getFlowReceiver(cell, 1u << direction);
 if (neighbor == NO_CELL) { continue; }
 let fraction = fractions[fractionsOffset + ${transport ? 'neighbor * 8u + ((direction + 4u) % 8u)' : 'cell * 8u + direction'}];
 if (!finite(fraction) || fraction < 0.0) { ready = false; continue; }
 if (fraction == 0.0) { continue; }
 hasReceiver = true;
 let bits = atomicLoad(&state[stateOffset + neighbor]);
 ready = ready && bits != WAITING && bits != INVALID;
 let downstream = bitcast<f32>(bits);
 ${
   props.edgeLengths
     ? 'let edgeLength = lengths[lengthsOffset + cell * 8u + direction];'
     : `let delta = vec2<f32>(f32(neighbor % GRID_WIDTH) - f32(cell % GRID_WIDTH), f32(neighbor / GRID_WIDTH) - f32(cell / GRID_WIDTH));
 let edgeLength = length(vec2<f32>(${affine[0]} * delta.x + ${affine[1]} * delta.y, ${affine[3]} * delta.x + ${affine[4]} * delta.y));`
 }
 ${
   transport || props.mode === 'fractional-catchment'
     ? 'value += fraction * downstream;'
     : `ready = ready && finite(edgeLength) && edgeLength >= 0.0;
 ${props.mode === 'minimum-distance' ? 'minimum = min(minimum, downstream + edgeLength);' : props.mode === 'maximum-distance' ? 'value = max(value, downstream + edgeLength);' : 'value += fraction * (downstream + edgeLength);'}`
 }
}
${props.mode === 'minimum-distance' ? 'value = select(0.0, minimum, hasReceiver);' : ''}
${props.mode === 'fractional-catchment' ? 'if (!hasReceiver) { value = source[sourceOffset + cell]; }' : ''}
ready = ready && finite(value);
if (!ready) { return false; }
 loop {
  let exchange = atomicCompareExchangeWeak(&state[stateOffset + cell], WAITING, bitcast<u32>(value));
  if (exchange.exchanged) { return true; }
  if (exchange.old_value != WAITING) { return false; }
 }
 return false;
}`;
    nodes.push(
      createRasterIterationResetNode(graph, {
        id: `${id}-reset`,
        operation: 'GPUTerrainBranchingFlow',
        state: iterationState
      })
    );
    nodes.push(
      ...createRasterGatedLoopNodes<Parameters>(graph, {
        id: `${id}-solve`,
        operation: 'GPUTerrainBranchingFlow',
        state: iterationState,
        maxIterations:
          props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024),
        createRound: ({nodeId, slotBinding, markChangedWGSL, condition, extraResources}) =>
          createWGSLKernelNode(graph, {
            id: nodeId,
            operation: 'GPUTerrainBranchingFlow',
            variant: props.mode,
            invocationCount: count,
            bindings: [
              terrainBinding('fractions', props.fractions),
              terrainBinding('source', props.values),
              {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'},
              ...lengths,
              {
                name: 'status',
                view: iterationState.status,
                type: 'atomic<u32>',
                access: 'read_write'
              },
              slotBinding
            ],
            declarations: solveDeclarations,
            condition,
            extraResources,
            body: `var current = index;
var changed = tryFinalize(current);
var published = changed;
for (var step = 0u; step < ${props.maximumWalkLength ?? 32}u && changed; step++) {
 var successor = NO_CELL;
 for (var direction = 0u; direction < 8u; direction++) {
  let neighbor = getFlowReceiver(current, 1u << direction);
  if (neighbor == NO_CELL) { continue; }
  let fraction = fractions[fractionsOffset + ${transport ? 'current * 8u + direction' : 'neighbor * 8u + ((direction + 4u) % 8u)'}];
  if (fraction > 0.0 && tryFinalize(neighbor)) { published = true; if (successor == NO_CELL) { successor = neighbor; } }
 }
 current = successor; changed = successor != NO_CELL;
}
if (published) { ${markChangedWGSL} }`
          })
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUTerrainBranchingFlow',
        invocationCount: count,
        bindings: [
          terrainBinding('state', state),
          terrainBinding('values', props.output.values, 'read_write'),
          terrainBinding('status', props.output.status, 'read_write')
        ],
        declarations: TERRAIN_FINITE_WGSL,
        body: `let bits = state[stateOffset + index];
let statusValue = select(select(1u, 0u, bits == 0xfffffffeu), 2u, bits == 0xffffffffu);
status[statusOffset + index] = statusValue;
values[valuesOffset + index] = select(missing(), bitcast<f32>(bits), statusValue == 1u);`
      }),
      ...createRasterCountNodes(graph, {
        id: `${id}-status`,
        operation: 'GPUTerrainBranchingFlow',
        values: props.output.status,
        count,
        condition: 'value != 1u',
        output: props.output.unresolvedCount
      })
    );
    return nodes;
  }
}
