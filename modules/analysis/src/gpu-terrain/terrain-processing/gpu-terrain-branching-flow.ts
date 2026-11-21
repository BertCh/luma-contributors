// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createRasterIterationState,
  createRasterIterationResetNode,
  createRasterGatedLoopNodes,
  validateRasterIterations
} from '../../raster-analysis/cost-distance/raster-relaxation';
import {
  validateOption,
  validateGraphPorts,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {createRasterCountNodes} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import {
  validateRasterGrid,
  validateRasterGridAxisAligned
} from '../../raster-analysis/raster-band-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTerrainBranchingFlowCommandNodeCount,
  validateTerrainBranchingFlowScratch
} from './terrain-branching-flow-budget';
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

/** Largest successor walk per invocation and sweep: bounds one thread's serial work so a sweep stays inside the GPU watchdog. */
export const TERRAIN_BRANCHING_MAXIMUM_WALK_LENGTH = 1024;

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
  /**
   * Optional eight edge lengths per cell, in the units you want the distance modes to report.
   * Otherwise the distance modes measure each edge from the grid: affine units by default, or
   * ground meters under `cellSizeMode`. `transport` and `fractional-catchment` are
   * dimensionless in length and ignore both.
   */
  edgeLengths?: GraphDataView<'float32'>;
  /**
   * Ground model for the default edge lengths. Default `'uniform'` (affine units; geographic
   * degrees are not ground meters). `'web-mercator'` and `'geographic'` give ground meters and
   * need an axis-aligned grid. Compile-time.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Optional per-cell validity (uint32, nonzero valid). A zero word marks `values` NaN: the cell is invalid (status 0). */
  validity?: GraphDataView<'uint32'>;
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

// Construction-time shader facts remain private to this module and internal composition.
const distanceDeclarationsByContributor = new WeakMap<GPUTerrainBranchingFlow, string>();

/** Deterministic DAG pulls with atomic value/readiness publication and GPU-gated successor walks. */
export class GPUTerrainBranchingFlow implements GPUCommandNodeProducer {
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUTerrainBranchingFlowProps) {
    validateGraphPorts(props.id ?? 'terrain-branching-flow', props, {
      fractions: {required: true, viewType: 'data', formats: ['float32']},
      values: {required: true, viewType: 'data', formats: ['float32']},
      edgeLengths: {required: false, viewType: 'data', formats: ['float32']},
      validity: {required: false, viewType: 'data', formats: ['uint32']},
      'output.values': {required: true, viewType: 'data', formats: ['float32']},
      'output.status': {required: true, viewType: 'data', formats: ['uint32']},
      'output.unresolvedCount': {required: true, viewType: 'data', formats: ['uint32']}
    });
    this.id = props.id ?? 'terrain-branching-flow';
    validateOption(this.id, 'mode', props.mode, [
      'transport',
      'expected-distance',
      'minimum-distance',
      'maximum-distance',
      'fractional-catchment'
    ]);
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    const measureDistances = props.mode !== 'transport' && props.mode !== 'fractional-catchment';
    let distanceDeclarations = '';
    if (measureDistances && !props.edgeLengths) {
      try {
        distanceDeclarations = getTerrainGroundMeasureWGSL(props.grid, props.cellSizeMode, false);
      } catch {
        validateRequirement(
          this.id,
          false,
          'grid distance parameters must be representable as finite float32 values'
        );
      }
    }
    distanceDeclarationsByContributor.set(this, distanceDeclarations);
    validateViewLength(this.id, 'fractions', props.fractions, count * 8);
    validateViewLength(this.id, 'values', props.values, count);
    if (props.edgeLengths) {
      validateViewLength(this.id, 'edgeLengths', props.edgeLengths, count * 8);
      validateSinglePackedView(props.edgeLengths, ['float32'], this.id);
    }
    validateTerrainValidityView(this.id, props.validity, count);
    validateViewLength(this.id, 'output.values', props.output.values, count);
    validateViewLength(this.id, 'output.status', props.output.status, count);
    validateViewLength(this.id, 'output.unresolvedCount', props.output.unresolvedCount, 1);
    validateRasterIterations(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024)
    );
    validateCapacity(
      this.id,
      'maximumWalkLength',
      props.maximumWalkLength ?? 32,
      1,
      TERRAIN_BRANCHING_MAXIMUM_WALK_LENGTH
    );
    this.estimatedCommandNodeCount = getTerrainBranchingFlowCommandNodeCount(
      props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024),
      Boolean(props.validity)
    );
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    checkTerrainViews(
      this.id,
      [
        props.fractions,
        props.values,
        ...(measureDistances && props.edgeLengths ? [props.edgeLengths] : []),
        ...(props.validity ? [props.validity] : [])
      ],
      Object.values(props.output),
      [
        {writes: [props.output.values, props.output.status]},
        {reads: [props.output.status], writes: [props.output.unresolvedCount]}
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    return getTerrainBranchingFlowNodes(graph, this).nodes;
  }
}

/** Builds the validated contributor with private gate status for resident terrain composition. @internal */
export function createTerrainBranchingFlowNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUTerrainBranchingFlowProps
): {nodes: readonly GPUCommandNode<Parameters>[]; iterationStatus: GraphDataView<'uint32'>} {
  return getTerrainBranchingFlowNodes(graph, new GPUTerrainBranchingFlow(props));
}

function getTerrainBranchingFlowNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUTerrainBranchingFlow
): {nodes: readonly GPUCommandNode<Parameters>[]; iterationStatus: GraphDataView<'uint32'>} {
  const {props, id} = contributor;
  const count = props.values.length;
  validateGraphViewsBelongToGraph(id, graph, [props.edgeLengths]);
  checkTerrainGraph(id, graph, [
    props.fractions,
    props.values,
    ...(props.mode !== 'transport' && props.mode !== 'fractional-catchment' && props.edgeLengths
      ? [props.edgeLengths]
      : []),
    ...(props.validity ? [props.validity] : []),
    ...Object.values(props.output)
  ]);
  validateTerrainBranchingFlowScratch(
    graph,
    id,
    count,
    props.maximumIterations ?? clamp(props.grid.width + props.grid.height, 128, 1024)
  );
  const masked = createTerrainValidityMask(graph, {
    id,
    operation: 'GPUTerrainBranchingFlow',
    values: props.values,
    validity: props.validity
  });
  const state = createTransientView(graph, `${id}-state`, 'uint32', count);
  const iterationState = createRasterIterationState(graph, id, 'GPUTerrainBranchingFlow', count);
  const declarations = `${getTerrainFlowTracingWGSL(props.grid.width, props.grid.height)}\n${TERRAIN_FINITE_WGSL}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    ...masked.nodes,
    createWGSLKernelNode(graph, {
      id: `${id}-initialize`,
      operation: 'GPUTerrainBranchingFlow',
      invocationCount: count,
      bindings: [
        terrainBinding('fractions', props.fractions),
        terrainBinding('source', masked.values),
        terrainBinding('state', state, 'read_write')
      ],
      declarations,
      body: `var valid = isFiniteValue(source[sourceOffset + index]) && source[sourceOffset + index] >= 0.0;
var total = 0.0;
for (var direction = 0u; direction < 8u; direction++) {
 let fraction = fractions[fractionsOffset + index * 8u + direction];
 valid = valid && isFiniteValue(fraction) && fraction >= 0.0;
 if (fraction > 0.0 && getFlowReceiver(index, 1u << direction) == NO_CELL) { valid = false; }
 total += fraction;
}
valid = valid && (total == 0.0 || abs(total - 1.0) <= 0.00001);
${props.mode === 'fractional-catchment' ? 'valid = valid && source[sourceOffset + index] <= 1.0;' : ''}
state[stateOffset + index] = select(0xfffffffeu, 0xffffffffu, valid);`
    })
  ];
  const transport = props.mode === 'transport';
  const measureDistances = !transport && props.mode !== 'fractional-catchment';
  const lengths =
    measureDistances && props.edgeLengths ? [terrainBinding('lengths', props.edgeLengths)] : [];
  const solveDeclarations = `${declarations}
${distanceDeclarationsByContributor.get(contributor)!}
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
 if (!isFiniteValue(fraction) || fraction < 0.0) { ready = false; continue; }
 if (fraction == 0.0) { continue; }
 hasReceiver = true;
 let bits = atomicLoad(&state[stateOffset + neighbor]);
 ready = ready && bits != WAITING && bits != INVALID;
 let downstream = bitcast<f32>(bits);
 ${
   !measureDistances
     ? ''
     : props.edgeLengths
       ? 'let edgeLength = lengths[lengthsOffset + cell * 8u + direction];'
       : 'let edgeLength = getGroundEdgeLength(cell, neighbor);'
 }
 ${
   transport || props.mode === 'fractional-catchment'
     ? 'value += fraction * downstream;'
     : `ready = ready && isFiniteValue(edgeLength) && edgeLength >= 0.0;
 ${props.mode === 'minimum-distance' ? 'minimum = min(minimum, downstream + edgeLength);' : props.mode === 'maximum-distance' ? 'value = max(value, downstream + edgeLength);' : 'value += fraction * (downstream + edgeLength);'}`
 }
}
${props.mode === 'minimum-distance' ? 'value = select(0.0, minimum, hasReceiver);' : ''}
${props.mode === 'fractional-catchment' ? 'if (!hasReceiver) { value = source[sourceOffset + cell]; }' : ''}
ready = ready && isFiniteValue(value);
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
            terrainBinding('source', masked.values),
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
values[valuesOffset + index] = select(getNaN(), bitcast<f32>(bits), statusValue == 1u);`
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
  return {nodes, iterationStatus: iterationState.status};
}
