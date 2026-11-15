// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  createRasterIterationState,
  createRasterIterationResetNode,
  createRasterIterationGateNode,
  createRasterIterationFinalizeNode,
  getRasterIterationCondition
} from '../../raster-analysis/cost-distance/raster-relaxation';
import {getTerrainFlowTracingWGSL} from './terrain-flow-tracing';
import {
  GPU_TERRAIN_DRAINAGE_UNRESOLVED,
  GPU_TERRAIN_DRAINAGE_MULTIPLE,
  type GPUTerrainHeightAboveDrainageOutput
} from './gpu-terrain-height-above-drainage';
import {GPU_TERRAIN_FLOW_STREAM_UNRESOLVED} from './gpu-terrain-flow';

/** Three words per cell keep readiness, float height bits and drainage identity in one binding. @internal */
export const TERRAIN_HAND_STATE_WORDS = 3;
const OPERATION = 'GPUTerrainHeightAboveDrainage';

/** Builds bounded, synchronous downstream dependency sweeps for the vertical average. @internal */
export function createTerrainFractionalHANDNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    grid: GPURasterMetadata;
    elevation: GraphDataView<'float32'>;
    fractions: GraphDataView<'float32'>;
    streams: GraphDataView<'uint32'>;
    output: GPUTerrainHeightAboveDrainageOutput;
    maximumIterations: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, grid, output, maximumIterations} = props;
  const cellCount = grid.width * grid.height;
  const current = createTransientView(
    graph,
    `${id}-current`,
    'uint32',
    cellCount * TERRAIN_HAND_STATE_WORDS
  );
  const next = createTransientView(
    graph,
    `${id}-next`,
    'uint32',
    cellCount * TERRAIN_HAND_STATE_WORDS
  );
  const iteration = createRasterIterationState(graph, `${id}-average`, OPERATION, cellCount);
  const declarations = `${getTerrainFlowTracingWGSL(grid.width, grid.height)}
${GET_QUIET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
const STATE_WORDS: u32 = ${TERRAIN_HAND_STATE_WORDS}u;
const WAITING: u32 = 0u;
const RESOLVED: u32 = 1u;
const NO_DRAINAGE: u32 = 2u;
const UNKNOWN: u32 = 3u;
const MULTIPLE: u32 = ${GPU_TERRAIN_DRAINAGE_MULTIPLE}u;
const UNRESOLVED: u32 = ${GPU_TERRAIN_DRAINAGE_UNRESOLVED}u;`;
  const elevationBinding: WGSLKernelBinding = {
    name: 'elevation',
    view: props.elevation,
    type: 'f32',
    access: 'read'
  };
  const fractionsBinding: WGSLKernelBinding = {
    name: 'fractions',
    view: props.fractions,
    type: 'f32',
    access: 'read'
  };
  const currentBinding: WGSLKernelBinding = {
    name: 'current',
    view: current,
    type: 'u32',
    access: 'read'
  };
  const nextBinding: WGSLKernelBinding = {
    name: 'next',
    view: next,
    type: 'u32',
    access: 'read_write'
  };
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-average-init`,
      operation: OPERATION,
      invocationCount: cellCount,
      declarations,
      bindings: [
        elevationBinding,
        fractionsBinding,
        {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
        {...currentBinding, access: 'read_write'}
      ],
      body: `let base = currentOffset + index * STATE_WORDS;
var state = WAITING;
var drainage = UNRESOLVED;
var value = getQuietNaN();
let stream = streams[streamsOffset + index];
if (!isFiniteValue(elevation[elevationOffset + index])) {
  state = NO_DRAINAGE; drainage = NO_CELL;
} else if (stream == ${GPU_TERRAIN_FLOW_STREAM_UNRESOLVED}u) {
  state = UNKNOWN;
} else if (stream != 0u) {
  state = RESOLVED; drainage = index; value = 0.0;
} else {
  var valid = true;
  var total = 0.0;
  for (var direction = 0u; direction < 8u; direction++) {
    let fraction = fractions[fractionsOffset + index * 8u + direction];
    valid = valid && isFiniteValue(fraction) && fraction >= 0.0;
    if (fraction > 0.0 && getFlowReceiver(index, 1u << direction) == NO_CELL) { valid = false; }
    total += fraction;
  }
  if (!valid || (total != 0.0 && abs(total - 1.0) > 0.00001)) {
    state = UNKNOWN;
  } else if (total == 0.0) {
    state = NO_DRAINAGE; drainage = NO_CELL;
  }
}
current[base] = state;
current[base + 1u] = bitcast<u32>(value);
current[base + 2u] = drainage;`
    }),
    createRasterIterationResetNode(graph, {
      id: `${id}-average-reset`,
      operation: OPERATION,
      state: iteration
    })
  ];
  for (let round = 0; round < maximumIterations; round++) {
    const nodeId = `${id}-average-round-${round}`;
    const gated = getRasterIterationCondition(iteration, nodeId);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: nodeId,
        operation: OPERATION,
        variant: 'fractional-hand',
        invocationCount: cellCount,
        declarations,
        ...gated,
        bindings: [
          currentBinding,
          nextBinding,
          elevationBinding,
          fractionsBinding,
          {name: 'status', view: iteration.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `let base = currentOffset + index * STATE_WORDS;
let nextBase = nextOffset + index * STATE_WORDS;
var state = current[base];
var value = bitcast<f32>(current[base + 1u]);
var drainage = current[base + 2u];
if (state == WAITING) {
  var waiting = false;
  var missing = false;
  var unknown = false;
  var total = 0.0;
  var average = 0.0;
  var firstDrainage = NO_CELL;
  for (var direction = 0u; direction < 8u; direction++) {
    let fraction = fractions[fractionsOffset + index * 8u + direction];
    if (fraction <= 0.0) { continue; }
    let neighbor = getFlowReceiver(index, 1u << direction);
    let neighborBase = currentOffset + neighbor * STATE_WORDS;
    let neighborState = current[neighborBase];
    waiting = waiting || neighborState == WAITING;
    missing = missing || neighborState == NO_DRAINAGE;
    unknown = unknown || neighborState == UNKNOWN;
    if (neighborState != RESOLVED) { continue; }
    let drop = elevation[elevationOffset + index] - elevation[elevationOffset + neighbor];
    average += fraction * (drop + bitcast<f32>(current[neighborBase + 1u]));
    total += fraction;
    let neighborDrainage = current[neighborBase + 2u];
    if (firstDrainage == NO_CELL) { firstDrainage = neighborDrainage; }
    else if (firstDrainage != neighborDrainage) { firstDrainage = MULTIPLE; }
  }
  // Every positive branch must reach a stream, matching TauDEM's contamination check.
  if (waiting) { /* Downstream dependencies must finish before classifying this row. */ }
  else if (missing) { state = NO_DRAINAGE; drainage = NO_CELL; }
  else if (unknown) { state = UNKNOWN; }
  else {
    value = average / total;
    if (isFiniteValue(value)) { state = RESOLVED; drainage = firstDrainage; }
    else { state = NO_DRAINAGE; drainage = NO_CELL; value = getQuietNaN(); }
  }
  // A cycle must consume its budget even when no readiness word changes.
  atomicOr(&status[statusOffset], 1u);
}
next[nextBase] = state;
next[nextBase + 1u] = bitcast<u32>(value);
next[nextBase + 2u] = drainage;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-average-copy-${round}`,
        operation: OPERATION,
        variant: 'fractional-hand-copy',
        invocationCount: cellCount,
        ...getRasterIterationCondition(iteration, `${nodeId}-copy`),
        bindings: [
          {...nextBinding, access: 'read'},
          {...currentBinding, access: 'read_write'}
        ],
        body: `for (var word = 0u; word < ${TERRAIN_HAND_STATE_WORDS}u; word++) {
  current[currentOffset + index * ${TERRAIN_HAND_STATE_WORDS}u + word] = next[nextOffset + index * ${TERRAIN_HAND_STATE_WORDS}u + word];
}`
      }),
      createRasterIterationGateNode(graph, {
        id: `${id}-average-gate-${round}`,
        operation: OPERATION,
        state: iteration,
        maxIterations: maximumIterations
      })
    );
  }
  const bindings: WGSLKernelBinding[] = [currentBinding];
  if (output.heightAboveDrainage)
    bindings.push({
      name: 'height',
      view: output.heightAboveDrainage,
      type: 'f32',
      access: 'read_write'
    });
  if (output.drainageCells)
    bindings.push({
      name: 'drainage',
      view: output.drainageCells,
      type: 'u32',
      access: 'read_write'
    });
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-average-publish`,
      operation: OPERATION,
      invocationCount: cellCount,
      bindings,
      body: `let base = currentOffset + index * ${TERRAIN_HAND_STATE_WORDS}u;
${output.heightAboveDrainage ? 'height[heightOffset + index] = bitcast<f32>(current[base + 1u]);' : ''}
${output.drainageCells ? 'drainage[drainageOffset + index] = current[base + 2u];' : ''}`
    })
  );
  if (
    output.status?.converged ||
    output.status?.iterationLimitReached ||
    output.status?.iterationCount
  ) {
    nodes.push(
      createRasterIterationFinalizeNode(graph, {
        id: `${id}-average-finalize`,
        operation: OPERATION,
        state: iteration,
        ...output.status
      })
    );
  }
  return nodes;
}
