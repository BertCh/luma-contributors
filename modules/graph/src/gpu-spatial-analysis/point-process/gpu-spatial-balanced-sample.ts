// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../utils/philox-random';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {validatePointProcessView} from './point-process-utils';

/** One stratum of the finite-point equal-probability design. */
export type GPUSpatialSampleAllocation = {stratumId: number; sampleCount: number};

/** Randomized hierarchical order followed by circular systematic selection. */
export type GPUSpatialBalancedSampleProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Stable user frame IDs; caller guarantees uniqueness. Returned verbatim. */
  frameIds: GraphDataView<'uint32'>;
  /** Optional per-row strata. Omission means stratum 0. Every row must match an allocation. */
  strata?: GraphDataView<'uint32'>;
  allocations: readonly GPUSpatialSampleAllocation[];
  /** Optional positive integer inclusion masses. Per-stratum sum must be divisible by sampleCount,
   * at most 2^24, and each mass <= sum/sampleCount. Gives exact unequal inclusion mass/step. */
  inclusionTickets?: GraphDataView<'uint32'>;
  /** [minX, minY, maxX, maxY], finite positive extent enclosing the complete frame. */
  bounds: readonly [number, number, number, number];
  seed: number;
  /** Optional resident [seed, replicate] overrides, mutable between encodings without recompilation. */
  parameters?: GraphDataView<'uint32'>;
  /** Independent survey replicate, default 0; reproducible regardless of dispatch partitioning. */
  replicate?: number;
  output: {
    frameIds: GraphDataView<'uint32'>;
    inclusionProbabilities: GraphDataView<'float32'>;
    designWeights: GraphDataView<'float32'>;
    /** Per stratum [selectedCount, status], status 0 success, 1 invalid frame/allocation, 2 RNG exhaustion. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Finite-point stratified spatially ordered probability sampling. Sixteen randomly rotated
 * quadtree levels produce an ordering; a uniform integer circular shift selects n distinct
 * approximately equally spaced ranks. Each stratum member has exact design inclusion n/N,
 * independently of coordinates, Morton collisions or random ordering. Rejection sampling
 * avoids modulo bias; 128 failed draws explicitly invalidate the stratum rather than bias it.
 *
 * This is a specified hierarchical circular-systematic design, not an assertion of GRTS parity.
 * Optional integer inclusion masses give exact unequal probabilities within each stratum. It excludes
 * continuous frames, replacement sites, minimum separation or Horvitz-Thompson variance estimates.
 * Invalid/out-of-bounds rows invalidate every stratum; they are never silently omitted. O(N²)
 * rank construction bounds this implementation to 4096 rows. Inputs stay caller-owned.
 */
export class GPUSpatialBalancedSample implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSpatialBalancedSampleProps;
  /** Replayable random stream and design provenance. */
  readonly provenance: Readonly<{
    generator: 'Philox4x32-10';
    purpose: number;
    seed: number;
    replicate: number;
    levels: 16;
    design: 'hierarchical-circular-systematic';
    parameterSource: 'constructor' | 'resident-view';
  }>;
  constructor(props: GPUSpatialBalancedSampleProps) {
    this.id = props.id ?? 'spatial-balanced-sample';
    this.props = props;
    const rows = props.positions.length;
    assert(rows > 0 && rows <= 4096);
    assert(props.allocations.length > 0 && props.allocations.length <= 256);
    const ids = new Set<number>();
    let sampleCount = 0;
    for (const allocation of props.allocations) {
      assert(
        Number.isInteger(allocation.stratumId) &&
          allocation.stratumId >= 0 &&
          allocation.stratumId < 16777216 &&
          !ids.has(allocation.stratumId)
      );
      assert(
        Number.isInteger(allocation.sampleCount) &&
          allocation.sampleCount > 0 &&
          allocation.sampleCount <= rows
      );
      ids.add(allocation.stratumId);
      sampleCount += allocation.sampleCount;
    }
    assert(sampleCount <= rows);
    if (!props.strata) {
      assert(props.allocations.length === 1 && props.allocations[0].stratumId === 0);
    }
    assert(props.bounds.every(value => Number.isFinite(Math.fround(value))));
    assert(
      Math.fround(props.bounds[2]) > Math.fround(props.bounds[0]) &&
        Math.fround(props.bounds[3]) > Math.fround(props.bounds[1])
    );
    assert(Number.isInteger(props.seed) && props.seed >= 0 && props.seed <= 0xffffffff);
    assert(
      Number.isInteger(props.replicate ?? 0) &&
        (props.replicate ?? 0) >= 0 &&
        (props.replicate ?? 0) <= 0xffffffff
    );
    if (props.parameters) {
      validatePointProcessView(props.parameters, 'uint32', 2);
    }
    validatePointProcessView(props.positions, 'float32x2', rows);
    validatePointProcessView(props.frameIds, 'uint32', rows);
    if (props.strata) {
      validatePointProcessView(props.strata, 'uint32', rows);
    }
    if (props.inclusionTickets) {
      validatePointProcessView(props.inclusionTickets, 'uint32', rows);
    }
    validatePointProcessView(props.output.frameIds, 'uint32', sampleCount);
    validatePointProcessView(props.output.inclusionProbabilities, 'float32', sampleCount);
    validatePointProcessView(props.output.designWeights, 'float32', sampleCount);
    validatePointProcessView(props.output.status, 'uint32', props.allocations.length * 2);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.frameIds,
      props.strata,
      props.inclusionTickets,
      props.parameters
    ]);
    // Status is written before publication, so graph hazards cannot reject cross-pass aliases.
    const outputs = Object.values(props.output);
    for (const [index, output] of outputs.entries()) {
      assert(!outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)));
    }
    this.provenance = Object.freeze({
      generator: 'Philox4x32-10',
      purpose: PHILOX_PURPOSES.spatialBalancedSampling.id,
      seed: props.seed,
      replicate: props.replicate ?? 0,
      levels: 16,
      design: 'hierarchical-circular-systematic',
      parameterSource: props.parameters ? 'resident-view' : 'constructor'
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.positions.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.frameIds,
      props.strata,
      props.inclusionTickets,
      props.parameters,
      ...Object.values(props.output)
    ]);
    const codes = createTransientView(graph, `${id}-codes`, 'uint32', rows);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', rows);
    const selectedRows = createTransientView(
      graph,
      `${id}-selected-rows`,
      'uint32',
      props.output.frameIds.length
    );
    const selectedProbabilities = createTransientView(
      graph,
      `${id}-selected-probabilities`,
      'float32',
      props.output.frameIds.length
    );
    const ticketBindings = props.inclusionTickets
      ? [
          {
            name: 'tickets',
            view: props.inclusionTickets,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const ticketDeclaration = `fn ticket(row: u32) -> u32 { return ${props.inclusionTickets ? 'tickets[ticketsOffset + row]' : '1u'}; }`;
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', rows);
    const stratumBindings = props.strata
      ? [{name: 'strata', view: props.strata, type: 'u32' as const, access: 'read' as const}]
      : [];
    const stratumDeclaration = `fn stratum(row: u32) -> u32 { return ${props.strata ? 'strata[strataOffset + row]' : '0u'}; }`;
    const parameterBindings = props.parameters
      ? [
          {
            name: 'parameters',
            view: props.parameters,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const replicateExpression = props.parameters
      ? 'parameters[parametersOffset + 1u]'
      : `${props.replicate ?? 0}u`;
    const randomDeclaration = `${PHILOX_WGSL}
fn random(counter: vec4<u32>) -> u32 { return philox4x32(counter, vec2<u32>(${props.parameters ? 'parameters[parametersOffset]' : `${props.seed}u`}, ${this.provenance.purpose}u)).x; }`;
    const minimum = props.bounds
      .slice(0, 2)
      .map(value => `${Math.fround(value)}f`)
      .join(', ');
    const maximum = props.bounds
      .slice(2)
      .map(value => `${Math.fround(value)}f`)
      .join(', ');
    // Scale large axes by an exact normal power of two before subtracting or dividing.
    // Besides overflowing extents, this avoids a flushed subnormal reciprocal in GPU division.
    const normalizationScale = [0, 1]
      .map(axis => {
        const magnitude = Math.max(Math.abs(props.bounds[axis]), Math.abs(props.bounds[axis + 2]));
        const exponent = magnitude >= 2 ** 64 ? Math.min(126, Math.floor(Math.log2(magnitude))) : 0;
        return `${2 ** -exponent}f`;
      })
      .join(', ');
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-hierarchy`,
        operation: 'GPUSpatialBalancedSample',
        variant: 'randomized-quadtree',
        invocationCount: rows,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          ...parameterBindings,
          ...stratumBindings,
          {name: 'codes', view: codes, type: 'u32', access: 'read_write'},
          {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}\n${stratumDeclaration}\n${randomDeclaration}`,
        body: `let point = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let minimum = vec2<f32>(${minimum});
  let maximum = vec2<f32>(${maximum});
  let valid = isFiniteFloat(point.x) && isFiniteFloat(point.y) && all(point >= minimum) && all(point <= maximum) && (${props.allocations.map(allocation => `stratum(index) == ${allocation.stratumId}u`).join(' || ')});
  validity[validityOffset + index] = select(0u, 1u, valid);
  var code = 0u;
  if (valid) {
    let normalizationScale = vec2<f32>(${normalizationScale});
    let coordinate = vec2<u32>(clamp((point * normalizationScale - minimum * normalizationScale) / (maximum * normalizationScale - minimum * normalizationScale), vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0);
    var prefix = 0u;
    for (var level = 0u; level < 16u; level++) {
      let shift = 15u - level;
      let child = ((coordinate.x >> shift) & 1u) | (((coordinate.y >> shift) & 1u) << 1u);
      let rotation = random(vec4<u32>(prefix, (stratum(index) << 4u) | level, ${replicateExpression}, 0u)) & 3u;
      code = (code << 2u) | ((child + rotation) & 3u);
      prefix = (prefix << 2u) | child;
    }
  }
  codes[codesOffset + index] = code;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-rank`,
        operation: 'GPUSpatialBalancedSample',
        variant: 'spatial-ranks',
        invocationCount: rows,
        bindings: [
          {name: 'codes', view: codes, type: 'u32', access: 'read'},
          ...stratumBindings,
          ...ticketBindings,
          {name: 'ranks', view: ranks, type: 'u32', access: 'read_write'}
        ],
        declarations: `${stratumDeclaration}\n${ticketDeclaration}`,
        body: `var rank = 0u;
  let code = codes[codesOffset + index];
  for (var other = 0u; other < ${rows}u; other++) {
    let otherCode = codes[codesOffset + other];
    if (stratum(other) == stratum(index) && (otherCode < code || (otherCode == code && other < index))) { rank = min(16777217u, rank + min(ticket(other), 16777217u)); }
  }
  ranks[ranksOffset + index] = rank;`
      })
    ];
    let firstOutput = 0;
    for (const [allocationIndex, allocation] of props.allocations.entries()) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-select-${allocationIndex}`,
          operation: 'GPUSpatialBalancedSample',
          variant: 'circular-systematic',
          invocationCount: 1,
          bindings: [
            {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
            {name: 'validity', view: validity, type: 'u32', access: 'read'},
            ...stratumBindings,
            ...ticketBindings,
            ...parameterBindings,
            {name: 'selected', view: selectedRows, type: 'u32', access: 'read_write'},
            {name: 'probabilities', view: selectedProbabilities, type: 'f32', access: 'read_write'},
            {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `${stratumDeclaration}\n${ticketDeclaration}\n${randomDeclaration}`,
          body: `var frameCount = 0u;
  var ticketTotal = 0u;
  var maximumTicket = 0u;
  var valid = true;
  for (var row = 0u; row < ${rows}u; row++) {
    valid = valid && validity[validityOffset + row] == 1u;
    if (stratum(row) == ${allocation.stratumId}u) {
      frameCount++;
      let value = ticket(row);
      valid = valid && value > 0u && value <= 16777216u - min(ticketTotal, 16777216u);
      ticketTotal = min(16777217u, ticketTotal + min(value, 16777217u));
      maximumTicket = max(maximumTicket, value);
    }
  }
  valid = valid && frameCount >= ${allocation.sampleCount}u;
  ${
    props.inclusionTickets
      ? `let step = ticketTotal / ${allocation.sampleCount}u;
  valid = valid && ticketTotal % ${allocation.sampleCount}u == 0u && maximumTicket <= step && step > 0u;
  let drawLimit = step;`
      : 'let drawLimit = frameCount;'
  }
  var shift = 0u;
  var found = false;
  if (valid) {
    let threshold = (0u - drawLimit) % drawLimit;
    for (var attempt = 0u; attempt < 128u; attempt++) {
      let draw = random(vec4<u32>(${allocation.stratumId}u, 0x80000000u | attempt, ${replicateExpression}, 0u));
      if (draw >= threshold) { shift = draw % drawLimit; found = true; break; }
    }
  }
  for (var sample = 0u; sample < ${allocation.sampleCount}u; sample++) {
    let slot = ${firstOutput}u + sample;
    selected[selectedOffset + slot] = 0xffffffffu;
    probabilities[probabilitiesOffset + slot] = bitcast<f32>(0x7fc00000u | (index & 0u));
    if (!valid || !found) { continue; }
    let targetRank = ${props.inclusionTickets ? 'shift + sample * step' : `(shift + sample * frameCount / ${allocation.sampleCount}u) % frameCount`};
    for (var row = 0u; row < ${rows}u; row++) {
      let first = ranks[ranksOffset + row];
      if (stratum(row) == ${allocation.stratumId}u && first <= targetRank && targetRank < first + ticket(row)) {
        selected[selectedOffset + slot] = row;
        probabilities[probabilitiesOffset + slot] = ${props.inclusionTickets ? 'f32(ticket(row)) / f32(step)' : `${allocation.sampleCount}.0 / f32(frameCount)`};
      }
    }
  }
  status[statusOffset + ${2 * allocationIndex}u] = select(0u, ${allocation.sampleCount}u, valid && found);
  status[statusOffset + ${2 * allocationIndex + 1}u] = select(1u, select(2u, 0u, found), valid);`
        })
      );
      firstOutput += allocation.sampleCount;
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUSpatialBalancedSample',
        variant: 'design-records',
        invocationCount: props.output.frameIds.length,
        bindings: [
          {name: 'selected', view: selectedRows, type: 'u32', access: 'read'},
          {name: 'probabilities', view: selectedProbabilities, type: 'f32', access: 'read'},
          {name: 'frameIds', view: props.frameIds, type: 'u32', access: 'read'},
          {name: 'ids', view: props.output.frameIds, type: 'u32', access: 'read_write'},
          {
            name: 'inclusion',
            view: props.output.inclusionProbabilities,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'weights', view: props.output.designWeights, type: 'f32', access: 'read_write'}
        ],
        body: `let row = selected[selectedOffset + index];
  var frameId = 0xffffffffu;
  if (row != 0xffffffffu) { frameId = frameIds[frameIdsOffset + row]; }
  let probability = probabilities[probabilitiesOffset + index];
  ids[idsOffset + index] = frameId;
  inclusion[inclusionOffset + index] = probability;
  weights[weightsOffset + index] = 1.0 / probability;`
      })
    );
    return nodes;
  }
}
