// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Binding} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  getGraphVectorData,
  getViewBinding,
  getViewElementOffset,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';

const WORKGROUP_SIZE = 64;

export type TripMembershipFilterProps = {
  id: string;
  /** Per-chunk CSR offsets, `tripCount + 1` words per chunk. */
  offsets: GraphVectorView<'uint32'>;
  /** Per-chunk member values, aligned with the offsets' chunks. */
  memberValues: GraphVectorView<'uint32'>;
  /** Per-chunk trip output; word is 1 when any member value is nonzero, otherwise 0. */
  output: GraphVectorView<'uint32'>;
  /** Float32 word aliases of the source vertices (3 words/vertex, 1 word/vertex) and outputs. */
  positions: GraphVectorView<'float32'>;
  times: GraphVectorView<'float32'>;
  filteredPositions: GraphVectorView<'float32'>;
  filteredTimes: GraphVectorView<'float32'>;
};

/**
 * Reduces member-row selection to one trip flag (visible when any member row is selected) and
 * writes a same-layout filtered path. Unselected vertices repeat the previous selected vertex's
 * position and time (leading ones repeat the first selected vertex), which produces zero-length,
 * zero-duration segments: the drawn polyline through selected vertices matches native tabular
 * Trips, which drops unselected rows, without changing vertex offsets or repacking sources.
 */
export class TripMembershipFilter {
  readonly id: string;
  constructor(private readonly props: TripMembershipFilterProps) {
    this.id = props.id;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const offsets = getGraphVectorData(this.props.offsets);
    const members = getGraphVectorData(this.props.memberValues);
    const outputs = getGraphVectorData(this.props.output);
    const positions = getGraphVectorData(this.props.positions);
    const times = getGraphVectorData(this.props.times);
    const filteredPositions = getGraphVectorData(this.props.filteredPositions);
    const filteredTimes = getGraphVectorData(this.props.filteredTimes);
    const nodes: GPUCommandNode<Parameters>[] = [];
    outputs.forEach((output, index) => {
      if (!output.length) return;
      nodes.push(
        this.makeNode<Parameters>(
          graph,
          `${this.id}-${index}`,
          offsets[index],
          members[index],
          output,
          [positions[index], times[index], filteredPositions[index], filteredTimes[index]]
        )
      );
    });
    return nodes;
  }

  private makeNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    id: string,
    offsets: GraphDataView<'uint32'>,
    members: GraphDataView<'uint32'>,
    output: GraphDataView<'uint32'>,
    vertices: readonly [
      GraphDataView<'float32'>,
      GraphDataView<'float32'>,
      GraphDataView<'float32'>,
      GraphDataView<'float32'>
    ]
  ): GPUCommandNode<Parameters> {
    const [positions, times, filteredPositions, filteredTimes] = vertices;
    const layout = getBoundedDispatchLayout(
      'TripMembershipFilter',
      output.length,
      WORKGROUP_SIZE,
      graph.device.limits.maxComputeWorkgroupsPerDimension
    );
    const source = `const TRIP_COUNT: u32 = ${output.length}u;
const OFFSET_BASE: u32 = ${getViewElementOffset(offsets)}u;
const MEMBER_BASE: u32 = ${getViewElementOffset(members)}u;
const OUTPUT_BASE: u32 = ${getViewElementOffset(output)}u;
const POSITION_BASE: u32 = ${getViewElementOffset(positions)}u;
const TIME_BASE: u32 = ${getViewElementOffset(times)}u;
const FILTERED_POSITION_BASE: u32 = ${getViewElementOffset(filteredPositions)}u;
const FILTERED_TIME_BASE: u32 = ${getViewElementOffset(filteredTimes)}u;
const NONE: u32 = 0xffffffffu;
@group(0) @binding(0) var<storage, read_write> outputWords: array<u32>;
@group(0) @binding(1) var<storage, read> offsetWords: array<u32>;
@group(0) @binding(2) var<storage, read> memberWords: array<u32>;
@group(0) @binding(3) var<storage, read> positionWords: array<u32>;
@group(0) @binding(4) var<storage, read> timeWords: array<u32>;
@group(0) @binding(5) var<storage, read_write> filteredPositionWords: array<u32>;
@group(0) @binding(6) var<storage, read_write> filteredTimeWords: array<u32>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= TRIP_COUNT) { return; }
  let first = offsetWords[OFFSET_BASE + index];
  let last = offsetWords[OFFSET_BASE + index + 1u];
  var anchor = NONE;
  for (var member = first; member < last; member++) {
    if (memberWords[MEMBER_BASE + member] != 0u) { anchor = member; break; }
  }
  outputWords[OUTPUT_BASE + index] = select(0u, 1u, anchor != NONE);
  for (var vertex = first; vertex < last; vertex++) {
    if (memberWords[MEMBER_BASE + vertex] != 0u) { anchor = vertex; }
    // Trips without selected members are hidden by their flag; copy them unchanged.
    let sourceVertex = select(anchor, vertex, anchor == NONE);
    for (var component = 0u; component < 3u; component++) {
      filteredPositionWords[FILTERED_POSITION_BASE + vertex * 3u + component] =
        positionWords[POSITION_BASE + sourceVertex * 3u + component];
    }
    filteredTimeWords[FILTERED_TIME_BASE + vertex] = timeWords[TIME_BASE + sourceVertex];
  }
}`;
    return createGPUComputeCommandNode<Parameters>({
      id,
      workload: {
        operation: 'TripMembershipFilter',
        commandCount: 1,
        maximumWorkgroupCount: layout.x * layout.y * layout.z,
        maximumInvocationCount: layout.x * layout.y * layout.z * WORKGROUP_SIZE,
        readByteLength: offsets.length * 4 + members.length * 4,
        writeByteLength: output.length * 4
      },
      resources: [
        {buffer: offsets, usage: 'storage-read'},
        {buffer: members, usage: 'storage-read'},
        {buffer: positions, usage: 'storage-read'},
        {buffer: times, usage: 'storage-read'},
        {buffer: output, usage: 'storage-write'},
        {buffer: filteredPositions, usage: 'storage-write'},
        {buffer: filteredTimes, usage: 'storage-write'}
      ],
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id,
          source,
          shaderLayout: {
            bindings: [
              {name: 'outputWords', type: 'storage', group: 0, location: 0},
              {name: 'offsetWords', type: 'read-only-storage', group: 0, location: 1},
              {name: 'memberWords', type: 'read-only-storage', group: 0, location: 2},
              {name: 'positionWords', type: 'read-only-storage', group: 0, location: 3},
              {name: 'timeWords', type: 'read-only-storage', group: 0, location: 4},
              {name: 'filteredPositionWords', type: 'storage', group: 0, location: 5},
              {name: 'filteredTimeWords', type: 'storage', group: 0, location: 6}
            ]
          }
        });
        return {
          encode: ({computePass, getBuffer}) => {
            const bindings: Record<string, Binding> = {
              outputWords: getViewBinding(output, getBuffer),
              offsetWords: getViewBinding(offsets, getBuffer),
              memberWords: getViewBinding(members, getBuffer),
              positionWords: getViewBinding(positions, getBuffer),
              timeWords: getViewBinding(times, getBuffer),
              filteredPositionWords: getViewBinding(filteredPositions, getBuffer),
              filteredTimeWords: getViewBinding(filteredTimes, getBuffer)
            };
            kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
          },
          destroy: () => kernel.destroy()
        };
      }
    });
  }
}
