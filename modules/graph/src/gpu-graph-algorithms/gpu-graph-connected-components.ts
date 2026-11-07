// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {withGPUGraphAlgorithmsRecordScope} from './gpu-graph-computation-record';
import {addGPUGraphAlgorithmsConnectedComponentsToGraphWithDispatchLimit} from './gpu-graph-connected-components-internals';
import type {GPUGraphAlgorithmsTopology} from './gpu-graph-topology';
import {
  isGPUGraphAlgorithmsViewColumn,
  usesGPUGraphAlgorithmsViews,
  validateGPUGraphAlgorithmsViewColumn,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';
import {getPhysicalBuffer, getAdjacencyVectors} from './gpu-graph-algorithms-utils';

const DEFAULT_COMPONENT_ITERATIONS = 32;
const MAXIMUM_COMPONENT_ITERATIONS = 1024;
const SCALAR_BYTE_LENGTH = 4;

/** Caller-owned GPU graph topology, component labels, and optional convergence status. */
export type GPUGraphAlgorithmsConnectedComponentsProps<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> = {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  id?: string;
  /** Existing forward graph adjacency; reverse adjacency is not required. */
  topology: Topology;
  /** One caller-owned packed unsigned component label for every graph vertex. */
  output: Column;
  /** Bounded number of compiled relaxation and pointer-jumping iterations. Defaults to 32. */
  iterations?: number;
  /** Optional caller-owned scalar publishing whether the final iteration reached a fixed point. */
  converged?: Column;
  /**
   * Publishes `0xffffffff` labels when the iteration budget ends before a fixed point, so
   * unfinished labels cannot be mistaken for components. Defaults to true when `iterations` is
   * omitted and to false for an explicit `iterations` (bounded, monotone partial labels).
   */
  invalidateOnNonConvergence?: boolean;
};

/**
 * Publishes weak graph components directly from existing GPU-resident forward adjacency.
 *
 * Directed edges connect both endpoints, so reverse adjacency is unnecessary. Once converged, each
 * component receives its lowest stable vertex identifier, and isolated vertices label themselves.
 * The optional convergence scalar is one only when the last compiled iteration reaches a fixed
 * point. Forward-adjacency overflow instead publishes `0xffffffff` labels and zero convergence.
 *
 * Iterations are statically encoded, but each is indirect-dispatch gated on the GPU: once an
 * iteration changes no label, later iterations launch zero workgroups (no CPU readback). With the default 32-iteration budget a
 * run that does not reach a fixed point publishes `0xffffffff` labels (see
 * `invalidateOnNonConvergence`) instead of silently returning unfinished labels.
 */
export class GPUGraphAlgorithmsConnectedComponents<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  readonly id: string;
  /** Existing caller-owned GPU graph topology. */
  readonly topology: Topology;
  /** Caller-owned vertex-aligned component labels. */
  readonly output: Column;
  /** Number of compiled, explicitly synchronized component-relaxation iterations. */
  readonly iterations: number;
  /** Optional caller-owned GPU-resident convergence status. */
  readonly converged?: Column;
  /** Whether a non-converged run publishes invalid labels instead of unfinished labels. */
  readonly invalidateOnNonConvergence: boolean;

  /** Validates caller-owned graph metadata without allocating, submitting, or reading GPU work. */
  constructor(props: GPUGraphAlgorithmsConnectedComponentsProps<Topology, Column>) {
    this.id = props.id ?? 'gpu-graph-connected-components';
    this.topology = props.topology;
    this.output = props.output;
    this.iterations = props.iterations ?? DEFAULT_COMPONENT_ITERATIONS;
    this.converged = props.converged;
    this.invalidateOnNonConvergence =
      props.invalidateOnNonConvergence ?? props.iterations === undefined;

    if (
      !Number.isSafeInteger(this.iterations) ||
      this.iterations < 1 ||
      this.iterations > MAXIMUM_COMPONENT_ITERATIONS
    ) {
      throw new Error(`${this.id} iterations must be a safe integer between one and 1024`);
    }
    validateColumn(this.output, this.topology.graph.vertexCount, `${this.id} output`);
    if (this.converged) {
      validateColumn(this.converged, 1, `${this.id} converged`);
    }
    if (!usesGPUGraphAlgorithmsViews(this.topology, [this.output, this.converged])) {
      validateDistinctComponentOutputs(this);
    }
  }

  /** Declares bounded weak-component passes without submitting commands or reading results. */
  addToGraph<Parameters>(commandGraph: GPUCommandGraph<Parameters>): void {
    withGPUGraphAlgorithmsRecordScope(
      commandGraph,
      'GPUGraphAlgorithmsConnectedComponents',
      this.id,
      () => {
        addGPUGraphAlgorithmsConnectedComponentsToGraphWithDispatchLimit(
          this,
          commandGraph,
          commandGraph.device.limits.maxComputeWorkgroupsPerDimension
        );
      }
    );
  }
}

/** Validates a physical or view column with its exact logical row count. @internal */
export function validateColumn(
  column: GPUGraphAlgorithmsColumn<'uint32'>,
  length: number,
  name: string
): void {
  if (isGPUGraphAlgorithmsViewColumn(column)) {
    validateGPUGraphAlgorithmsViewColumn(column, 'uint32', length, name);
  } else {
    validateComponentVector(column, length, name);
  }
}

/** Requires one packed, aligned unsigned output chunk with its exact logical row count. */
function validateComponentVector(vector: GPUVector<'uint32'>, length: number, name: string): void {
  if (
    vector.data.length !== 1 ||
    vector.format !== 'uint32' ||
    vector.stride !== 1 ||
    vector.byteStride !== SCALAR_BYTE_LENGTH ||
    vector.rowByteLength !== SCALAR_BYTE_LENGTH ||
    vector.valueLength !== vector.length ||
    vector.bufferLayout
  ) {
    throw new Error(`${name} must contain exactly one packed uint32 chunk`);
  }
  if (vector.length !== length) {
    throw new Error(`${name} must contain exactly ${length} uint32 rows`);
  }

  const chunk = vector.data[0];
  if (
    chunk.format !== 'uint32' ||
    chunk.length !== length ||
    chunk.stride !== 1 ||
    chunk.byteStride !== SCALAR_BYTE_LENGTH ||
    chunk.rowByteLength !== SCALAR_BYTE_LENGTH ||
    chunk.valueLength !== chunk.length ||
    !Number.isSafeInteger(chunk.byteOffset) ||
    chunk.byteOffset < 0 ||
    chunk.byteOffset % SCALAR_BYTE_LENGTH !== 0
  ) {
    throw new Error(`${name} must contain one packed, uint32-aligned chunk`);
  }
}

/** Keeps component labels and optional convergence status disjoint from all graph allocations. */
function validateDistinctComponentOutputs(
  components: GPUGraphAlgorithmsConnectedComponents<
    GPUGraphAlgorithmsTopologyLike,
    GPUGraphAlgorithmsColumn<'uint32'>
  >
): void {
  // Only called when every topology input and output is a physical GPUVector.
  const topology = components.topology as GPUGraphAlgorithmsTopology;
  const inputVectors = [
    topology.graph.sourceVertices,
    topology.graph.targetVertices,
    ...(topology.graph.edgeWeights ? [topology.graph.edgeWeights] : []),
    ...(topology.graph.edgeIds ? [topology.graph.edgeIds] : []),
    ...getAdjacencyVectors(topology.forward),
    ...(topology.reverse ? getAdjacencyVectors(topology.reverse) : []),
    topology.invalidEdgeCount
  ];
  const allocations = new Set<Buffer>();
  for (const vector of inputVectors) {
    for (const chunk of vector.data) {
      allocations.add(getPhysicalBuffer(chunk));
    }
  }

  const outputs = [
    {name: 'output', vector: components.output as GPUVector<'uint32'>},
    ...(components.converged
      ? [{name: 'converged', vector: components.converged as GPUVector<'uint32'>}]
      : [])
  ];
  for (const {name, vector} of outputs) {
    const buffer = getPhysicalBuffer(vector.data[0]);
    if (allocations.has(buffer)) {
      throw new Error(`${components.id} ${name} must use a distinct physical buffer allocation`);
    }
    allocations.add(buffer);
  }
}
