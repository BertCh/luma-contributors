// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {withGPUGraphAlgorithmsRecordScope} from './gpu-graph-computation-record';
import {addGPUGraphAlgorithmsCoreNumberToGraphWithDispatchLimit} from './gpu-graph-core-number-internals';
import type {GPUGraphAlgorithmsTopology} from './gpu-graph-topology';
import {
  isGPUGraphAlgorithmsViewColumn,
  usesGPUGraphAlgorithmsViews,
  validateGPUGraphAlgorithmsViewColumn,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';
import {getPhysicalBuffer, getAdjacencyVectors} from './gpu-graph-algorithms-utils';

const DEFAULT_CORE_NUMBER_ITERATIONS = 32;
const MAXIMUM_CORE_NUMBER_ITERATIONS = 1024;
const SCALAR_BYTE_LENGTH = 4;

/** Existing graph topology and caller-owned k-core decomposition destinations. */
export type GPUGraphAlgorithmsCoreNumberProps<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> = {
  /** Prefix for generated command-graph nodes and imported-resource identifiers. */
  id?: string;
  /** Existing compressed adjacency; directed weak neighborhoods require reverse CSR. */
  topology: Topology;
  /** One caller-owned unsigned core-number row per graph vertex. */
  output: Column;
  /** Maximum synchronized core-refinement rounds; defaults to 32 and is bounded by 1,024. */
  iterations?: number;
  /** Optional caller-owned scalar reporting whether the core numbers reached a fixed point. */
  converged?: Column;
  /** Optional caller-owned scalar receiving the maximum currently published core number. */
  degeneracy?: Column;
};

/**
 * Computes the standard simple, undirected k-core numbers of GPU-resident weak neighborhoods.
 *
 * Directed graphs require reverse adjacency and are interpreted as undirected. Reciprocal and
 * duplicate edges are deduplicated, self-loops are ignored, and edge weights have no effect.
 * Synchronous H-index refinements monotonically lower distinct-degree upper bounds until the
 * exact core numbers are reached. If the bounded rounds stop first, outputs are valid upper
 * bounds and the optional convergence scalar remains zero.
 *
 * An isolated vertex has core number zero. Overflow in either required adjacency publishes
 * `0xffffffff` output and optional degeneracy sentinels and leaves convergence at zero.
 */
export class GPUGraphAlgorithmsCoreNumber<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> {
  /** Prefix for generated command-graph nodes and imported resources. */
  readonly id: string;
  /** Existing caller-owned GPU graph topology. */
  readonly topology: Topology;
  /** Caller-owned, vertex-aligned unsigned core numbers. */
  readonly output: Column;
  /** Maximum number of compiled, globally synchronized refinement rounds. */
  readonly iterations: number;
  /** Optional caller-owned fixed-point convergence status. */
  readonly converged?: Column;
  /** Optional caller-owned maximum currently published core number. */
  readonly degeneracy?: Column;

  /** Validates caller-owned metadata without allocating, submitting, or reading GPU work. */
  constructor(props: GPUGraphAlgorithmsCoreNumberProps<Topology, Column>) {
    this.id = props.id ?? 'gpu-graph-core-number';
    this.topology = props.topology;
    this.output = props.output;
    this.iterations = props.iterations ?? DEFAULT_CORE_NUMBER_ITERATIONS;
    this.converged = props.converged;
    this.degeneracy = props.degeneracy;

    if (this.topology.graph.directed && !this.topology.reverse) {
      throw new Error(`${this.id} directed weak-neighbor core numbers require reverse adjacency`);
    }
    if (
      !Number.isSafeInteger(this.iterations) ||
      this.iterations < 0 ||
      this.iterations > MAXIMUM_CORE_NUMBER_ITERATIONS
    ) {
      throw new Error(`${this.id} iterations must be a safe integer between zero and 1024`);
    }

    validateColumn(this.output, this.topology.graph.vertexCount, `${this.id} output`);
    if (this.converged) {
      validateColumn(this.converged, 1, `${this.id} converged`);
    }
    if (this.degeneracy) {
      validateColumn(this.degeneracy, 1, `${this.id} degeneracy`);
    }
    if (
      !usesGPUGraphAlgorithmsViews(this.topology, [this.output, this.converged, this.degeneracy])
    ) {
      validateDistinctCoreNumberOutputs(this);
    }
  }

  /** Declares bounded GPU core refinement without queue submission or CPU synchronization. */
  addToGraph<Parameters>(commandGraph: GPUCommandGraph<Parameters>): void {
    withGPUGraphAlgorithmsRecordScope(commandGraph, 'GPUGraphAlgorithmsCoreNumber', this.id, () => {
      addGPUGraphAlgorithmsCoreNumberToGraphWithDispatchLimit(
        this,
        commandGraph,
        commandGraph.device.limits.maxComputeWorkgroupsPerDimension
      );
    });
  }
}

/** Validates a physical or view column with its exact logical row count. */
function validateColumn(
  column: GPUGraphAlgorithmsColumn<'uint32'>,
  length: number,
  name: string
): void {
  if (isGPUGraphAlgorithmsViewColumn(column)) {
    validateGPUGraphAlgorithmsViewColumn(column, 'uint32', length, name);
  } else {
    validateCoreNumberVector(column, length, name);
  }
}

/** Requires exactly one packed, uint32-aligned caller-owned scalar output chunk. */
function validateCoreNumberVector(vector: GPUVector<'uint32'>, length: number, name: string): void {
  if (
    vector.format !== 'uint32' ||
    vector.data.length !== 1 ||
    vector.length !== length ||
    vector.stride !== 1 ||
    vector.byteStride !== SCALAR_BYTE_LENGTH ||
    vector.rowByteLength !== SCALAR_BYTE_LENGTH ||
    vector.valueLength !== vector.length ||
    vector.bufferLayout
  ) {
    throw new Error(`${name} must contain exactly ${length} packed uint32 rows in one chunk`);
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

/** Keeps caller-visible core outputs disjoint from graph sources, CSR, status, and peers. */
function validateDistinctCoreNumberOutputs(
  coreNumber: GPUGraphAlgorithmsCoreNumber<
    GPUGraphAlgorithmsTopologyLike,
    GPUGraphAlgorithmsColumn<'uint32'>
  >
): void {
  // Only called when every topology input and output is a physical GPUVector.
  const topology = coreNumber.topology as GPUGraphAlgorithmsTopology;
  const inputVectors = [
    topology.graph.sourceVertices,
    topology.graph.targetVertices,
    ...(topology.graph.edgeWeights ? [topology.graph.edgeWeights] : []),
    ...(topology.graph.edgeIds ? [topology.graph.edgeIds] : []),
    ...getAdjacencyVectors(topology.forward),
    ...(topology.reverse ? getAdjacencyVectors(topology.reverse) : []),
    topology.invalidEdgeCount
  ];
  const physicalAllocations = new Set<Buffer>();
  for (const vector of inputVectors) {
    for (const chunk of vector.data) {
      physicalAllocations.add(getPhysicalBuffer(chunk));
    }
  }

  const outputs = [
    {name: 'output', vector: coreNumber.output as GPUVector<'uint32'>},
    ...(coreNumber.converged
      ? [{name: 'converged', vector: coreNumber.converged as GPUVector<'uint32'>}]
      : []),
    ...(coreNumber.degeneracy
      ? [{name: 'degeneracy', vector: coreNumber.degeneracy as GPUVector<'uint32'>}]
      : [])
  ];
  for (const {name, vector} of outputs) {
    const physicalBuffer = getPhysicalBuffer(vector.data[0]);
    if (physicalAllocations.has(physicalBuffer)) {
      throw new Error(`${coreNumber.id} ${name} must use a distinct physical buffer allocation`);
    }
    physicalAllocations.add(physicalBuffer);
  }
}
