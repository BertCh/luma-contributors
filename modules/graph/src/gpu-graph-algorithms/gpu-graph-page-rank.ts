// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {Buffer} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {withGPUGraphAlgorithmsRecordScope} from './gpu-graph-computation-record';
import {addGPUGraphAlgorithmsPageRankToGraphWithDispatchLimit} from './gpu-graph-page-rank-internals';
import type {GPUGraphAlgorithmsTopology} from './gpu-graph-topology';
import {
  GPUGraphAlgorithmsTopologyView,
  isGPUGraphAlgorithmsViewColumn,
  validateGPUGraphAlgorithmsViewColumn,
  type GPUGraphAlgorithmsColumn,
  type GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';
import {getPhysicalBuffer, getAdjacencyVectors} from './gpu-graph-algorithms-utils';

const DEFAULT_PAGE_RANK_DAMPING = 0.85;
const DEFAULT_PAGE_RANK_ITERATIONS = 40;
const MAXIMUM_PAGE_RANK_ITERATIONS = 1024;
const SCALAR_BYTE_LENGTH = 4;

/** Existing graph topology, caller-owned PageRank scores, and optional residual. */
export type GPUGraphAlgorithmsPageRankProps<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'float32'> = GPUVector<'float32'>
> = {
  /** Prefix for generated command-graph nodes and imported resources. */
  id?: string;
  /**
   * Existing GPU-resident graph topology or {@link GPUGraphAlgorithmsTopologyView}; directed graphs require
   * reverse adjacency.
   */
  topology: Topology;
  /** One caller-owned, packed floating-point PageRank score for each graph vertex. */
  output: Column;
  /** Probability of following an outgoing edge rather than teleporting. Defaults to 0.85. */
  damping?: number;
  /** Bounded number of compiled, normalized PageRank iterations. Defaults to 40. */
  iterations?: number;
  /** Optional caller-owned scalar receiving the last executed iteration's absolute rank change. */
  residual?: Column;
  /**
   * Optional early-exit threshold. When positive, every iteration measures its absolute rank
   * change on the GPU and all later iterations launch zero workgroups once the change is at most
   * this value; no value is read back. Defaults to zero, which always runs every iteration.
   */
  tolerance?: number;
  /**
   * Divides each iteration by the measured sum of the new scores. Defaults to true. Dangling mass
   * is redistributed exactly, so the unnormalized recurrence already preserves total mass up to
   * rounding; `false` skips that second full reduction per iteration (about 40 percent fewer
   * dispatches) and lets rounding drift accumulate instead of correcting it every iteration.
   */
  renormalize?: boolean;
};

/**
 * Publishes normalized, unweighted PageRank scores entirely from existing GPU graph topology.
 *
 * Directed graphs require reverse adjacency for incoming-edge gathers; undirected graphs reuse
 * their symmetric forward adjacency. Each iteration redistributes dangling-vertex mass before
 * normalizing the published scores. Existing edge weights do not affect this unweighted metric.
 * Overflow in either required adjacency instead publishes zero scores and a zero residual.
 */
export class GPUGraphAlgorithmsPageRank<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'float32'> = GPUVector<'float32'>
> {
  /** Prefix for generated command-graph nodes and imported resources. */
  readonly id: string;
  /** Existing caller-owned GPU graph topology. */
  readonly topology: Topology;
  /** Caller-owned, vertex-aligned floating-point PageRank scores. */
  readonly output: Column;
  /** Probability of following an outgoing edge rather than teleporting. */
  readonly damping: number;
  /** Number of compiled, synchronized PageRank iterations. */
  readonly iterations: number;
  /** Optional caller-owned GPU-resident final absolute rank-change scalar. */
  readonly residual?: Column;
  /** Absolute rank-change threshold that disables later iterations on the GPU; zero disables it. */
  readonly tolerance: number;
  /** Whether each iteration renormalizes scores by their measured sum. */
  readonly renormalize: boolean;

  /** Validates existing caller-owned metadata without allocating, submitting, or reading work. */
  constructor(props: GPUGraphAlgorithmsPageRankProps<Topology, Column>) {
    this.id = props.id ?? 'gpu-graph-page-rank';
    this.topology = props.topology;
    this.output = props.output;
    this.damping = props.damping ?? DEFAULT_PAGE_RANK_DAMPING;
    this.iterations = props.iterations ?? DEFAULT_PAGE_RANK_ITERATIONS;
    this.residual = props.residual;
    this.tolerance = props.tolerance ?? 0;
    this.renormalize = props.renormalize ?? true;

    if (this.topology.graph.directed && !this.topology.reverse) {
      throw new Error(`${this.id} directed PageRank requires reverse adjacency`);
    }
    if (!Number.isFinite(this.damping) || this.damping < 0 || this.damping > 1) {
      throw new Error(`${this.id} damping must be a finite number between zero and one`);
    }
    if (
      !Number.isSafeInteger(this.iterations) ||
      this.iterations < 1 ||
      this.iterations > MAXIMUM_PAGE_RANK_ITERATIONS
    ) {
      throw new Error(`${this.id} iterations must be a safe integer between one and 1024`);
    }

    if (!Number.isFinite(this.tolerance) || this.tolerance < 0) {
      throw new Error(`${this.id} tolerance must be a finite non-negative number`);
    }

    validatePageRankColumn(this.output, this.topology.graph.vertexCount, `${this.id} output`);
    if (this.residual) {
      validatePageRankColumn(this.residual, 1, `${this.id} residual`);
    }
    const {topology, output, residual} = this;
    if (
      !(topology instanceof GPUGraphAlgorithmsTopologyView) &&
      !isGPUGraphAlgorithmsViewColumn(output) &&
      !(residual && isGPUGraphAlgorithmsViewColumn(residual))
    ) {
      // Every input and output is physical here; generic narrowing does not prove it to TypeScript.
      validateDistinctPageRankOutputs(
        this.id,
        topology,
        output as GPUVector<'float32'>,
        residual as GPUVector<'float32'> | undefined
      );
    }
  }

  /** Declares bounded graph ranking work without submitting commands or reading results. */
  addToGraph<Parameters>(commandGraph: GPUCommandGraph<Parameters>): void {
    withGPUGraphAlgorithmsRecordScope(commandGraph, 'GPUGraphAlgorithmsPageRank', this.id, () => {
      addGPUGraphAlgorithmsPageRankToGraphWithDispatchLimit(
        this,
        commandGraph,
        commandGraph.device.limits.maxComputeWorkgroupsPerDimension
      );
    });
  }
}

/** Validates a physical or view column with its exact logical row count. */
function validatePageRankColumn(
  column: GPUGraphAlgorithmsColumn<'float32'>,
  length: number,
  name: string
): void {
  if (isGPUGraphAlgorithmsViewColumn(column)) {
    validateGPUGraphAlgorithmsViewColumn(column, 'float32', length, name);
  } else {
    validatePageRankVector(column, length, name);
  }
}

/** Requires one packed, aligned floating-point output chunk with its exact logical row count. */
function validatePageRankVector(vector: GPUVector<'float32'>, length: number, name: string): void {
  if (
    vector.data.length !== 1 ||
    vector.format !== 'float32' ||
    vector.stride !== 1 ||
    vector.byteStride !== SCALAR_BYTE_LENGTH ||
    vector.rowByteLength !== SCALAR_BYTE_LENGTH ||
    vector.valueLength !== vector.length ||
    vector.bufferLayout
  ) {
    throw new Error(`${name} must contain exactly one packed float32 chunk`);
  }
  if (vector.length !== length) {
    throw new Error(`${name} must contain exactly ${length} float32 rows`);
  }

  const chunk = vector.data[0];
  if (
    chunk.format !== 'float32' ||
    chunk.length !== length ||
    chunk.stride !== 1 ||
    chunk.byteStride !== SCALAR_BYTE_LENGTH ||
    chunk.rowByteLength !== SCALAR_BYTE_LENGTH ||
    chunk.valueLength !== chunk.length ||
    !Number.isSafeInteger(chunk.byteOffset) ||
    chunk.byteOffset < 0 ||
    chunk.byteOffset % SCALAR_BYTE_LENGTH !== 0
  ) {
    throw new Error(`${name} must contain one packed, float32-aligned chunk`);
  }
}

/** Keeps ranking scores and optional residual disjoint from every existing graph allocation. */
function validateDistinctPageRankOutputs(
  id: string,
  topology: GPUGraphAlgorithmsTopology,
  output: GPUVector<'float32'>,
  residual: GPUVector<'float32'> | undefined
): void {
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
    {name: 'output', vector: output},
    ...(residual ? [{name: 'residual', vector: residual}] : [])
  ];
  for (const {name, vector} of outputs) {
    const buffer = getPhysicalBuffer(vector.data[0]);
    if (allocations.has(buffer)) {
      throw new Error(`${id} ${name} must use a distinct physical buffer allocation`);
    }
    allocations.add(buffer);
  }
}
