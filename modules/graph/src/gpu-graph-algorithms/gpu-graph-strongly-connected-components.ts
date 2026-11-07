// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuGraph.

import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import type {GPUCommandGraph} from '../gpu-core/gpu-command-graph';
import {withGPUGraphAlgorithmsRecordScope} from './gpu-graph-computation-record';
import {validateColumn} from './gpu-graph-connected-components';
import {addGPUGraphAlgorithmsStronglyConnectedComponentsToGraph} from './gpu-graph-strongly-connected-components-internals';
import type {GPUGraphAlgorithmsTopology} from './gpu-graph-topology';
import type {
  GPUGraphAlgorithmsColumn,
  GPUGraphAlgorithmsTopologyLike
} from './gpu-graph-topology-view';

const DEFAULT_ITERATIONS = 256;
const MAXIMUM_ITERATIONS = 4096;

/** Caller-owned GPU graph topology, strong-component labels, and optional convergence status. */
export type GPUGraphAlgorithmsStronglyConnectedComponentsProps<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> = {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  id?: string;
  /** Existing forward graph adjacency; reverse adjacency is not required. */
  topology: Topology;
  /** One caller-owned packed unsigned component label for every graph vertex. */
  output: Column;
  /**
   * Bounded number of compiled state-machine steps. Every step is indirect-dispatch gated, so
   * unused steps cost no GPU work after convergence. Defaults to 256, maximum 4096.
   */
  iterations?: number;
  /** Optional caller-owned scalar that is one only when every component was resolved. */
  converged?: Column;
  /**
   * Publishes `0xffffffff` for every vertex when the budget ends before all components are
   * resolved. When false, resolved components keep their exact labels and unresolved vertices
   * publish `0xffffffff`. Defaults to true.
   */
  invalidateOnNonConvergence?: boolean;
};

/**
 * Publishes exact directed strongly connected components over GPU-resident forward CSR adjacency.
 *
 * Uses O(V) scratch (no O(V^2) tables). Each component is labelled with its lowest vertex
 * identifier, matching `GPUNetworkStructure.strongComponents`. The algorithm repeats a GPU-side
 * state machine of trimming (vertices without active in- or out-neighbours are singleton
 * components), minimum-colour propagation along forward edges, and backward reachability from each
 * colour root. Phase transitions are decided on the GPU with no CPU readback. Long chains and very
 * deep components need roughly one step per level, so raise `iterations` for such graphs;
 * a run that exhausts the budget fails closed (see `invalidateOnNonConvergence`).
 */
export class GPUGraphAlgorithmsStronglyConnectedComponents<
  Topology extends GPUGraphAlgorithmsTopologyLike = GPUGraphAlgorithmsTopology,
  Column extends GPUGraphAlgorithmsColumn<'uint32'> = GPUVector<'uint32'>
> {
  /** Prefix for generated command-graph node and imported-resource identifiers. */
  readonly id: string;
  /** Existing caller-owned GPU graph topology. */
  readonly topology: Topology;
  /** Caller-owned vertex-aligned component labels. */
  readonly output: Column;
  /** Number of compiled state-machine steps. */
  readonly iterations: number;
  /** Optional caller-owned GPU-resident convergence status. */
  readonly converged?: Column;
  /** Whether a non-converged run publishes invalid labels for every vertex. */
  readonly invalidateOnNonConvergence: boolean;

  /** Validates caller-owned graph metadata without allocating, submitting, or reading GPU work. */
  constructor(props: GPUGraphAlgorithmsStronglyConnectedComponentsProps<Topology, Column>) {
    this.id = props.id ?? 'gpu-graph-strongly-connected-components';
    this.topology = props.topology;
    this.output = props.output;
    this.iterations = props.iterations ?? DEFAULT_ITERATIONS;
    this.converged = props.converged;
    this.invalidateOnNonConvergence = props.invalidateOnNonConvergence ?? true;

    if (
      !Number.isSafeInteger(this.iterations) ||
      this.iterations < 1 ||
      this.iterations > MAXIMUM_ITERATIONS
    ) {
      throw new Error(`${this.id} iterations must be a safe integer between one and 4096`);
    }
    validateColumn(this.output, this.topology.graph.vertexCount, `${this.id} output`);
    if (this.converged) {
      validateColumn(this.converged, 1, `${this.id} converged`);
    }
  }

  /** Declares the bounded strong-component state machine without submitting or reading back. */
  addToGraph<Parameters>(commandGraph: GPUCommandGraph<Parameters>): void {
    withGPUGraphAlgorithmsRecordScope(
      commandGraph,
      'GPUGraphAlgorithmsStronglyConnectedComponents',
      this.id,
      () => {
        addGPUGraphAlgorithmsStronglyConnectedComponentsToGraph(this, commandGraph);
      }
    );
  }
}
