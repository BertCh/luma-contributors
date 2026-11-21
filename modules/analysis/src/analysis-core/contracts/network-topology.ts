// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {DynamicBuffer} from '@luma.gl/core/dynamic';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {validateSinglePackedView} from '../kernels/gpu-contributor-utils';
import type {
  GPUCandidateStatusPort,
  GPUCardinalityPort,
  GPUOverflowStatusPort,
  GPUStatusScalar,
  GPUVertexCountStatusPort
} from './status';

/**
 * A network in compressed sparse row (CSR) form, the one network input of every `gpu-network`
 * contributor.
 *
 * Node `u`'s outgoing edges occupy slots `[offsets[u], offsets[u + 1])` of `neighbors` and
 * `weights`, so `offsets` has `nodeCount + 1` rows and `neighbors.length` is the edge capacity.
 * An undirected network lists every edge in both directions.
 *
 * The CSR part (`offsets`, `neighbors`, `weights`) has the names and formats of
 * `GPUSpatialWeights`, but reuse as spatial weights requires strictly ascending, unique neighbor
 * rows within each CSR row, no self pairs, and finite non-negative weights. These invariants are
 * not required of a network; check them with `GPUSpatialWeightsValidation` (or
 * `computeSpatialWeightsValidationOnCPU`) before the handoff and explicitly choose the statistical
 * meaning of each weight (travel cost is not automatically statistical influence). Parallel edges
 * require an explicit aggregation policy before reuse.
 *
 * This explicitly sized topology treats every row up to `offsets.length - 1` as a node. For a
 * noding result with spare node capacity, retain {@link GPUNodedNetworkTopologyPort} instead.
 */
export type GPUNetworkTopology = {
  /** `nodeCount + 1` exclusive row offsets into the edge slots. */
  offsets: GraphDataView<'uint32'>;
  /** Target node per edge slot. Targets `>= nodeCount` are skipped by every contributor. */
  neighbors: GraphDataView<'uint32'>;
  /** Travel cost per edge slot, aligned with `neighbors`. Negative or NaN costs are impassable. */
  weights: GraphDataView<'float32'>;
  /** Optional position per node, required by the contributors that measure geometry. */
  nodePositions?: GraphDataView<'float32x2'>;
};

/**
 * Non-owning handoff of a noded network, retaining its live node count and supplied build status.
 * A `GPUNetworkNodingOutput` is assignable directly, preserving every view by identity.
 *
 * Only node rows in `[0, nodes.status.count)` are live; `topology.offsets.length - 1` is node
 * capacity. Consumers must bound start/target rows by the GPU-written count and propagate or
 * refuse supplied build loss (`status` and `pieces.status`) independently of search convergence.
 * Overall status remains optional; its omission does not certify a complete source network.
 * Piece count and overflow are mandatory, as on the producer, so its guaranteed loss evidence
 * cannot be omitted from this handoff.
 */
export type GPUNodedNetworkTopologyPort = {
  topology: GPUNetworkTopology;
  /** Live node cardinality, retained from the noding output's `nodes.status`. */
  nodes: {status: GPUCardinalityPort};
  /** Producer-specific build diagnostics, not canonical fallback or final-capacity status. */
  status?: {
    /** Legacy producer spelling: uncertified segment pairs left unsplit, not fallback successes. */
    uncertainCount?: GPUStatusScalar;
    /** One after piece/vertex/intersection/node capacity loss OR tolerance-grid range failure. */
    overflow?: GPUStatusScalar;
  };
  /** Piece-generation loss, including final piece/vertex loss and intermediate candidate loss. */
  pieces: {
    status: GPUCardinalityPort &
      GPUOverflowStatusPort &
      GPUCandidateStatusPort &
      GPUVertexCountStatusPort;
  };
};

/**
 * A {@link GPUNetworkTopology} read for its connectivity, where edge weights are optional. Every
 * `GPUNetworkTopology` is a `GPUNetworkAdjacency`, so one network object feeds every contributor.
 */
export type GPUNetworkAdjacency = Omit<GPUNetworkTopology, 'weights'> & {
  /** Optional travel cost or weight per edge slot, aligned with `neighbors`. */
  weights?: GraphDataView<'float32'>;
};

/** Which optional topology fields a contributor requires. @internal */
export type GPUNetworkTopologyRequirements = {
  /** Require `weights`. */
  weights?: boolean;
  /** Require `nodePositions`. */
  nodePositions?: boolean;
};

/** Node and edge-slot counts of a validated topology. @internal */
export type GPUNetworkTopologyShape = {
  /** `offsets.length - 1`. */
  nodeCount: number;
  /** `neighbors.length`, the edge-slot capacity. */
  edgeCount: number;
};

/**
 * Validates the formats and lengths of a network topology and returns its node and edge counts.
 *
 * @internal
 * @throws If a view is not packed in its format, the topology has no node, or `weights` or
 * `nodePositions` are misaligned or missing while required.
 */
export function validateGPUNetworkTopology(
  id: string,
  topology: GPUNetworkAdjacency,
  requirements: GPUNetworkTopologyRequirements = {}
): GPUNetworkTopologyShape {
  if (!topology) {
    throw new Error(`${id} topology is required`);
  }
  validateSinglePackedView(topology.offsets, ['uint32'], `${id} topology.offsets`);
  validateSinglePackedView(topology.neighbors, ['uint32'], `${id} topology.neighbors`);
  const nodeCount = topology.offsets.length - 1;
  const edgeCount = topology.neighbors.length;
  if (nodeCount < 1) {
    throw new Error(`${id} topology.offsets must contain at least two rows`);
  }
  if (topology.weights) {
    validateSinglePackedView(topology.weights, ['float32'], `${id} topology.weights`);
    if (topology.weights.length !== edgeCount) {
      throw new Error(`${id} topology.weights length must equal topology.neighbors length`);
    }
  } else if (requirements.weights) {
    throw new Error(`${id} topology.weights is required`);
  }
  if (topology.nodePositions) {
    validateSinglePackedView(topology.nodePositions, ['float32x2'], `${id} topology.nodePositions`);
    if (topology.nodePositions.length !== nodeCount) {
      throw new Error(`${id} topology.nodePositions must contain one row per node`);
    }
  } else if (requirements.nodePositions) {
    throw new Error(`${id} topology.nodePositions is required`);
  }
  return {nodeCount, edgeCount};
}

/** Returns the views of a topology, for graph-membership and aliasing checks. @internal */
export function getGPUNetworkTopologyViews(
  topology: GPUNetworkAdjacency
): (GraphDataView | undefined)[] {
  return [topology.offsets, topology.neighbors, topology.weights, topology.nodePositions];
}

/**
 * Returns the physical buffer behind a graph view's default import.
 *
 * @internal
 * @throws If the view was not imported with a default buffer, as transient views are not.
 */
export function getGraphViewDefaultBuffer(
  id: string,
  name: string,
  view: GraphDataView
): Buffer | DynamicBuffer {
  const defaultBuffer = view.buffer.defaultBuffer;
  if (!defaultBuffer) {
    throw new Error(`${id} ${name} must be imported with a default buffer`);
  }
  return defaultBuffer;
}

/**
 * Resolves a default import to its current core buffer, unwrapping a `DynamicBuffer`.
 *
 * @internal
 */
export function getCoreDefaultBuffer(buffer: Buffer | DynamicBuffer): Buffer {
  // The published dynamic wrapper exposes its current core allocation through `buffer`.
  // Constructor identity differs between ESM and CommonJS; core Buffer has no such property.
  return 'buffer' in buffer ? buffer.buffer : buffer;
}

/**
 * Wraps a packed, imported graph view in a `GPUVector` over the same physical buffer.
 *
 * `GPUCommandGraph.importGPUVector` de-duplicates on the physical buffer, so gpu-graph algorithms
 * that import this vector resolve to the handle the caller already imported and hazards stay
 * tracked between them and the contributor's own nodes.
 *
 * @internal
 * @throws If the view was not imported with a default buffer.
 */
export function getGraphViewGPUVector<Format extends 'uint32' | 'float32'>(
  id: string,
  name: string,
  view: GraphDataView<Format>
): GPUVector<Format> {
  return new GPUVector<Format>({
    type: 'buffer',
    name,
    buffer: getGraphViewDefaultBuffer(id, name, view),
    format: view.format,
    length: view.length,
    byteOffset: view.byteOffset
  });
}
