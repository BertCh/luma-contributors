// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUMapMatchingInput} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {
  getGPUSphereRadius,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts/spatial-context';
import {getGPUMapMatchingParameterValues} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import type {MapMatchingOperationProps} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import type {OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {getHaversineDistance, wrapLongitudeDelta} from '../network-metric-oracle';
import type {AccessibilityCSR} from '../network-accessibility/network-accessibility-oracle';

/** The junction represented by an exact edge endpoint, or undefined for an interior/missing snap. */
export function getMatchedEndpointNode(
  edge: number,
  fraction: number,
  topology: Pick<AccessibilityCSR, 'sources' | 'neighbors'>
): number | undefined {
  if (fraction === 0) return topology.sources[edge];
  if (fraction === 1) return topology.neighbors[edge];
  return undefined;
}

/** Finite emissions below the contributor's score-validity cutoff. */
export function createCutoffMatchingFixture(): ReturnType<typeof createMatchingFixture> {
  const {input, props} = createMatchingFixture();
  input.topology = {
    nodePositions: new Float32Array([0, 0, 1, 0]),
    offsets: new Uint32Array([0, 1, 1]),
    neighbors: new Uint32Array([1])
  };
  input.positions = new Float32Array([0, 1e9, 1, 1e9]);
  input.trackOffsets = new Uint32Array([0, 2]);
  input.parameters = getGPUMapMatchingParameterValues({sigma: 1e-6, beta: 1, searchRadius: 1.1e9});
  props.nodeCount = 2;
  props.edgeCount = 1;
  props.pointCount = 2;
  props.trackCount = 1;
  props.topology = {...props.topology, weights: undefined};
  props.candidateCount = 1;
  props.cellSize = 1e9;
  props.bounds = {minimum: [-1, -1], maximum: [2, 1.2e9]};
  return {input, props};
}

function getFloat32Ulp(value: number): number {
  return Math.max(2 ** -149, 2 ** (Math.floor(Math.log2(Math.abs(value))) - 23));
}

/**
 * Coordinate comparison in the context's length unit. Two output/endpoint ULPs per axis cover
 * interpolation rounding; eight f32 epsilons of the largest edge span cover projection error.
 * Longitude differences wrap, so equivalent seam coordinates are compared as the same location.
 */
export function getGeographicSnapComparison(
  actual: readonly number[],
  expected: readonly number[],
  topology: CPUMapMatchingInput['topology'],
  context: GPUSpatialContext
): {distance: number; tolerance: number} {
  const coordinateUlps = [getFloat32Ulp(expected[0]), getFloat32Ulp(expected[1])];
  const segmentExtents = [0, 0];
  for (let source = 0; source + 1 < topology.offsets.length; source++) {
    for (const axis of [0, 1])
      coordinateUlps[axis] = Math.max(
        coordinateUlps[axis],
        getFloat32Ulp(topology.nodePositions[source * 2 + axis])
      );
    for (
      let edge = topology.offsets[source];
      edge < Math.min(topology.offsets[source + 1], topology.neighbors.length);
      edge++
    ) {
      const target = topology.neighbors[edge];
      if (target >= topology.offsets.length - 1) continue;
      segmentExtents[0] = Math.max(
        segmentExtents[0],
        Math.abs(
          wrapLongitudeDelta(
            topology.nodePositions[target * 2] - topology.nodePositions[source * 2]
          )
        )
      );
      segmentExtents[1] = Math.max(
        segmentExtents[1],
        Math.abs(topology.nodePositions[target * 2 + 1] - topology.nodePositions[source * 2 + 1])
      );
    }
  }
  const degreeErrors = coordinateUlps.map(
    (ulp, axis) => 2 * ulp + 8 * 2 ** -23 * segmentExtents[axis]
  );
  const radius = getGPUSphereRadius(context);
  const radians = Math.PI / 180;
  return {
    distance: getHaversineDistance(actual, expected, radius),
    tolerance:
      radius *
      radians *
      Math.hypot(degreeErrors[0] * Math.cos(expected[1] * radians), degreeErrors[1])
  };
}

/** Two disconnected parallel one-way roads; nearest-road choices switch within track zero. */
export function createMatchingFixture(): {
  input: CPUMapMatchingInput;
  props: MapMatchingOperationProps;
} {
  return {
    input: {
      positions: new Float32Array([2, 0.5, 5, 2.5, 28, 0.5, 2, 2.5, 28, 2]),
      trackOffsets: new Uint32Array([0, 3, 5]),
      parameters: getGPUMapMatchingParameterValues({
        sigma: 1,
        beta: 2,
        searchRadius: 4,
        routeFactor: 2,
        routeSlack: 5
      }),
      topology: {
        nodePositions: new Float32Array([0, 0, 10, 0, 20, 0, 30, 0, 0, 3, 10, 3, 20, 3, 30, 3]),
        offsets: new Uint32Array([0, 1, 2, 3, 3, 4, 5, 6, 6]),
        neighbors: new Uint32Array([1, 2, 3, 5, 6, 7]),
        // Map matching must ignore these nonuniform costs and use geometric edge lengths.
        weights: new Float32Array([90, 1, 55, 3, 80, 2])
      }
    },
    props: {
      id: 'match-test',
      pointCount: 5,
      trackCount: 2,
      nodeCount: 8,
      edgeCount: 6,
      positions: 'positions',
      trackOffsets: 'trackOffsets',
      parameters: 'parameters',
      topology: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        nodePositions: 'nodePositions',
        weights: 'weights'
      },
      candidateCount: 2,
      cellSize: 10,
      bounds: {minimum: [-5, -5], maximum: [35, 8]},
      traversedEdgeCapacity: 4,
      output: {
        matchedEdges: 'matchedEdges',
        matchedFractions: 'matchedFractions',
        matchedOffsets: 'matchedOffsets',
        snapDistances: 'snapDistances',
        snappedPositions: 'snappedPositions',
        breaks: 'breaks',
        trackLogLikelihoods: 'trackLogLikelihoods',
        matchedCount: 'matchedCount',
        breakCount: 'breakCount',
        status: {
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow',
          routeOverflow: 'routeOverflow'
        },
        traversedEdges: {
          edges: 'traversedEdges',
          counts: 'traversedCounts',
          status: {overflow: 'traversedOverflow', candidateOverflow: 'traversedCandidateOverflow'}
        }
      }
    }
  };
}

/** Same typed arrays supplied to both graph operations. */
export function getMatchingGraphInputs(
  input: CPUMapMatchingInput
): Record<string, OperationGraphInput> {
  const result: Record<string, OperationGraphInput> = {
    positions: {format: 'float32x2', values: input.positions},
    trackOffsets: {format: 'uint32', values: input.trackOffsets},
    parameters: {format: 'float32', values: input.parameters},
    nodePositions: {format: 'float32x2', values: input.topology.nodePositions},
    offsets: {format: 'uint32', values: input.topology.offsets},
    neighbors: {format: 'uint32', values: input.topology.neighbors}
  };
  if (input.topology.weights)
    result['weights'] = {format: 'float32', values: input.topology.weights};
  if (input.routeTable) {
    const table = input.routeTable;
    result['tableOffsets'] = {format: 'uint32', values: table.offsets};
    result['tableTargets'] = {format: 'uint32', values: table.targets};
    result['tableCosts'] = {format: 'float32', values: table.costs};
    result['tableCostLimit'] = {format: 'float32', values: table.costLimit};
    if (table.complete) result['tableComplete'] = {format: 'uint32', values: table.complete};
  }
  return result;
}
