// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import type {VertexFormat} from '@luma.gl/core';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {GPUNetworkFlow} from '../../../src/gpu-network/network-optimization/gpu-network-flow';
import {GPUNetworkLouvain} from '../../../src/gpu-network/network-optimization/gpu-network-louvain';
import {
  GPUNetworkTimeRouting,
  prepareGPUNetworkRestrictionStates
} from '../../../src/gpu-network/network-optimization/gpu-network-time-routing';
import {GPUNetworkPointRouting} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing';
import {GPUNetworkFacilityOptimization} from '../../../src/gpu-network/network-optimization/gpu-network-facility-optimization';
import {GPUNetworkRationalAccess} from '../../../src/gpu-network/network-optimization/gpu-network-rational-access';
import {GPUTrajectoryKalman} from '../../../src/gpu-spatial-analysis/trajectory-estimation/gpu-trajectory-kalman';
import {GPUTrajectoryRestriction} from '../../../src/gpu-spatial-analysis/trajectory-restriction/gpu-trajectory-restriction';
import {GPUTrackSimilaritySearch} from '../../../src/gpu-spatial-analysis/track-similarity/gpu-track-similarity-search';

it('network and movement kernels parse, avoid reserved names, and respect portable binding limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `audit-${serial++}`, format, length);
  const contributors: GPUCommandNodeProducer[] = [
    new GPUNetworkFlow({
      nodeCount: 4,
      edges: view('uint32x2', 6),
      capacities: view('uint32', 6),
      costs: view('uint32', 6),
      parameters: view('uint32', 3),
      output: {flow: view('uint32', 6), cut: view('uint32', 4), summary: view('uint32', 8)}
    }),
    new GPUNetworkLouvain({
      nodeCount: 4,
      weights: view('float32', 16),
      parameters: view('float32', 2),
      output: {
        communities: view('uint32', 4),
        objectiveTrace: view('float32', 129),
        summary: view('uint32', 4)
      }
    }),
    new GPUNetworkTimeRouting({
      stateVertices: view('uint32', 4),
      transitions: view('uint32x2', 6),
      costs: view('float32x2', 6),
      endpoints: view('uint32', 2),
      departure: view('float32', 1),
      output: {
        arrivals: view('float32', 4),
        predecessorTransitions: view('uint32', 4),
        summary: view('uint32', 4)
      }
    }),
    new GPUNetworkPointRouting({
      turns: view('uint32x2', 6),
      costs: view('float32', 4),
      queries: view('uint32x2', 2),
      fractions: view('float32x2', 2),
      output: {
        costs: view('float32', 2),
        status: view('uint32', 2),
        predecessorEdges: view('uint32', 8)
      }
    }),
    ...(['backup-coverage', 'p-dispersion', 'capacitated-p-median'] as const).map(
      operation =>
        new GPUNetworkFacilityOptimization({
          operation,
          facilityCount: 2,
          selectCount: 2,
          costs: view('float32', 4),
          demand: view('float32', 2),
          capacities: view('float32', 2),
          threshold: view('float32', 1),
          output: {
            selectedFacilities: view('uint32', 2),
            assignments: view('uint32', 2),
            objective: view('float32', 1),
            summary: view('uint32', 4)
          }
        })
    ),
    new GPUNetworkRationalAccess({
      costs: view('float32', 4),
      demand: view('float32', 2),
      supply: view('float32', 2),
      parameters: view('float32', 6),
      output: {
        assignments: view('float32', 4),
        scores: view('float32', 2),
        unservedDemand: view('float32', 2),
        summary: view('uint32', 4)
      }
    }),
    new GPUTrajectoryKalman({
      positions: view('float32x2', 4),
      times: view('float32', 4),
      offsets: view('uint32', 3),
      parameters: view('float32', 4),
      output: {
        filteredState: view('float32x4', 4),
        filteredCovariance: view('float32x4', 4),
        status: view('uint32', 4),
        smoothedState: view('float32x4', 4),
        smoothedCovariance: view('float32x4', 4)
      }
    }),
    new GPUTrajectoryRestriction({
      positions: view('float32x2', 4),
      times: view('float32', 4),
      offsets: view('uint32', 3),
      parameters: view('float32', 7),
      output: {
        intervals: view('float32x2', 4),
        positions: view('float32x4', 4),
        status: view('uint32', 4),
        relationships: view('uint32x4', 2)
      }
    }),
    ...(['dynamic-time-warping', 'frechet', 'longest-common-subsequence'] as const).map(
      metric =>
        new GPUTrackSimilaritySearch({
          id: `search-${metric}`,
          metric,
          linesA: {kind: 'lines', positions: view('float32x2', 4), lineOffsets: view('uint32', 3)},
          pairA: view('uint32', 2),
          pairB: view('uint32', 2),
          threshold: view('float32', 1),
          maximumFrechetVertexCount: 8,
          output: {
            distance: view('float32', 2),
            lowerBound: view('float32', 2),
            similarity: view('float32', 2),
            upperBound: view('float32', 2),
            matched: view('uint32', 2),
            status: view('uint32', 2)
          }
        })
    )
  ];
  try {
    for (const contributor of contributors) {
      for (const kernel of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
        expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
    }
  } finally {
    device.destroy();
  }
});

it('restriction-state topology excludes complete forbidden histories while preserving parallel edge IDs', () => {
  const prepared = prepareGPUNetworkRestrictionStates(
    3,
    [
      {source: 0, target: 1},
      {source: 1, target: 2},
      {source: 0, target: 1},
      {source: 2, target: 0}
    ],
    [
      [0, 1],
      [1, 3, 0]
    ]
  );
  const walk = (edges: number[]) => {
    let state = 0;
    for (const originalEdge of edges) {
      let next = -1;
      for (let transition = 0; transition < prepared.originalEdges.length; transition++) {
        if (
          prepared.transitions[2 * transition] === state &&
          prepared.originalEdges[transition] === originalEdge
        )
          next = prepared.transitions[2 * transition + 1];
      }
      if (next < 0) return false;
      state = next;
    }
    return true;
  };
  expect(walk([0, 1])).toBe(false);
  expect(walk([2, 1])).toBe(true);
  expect(walk([2, 1, 3, 0])).toBe(false);
  expect(walk([2, 1, 3, 2])).toBe(true);
});

it('explicit WGS84/local epoch preparation matches pinned pyproj ECEF and preserves nanosecond differences', async () => {
  const {prepareGPUTrajectoryObservations} = await import(
    '../../../src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations'
  );
  const {default: fixture} = await import('../../oracles/fixtures/network-movement-advanced.json');
  const reference = fixture.projection;
  const epoch = 1700000000000000000n;
  const prepared = prepareGPUTrajectoryObservations({
    longitudeLatitude: reference.longitudeLatitude.flat(),
    timestamps: [epoch, epoch + 1n, epoch + 1000000000n],
    epoch,
    unitsPerSecond: 1000000000n,
    origin: [reference.origin[0], reference.origin[1]]
  });
  prepared.positions.forEach((value, index) =>
    expect(value).toBeCloseTo(reference.positions.flat()[index], 5)
  );
  expect(Array.from(prepared.times)).toEqual([0, Math.fround(1e-9), 1]);
  expect(prepared.projection.units).toBe('metres');
});

it('math.gl trajectory preparation matches the legacy hand-rolled WGS84 ENU conversion', async () => {
  const {prepareGPUTrajectoryObservations} = await import(
    '../../../src/gpu-spatial-analysis/trajectory-estimation/prepare-trajectory-observations'
  );
  const radians = Math.PI / 180;
  const eccentricitySquared = 6.6943799901413165e-3;
  const convert = (longitudeDegrees: number, latitudeDegrees: number) => {
    const longitude = longitudeDegrees * radians;
    const latitude = latitudeDegrees * radians;
    const sine = Math.sin(latitude);
    const radius = 6378137 / Math.sqrt(1 - eccentricitySquared * sine ** 2);
    return [
      radius * Math.cos(latitude) * Math.cos(longitude),
      radius * Math.cos(latitude) * Math.sin(longitude),
      radius * (1 - eccentricitySquared) * sine
    ];
  };
  const origin: [number, number] = [-122.4, 37.8];
  const coordinates = [-122.4, 37.8, -122.3, 37.9, -121.9, 38.2, 10, -45, -122.41, 37.79];
  const originLongitude = origin[0] * radians;
  const originLatitude = origin[1] * radians;
  const anchor = convert(origin[0], origin[1]);
  const timestamps = coordinates.slice(0, coordinates.length / 2).map(() => 0n);
  const prepared = prepareGPUTrajectoryObservations({
    longitudeLatitude: coordinates,
    timestamps,
    epoch: 0n,
    unitsPerSecond: 1n,
    origin
  });
  for (let row = 0; row < timestamps.length; row++) {
    const location = convert(coordinates[row * 2], coordinates[row * 2 + 1]);
    const [dx, dy, dz] = location.map((value, index) => value - anchor[index]);
    const east = -Math.sin(originLongitude) * dx + Math.cos(originLongitude) * dy;
    const north =
      -Math.sin(originLatitude) * Math.cos(originLongitude) * dx -
      Math.sin(originLatitude) * Math.sin(originLongitude) * dy +
      Math.cos(originLatitude) * dz;
    expect(Math.abs(prepared.positions[row * 2] - Math.fround(east))).toBeLessThan(
      1e-6 + 1e-7 * Math.abs(east)
    );
    expect(Math.abs(prepared.positions[row * 2 + 1] - Math.fround(north))).toBeLessThan(
      1e-6 + 1e-7 * Math.abs(north)
    );
  }
  const missing = prepareGPUTrajectoryObservations({
    longitudeLatitude: [NaN, NaN],
    timestamps: [0n],
    epoch: 0n,
    unitsPerSecond: 1n,
    origin
  });
  expect(Number.isNaN(missing.positions[0])).toBe(true);
});

it('balanced flow preparation preserves original edges and exposes explicit demand arcs', async () => {
  const {prepareGPUNetworkFlowDemands} = await import(
    '../../../src/gpu-network/network-optimization/prepare-flow-demands'
  );
  const prepared = prepareGPUNetworkFlowDemands({
    nodeCount: 3,
    edges: [
      [0, 1],
      [1, 2]
    ],
    capacities: [7, 7],
    costs: [2, 3],
    demands: [-5, 0, 5]
  });
  expect(prepared.nodeCount).toBe(5);
  expect(prepared.originalEdgeCount).toBe(2);
  expect(Array.from(prepared.edges)).toEqual([0, 1, 1, 2, 3, 0, 2, 4]);
  expect(Array.from(prepared.parameters)).toEqual([3, 4, 5]);
  expect(Array.from(prepared.costs!)).toEqual([2, 3, 0, 0]);
  expect(() =>
    prepareGPUNetworkFlowDemands({nodeCount: 2, edges: [], capacities: [], demands: [-1, 0]})
  ).toThrow();
});

it('multi-pass movement producers reject aliases that overwrite independently returned outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends VertexFormat>(format: Format, length: number) =>
    createTransientView(graph, `alias-${serial++}`, format, length);
  try {
    const values = view('float32', 1);
    expect(
      () =>
        new GPUTrackSimilaritySearch({
          metric: 'frechet',
          linesA: {kind: 'lines', positions: view('float32x2', 2), lineOffsets: view('uint32', 2)},
          pairA: view('uint32', 1),
          pairB: view('uint32', 1),
          threshold: view('float32', 1),
          output: {
            distance: values,
            lowerBound: values,
            matched: view('uint32', 1),
            status: view('uint32', 1)
          }
        })
    ).toThrow();
    const intervals = view('float32x2', 2);
    expect(
      () =>
        new GPUTrajectoryRestriction({
          positions: view('float32x2', 2),
          times: view('float32', 2),
          offsets: view('uint32', 2),
          parameters: view('float32', 7),
          output: {
            intervals,
            positions: view('float32x4', 2),
            status: view('uint32', 2),
            relationships: view('uint32x4', 1),
            compactIntervals: intervals,
            compactPositions: view('float32x4', 2),
            compactSources: view('uint32x2', 2),
            compactStatus: view('uint32', 2),
            compactSummary: view('uint32', 3)
          }
        })
    ).toThrow();
  } finally {
    device.destroy();
  }
});
