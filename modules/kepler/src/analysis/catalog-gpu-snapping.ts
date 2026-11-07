// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUNetworkSnapping} from '@luma-contributors/analysis/gpu-network';
import type {CatalogGpuContext} from './catalog-binding-types';
import type {CatalogPosition} from './catalog-geojson';
import {
  makeProjector,
  snapPointToNetwork,
  type NetworkGraph,
  type SnappedPoint
} from './catalog-network';
import {withCatalogGpuSession} from './catalog-gpu-session';

/** Sentinel written by GPUNetworkSnapping when a point snaps to no edge. */
const NO_EDGE = 0xffffffff;

/** Point-by-segment products at or above this size use the BVH candidate search instead of the full scan. */
export const SNAP_INDEX_PRODUCT_THRESHOLD = 1 << 16;

/** Candidate edges reserved per point on the BVH path; an overflow reruns the exact scan. */
const SNAP_CANDIDATES_PER_POINT = 16;

export type CatalogSnapIndexMode = 'auto' | 'bvh' | 'scan';

export type CatalogSnapOptions = {
  /** `auto` (default) indexes large point-by-segment products, `bvh` always indexes, `scan` never does. */
  spatialIndex?: CatalogSnapIndexMode;
  /** Receives which search ran: `bvh`, `scan` or `scan-after-overflow`. */
  onSearch?: (search: 'bvh' | 'scan' | 'scan-after-overflow') => void;
};

export type CatalogSnapResult = {
  snapped: (SnappedPoint | null)[];
  engine: 'gpu' | 'cpu-reference';
  /** Why the GPU did not run when engine is cpu-reference. */
  fallbackReason?: string;
};

/**
 * Snaps points to a network on the GPU with GPUNetworkSnapping.
 *
 * Graph edges are polylines while the contributor snaps to straight node-to-node edges, so each
 * polyline segment becomes its own two-node edge of a disconnected segment topology. The nearest
 * segment per point is then mapped back to its owning edge and the fraction along the whole edge
 * is rebuilt from cumulative lengths, which is exactly the CPU rule. Distances are meters in the
 * same local frame the CPU snap uses (float32 on the GPU, ties go to the smallest segment row).
 */
export async function snapCatalogPointsOnGpu(
  context: CatalogGpuContext,
  graph: NetworkGraph,
  points: readonly CatalogPosition[],
  maximumDistance: number,
  options: CatalogSnapOptions = {}
): Promise<(SnappedPoint | null)[]> {
  if (!points.length) return [];
  const project = makeProjector(graph.coordinateSystem, graph.nodes);
  const segmentEdge: number[] = [];
  const segmentIndex: number[] = [];
  const nodeValues: number[] = [];
  const neighborValues: number[] = [];
  const weightValues: number[] = [];
  graph.edges.forEach((edge, edgeIndex) => {
    for (let index = 1; index < edge.coordinates.length; index++) {
      const [ax, ay] = project(edge.coordinates[index - 1]);
      const [bx, by] = project(edge.coordinates[index]);
      const segment = segmentEdge.length;
      segmentEdge.push(edgeIndex);
      segmentIndex.push(index);
      nodeValues.push(ax, ay, bx, by);
      // Node 2s owns one slot to node 2s + 1, so slot s is segment s; node 2s + 1 owns none.
      neighborValues.push(2 * segment + 1);
      weightValues.push(Math.hypot(bx - ax, by - ay));
    }
  });
  if (!segmentEdge.length) return points.map(() => null);
  const offsets = new Uint32Array(2 * segmentEdge.length + 1);
  for (let segment = 0; segment < segmentEdge.length; segment++) {
    offsets[2 * segment] = segment;
    offsets[2 * segment + 1] = segment + 1;
  }
  offsets[2 * segmentEdge.length] = segmentEdge.length;
  const projectedPoints = new Float32Array(points.length * 2);
  points.forEach((point, pointIndex) => {
    const [x, y] = project(point);
    projectedPoints[2 * pointIndex] = x;
    projectedPoints[2 * pointIndex + 1] = y;
  });
  const segmentCount = segmentEdge.length;
  const mode = options.spatialIndex ?? 'auto';
  // The BVH path needs a positive radius; zero or non-finite distances keep the exact scan.
  const canIndex = maximumDistance > 0 && Number.isFinite(maximumDistance);
  const useIndex =
    canIndex &&
    (mode === 'bvh' ||
      (mode === 'auto' && points.length * segmentCount >= SNAP_INDEX_PRODUCT_THRESHOLD));
  const runSnap = (indexed: boolean) =>
    withCatalogGpuSession(context.device, 'catalog-snapping', async session => {
      const snappedEdges = session.output('snapped-edges', 'uint32', points.length);
      const snapFractions = session.output('snap-fractions', 'float32', points.length);
      const snapDistances = session.output('snap-distances', 'float32', points.length);
      const overflow = session.output('snap-overflow', 'uint32', 1);
      session.graph.add(
        new GPUNetworkSnapping({
          positions: session.input('points', projectedPoints, 'float32x2'),
          topology: {
            offsets: session.input('offsets', offsets, 'uint32'),
            neighbors: session.input('neighbors', Uint32Array.from(neighborValues), 'uint32'),
            weights: session.input('weights', Float32Array.from(weightValues), 'float32'),
            nodePositions: session.input('nodes', Float32Array.from(nodeValues), 'float32x2')
          },
          maximumSnapDistance: session.input(
            'maximum-distance',
            Float32Array.of(maximumDistance),
            'float32'
          ),
          ...(indexed
            ? {
                candidateCapacity: points.length * SNAP_CANDIDATES_PER_POINT,
                spatialSort: true
              }
            : {}),
          output: {
            snappedEdges: snappedEdges.view,
            snapFractions: snapFractions.view,
            snapDistances: snapDistances.view,
            ...(indexed ? {status: {overflow: overflow.view}} : {})
          }
        })
      );
      session.run();
      return {
        edges: await session.readUint32(snappedEdges),
        fractions: await session.readFloat32(snapFractions),
        distances: await session.readFloat32(snapDistances),
        overflowed: indexed ? (await session.readScalar(overflow)) !== 0 : false
      };
    });
  let search: 'bvh' | 'scan' | 'scan-after-overflow' = useIndex ? 'bvh' : 'scan';
  let result = await runSnap(useIndex).catch(failure => {
    if (!useIndex) throw failure;
    search = 'scan-after-overflow';
    return undefined;
  });
  if (!result) result = await runSnap(false);
  else if (useIndex && result.overflowed) {
    // Candidate capacity dropped work: the exact scan is the fallback, never a wrong answer.
    result = await runSnap(false);
    search = 'scan-after-overflow';
  }
  options.onSearch?.(search);
  return points.map((_, pointIndex) => {
    const segment = result.edges[pointIndex];
    if (segment === NO_EDGE || segment >= segmentEdge.length) return null;
    const edgeIndex = segmentEdge[segment];
    const edge = graph.edges[edgeIndex];
    const index = segmentIndex[segment];
    const u = Math.max(0, Math.min(1, result.fractions[pointIndex]));
    const along =
      edge.cumulative[index - 1] + u * (edge.cumulative[index] - edge.cumulative[index - 1]);
    const start = edge.coordinates[index - 1];
    const end = edge.coordinates[index];
    return {
      edge: edgeIndex,
      fraction: edge.length > 0 ? along / edge.length : 0,
      distance: result.distances[pointIndex],
      position: [start[0] + (end[0] - start[0]) * u, start[1] + (end[1] - start[1]) * u]
    };
  });
}

/** GPU snapping with the CPU snap as fallback (no device failure, unsupported size, any GPU error). */
export async function snapCatalogPoints(
  context: CatalogGpuContext | undefined,
  graph: NetworkGraph,
  points: readonly CatalogPosition[],
  maximumDistance: number,
  options: CatalogSnapOptions = {}
): Promise<CatalogSnapResult> {
  const snapOnCpu = (fallbackReason?: string): CatalogSnapResult => ({
    snapped: points.map(point => snapPointToNetwork(graph, point, maximumDistance)),
    engine: 'cpu-reference',
    ...(fallbackReason ? {fallbackReason} : {})
  });
  if (!context) return snapOnCpu('no GPU device');
  try {
    return {
      snapped: await snapCatalogPointsOnGpu(context, graph, points, maximumDistance, options),
      engine: 'gpu'
    };
  } catch (error) {
    return snapOnCpu(error instanceof Error ? error.message : 'GPU snapping failed');
  }
}
