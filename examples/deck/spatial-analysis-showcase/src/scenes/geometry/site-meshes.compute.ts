// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPUConcaveHull,
  GPUConstrainedDelaunayTriangulation,
  GPUDelaunayTessellation,
  GPUGroupConcaveHull,
  GPUPolygonTriangulation,
  GPURatioConcaveHull,
  GPUVoronoiDiagram,
  getDelaunayMaximumTriangleCount,
  getPolygonTriangulationIndexCount,
  getVoronoiMaximumSegmentCount
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildSiteMeshesData, type SiteMeshesData} from './site-meshes-data';

export type SiteMeshesOptions = {
  view: 'mesh' | 'alpha' | 'ratio' | 'groups' | 'courtyard';
  alphaRadius: number;
  ratio: number;
  allowHoles: boolean;
  maximumFlips: number;
};

type MeshView = {
  kind: 'mesh';
  trianglePaths: [number, number][][];
  voronoiPaths: [number, number][][];
  circumcentres: [number, number][];
  status: string;
  count: string;
};
type HullView = {
  kind: 'hull';
  edgePaths: [number, number][][];
  groupPaths: [number, number][][][];
  status: string;
  count: string;
};
type CourtyardView = {
  kind: 'courtyard';
  earcutPaths: [number, number][][];
  constrainedPaths: [number, number][][];
  status: string;
  count: string;
};
type SiteView = MeshView | HullView | CourtyardView;
type Variant = {
  key: string;
  view: SiteMeshesOptions['view'];
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  dirty: boolean;
  result: SiteView;
};

type ViewImport = <
  Format extends 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x3'
>(
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
) => GraphDataView<Format>;

function trianglePathsFromIndices(
  positions: Float32Array,
  triangles: Uint32Array,
  triangleCount: number,
  projection: SiteMeshesData['projection']
): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const first = triangles[triangle * 3];
    const second = triangles[triangle * 3 + 1];
    const third = triangles[triangle * 3 + 2];
    paths.push([
      projection.unproject(positions[first * 2], positions[first * 2 + 1]),
      projection.unproject(positions[second * 2], positions[second * 2 + 1]),
      projection.unproject(positions[third * 2], positions[third * 2 + 1]),
      projection.unproject(positions[first * 2], positions[first * 2 + 1])
    ]);
  }
  return paths;
}

function trianglePathsFromFlatIndices(
  positions: Float32Array,
  indices: Uint32Array,
  triangleCount: number,
  projection: SiteMeshesData['projection']
): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let triangle = 0; triangle < triangleCount; triangle++) {
    const first = indices[triangle * 3];
    const second = indices[triangle * 3 + 1];
    const third = indices[triangle * 3 + 2];
    paths.push([
      projection.unproject(positions[first * 2], positions[first * 2 + 1]),
      projection.unproject(positions[second * 2], positions[second * 2 + 1]),
      projection.unproject(positions[third * 2], positions[third * 2 + 1]),
      projection.unproject(positions[first * 2], positions[first * 2 + 1])
    ]);
  }
  return paths;
}

function makeBufferView<
  Format extends 'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x3'
>(
  resources: SpatialAnalysisResources,
  importView: ViewImport,
  prefix: string,
  format: Format,
  length: number,
  bytesPerRow: number
): {buffer: Buffer; view: GraphDataView<Format>} {
  const buffer = resources.createBuffer(prefix, length * bytesPerRow);
  return {buffer, view: importView(prefix, buffer, format, length)};
}

function decodeFloats(bytes: ArrayBuffer, offset: number, count: number): Float32Array {
  return new Float32Array(bytes.slice(offset, offset + count * 4));
}
function decodeUint32(bytes: ArrayBuffer, offset: number, count: number): Uint32Array {
  return new Uint32Array(bytes.slice(offset, offset + count * 4));
}

function circumcentresFromTriangles(
  positions: Float32Array,
  triangles: Uint32Array,
  triangleCount: number,
  projection: SiteMeshesData['projection']
): [number, number][] {
  const centres: [number, number][] = [];
  for (let row = 0; row < triangleCount; row++) {
    const firstIndex = triangles[row * 3];
    const secondIndex = triangles[row * 3 + 1];
    const thirdIndex = triangles[row * 3 + 2];
    const firstX = positions[firstIndex * 2];
    const firstY = positions[firstIndex * 2 + 1];
    const secondX = positions[secondIndex * 2];
    const secondY = positions[secondIndex * 2 + 1];
    const thirdX = positions[thirdIndex * 2];
    const thirdY = positions[thirdIndex * 2 + 1];
    const denominator =
      2 * (firstX * (secondY - thirdY) + secondX * (thirdY - firstY) + thirdX * (firstY - secondY));
    if (Math.abs(denominator) < 1e-8) continue;
    const firstSquared = firstX * firstX + firstY * firstY;
    const secondSquared = secondX * secondX + secondY * secondY;
    const thirdSquared = thirdX * thirdX + thirdY * thirdY;
    const centerX =
      (firstSquared * (secondY - thirdY) +
        secondSquared * (thirdY - firstY) +
        thirdSquared * (firstY - secondY)) /
      denominator;
    const centerY =
      (firstSquared * (thirdX - secondX) +
        secondSquared * (firstX - thirdX) +
        thirdSquared * (secondX - firstX)) /
      denominator;
    centres.push(projection.unproject(centerX, centerY));
  }
  return centres;
}

export async function createSiteMeshes(
  ctx: SceneContext<SiteMeshesOptions>
): Promise<SceneInstance<SiteMeshesOptions>> {
  const facilities = ctx.datasets.get('chicago-facilities');
  const buildings = ctx.datasets.get('chicago-buildings');
  const nameTable = await facilities.loadTable(undefined, ctx.signal);
  const data = buildSiteMeshesData(facilities, buildings, nameTable.records);
  const shared = new SpatialAnalysisResources(ctx.device, 'site-meshes-input');
  const libraryPositionBuffer = shared.createBuffer('library-sites', data.libraryPositions);
  const groupedPositionBuffer = shared.createBuffer(
    'library-and-fire-sites',
    data.groupedPositions
  );
  const groupIdBuffer = shared.createBuffer('site-group-labels', data.groupIds);
  const clipBoundsBuffer = shared.createBuffer('site-voronoi-clip-bounds', data.clipBounds);
  const buildingPositionBuffer = shared.createBuffer(
    'courtyard-building-positions',
    data.buildingPositions
  );
  const buildingRingBuffer = shared.createBuffer(
    'courtyard-building-ring-offsets',
    data.buildingRingOffsets
  );
  const buildingPolygonBuffer = shared.createBuffer(
    'courtyard-building-polygon-offsets',
    data.buildingPolygonOffsets
  );
  const buildingFeatureBuffer = shared.createBuffer(
    'courtyard-building-feature-offsets',
    data.buildingFeatureOffsets
  );
  ctx.setReadout(
    'librarySites',
    `${formatCount(data.libraryLngLat.length)} archived library locations`
  );
  ctx.setReadout(
    'fireSites',
    `${formatCount(data.groupedLngLat.length - data.libraryLngLat.length)} archived fire stations in optional comparison`
  );
  ctx.setReadout('duplicateCoordinates', '0 exact duplicate coordinates among 82 library sites');
  ctx.setReadout(
    'courtyardSource',
    `Source building ${data.buildingId}: 1 polygon, ${data.buildingRingOffsets.length - 1} rings, ${data.buildingPositions.length / 2} stored vertices`
  );
  ctx.setReadout('meshStatus', 'mesh variant has not been compiled yet');
  ctx.setReadout('meshCounts', 'pending GPU output');
  ctx.setReadout('voronoiPairs', 'pending GPU output');
  ctx.setReadout('hullStatus', 'hull variant has not been compiled yet');
  ctx.setReadout('hullCounts', 'pending GPU output');
  ctx.setReadout('hullParameter', 'select a hull rule');
  ctx.setReadout('courtyardStatus', 'triangulation variant has not been compiled yet');
  ctx.setReadout('courtyardCounts', 'pending GPU output');
  const variants = new Map<string, Variant>();
  const graphSwap = shared.track(new GraphSwap());
  let active: Variant | null = null;
  let destroyed = false;
  let revision = 0;

  const buildVariant = async (options: Readonly<SiteMeshesOptions>): Promise<Variant> => {
    const key = currentKey(options);
    const resources = new SpatialAnalysisResources(ctx.device, `site-meshes-${key}`);
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `site-meshes-${key}`});
    const imported = new Map<
      Buffer,
      GraphDataView<'float32x2' | 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2' | 'uint32x3'>
    >();
    const importView: ViewImport = (id, buffer, format, length) => {
      const existing = imported.get(buffer);
      if (existing) return existing as GraphDataView<typeof format>;
      const view = importGraphBuffer(graph, id, buffer, format, length);
      imported.set(buffer, view);
      return view;
    };
    const variantBuffers: {buffer: Buffer; size: number}[] = [];
    const inputView = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importView(id, buffer, format, length);
    let result: SiteView;

    if (options.view === 'mesh') {
      const triangleCapacity = getDelaunayMaximumTriangleCount(data.libraryLngLat.length);
      const positions = inputView(
        'library-sites',
        libraryPositionBuffer,
        'float32x2',
        data.libraryLngLat.length
      );
      const triangleOutput = makeBufferView(
        resources,
        importView,
        'delaunay-triangles',
        'uint32x3',
        triangleCapacity,
        12
      );
      const triangleCount = makeBufferView(resources, importView, 'delaunay-count', 'uint32', 1, 4);
      const triangleRequired = makeBufferView(
        resources,
        importView,
        'delaunay-required',
        'uint32',
        1,
        4
      );
      const triangleOverflow = makeBufferView(
        resources,
        importView,
        'delaunay-overflow',
        'uint32',
        1,
        4
      );
      const triangleInvalid = makeBufferView(
        resources,
        importView,
        'delaunay-invalid',
        'uint32',
        1,
        4
      );
      const duplicateCount = makeBufferView(
        resources,
        importView,
        'delaunay-duplicates',
        'uint32',
        1,
        4
      );
      const converged = makeBufferView(resources, importView, 'delaunay-converged', 'uint32', 1, 4);
      graph.add(
        new GPUDelaunayTessellation({
          id: 'library-delaunay',
          spatialContext: data.projection.spatialContext,
          positions,
          output: {
            triangles: triangleOutput.view,
            status: {
              count: triangleCount.view,
              requiredCount: triangleRequired.view,
              overflow: triangleOverflow.view,
              invalidCount: triangleInvalid.view,
              duplicateCount: duplicateCount.view,
              converged: converged.view
            }
          }
        })
      );
      const voronoiCapacity = getVoronoiMaximumSegmentCount(triangleCapacity);
      const clipBounds = inputView('site-voronoi-clip-bounds', clipBoundsBuffer, 'float32x4', 1);
      const segmentOutput = makeBufferView(
        resources,
        importView,
        'voronoi-segments',
        'float32x4',
        voronoiCapacity,
        16
      );
      const siteIdOutput = makeBufferView(
        resources,
        importView,
        'voronoi-site-pairs',
        'uint32x2',
        voronoiCapacity,
        8
      );
      const voronoiCount = makeBufferView(resources, importView, 'voronoi-count', 'uint32', 1, 4);
      const voronoiRequired = makeBufferView(
        resources,
        importView,
        'voronoi-required',
        'uint32',
        1,
        4
      );
      const voronoiOverflow = makeBufferView(
        resources,
        importView,
        'voronoi-overflow',
        'uint32',
        1,
        4
      );
      const voronoiInvalid = makeBufferView(
        resources,
        importView,
        'voronoi-invalid',
        'uint32',
        1,
        4
      );
      graph.add(
        new GPUVoronoiDiagram({
          id: 'library-voronoi',
          spatialContext: data.projection.spatialContext,
          positions,
          triangles: triangleOutput.view,
          triangleCount: triangleCount.view,
          clipBounds,
          output: {
            segments: segmentOutput.view,
            siteIds: siteIdOutput.view,
            status: {
              count: voronoiCount.view,
              requiredCount: voronoiRequired.view,
              overflow: voronoiOverflow.view,
              invalidCount: voronoiInvalid.view
            }
          }
        })
      );
      variantBuffers.push(
        ...[
          triangleOutput,
          triangleCount,
          triangleRequired,
          triangleOverflow,
          triangleInvalid,
          duplicateCount,
          converged
        ].map(item => ({buffer: item.buffer, size: item.buffer.byteLength})),
        ...[
          segmentOutput,
          siteIdOutput,
          voronoiCount,
          voronoiRequired,
          voronoiOverflow,
          voronoiInvalid
        ].map(item => ({buffer: item.buffer, size: item.buffer.byteLength}))
      );
      result = {
        kind: 'mesh',
        trianglePaths: [],
        voronoiPaths: [],
        circumcentres: [],
        status: 'Delaunay and Voronoi are being calculated…',
        count: 'pending'
      };
    } else if (options.view === 'alpha' || options.view === 'ratio') {
      const triangleCapacity = getDelaunayMaximumTriangleCount(data.libraryLngLat.length);
      const positions = inputView(
        'library-sites',
        libraryPositionBuffer,
        'float32x2',
        data.libraryLngLat.length
      );
      const triangles = makeBufferView(
        resources,
        importView,
        'hull-triangles',
        'uint32x3',
        triangleCapacity,
        12
      );
      const triangleValidity = makeBufferView(
        resources,
        importView,
        'hull-triangle-validity',
        'uint32',
        triangleCapacity,
        4
      );
      const boundaryEdges = makeBufferView(
        resources,
        importView,
        'hull-boundary-edges',
        'uint32x2',
        triangleCapacity * 3,
        8
      );
      const edgeValidity = makeBufferView(
        resources,
        importView,
        'hull-edge-validity',
        'uint32',
        triangleCapacity * 3,
        4
      );
      const hullConverged = makeBufferView(resources, importView, 'hull-converged', 'uint32', 1, 4);
      const output = {
        triangles: triangles.view,
        triangleValidity: triangleValidity.view,
        boundaryEdges: boundaryEdges.view,
        edgeValidity: edgeValidity.view,
        status: {converged: hullConverged.view}
      };
      if (options.view === 'alpha') {
        graph.add(
          new GPUConcaveHull({
            id: 'library-alpha-complex',
            spatialContext: data.projection.spatialContext,
            positions,
            maximumCircumradius: Number(options.alphaRadius),
            output
          })
        );
      } else {
        graph.add(
          new GPURatioConcaveHull({
            id: 'library-ratio-hull',
            spatialContext: data.projection.spatialContext,
            positions,
            ratio: Number(options.ratio),
            allowHoles: options.allowHoles,
            maximumIterations: triangleCapacity,
            output
          })
        );
      }
      variantBuffers.push(
        ...[triangleValidity, boundaryEdges, edgeValidity, hullConverged].map(item => ({
          buffer: item.buffer,
          size: item.buffer.byteLength
        }))
      );
      result = {
        kind: 'hull',
        edgePaths: [],
        groupPaths: [],
        status: 'Hull is being calculated…',
        count: 'pending'
      };
    } else if (options.view === 'groups') {
      const groupTriangleCapacity = getDelaunayMaximumTriangleCount(data.groupIds.length) * 2;
      const positions = inputView(
        'library-and-fire-sites',
        groupedPositionBuffer,
        'float32x2',
        data.groupIds.length
      );
      const groupIds = inputView(
        'site-group-labels',
        groupIdBuffer,
        'uint32',
        data.groupIds.length
      );
      const triangles = makeBufferView(
        resources,
        importView,
        'group-hull-triangles',
        'uint32x3',
        groupTriangleCapacity,
        12
      );
      const triangleValidity = makeBufferView(
        resources,
        importView,
        'group-hull-triangle-validity',
        'uint32',
        groupTriangleCapacity,
        4
      );
      const boundaryEdges = makeBufferView(
        resources,
        importView,
        'group-hull-boundaries',
        'uint32x2',
        groupTriangleCapacity * 3,
        8
      );
      const edgeValidity = makeBufferView(
        resources,
        importView,
        'group-hull-edge-validity',
        'uint32',
        groupTriangleCapacity * 3,
        4
      );
      const groupConverged = makeBufferView(
        resources,
        importView,
        'group-hull-converged',
        'uint32',
        2,
        4
      );
      graph.add(
        new GPUGroupConcaveHull({
          id: 'facility-group-hulls',
          spatialContext: data.projection.spatialContext,
          positions,
          groupIds,
          groupCount: 2,
          maximumCircumradius: Number(options.alphaRadius),
          output: {
            triangles: triangles.view,
            triangleValidity: triangleValidity.view,
            boundaryEdges: boundaryEdges.view,
            edgeValidity: edgeValidity.view,
            status: {converged: groupConverged.view}
          }
        })
      );
      variantBuffers.push(
        ...[triangleValidity, boundaryEdges, edgeValidity, groupConverged].map(item => ({
          buffer: item.buffer,
          size: item.buffer.byteLength
        }))
      );
      result = {
        kind: 'hull',
        edgePaths: [],
        groupPaths: [[], []],
        status: 'Group hulls are being calculated…',
        count: 'pending'
      };
    } else {
      const polygonCount = data.buildingPolygonOffsets.length - 1;
      const ringCount = data.buildingRingOffsets.length - 1;
      const positionCount = data.buildingPositions.length / 2;
      const positions = inputView(
        'courtyard-building-positions',
        buildingPositionBuffer,
        'float32x2',
        positionCount
      );
      const ringOffsets = inputView(
        'courtyard-building-rings',
        buildingRingBuffer,
        'uint32',
        ringCount + 1
      );
      const polygonOffsets = inputView(
        'courtyard-building-polygons',
        buildingPolygonBuffer,
        'uint32',
        polygonCount + 1
      );
      const featureOffsets = inputView(
        'courtyard-building-features',
        buildingFeatureBuffer,
        'uint32',
        2
      );
      const indexCount = getPolygonTriangulationIndexCount(positionCount, ringCount, polygonCount);
      const makeTriangulation = (prefix: string, constrained: boolean) => {
        const indices = makeBufferView(
          resources,
          importView,
          `${prefix}-indices`,
          'uint32',
          indexCount,
          4
        );
        const valid = makeBufferView(
          resources,
          importView,
          `${prefix}-valid`,
          'uint32',
          polygonCount,
          4
        );
        const triangleCount = makeBufferView(
          resources,
          importView,
          `${prefix}-triangle-count`,
          'uint32',
          polygonCount,
          4
        );
        const triangleOffsets = makeBufferView(
          resources,
          importView,
          `${prefix}-triangle-offsets`,
          'uint32',
          polygonCount + 1,
          4
        );
        const converged = constrained
          ? makeBufferView(resources, importView, `${prefix}-converged`, 'uint32', polygonCount, 4)
          : null;
        const common = {
          id: prefix,
          spatialContext: data.projection.spatialContext,
          positions,
          featureOffsets,
          polygonOffsets,
          ringOffsets,
          maximumWork: 100_000
        };
        if (converged) {
          graph.add(
            new GPUConstrainedDelaunayTriangulation({
              ...common,
              maximumFlips: Number(options.maximumFlips),
              output: {
                indices: indices.view,
                valid: valid.view,
                triangleCount: triangleCount.view,
                triangleOffsets: triangleOffsets.view,
                status: {converged: converged.view}
              }
            })
          );
        } else {
          graph.add(
            new GPUPolygonTriangulation({
              ...common,
              output: {
                indices: indices.view,
                valid: valid.view,
                triangleCount: triangleCount.view,
                triangleOffsets: triangleOffsets.view
              }
            })
          );
        }
        const buffers = [indices, valid, triangleCount, ...(converged ? [converged] : [])];
        variantBuffers.push(
          ...buffers.map(item => ({buffer: item.buffer, size: item.buffer.byteLength}))
        );
      };
      makeTriangulation('earcut-triangulation', false);
      makeTriangulation('constrained-delaunay-triangulation', true);
      result = {
        kind: 'courtyard',
        earcutPaths: [],
        constrainedPaths: [],
        status: 'Courtyard meshes are being calculated…',
        count: 'pending'
      };
    }

    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = resources.track(await graph.compileAsync());
    } catch (error) {
      resources.destroy();
      throw error;
    }
    let builtVariant: Variant | null = null;
    const reader = new SummaryReader(resources, `site-meshes-${key}`, variantBuffers, bytes => {
      if (!builtVariant || active !== builtVariant || destroyed || currentKey(ctx.options) !== key)
        return;
      let offset = 0;
      const readUint = (count: number) => {
        const values = decodeUint32(bytes, offset, count);
        offset += count * 4;
        return values;
      };
      const readFloat = (count: number) => {
        const values = decodeFloats(bytes, offset, count);
        offset += count * 4;
        return values;
      };
      if (options.view === 'mesh') {
        const triangleCapacity = getDelaunayMaximumTriangleCount(data.libraryLngLat.length);
        const triangles = readUint(triangleCapacity * 3);
        const triangleCount = readUint(1)[0];
        const required = readUint(1)[0];
        const overflow = readUint(1)[0];
        const invalid = readUint(1)[0];
        const duplicates = readUint(1)[0];
        const converged = readUint(1)[0];
        const voronoiCapacity = getVoronoiMaximumSegmentCount(triangleCapacity);
        const segments = readFloat(voronoiCapacity * 4);
        const sitePairs = readUint(voronoiCapacity * 2);
        const voronoiCount = readUint(1)[0];
        const voronoiRequired = readUint(1)[0];
        const voronoiOverflow = readUint(1)[0];
        const voronoiInvalid = readUint(1)[0];
        const complete =
          !overflow && !invalid && converged !== 0 && !voronoiOverflow && !voronoiInvalid;
        const trianglePaths = complete
          ? trianglePathsFromIndices(
              data.libraryPositions,
              triangles,
              triangleCount,
              data.projection
            )
          : [];
        const voronoiPaths: [number, number][][] = [];
        const pairs: string[] = [];
        if (complete)
          for (let row = 0; row < voronoiCount; row++) {
            const positionX = segments[row * 4];
            const positionY = segments[row * 4 + 1];
            const endX = segments[row * 4 + 2];
            const endY = segments[row * 4 + 3];
            voronoiPaths.push([
              data.projection.unproject(positionX, positionY),
              data.projection.unproject(endX, endY)
            ]);
            if (pairs.length < 8)
              pairs.push(
                `${data.libraryNames[sitePairs[row * 2]]} ↔ ${data.libraryNames[sitePairs[row * 2 + 1]]}`
              );
          }
        const circumcentres = complete
          ? circumcentresFromTriangles(
              data.libraryPositions,
              triangles,
              triangleCount,
              data.projection
            )
          : [];
        builtVariant.result = {
          kind: 'mesh',
          trianglePaths,
          voronoiPaths,
          circumcentres,
          status: complete
            ? `required triangles ${required}; exact duplicate sites ${duplicates}; Voronoi required segments ${voronoiRequired}`
            : `incomplete: Delaunay overflow ${overflow}, invalid ${invalid}, converged ${converged}; Voronoi overflow ${voronoiOverflow}, invalid ${voronoiInvalid}`,
          count: `${triangleCount} triangles, ${voronoiCount} clipped Voronoi edges`
        };
        ctx.setReadout('meshStatus', builtVariant.result.status);
        ctx.setReadout('meshCounts', builtVariant.result.count);
        ctx.setReadout(
          'voronoiPairs',
          complete ? pairs.join('\n') : 'withheld until topology completes'
        );
        ctx.setChart('meshCountsChart', {
          kind: 'bars',
          labels: ['Delaunay triangles', 'Voronoi edges'],
          values: [triangleCount, voronoiCount],
          description: 'Counts reported by Delaunay and clipped Voronoi GPU outputs.'
        });
      } else if (
        options.view === 'alpha' ||
        options.view === 'ratio' ||
        options.view === 'groups'
      ) {
        const groupMode = options.view === 'groups';
        const pointPositions = groupMode ? data.groupedPositions : data.libraryPositions;
        const triangleCapacity = groupMode
          ? getDelaunayMaximumTriangleCount(data.groupIds.length) * 2
          : getDelaunayMaximumTriangleCount(data.libraryLngLat.length);
        const triangleValidity = readUint(triangleCapacity);
        const boundaryEdges = readUint(triangleCapacity * 3 * 2);
        const edgeValidity = readUint(triangleCapacity * 3);
        const converged = readUint(groupMode ? 2 : 1);
        const complete = converged.every(value => value !== 0);
        const groupPaths: [number, number][][][] = [[], []];
        let edgeCount = 0;
        if (complete) {
          for (let edge = 0; edge < edgeValidity.length; edge++) {
            if (edgeValidity[edge] === 0) continue;
            const first = boundaryEdges[edge * 2];
            const second = boundaryEdges[edge * 2 + 1];
            const path = [
              data.projection.unproject(pointPositions[first * 2], pointPositions[first * 2 + 1]),
              data.projection.unproject(pointPositions[second * 2], pointPositions[second * 2 + 1])
            ];
            const group = groupMode
              ? edge < getDelaunayMaximumTriangleCount(data.groupIds.length) * 3
                ? 0
                : 1
              : 0;
            groupPaths[group].push(path);
            edgeCount++;
          }
        }
        builtVariant.result = {
          kind: 'hull',
          edgePaths: groupPaths[0],
          groupPaths,
          status: complete
            ? `all ${converged.length} Delaunay/hull stage(s) converged`
            : `incomplete; converged flags ${Array.from(converged).join(', ')}`,
          count: `${edgeCount} boundary edges; ${triangleValidity.reduce((sum, value) => sum + Number(value !== 0), 0)} retained triangles`
        };
        ctx.setReadout('hullStatus', builtVariant.result.status);
        ctx.setReadout('hullCounts', builtVariant.result.count);
        ctx.setReadout(
          'hullParameter',
          options.view === 'alpha' || groupMode
            ? `maximum circumradius ${options.alphaRadius} m`
            : `edge-length ratio ${options.ratio}; holes ${options.allowHoles ? 'allowed' : 'not allowed'}`
        );
        ctx.setChart('hullCountsChart', {
          kind: 'bars',
          labels: ['Retained triangles', 'Boundary edges'],
          values: [
            triangleValidity.reduce((sum, value) => sum + Number(value !== 0), 0),
            edgeCount
          ],
          description: 'GPU validity masks and boundary-edge masks for the selected hull.'
        });
      } else {
        const indexCount = getPolygonTriangulationIndexCount(
          data.buildingPositions.length / 2,
          data.buildingRingOffsets.length - 1,
          data.buildingPolygonOffsets.length - 1
        );
        const earcutIndices = readUint(indexCount);
        const earcutValid = readUint(1)[0];
        const earcutCount = readUint(1)[0];
        const cdtIndices = readUint(indexCount);
        const cdtValid = readUint(1)[0];
        const cdtCount = readUint(1)[0];
        const cdtConverged = readUint(1)[0];
        const earcutPaths = earcutValid
          ? trianglePathsFromFlatIndices(
              data.buildingPositions,
              earcutIndices,
              earcutCount,
              data.projection
            )
          : [];
        const constrainedPaths = cdtValid
          ? trianglePathsFromFlatIndices(
              data.buildingPositions,
              cdtIndices,
              cdtCount,
              data.projection
            )
          : [];
        const constrainedStatus = !cdtValid
          ? `Earcut valid ${earcutValid}; CDT polygon valid ${cdtValid}, convergence ${cdtConverged}`
          : cdtConverged
            ? `Earcut valid ${earcutValid}; CDT polygon valid and flip process converged`
            : `Earcut valid ${earcutValid}; CDT polygon valid, but flip process did not converge`;
        builtVariant.result = {
          kind: 'courtyard',
          earcutPaths,
          constrainedPaths,
          status: constrainedStatus,
          count: `Earcut ${earcutCount} triangles; CDT ${cdtCount} triangles; source has ${data.buildingRingOffsets.length - 1} rings`
        };
        ctx.setReadout('courtyardStatus', builtVariant.result.status);
        ctx.setReadout('courtyardCounts', builtVariant.result.count);
        ctx.setChart('courtyardCountsChart', {
          kind: 'bars',
          labels: ['Earcut triangles', 'CDT triangles'],
          values: [earcutCount, cdtCount],
          description:
            'Triangle counts read from GPU triangulation outputs. The real footprint includes an interior courtyard ring.'
        });
      }
      ctx.requestLayers();
    });
    builtVariant = {
      key,
      view: options.view,
      graph: compiled,
      resources,
      reader,
      dirty: true,
      result
    };
    return builtVariant;
  };

  const currentKey = (options: Readonly<SiteMeshesOptions>) => {
    if (options.view === 'alpha' || options.view === 'groups')
      return `${options.view}-${options.alphaRadius}`;
    if (options.view === 'ratio') return `ratio-${options.ratio}-${Number(options.allowHoles)}`;
    if (options.view === 'courtyard') return `courtyard-${options.maximumFlips}`;
    return 'mesh';
  };
  const requestVariant = (options: Readonly<SiteMeshesOptions>) => {
    const requestedKey = currentKey(options);
    revision++;
    const requestedRevision = revision;
    ctx.setReadout('meshStatus', 'building GPU mesh view…');
    ctx.setReadout('hullStatus', 'building GPU hull…');
    ctx.setReadout('courtyardStatus', 'building GPU triangulations…');
    ctx.setReadout('meshCounts', 'pending GPU output');
    ctx.setReadout('voronoiPairs', 'pending GPU output');
    ctx.setReadout('hullCounts', 'pending GPU output');
    ctx.setReadout('courtyardCounts', 'pending GPU output');
    ctx.setChart('meshCountsChart', null);
    ctx.setChart('hullCountsChart', null);
    ctx.setChart('courtyardCountsChart', null);
    void graphSwap
      .request({
        build: async () => variants.get(requestedKey) ?? buildVariant(options),
        commit: next => {
          if (
            destroyed ||
            requestedRevision !== revision ||
            currentKey(ctx.options) !== requestedKey
          )
            return;
          variants.delete(requestedKey);
          variants.set(requestedKey, next);
          active = next;
          next.dirty = true;
          while (variants.size > 8) {
            const oldestKey = variants.keys().next().value;
            if (!oldestKey || oldestKey === requestedKey) break;
            const oldest = variants.get(oldestKey);
            variants.delete(oldestKey);
            if (oldest) {
              oldest.reader.release();
              oldest.resources.destroy();
            }
          }
          ctx.requestLayers();
        },
        discard: next => {
          if (!variants.has(requestedKey)) {
            next.reader.stop();
            next.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && requestedRevision === revision)
          ctx.setReadout('meshStatus', `GPU graph failed: ${String(error)}`);
      });
  };
  requestVariant(ctx.options);
  const statusLayer = () => {
    const result = active && currentKey(ctx.options) === active.key ? active.result : null;
    if (!result) return [];
    if (result.kind === 'mesh')
      return [
        new PathLayer({
          id: 'site-meshes-delaunay',
          data: result.trianglePaths,
          getPath: (path: [number, number][]) => path,
          getColor: [59, 110, 121, 105],
          getWidth: 1,
          widthUnits: 'pixels',
          pickable: false
        }),
        new PathLayer({
          id: 'site-meshes-voronoi',
          data: result.voronoiPaths,
          getPath: (path: [number, number][]) => path,
          getColor: [181, 126, 87, 205],
          getWidth: 2,
          widthUnits: 'pixels',
          pickable: false
        }),
        new ScatterplotLayer({
          id: 'site-meshes-circumcentres',
          data: result.circumcentres,
          getPosition: (point: [number, number]) => point,
          getFillColor: [190, 103, 67, 215],
          getRadius: 3,
          radiusUnits: 'pixels',
          pickable: false
        })
      ];
    if (result.kind === 'courtyard')
      return [
        new PathLayer({
          id: 'site-meshes-earcut',
          data: result.earcutPaths,
          getPath: (path: [number, number][]) => path,
          getColor: [65, 129, 135, 190],
          getWidth: 1,
          widthUnits: 'pixels',
          pickable: false
        }),
        new PathLayer({
          id: 'site-meshes-constrained',
          data: result.constrainedPaths,
          getPath: (path: [number, number][]) => path,
          getColor: [189, 122, 73, 215],
          getWidth: 2,
          widthUnits: 'pixels',
          pickable: false
        })
      ];
    return result.groupPaths.length > 1 && active?.view === 'groups'
      ? [
          new PathLayer({
            id: 'site-meshes-library-group',
            data: result.groupPaths[0],
            getPath: (path: [number, number][]) => path,
            getColor: [54, 122, 135, 210],
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: false
          }),
          new PathLayer({
            id: 'site-meshes-fire-group',
            data: result.groupPaths[1],
            getPath: (path: [number, number][]) => path,
            getColor: [187, 110, 72, 220],
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: false
          })
        ]
      : [
          new PathLayer({
            id: `site-meshes-${active?.view}`,
            data: result.edgePaths,
            getPath: (path: [number, number][]) => path,
            getColor: [178, 111, 72, 225],
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: false
          })
        ];
  };
  const siteLayers = () => {
    const activeGroups = ctx.options.view === 'groups';
    return [
      ...statusLayer(),
      ...(ctx.options.view === 'mesh'
        ? [
            new PathLayer({
              id: 'site-meshes-voronoi-clip-box',
              data: [data.clipPath],
              getPath: (path: [number, number][]) => path,
              getColor: [97, 104, 100, 150],
              getWidth: 1,
              widthUnits: 'pixels',
              pickable: false
            })
          ]
        : []),
      new ScatterplotLayer({
        id: 'site-meshes-library-sites',
        data: data.libraryLngLat,
        getPosition: (point: [number, number]) => point,
        getFillColor: [37, 114, 123, 230],
        getLineColor: [242, 235, 219, 255],
        lineWidthMinPixels: 1,
        getRadius: 5,
        radiusUnits: 'pixels',
        stroked: true,
        pickable: false
      }),
      ...(activeGroups
        ? [
            new ScatterplotLayer({
              id: 'site-meshes-fire-stations',
              data: data.groupedLngLat.slice(data.libraryLngLat.length),
              getPosition: (point: [number, number]) => point,
              getFillColor: [194, 107, 71, 225],
              getRadius: 4,
              radiusUnits: 'pixels',
              pickable: false
            })
          ]
        : []),
      ...(ctx.options.view === 'courtyard'
        ? [
            new PathLayer({
              id: 'site-meshes-courtyard-source',
              data: data.buildingPath,
              getPath: (path: [number, number][]) => path,
              getColor: [34, 104, 113, 235],
              getWidth: 3,
              widthUnits: 'pixels',
              pickable: false
            })
          ]
        : [])
    ];
  };
  return {
    getCompiledGraphs: () => Array.from(variants.values(), variant => variant.graph),
    getLayers: siteLayers,
    encode: (encoder, _frame) => {
      if (active) {
        active.graph.encode(encoder, {parameters: undefined});
        if (active.dirty) {
          active.dirty = false;
          active.reader.markStale();
        }
        active.reader.flush(encoder);
      }
    },
    setOption: (_id, _value, options) => requestVariant(options),
    destroy: () => {
      destroyed = true;
      graphSwap.destroy();
      for (const variant of variants.values()) {
        variant.reader.stop();
        variant.resources.destroy();
      }
      shared.destroy();
    }
  };
}
