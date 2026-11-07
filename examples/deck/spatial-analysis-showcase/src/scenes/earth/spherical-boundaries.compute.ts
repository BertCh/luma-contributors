// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {PathLayer, ScatterplotLayer, TextLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPUSphericalPolygonContains,
  GPUSphericalPolygonOverlay,
  GPUS2PolygonCovering,
  GPUCellGeometry,
  GPUS2CellMeasures
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GraphSwap} from '../../engine/graph-swap';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  loadSphericalBoundaryInputs,
  type SphericalBoundarySource
} from './spherical-boundaries-data';
import {unwrapLongitudeNear} from './ocean-drifters-data';
import {clamp, toDegrees, toRadians} from '@math.gl/core';

export type SphericalBoundariesOptions = {
  view: 'source' | 'contains' | 'overlay' | 'cover';
  boundary: 'closed' | 'open';
  complement: boolean;
  operation: 'intersection' | 'union' | 'difference' | 'symmetric-difference';
  coverMode: 'exterior' | 'interior';
  minimumLevel: '3' | '5';
  maximumLevel: '6' | '8' | '10';
};

type OverlayResult = {
  geometry: [number, number][][];
  boundarySource: [number, number][][];
  boundaryWindow: [number, number][][];
  status: string;
  featureCount: number;
  requiredCount: number;
  complete: boolean;
};

type CoverResult = {
  paths: [number, number][][];
  count: number;
  required: number;
  visited: number;
  areaKm2: number;
  status: string;
  complete: boolean;
};

type OverlayBuffers = {
  positions: Buffer;
  ringOffsets: Buffer;
  polygonOffsets: Buffer;
  featureOffsets: Buffer;
  sourceIds: Buffer;
  count: Buffer;
  required: Buffer;
  overflow: Buffer;
  candidateOverflow: Buffer;
  boundaryEndpoints: Buffer;
  boundaryOperand: Buffer;
  boundaryFeature: Buffer;
  boundaryCount: Buffer;
  boundaryRequired: Buffer;
  boundaryOverflow: Buffer;
  boundaryCandidate: Buffer;
  complement: Buffer;
  uncertain: Buffer;
};

type CoverBuffers = {
  cells: Buffer;
  counts: Buffer;
  status: Buffer;
  visited: Buffer;
  boundaries: Buffer;
  vertexCounts: Buffer;
  measures: Buffer;
  measureStatus: Buffer;
};

const OVERLAY_VERTEX_CAPACITY = 512;
const OVERLAY_SEGMENT_CAPACITY = 1024;
const COVER_CAPACITY = 4096;
const SPHERICAL_BOUNDARY_CHART_CENTER: [number, number, number] = [
  Math.cos(toRadians(-17)) * Math.cos(toRadians(179)),
  Math.cos(toRadians(-17)) * Math.sin(toRadians(179)),
  Math.sin(toRadians(-17))
];
const OVERLAY_OPERATIONS = ['intersection', 'union', 'difference', 'symmetric-difference'] as const;

function allocateOverlay(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  operation: string
): {
  output: ConstructorParameters<typeof GPUSphericalPolygonOverlay>[0]['output'];
  buffers: OverlayBuffers;
} {
  const prefix = `overlay-${operation}`;
  const view = <Format extends 'float32x3' | 'float32x4' | 'float32x2' | 'uint32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `${prefix}-${name}`, buffer, format, length);
  const positions = resources.createBuffer(`${prefix}-positions`, OVERLAY_VERTEX_CAPACITY * 12);
  const ringOffsets = resources.createBuffer(`${prefix}-rings`, (OVERLAY_SEGMENT_CAPACITY + 1) * 4);
  const polygonOffsets = resources.createBuffer(
    `${prefix}-polygons`,
    (OVERLAY_SEGMENT_CAPACITY + 1) * 4
  );
  const featureOffsets = resources.createBuffer(
    `${prefix}-features`,
    (OVERLAY_SEGMENT_CAPACITY + 1) * 4
  );
  const sourceIds = resources.createBuffer(`${prefix}-sources`, OVERLAY_SEGMENT_CAPACITY * 4);
  const count = resources.createBuffer(`${prefix}-count`, 4);
  const required = resources.createBuffer(`${prefix}-required`, 4);
  const overflow = resources.createBuffer(`${prefix}-overflow`, 4);
  const candidateOverflow = resources.createBuffer(`${prefix}-candidate-overflow`, 4);
  const boundaryEndpoints = resources.createBuffer(
    `${prefix}-boundary-endpoints`,
    OVERLAY_SEGMENT_CAPACITY * 16
  );
  const boundaryOperand = resources.createBuffer(
    `${prefix}-boundary-operands`,
    OVERLAY_SEGMENT_CAPACITY * 4
  );
  const boundaryFeature = resources.createBuffer(
    `${prefix}-boundary-features`,
    OVERLAY_SEGMENT_CAPACITY * 4
  );
  const boundaryCount = resources.createBuffer(`${prefix}-boundary-count`, 4);
  const boundaryRequired = resources.createBuffer(`${prefix}-boundary-required`, 4);
  const boundaryOverflow = resources.createBuffer(`${prefix}-boundary-overflow`, 4);
  const boundaryCandidate = resources.createBuffer(`${prefix}-boundary-candidate`, 4);
  const complement = resources.createBuffer(`${prefix}-complement`, 4);
  const uncertain = resources.createBuffer(`${prefix}-uncertain`, 4);
  const buffers: OverlayBuffers = {
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    sourceIds,
    count,
    required,
    overflow,
    candidateOverflow,
    boundaryEndpoints,
    boundaryOperand,
    boundaryFeature,
    boundaryCount,
    boundaryRequired,
    boundaryOverflow,
    boundaryCandidate,
    complement,
    uncertain
  };
  return {
    buffers,
    output: {
      geometry: {
        kind: 'polygons',
        positions: view('positions', positions, 'float32x3', OVERLAY_VERTEX_CAPACITY),
        ringOffsets: view('ring-offsets', ringOffsets, 'uint32', OVERLAY_SEGMENT_CAPACITY + 1),
        polygonOffsets: view(
          'polygon-offsets',
          polygonOffsets,
          'uint32',
          OVERLAY_SEGMENT_CAPACITY + 1
        ),
        featureOffsets: view(
          'feature-offsets',
          featureOffsets,
          'uint32',
          OVERLAY_SEGMENT_CAPACITY + 1
        ),
        sourceIds: view('source-ids', sourceIds, 'uint32', OVERLAY_SEGMENT_CAPACITY)
      },
      complement: view('complement', complement, 'uint32', 1),
      status: {
        count: view('count', count, 'uint32', 1),
        requiredCount: view('required', required, 'uint32', 1),
        overflow: view('overflow', overflow, 'uint32', 1),
        candidateOverflow: view('candidate-overflow', candidateOverflow, 'uint32', 1),
        uncertainCount: view('uncertain', uncertain, 'uint32', 1)
      },
      boundary: {
        endpoints: view(
          'boundary-endpoints',
          boundaryEndpoints,
          'float32x4',
          OVERLAY_SEGMENT_CAPACITY
        ),
        operandIds: view('boundary-operands', boundaryOperand, 'uint32', OVERLAY_SEGMENT_CAPACITY),
        sourceFeatureIds: view(
          'boundary-features',
          boundaryFeature,
          'uint32',
          OVERLAY_SEGMENT_CAPACITY
        ),
        status: {
          count: view('boundary-count', boundaryCount, 'uint32', 1),
          requiredCount: view('boundary-required', boundaryRequired, 'uint32', 1),
          overflow: view('boundary-overflow', boundaryOverflow, 'uint32', 1),
          candidateOverflow: view('boundary-candidate', boundaryCandidate, 'uint32', 1)
        }
      }
    }
  };
}

function allocateCover(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  mode: 'exterior' | 'interior'
): {
  cover: CoverBuffers;
  views: {
    cells: GraphDataView<'uint32x2'>;
    boundaries: GraphDataView<'float32x2'>;
    vertexCounts: GraphDataView<'uint32'>;
    measures: GraphDataView<'float32x2'>;
    measureStatus: GraphDataView<'uint32'>;
  };
} {
  const prefix = `cover-${mode}`;
  const cells = resources.createBuffer(`${prefix}-cells`, COVER_CAPACITY * 8);
  const counts = resources.createBuffer(`${prefix}-counts`, 8);
  const status = resources.createBuffer(`${prefix}-status`, 4);
  const visited = resources.createBuffer(`${prefix}-visited`, 4);
  const boundaries = resources.createBuffer(`${prefix}-boundaries`, COVER_CAPACITY * 8 * 4);
  const vertexCounts = resources.createBuffer(`${prefix}-vertex-counts`, COVER_CAPACITY * 4);
  const measures = resources.createBuffer(`${prefix}-measures`, COVER_CAPACITY * 8);
  const measureStatus = resources.createBuffer(`${prefix}-measure-status`, COVER_CAPACITY * 4);
  const cover: CoverBuffers = {
    cells,
    counts,
    status,
    visited,
    boundaries,
    vertexCounts,
    measures,
    measureStatus
  };
  return {
    cover,
    views: {
      cells: importGraphBuffer(graph, `${prefix}-cells`, cells, 'uint32x2', COVER_CAPACITY),
      boundaries: importGraphBuffer(
        graph,
        `${prefix}-boundaries`,
        boundaries,
        'float32x2',
        COVER_CAPACITY * 4
      ),
      vertexCounts: importGraphBuffer(
        graph,
        `${prefix}-vertex-counts`,
        vertexCounts,
        'uint32',
        COVER_CAPACITY
      ),
      measures: importGraphBuffer(
        graph,
        `${prefix}-measures`,
        measures,
        'float32x2',
        COVER_CAPACITY
      ),
      measureStatus: importGraphBuffer(
        graph,
        `${prefix}-measure-status`,
        measureStatus,
        'uint32',
        COVER_CAPACITY
      )
    }
  };
}

function xyzToLonLat(positionX: number, positionY: number, positionZ: number): [number, number] {
  return [
    toDegrees(Math.atan2(positionY, positionX)),
    toDegrees(Math.atan2(positionZ, Math.hypot(positionX, positionY)))
  ];
}

function densifyPath(path: [number, number][], origin: number, steps = 8): [number, number][] {
  const openPath =
    path.length > 1 && path[0][0] === path.at(-1)![0] && path[0][1] === path.at(-1)![1]
      ? path.slice(0, -1)
      : path;
  const output: [number, number][] = [];
  for (let edge = 0; edge < openPath.length; edge++) {
    const first = openPath[edge];
    const second = openPath[(edge + 1) % openPath.length];
    const firstLongitude = toRadians(first[0]);
    const firstLatitude = toRadians(first[1]);
    const secondLongitude = toRadians(second[0]);
    const secondLatitude = toRadians(second[1]);
    const start = [
      Math.cos(firstLatitude) * Math.cos(firstLongitude),
      Math.cos(firstLatitude) * Math.sin(firstLongitude),
      Math.sin(firstLatitude)
    ];
    const end = [
      Math.cos(secondLatitude) * Math.cos(secondLongitude),
      Math.cos(secondLatitude) * Math.sin(secondLongitude),
      Math.sin(secondLatitude)
    ];
    const angle = Math.acos(
      clamp(start[0] * end[0] + start[1] * end[1] + start[2] * end[2], -1, 1)
    );
    const sine = Math.sin(angle);
    for (let step = 0; step < steps; step++) {
      const fraction = step / steps;
      const firstWeight = sine < 1e-8 ? 1 - fraction : Math.sin((1 - fraction) * angle) / sine;
      const secondWeight = sine < 1e-8 ? fraction : Math.sin(fraction * angle) / sine;
      const positionX = firstWeight * start[0] + secondWeight * end[0];
      const positionY = firstWeight * start[1] + secondWeight * end[1];
      const positionZ = firstWeight * start[2] + secondWeight * end[2];
      const point = xyzToLonLat(positionX, positionY, positionZ);
      output.push([unwrapLongitudeNear(origin, point[0]), point[1]]);
    }
  }
  if (output.length) output.push(output[0]);
  return output;
}

function sourcePaths(source: SphericalBoundarySource): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let ring = 0; ring < source.ringOffsets.length - 1; ring++) {
    const path = source.lonLat.slice(source.ringOffsets[ring], source.ringOffsets[ring + 1]);
    paths.push(densifyPath(path, 179));
  }
  return paths;
}

function sourcePartPaths(sources: SphericalBoundarySource[]): [number, number][][] {
  return sources.flatMap(source => sourcePaths(source));
}

function getOverlayResult(bytes: ArrayBuffer, operation: string): OverlayResult {
  const decoder = new DataView(bytes);
  const count = decoder.getUint32(0, true);
  const required = decoder.getUint32(4, true);
  const overflow = decoder.getUint32(8, true);
  const candidateOverflow = decoder.getUint32(12, true);
  const boundaryCount = decoder.getUint32(16, true);
  const boundaryRequired = decoder.getUint32(20, true);
  const boundaryOverflow = decoder.getUint32(24, true);
  const boundaryCandidate = decoder.getUint32(28, true);
  const uncertain = decoder.getUint32(32, true);
  const complement = decoder.getUint32(36, true);
  const featureOffset = 40;
  const polygonOffset = featureOffset + (OVERLAY_SEGMENT_CAPACITY + 1) * 4;
  const ringOffset = polygonOffset + (OVERLAY_SEGMENT_CAPACITY + 1) * 4;
  const positionsOffset = ringOffset + (OVERLAY_SEGMENT_CAPACITY + 1) * 4;
  const endpointOffset = positionsOffset + OVERLAY_VERTEX_CAPACITY * 12;
  const featureOffsets = new Uint32Array(bytes, featureOffset, OVERLAY_SEGMENT_CAPACITY + 1);
  const polygonOffsets = new Uint32Array(bytes, polygonOffset, OVERLAY_SEGMENT_CAPACITY + 1);
  const ringOffsets = new Uint32Array(bytes, ringOffset, OVERLAY_SEGMENT_CAPACITY + 1);
  const positions = new Float32Array(bytes, positionsOffset, OVERLAY_VERTEX_CAPACITY * 3);
  const paths: [number, number][][] = [];
  const ringCount = Math.min(
    OVERLAY_SEGMENT_CAPACITY,
    polygonOffsets[featureOffsets[Math.min(count, OVERLAY_SEGMENT_CAPACITY)]]
  );
  for (let ring = 0; ring < ringCount; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    if (last - first < 3) continue;
    const path = Array.from({length: last - first}, (_, index) => {
      const point = xyzToLonLat(
        positions[(first + index) * 3],
        positions[(first + index) * 3 + 1],
        positions[(first + index) * 3 + 2]
      );
      return [unwrapLongitudeNear(179, point[0]), point[1]] as [number, number];
    });
    paths.push(densifyPath(path, 179, 4));
  }
  const endpoints = new Float32Array(bytes, endpointOffset, OVERLAY_SEGMENT_CAPACITY * 4);
  const operands = new Uint32Array(
    bytes,
    endpointOffset + OVERLAY_SEGMENT_CAPACITY * 16,
    OVERLAY_SEGMENT_CAPACITY
  );
  const center = SPHERICAL_BOUNDARY_CHART_CENTER;
  const axis: [number, number, number] = Math.abs(center[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  const horizontal = [
    axis[1] * center[2] - axis[2] * center[1],
    axis[2] * center[0] - axis[0] * center[2],
    axis[0] * center[1] - axis[1] * center[0]
  ];
  const horizontalLength = Math.hypot(...horizontal);
  const basisHorizontal = horizontal.map(value => value / horizontalLength);
  const basisVertical = [
    center[1] * basisHorizontal[2] - center[2] * basisHorizontal[1],
    center[2] * basisHorizontal[0] - center[0] * basisHorizontal[2],
    center[0] * basisHorizontal[1] - center[1] * basisHorizontal[0]
  ];
  const lift = (x: number, y: number): [number, number] => {
    const vector = [
      center[0] + x * basisHorizontal[0] + y * basisVertical[0],
      center[1] + x * basisHorizontal[1] + y * basisVertical[1],
      center[2] + x * basisHorizontal[2] + y * basisVertical[2]
    ];
    const length = Math.hypot(...vector);
    return xyzToLonLat(vector[0] / length, vector[1] / length, vector[2] / length);
  };
  const boundarySource: [number, number][][] = [];
  const boundaryWindow: [number, number][][] = [];
  for (let index = 0; index < Math.min(boundaryCount, OVERLAY_SEGMENT_CAPACITY); index++) {
    const first = lift(endpoints[index * 4], endpoints[index * 4 + 1]);
    const second = lift(endpoints[index * 4 + 2], endpoints[index * 4 + 3]);
    const path: [number, number][] = [
      [unwrapLongitudeNear(179, first[0]), first[1]],
      [unwrapLongitudeNear(179, second[0]), second[1]]
    ];
    (operands[index] === 0 ? boundarySource : boundaryWindow).push(path);
  }
  const complete =
    overflow === 0 &&
    candidateOverflow === 0 &&
    uncertain === 0 &&
    boundaryOverflow === 0 &&
    boundaryCandidate === 0 &&
    complement === 0;
  const status =
    `${operation}: ${formatCount(count)} output polygon(s), ${formatCount(boundaryCount)} selected boundary edges; ` +
    `output status ${overflow}, required ring count ${required}, candidate overflow ${candidateOverflow}; ` +
    `boundary status ${boundaryOverflow}, required edge count ${boundaryRequired}, candidate overflow ${boundaryCandidate}; ` +
    `${uncertain} uncertain segment pairs; complement bit ${complement}. ` +
    (complete
      ? 'Output is complete.'
      : 'Output is incomplete or uses a complement the map cannot render; geometry withheld.');
  return {
    geometry: paths,
    boundarySource,
    boundaryWindow,
    status,
    featureCount: count,
    requiredCount: required,
    complete
  };
}

async function buildSphericalBoundariesVariant(
  ctx: SceneContext<SphericalBoundariesOptions>,
  isCurrentVariant: () => boolean
): Promise<SceneInstance<SphericalBoundariesOptions>> {
  const input = await loadSphericalBoundaryInputs(ctx.datasets.get('natural-earth'), ctx.signal);
  const resources = new SpatialAnalysisResources(ctx.device, 'spherical-boundaries');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'spherical-boundaries'});
  const importView = <Format extends 'float32x3' | 'float32x2' | 'uint32' | 'uint32x2'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const sourcePositionsBuffer = resources.createBuffer('source-positions', input.source.positions);
  const sourceRingsBuffer = resources.createBuffer('source-rings', input.source.ringOffsets);
  const sourcePolygonsBuffer = resources.createBuffer(
    'source-polygons',
    input.source.polygonOffsets
  );
  const windowPositionsBuffer = resources.createBuffer('window-positions', input.window.positions);
  const windowRingsBuffer = resources.createBuffer('window-rings', input.window.ringOffsets);
  const windowPolygonsBuffer = resources.createBuffer(
    'window-polygons',
    input.window.polygonOffsets
  );
  const queryBuffer = resources.createBuffer('query-positions', input.probePositions);
  const sourcePositions = importView(
    'source-positions',
    sourcePositionsBuffer,
    'float32x3',
    input.source.positions.length / 3
  );
  const sourceRings = importView(
    'source-rings',
    sourceRingsBuffer,
    'uint32',
    input.source.ringOffsets.length
  );
  const sourcePolygons = importView(
    'source-polygons',
    sourcePolygonsBuffer,
    'uint32',
    input.source.polygonOffsets.length
  );
  const windowPositions = importView(
    'window-positions',
    windowPositionsBuffer,
    'float32x3',
    input.window.positions.length / 3
  );
  const windowRings = importView(
    'window-rings',
    windowRingsBuffer,
    'uint32',
    input.window.ringOffsets.length
  );
  const windowPolygons = importView(
    'window-polygons',
    windowPolygonsBuffer,
    'uint32',
    input.window.polygonOffsets.length
  );
  const queryPositions = importView(
    'query-positions',
    queryBuffer,
    'float32x3',
    input.probePositions.length / 3
  );
  const queryIndicesBuffer = resources.createBuffer(
    'query-polygon-indices',
    new Uint32Array(input.probeLonLat.length)
  );
  const queryIndices = importView(
    'query-polygon-indices',
    queryIndicesBuffer,
    'uint32',
    input.probeLonLat.length
  );
  const containsOutputs: Array<{
    boundary: string;
    complement: boolean;
    values: Buffer;
    status: Buffer;
  }> = [];
  for (const boundary of ['closed', 'open'] as const) {
    for (const complement of [false, true]) {
      const values = resources.createBuffer(
        `contains-${boundary}-${complement}-values`,
        input.probeLonLat.length * 4
      );
      const status = resources.createBuffer(
        `contains-${boundary}-${complement}-status`,
        input.probeLonLat.length * 4
      );
      const output = {
        contains: importView(
          `contains-${boundary}-${complement}-values`,
          values,
          'uint32',
          input.probeLonLat.length
        ),
        status: importView(
          `contains-${boundary}-${complement}-status`,
          status,
          'uint32',
          input.probeLonLat.length
        )
      };
      graph.add(
        new GPUSphericalPolygonContains({
          id: `contains-${boundary}-${complement}`,
          positions: sourcePositions,
          ringOffsets: sourceRings,
          polygonOffsets: sourcePolygons,
          queries: queryPositions,
          polygonIndices: queryIndices,
          boundary,
          complement,
          tolerance: 1e-6,
          hemisphereCenter: SPHERICAL_BOUNDARY_CHART_CENTER,
          output
        })
      );
      containsOutputs.push({boundary, complement, values, status});
    }
  }

  const overlayResults = new Map<
    string,
    {
      buffers: OverlayBuffers;
      output: ConstructorParameters<typeof GPUSphericalPolygonOverlay>[0]['output'];
    }
  >();
  for (const operation of OVERLAY_OPERATIONS) {
    const allocated = allocateOverlay(graph, resources, operation);
    const leftFeatureOffsets = resources.createBuffer(
      `overlay-${operation}-left-features`,
      Uint32Array.of(0, 1)
    );
    const rightFeatureOffsets = resources.createBuffer(
      `overlay-${operation}-right-features`,
      Uint32Array.of(0, 1)
    );
    const left = {
      kind: 'polygons' as const,
      positions: sourcePositions,
      ringOffsets: sourceRings,
      polygonOffsets: sourcePolygons,
      featureOffsets: importView(
        `overlay-${operation}-left-feature-offsets`,
        leftFeatureOffsets,
        'uint32',
        2
      )
    };
    const right = {
      kind: 'polygons' as const,
      positions: windowPositions,
      ringOffsets: windowRings,
      polygonOffsets: windowPolygons,
      featureOffsets: importView(
        `overlay-${operation}-right-feature-offsets`,
        rightFeatureOffsets,
        'uint32',
        2
      )
    };
    graph.add(
      new GPUSphericalPolygonOverlay({
        id: `overlay-${operation}`,
        left,
        right,
        operation,
        chartCenter: SPHERICAL_BOUNDARY_CHART_CENTER,
        minimumCosine: 0.5,
        vertexTolerance: 2 ** -18,
        capacity: {intersections: 64, nodedSegments: 256},
        output: allocated.output
      })
    );
    overlayResults.set(operation, allocated);
  }

  const covers = new Map<'exterior' | 'interior', CoverBuffers>();
  for (const mode of ['exterior', 'interior'] as const) {
    const allocated = allocateCover(graph, resources, mode);
    const counts = importView(`cover-${mode}-counts`, allocated.cover.counts, 'uint32x2', 1);
    const status = importView(`cover-${mode}-status`, allocated.cover.status, 'uint32', 1);
    const visited = importView(`cover-${mode}-visited`, allocated.cover.visited, 'uint32', 1);
    graph.add(
      new GPUS2PolygonCovering({
        id: `s2-cover-${mode}`,
        positions: sourcePositions,
        ringOffsets: sourceRings,
        minimumLevel: Number(ctx.options.minimumLevel),
        maximumLevel: Number(ctx.options.maximumLevel),
        mode,
        capacity: COVER_CAPACITY,
        maximumVisitedCells: 65536,
        tolerance: 1e-6,
        hemisphereCenter: SPHERICAL_BOUNDARY_CHART_CENTER,
        output: {cells: allocated.views.cells, counts, status, visitedCount: visited}
      })
    );
    graph.add(
      new GPUCellGeometry({
        id: `s2-geometry-${mode}`,
        family: 's2',
        cells: allocated.views.cells,
        maximumVertexCount: 4,
        output: {boundaries: allocated.views.boundaries, vertexCounts: allocated.views.vertexCounts}
      })
    );
    graph.add(
      new GPUS2CellMeasures({
        id: `s2-measures-${mode}`,
        cells: allocated.views.cells,
        output: {measures: allocated.views.measures, status: allocated.views.measureStatus}
      })
    );
    covers.set(mode, allocated.cover);
  }

  let compiled: CompiledGPUCommandGraph<void>;
  try {
    compiled = resources.track(await graph.compileAsync());
  } catch (error) {
    resources.destroy();
    throw error;
  }
  const sourceGeometry = sourcePaths(input.source);
  const contextualSourceGeometry = sourcePartPaths(
    input.sourceParts.filter(source => source !== input.source)
  );
  const windowGeometry = sourcePaths(input.window);
  const containsSummaries = new Map<string, {values: Uint32Array; statuses: Uint32Array}>();
  const readers: SummaryReader[] = [];
  for (const item of containsOutputs) {
    const revisionKey = `${item.boundary}-${item.complement}`;
    readers.push(
      new SummaryReader(
        resources,
        `contains-${revisionKey}`,
        [
          {buffer: item.values, size: input.probeLonLat.length * 4},
          {buffer: item.status, size: input.probeLonLat.length * 4}
        ],
        bytes => {
          containsSummaries.set(revisionKey, {
            values: new Uint32Array(bytes.slice(0, input.probeLonLat.length * 4)),
            statuses: new Uint32Array(bytes.slice(input.probeLonLat.length * 4))
          });
          const accepted = new Uint32Array(bytes.slice(0, input.probeLonLat.length * 4));
          const failure = new Uint32Array(bytes.slice(input.probeLonLat.length * 4));
          containsSummaries.set(revisionKey, {values: accepted, statuses: failure});
          if (
            isCurrentVariant() &&
            ctx.options.view === 'contains' &&
            ctx.options.boundary === item.boundary &&
            ctx.options.complement === item.complement
          ) {
            publishSelectedResult();
            ctx.requestLayers();
          }
        }
      )
    );
  }
  const overlaySummaries = new Map<string, OverlayResult>();
  for (const [operation, allocated] of overlayResults) {
    const buffers = allocated.buffers;
    const sources = [
      {buffer: buffers.count, size: 4},
      {buffer: buffers.required, size: 4},
      {buffer: buffers.overflow, size: 4},
      {buffer: buffers.candidateOverflow, size: 4},
      {buffer: buffers.boundaryCount, size: 4},
      {buffer: buffers.boundaryRequired, size: 4},
      {buffer: buffers.boundaryOverflow, size: 4},
      {buffer: buffers.boundaryCandidate, size: 4},
      {buffer: buffers.uncertain, size: 4},
      {buffer: buffers.complement, size: 4},
      {buffer: buffers.featureOffsets, size: (OVERLAY_SEGMENT_CAPACITY + 1) * 4},
      {buffer: buffers.polygonOffsets, size: (OVERLAY_SEGMENT_CAPACITY + 1) * 4},
      {buffer: buffers.ringOffsets, size: (OVERLAY_SEGMENT_CAPACITY + 1) * 4},
      {buffer: buffers.positions, size: OVERLAY_VERTEX_CAPACITY * 12},
      {buffer: buffers.boundaryEndpoints, size: OVERLAY_SEGMENT_CAPACITY * 16},
      {buffer: buffers.boundaryOperand, size: OVERLAY_SEGMENT_CAPACITY * 4}
    ];
    readers.push(
      new SummaryReader(resources, `overlay-${operation}`, sources, bytes => {
        const result = getOverlayResult(bytes, operation);
        overlaySummaries.set(operation, result);
        if (
          isCurrentVariant() &&
          ctx.options.view === 'overlay' &&
          ctx.options.operation === operation
        ) {
          publishSelectedResult();
          ctx.requestLayers();
        }
      })
    );
  }
  const coverSummaries = new Map<'exterior' | 'interior', CoverResult>();
  for (const [mode, cover] of covers) {
    readers.push(
      new SummaryReader(
        resources,
        `cover-${mode}`,
        [
          {buffer: cover.counts, size: 8},
          {buffer: cover.status, size: 4},
          {buffer: cover.visited, size: 4},
          {buffer: cover.boundaries, size: COVER_CAPACITY * 8 * 4},
          {buffer: cover.vertexCounts, size: COVER_CAPACITY * 4},
          {buffer: cover.measures, size: COVER_CAPACITY * 8},
          {buffer: cover.measureStatus, size: COVER_CAPACITY * 4}
        ],
        bytes => {
          const values = new DataView(bytes);
          const written = values.getUint32(0, true);
          const required = values.getUint32(4, true);
          const status = values.getUint32(8, true);
          const visited = values.getUint32(12, true);
          const boundaryOffset = 16;
          const countOffset = boundaryOffset + COVER_CAPACITY * 32;
          const measureOffset = countOffset + COVER_CAPACITY * 4;
          const vertexCounts = new Uint32Array(bytes, countOffset, COVER_CAPACITY);
          const boundaries = new Float32Array(bytes, boundaryOffset, COVER_CAPACITY * 8);
          const measures = new Float32Array(bytes, measureOffset, COVER_CAPACITY * 2);
          const paths: [number, number][][] = [];
          let areaSteradians = 0;
          for (let row = 0; row < Math.min(written, COVER_CAPACITY); row++) {
            if (vertexCounts[row] >= 3) {
              const path: [number, number][] = [];
              for (let vertex = 0; vertex < vertexCounts[row]; vertex++)
                path.push([
                  unwrapLongitudeNear(179, boundaries[row * 8 + vertex * 2]),
                  boundaries[row * 8 + vertex * 2 + 1]
                ]);
              paths.push(densifyPath(path, 179, 4));
            }
          }
          const measureErrors = new Uint32Array(
            bytes,
            measureOffset + COVER_CAPACITY * 8,
            COVER_CAPACITY
          );
          const badMeasures = measureErrors
            .subarray(0, Math.min(written, COVER_CAPACITY))
            .reduce((sum, value) => sum + value, 0);
          if (badMeasures === 0) {
            for (let row = 0; row < Math.min(written, COVER_CAPACITY); row++) {
              if (vertexCounts[row] >= 3) areaSteradians += measures[row * 2];
            }
          }
          const areaKm2 =
            (areaSteradians * GPU_EARTH_MEAN_RADIUS_METERS * GPU_EARTH_MEAN_RADIUS_METERS) / 1e6;
          const complete = status === 0 && badMeasures === 0;
          const statusText =
            `${mode} adaptive S2: ${formatCount(written)} written, ${formatCount(required)} required so far, ` +
            `${formatCount(visited)} hierarchy cells visited; status ${status} (1 invalid geometry, 2 capacity exhausted, 4 traversal exhausted); ` +
            `${badMeasures} cell measure errors. ` +
            (complete
              ? 'Cover and measures are complete.'
              : 'Partial geometry withheld; area is unavailable.');
          coverSummaries.set(mode, {
            paths: complete ? paths : [],
            count: written,
            required,
            visited,
            areaKm2,
            status: statusText,
            complete
          });
          if (
            isCurrentVariant() &&
            ctx.options.view === 'cover' &&
            ctx.options.coverMode === mode
          ) {
            publishSelectedResult();
            ctx.requestLayers();
          }
        }
      )
    );
  }

  function publishSelectedResult(): void {
    if (!isCurrentVariant()) return;
    ctx.setReadout('insideCount', '');
    ctx.setReadout('boundaryCount', '');
    ctx.setReadout('invalidCount', '');
    ctx.setReadout('overlayFeatures', '');
    ctx.setReadout('overlayStatus', '');
    ctx.setReadout('coverCells', '');
    ctx.setReadout('coverArea', '');
    ctx.setReadout('coverStatus', '');
    if (ctx.options.view === 'source') {
      ctx.setReadout(
        'computeStatus',
        'Natural Earth 1:110m Fiji: all three published shells are drawn; calculations use the selected Viti Levu shell only. The synthetic window is a probe.'
      );
      return;
    }
    if (ctx.options.view === 'contains') {
      const result = containsSummaries.get(`${ctx.options.boundary}-${ctx.options.complement}`);
      if (!result) {
        ctx.setReadout('computeStatus', 'Containment result is pending asynchronous GPU readback.');
        return;
      }
      const insideCount = result.values.reduce((sum, value) => sum + Number(value > 0), 0);
      const boundaryCount = result.statuses.reduce((sum, value) => sum + Number(value === 2), 0);
      const invalidCount = result.statuses.reduce((sum, value) => sum + Number(value === 1), 0);
      ctx.setReadout('insideCount', formatCount(insideCount));
      ctx.setReadout('boundaryCount', formatCount(boundaryCount));
      ctx.setReadout('invalidCount', formatCount(invalidCount));
      ctx.setReadout(
        'computeStatus',
        `Containment: ${formatCount(result.values.length)} deterministic grid and exact-vertex probes; ${formatCount(insideCount)} accepted; ${formatCount(boundaryCount)} boundary, ${formatCount(invalidCount)} invalid. Tolerance 1e-6 in unit-vector space.`
      );
      return;
    }
    if (ctx.options.view === 'overlay') {
      const result = overlaySummaries.get(ctx.options.operation);
      if (!result) {
        ctx.setReadout('overlayStatus', 'Overlay result is pending asynchronous GPU readback.');
        ctx.setReadout('computeStatus', 'Overlay result is pending asynchronous GPU readback.');
        return;
      }
      if (result.complete) ctx.setReadout('overlayFeatures', formatCount(result.featureCount));
      ctx.setReadout('overlayStatus', result.status);
      ctx.setReadout(
        'computeStatus',
        result.complete
          ? `${formatCount(result.featureCount)} complete output polygon(s). The separate required count is required rings (shells plus holes), not polygons.`
          : result.status
      );
      return;
    }
    const result = coverSummaries.get(ctx.options.coverMode);
    if (!result) {
      ctx.setReadout('coverStatus', 'Cover result is pending asynchronous GPU readback.');
      ctx.setReadout('computeStatus', 'Cover result is pending asynchronous GPU readback.');
      return;
    }
    ctx.setReadout('coverCells', formatCount(result.count));
    if (result.complete)
      ctx.setReadout('coverArea', `${formatCount(Math.round(result.areaKm2))} km²`);
    ctx.setReadout('coverStatus', result.status);
    ctx.setReadout('computeStatus', result.status);
  }

  publishSelectedResult();
  let dirty = true;
  let destroyed = false;
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (destroyed) return;
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        for (const reader of readers) reader.markStale();
      }
      for (const reader of readers) reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [
        new PathLayer({
          id: 'fiji-source-shell',
          data: sourceGeometry,
          getPath: path => path,
          getColor: [18, 91, 122, 255],
          getWidth: 3.2,
          widthUnits: 'pixels',
          pickable: false
        })
      ];
      if (contextualSourceGeometry.length) {
        layers.push(
          new PathLayer({
            id: 'fiji-context-shells',
            data: contextualSourceGeometry,
            getPath: path => path,
            getColor: [116, 138, 144, 170],
            getWidth: 1.7,
            widthUnits: 'pixels',
            pickable: false
          })
        );
      }
      if (ctx.options.view === 'source' || ctx.options.view === 'overlay') {
        layers.push(
          new PathLayer({
            id: 'synthetic-window',
            data: windowGeometry,
            getPath: path => path,
            getColor: [231, 167, 76, 210],
            getWidth: 2,
            widthUnits: 'pixels',
            dashJustified: true,
            pickable: false
          })
        );
      }
      if (ctx.options.view === 'contains') {
        const key = `${ctx.options.boundary}-${ctx.options.complement}`;
        const result = containsSummaries.get(key);
        if (result) {
          const inside: [number, number][] = [];
          const outside: [number, number][] = [];
          const boundary: [number, number][] = [];
          const invalid: [number, number][] = [];
          for (let index = 0; index < input.probeLonLat.length; index++) {
            const point = input.probeLonLat[index];
            const failure = result.statuses[index];
            if (failure === 1) invalid.push(point);
            else if (failure === 2) boundary.push(point);
            else if (result.values[index]) inside.push(point);
            else outside.push(point);
          }
          for (const [id, data, color] of [
            ['inside', inside, [230, 163, 71, 230]],
            ['outside', outside, [119, 150, 162, 150]],
            ['boundary', boundary, [13, 139, 141, 255]],
            ['invalid', invalid, [196, 75, 64, 255]]
          ] as const) {
            layers.push(
              new ScatterplotLayer({
                id: `probe-${id}`,
                data,
                getPosition: point => point,
                getRadius: id === 'boundary' ? 4.4 : 2.5,
                radiusUnits: 'pixels',
                getFillColor: color,
                stroked: id === 'boundary',
                getLineColor: [251, 246, 230, 255],
                getLineWidth: 0.8,
                lineWidthUnits: 'pixels',
                pickable: false
              })
            );
          }
        }
      }
      if (ctx.options.view === 'overlay') {
        const result = overlaySummaries.get(ctx.options.operation);
        if (result?.complete) {
          layers.push(
            new PathLayer({
              id: 'overlay-result',
              data: result.geometry,
              getPath: path => path,
              getColor: [18, 142, 139, 245],
              getWidth: 3,
              widthUnits: 'pixels',
              pickable: false
            })
          );
          layers.push(
            new PathLayer({
              id: 'overlay-source-provenance',
              data: result.boundarySource,
              getPath: path => path,
              getColor: [237, 176, 75, 255],
              getWidth: 1.2,
              widthUnits: 'pixels',
              pickable: false
            })
          );
          layers.push(
            new PathLayer({
              id: 'overlay-window-provenance',
              data: result.boundaryWindow,
              getPath: path => path,
              getColor: [18, 142, 139, 255],
              getWidth: 1.2,
              widthUnits: 'pixels',
              pickable: false
            })
          );
        }
      }
      if (ctx.options.view === 'cover') {
        const result = coverSummaries.get(ctx.options.coverMode);
        if (result?.complete)
          layers.push(
            new PathLayer({
              id: 'adaptive-s2-cover',
              data: result.paths,
              getPath: path => path,
              getColor:
                ctx.options.coverMode === 'exterior' ? [42, 125, 152, 175] : [12, 145, 137, 190],
              getWidth: 1.1,
              widthUnits: 'pixels',
              pickable: false
            })
          );
      }
      layers.push(
        new TextLayer<{position: [number, number]; label: string}>({
          id: 'fiji-geographic-labels',
          data: [
            {position: [177.9, -18.43], label: 'VITI LEVU'},
            {position: [179.65, -16.98], label: 'VANUA LEVU'},
            ...(ctx.options.view === 'source' || ctx.options.view === 'overlay'
              ? [{position: [178.85, -16.08] as [number, number], label: 'SYNTHETIC WINDOW'}]
              : [])
          ],
          getPosition: row => row.position,
          getText: row => row.label,
          getColor: [42, 76, 84, 240],
          getSize: 12,
          fontFamily: 'sans-serif',
          fontWeight: 600,
          getTextAnchor: 'middle',
          getAlignmentBaseline: 'center',
          background: true,
          getBackgroundColor: [244, 241, 232, 220],
          backgroundPadding: [5, 3],
          pickable: false
        })
      );
      return layers;
    },
    setOption(id) {
      if (
        id === 'view' ||
        id === 'operation' ||
        id === 'coverMode' ||
        id === 'boundary' ||
        id === 'complement'
      ) {
        publishSelectedResult();
        ctx.requestLayers();
      }
    },
    destroy() {
      destroyed = true;
      for (const reader of readers) reader.stop();
      resources.destroy();
    }
  };
}

export async function createSphericalBoundaries(
  ctx: SceneContext<SphericalBoundariesOptions>
): Promise<SceneInstance<SphericalBoundariesOptions>> {
  let destroyed = false;
  let activeGeneration = 0;
  let active = await buildSphericalBoundariesVariant(
    ctx,
    () => !destroyed && activeGeneration === 0
  );
  let pendingLevelVariant = false;
  const variantSwap = new GraphSwap();

  function requestLevelVariant(): void {
    if (destroyed) return;
    const requestedGeneration = ++activeGeneration;
    pendingLevelVariant = true;
    ctx.setReadout('coverCells', '');
    ctx.setReadout('coverArea', '');
    ctx.setReadout('coverStatus', 'Building the requested S2 level variant.');
    ctx.setReadout('computeStatus', 'Building the requested S2 level variant.');
    ctx.requestLayers();
    void variantSwap
      .request({
        build: () =>
          buildSphericalBoundariesVariant(
            ctx,
            () => !destroyed && activeGeneration === requestedGeneration
          ),
        commit: next => {
          const previous = active;
          active = next;
          pendingLevelVariant = false;
          previous.destroy?.();
          ctx.requestLayers();
        },
        discard: next => next.destroy?.()
      })
      .catch(error => {
        if (destroyed || activeGeneration !== requestedGeneration) return;
        ctx.setReadout('coverCells', '');
        ctx.setReadout('coverArea', '');
        ctx.setReadout('coverStatus', `S2 level variant failed: ${String(error)}`);
        ctx.setReadout('computeStatus', `S2 level variant failed: ${String(error)}`);
      });
  }

  return {
    getCompiledGraphs: () => active.getCompiledGraphs(),
    encode(commandEncoder, frame) {
      active.encode(commandEncoder, frame);
    },
    getLayers: () => {
      const layers = active.getLayers();
      return pendingLevelVariant && ctx.options.view === 'cover'
        ? layers.filter(layer => layer.id !== 'adaptive-s2-cover')
        : layers;
    },
    setOption: (id, value, state) => {
      if (id === 'minimumLevel' || id === 'maximumLevel') requestLevelVariant();
      else active.setOption?.(id, value, state);
    },
    destroy() {
      destroyed = true;
      activeGeneration++;
      variantSwap.destroy();
      active.destroy?.();
    }
  };
}
