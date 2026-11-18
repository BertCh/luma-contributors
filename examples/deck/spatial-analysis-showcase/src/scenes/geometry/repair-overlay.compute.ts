// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPUGeometryValidity,
  GPUMakeValid,
  GPUPolygonOverlay,
  type GPUPolygonOverlayOperation,
  type GPUPolygonOverlayOutput,
  type GPUPolygonizeOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildRepairOverlayData, type PackedPolygonData} from './repair-overlay-data';

export type RepairOverlayOptions = {
  mapView: 'repair' | 'overlay';
  inputVariant: 'source' | 'simulated-crossing';
  repairView: 'input' | 'repaired';
  overlayOperation: GPUPolygonOverlayOperation;
};

const ORIGIN: [number, number] = [-87.6365, 41.882];
const INTERSECTION_CAPACITY = 32;
const SEGMENT_CAPACITY = 1024;
const RING_CAPACITY = 1024;
const VERTEX_CAPACITY = 4096;
const OVERLAY_CAPACITY = {intersections: 256, nodedSegments: 2048};

type Variant = {
  operation: GPUPolygonOverlayOperation;
  inputVariant: RepairOverlayOptions['inputVariant'];
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  dirty: boolean;
  repairPaths: [number, number][][];
  overlayPaths: [number, number][][];
};

type GeometryViews = {
  kind: 'polygons';
  positions: GraphDataView<'float32x2'>;
  featureOffsets: GraphDataView<'uint32'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
  sourceIds: GraphDataView<'uint32'>;
};

function addGeometryViews(
  graph: GPUCommandGraph<void>,
  buffers: {
    positions: Buffer;
    featureOffsets: Buffer;
    polygonOffsets: Buffer;
    ringOffsets: Buffer;
    sourceIds: Buffer;
  },
  lengths: {vertices: number; features: number; polygons: number; rings: number},
  name: string,
  importView: <Format extends 'float32x2' | 'uint32'>(
    id: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => GraphDataView<Format>
): GeometryViews {
  return {
    kind: 'polygons',
    positions: importView(`${name}-positions`, buffers.positions, 'float32x2', lengths.vertices),
    featureOffsets: importView(
      `${name}-features`,
      buffers.featureOffsets,
      'uint32',
      lengths.features + 1
    ),
    polygonOffsets: importView(
      `${name}-polygons`,
      buffers.polygonOffsets,
      'uint32',
      lengths.polygons + 1
    ),
    ringOffsets: importView(`${name}-rings`, buffers.ringOffsets, 'uint32', lengths.rings + 1),
    sourceIds: importView(`${name}-source-ids`, buffers.sourceIds, 'uint32', lengths.features)
  };
}

function makePolygonizeOutput(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  importView: <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
    id: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => GraphDataView<Format>,
  prefix: string,
  segmentCapacity: number,
  ringCapacity: number,
  vertexCapacity: number
): {output: GPUPolygonizeOutput; buffers: Buffer[]} {
  const make = (name: string, byteLength: number) =>
    resources.createBuffer(`${prefix}-${name}`, byteLength);
  const status = (name: string) => {
    const count = make(`${name}-count`, 4);
    const requiredCount = make(`${name}-required`, 4);
    const overflow = make(`${name}-overflow`, 4);
    const candidateOverflow = make(`${name}-candidate-overflow`, 4);
    const uncertainCount = make(`${name}-uncertain`, 4);
    return {
      views: {
        count: importView(`${prefix}-${name}-count`, count, 'uint32', 1),
        requiredCount: importView(`${prefix}-${name}-required`, requiredCount, 'uint32', 1),
        overflow: importView(`${prefix}-${name}-overflow`, overflow, 'uint32', 1),
        candidateOverflow: importView(
          `${prefix}-${name}-candidate-overflow`,
          candidateOverflow,
          'uint32',
          1
        ),
        uncertainCount: importView(`${prefix}-${name}-uncertain`, uncertainCount, 'uint32', 1)
      },
      buffers: [count, requiredCount, overflow, candidateOverflow, uncertainCount]
    };
  };
  const finalStatus = status('final-status');
  const {requiredCount: requiredRingCount, ...featureStatus} = finalStatus.views;
  const nodedStatus = status('noded-status');
  const directedStatus = status('directed-status');
  const directedCapacity = segmentCapacity * 2;
  const positions = make('positions', vertexCapacity * 8);
  const featureOffsets = make('feature-offsets', (ringCapacity + 1) * 4);
  const polygonOffsets = make('polygon-offsets', (ringCapacity + 1) * 4);
  const ringOffsets = make('ring-offsets', (ringCapacity + 1) * 4);
  const sourceIds = make('source-ids', ringCapacity * 4);
  const directedEdges = {
    endpoints: importView(
      'repair-directed-endpoints',
      make('directed-endpoints', directedCapacity * 16),
      'float32x4',
      directedCapacity
    ),
    sourceSegmentIds: importView(
      'repair-directed-segments',
      make('directed-segments', directedCapacity * 4),
      'uint32',
      directedCapacity
    ),
    sourceFeatureIds: importView(
      'repair-directed-features',
      make('directed-features', directedCapacity * 4),
      'uint32',
      directedCapacity
    ),
    twinIds: importView(
      'repair-directed-twins',
      make('directed-twins', directedCapacity * 4),
      'uint32',
      directedCapacity
    ),
    ringIds: importView(
      'repair-directed-rings',
      make('directed-rings', directedCapacity * 4),
      'uint32',
      directedCapacity
    ),
    flags: importView(
      'repair-directed-flags',
      make('directed-flags', directedCapacity * 4),
      'uint32',
      directedCapacity
    ),
    status: directedStatus.views
  };
  const diagnosticBuffers = [
    make('closed-rings', 4),
    make('open-chains', 4),
    make('cut-edges', 4),
    make('dangles', 4),
    make('unused-edges', 4)
  ];
  const diagnostics = {
    edgeClasses: importView(
      'repair-edge-classes',
      make('edge-classes', segmentCapacity * 4),
      'uint32',
      segmentCapacity
    ),
    closedRingCount: importView('repair-closed-rings', diagnosticBuffers[0], 'uint32', 1),
    openChainCount: importView('repair-open-chains', diagnosticBuffers[1], 'uint32', 1),
    cutEdgeCount: importView('repair-cut-edges', diagnosticBuffers[2], 'uint32', 1),
    dangleCount: importView('repair-dangles', diagnosticBuffers[3], 'uint32', 1),
    unusedEdgeCount: importView('repair-unused-edges', diagnosticBuffers[4], 'uint32', 1)
  };
  const output: GPUPolygonizeOutput = {
    nodedSegments: {
      endpoints: importView(
        'repair-noded-endpoints',
        make('noded-endpoints', segmentCapacity * 16),
        'float32x4',
        segmentCapacity
      ),
      sourceStartParameters: importView(
        'repair-noded-starts',
        make('noded-starts', segmentCapacity * 4),
        'float32',
        segmentCapacity
      ),
      sourceEndParameters: importView(
        'repair-noded-ends',
        make('noded-ends', segmentCapacity * 4),
        'float32',
        segmentCapacity
      ),
      sourceFeatureIds: importView(
        'repair-noded-features',
        make('noded-features', segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      sourceRingIds: importView(
        'repair-noded-rings',
        make('noded-rings', segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      sourceEdgeIds: importView(
        'repair-noded-edge-ids',
        make('noded-edge-ids', segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      status: nodedStatus.views
    },
    directedEdges,
    polygons: {
      kind: 'polygons',
      positions: importView('repair-output-positions', positions, 'float32x2', vertexCapacity),
      featureOffsets: importView(
        'repair-output-features',
        featureOffsets,
        'uint32',
        ringCapacity + 1
      ),
      polygonOffsets: importView(
        'repair-output-polygons',
        polygonOffsets,
        'uint32',
        ringCapacity + 1
      ),
      ringOffsets: importView('repair-output-rings', ringOffsets, 'uint32', ringCapacity + 1),
      sourceIds: importView('repair-output-source-ids', sourceIds, 'uint32', ringCapacity)
    },
    diagnostics,
    status: {...featureStatus, requiredRingCount}
  };
  return {
    output,
    buffers: [
      ...finalStatus.buffers,
      positions,
      featureOffsets,
      polygonOffsets,
      ringOffsets,
      sourceIds,
      ...diagnosticBuffers
    ]
  };
}

function unpackPaths(
  positions: Float32Array,
  ringOffsets: Uint32Array,
  ringCount: number,
  unproject: (x: number, y: number) => [number, number]
): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    if (last - first < 3) continue;
    const path = Array.from({length: last - first}, (_, index) =>
      unproject(positions[(first + index) * 2], positions[(first + index) * 2 + 1])
    );
    if (path[0][0] !== path.at(-1)![0] || path[0][1] !== path.at(-1)![1]) path.push(path[0]);
    paths.push(path);
  }
  return paths;
}

function describeMask(mask: number): string {
  if (mask === 0) return 'no listed validity issue';
  const issues = Object.entries(GPU_GEOMETRY_VALIDITY_BIT)
    .filter(([, bit]) => (mask & bit) !== 0)
    .map(([name]) => name);
  return issues.join(', ');
}

export async function createRepairOverlay(
  ctx: SceneContext<RepairOverlayOptions>
): Promise<SceneInstance<RepairOverlayOptions>> {
  const dataset = ctx.datasets.get('chicago-buildings');
  const data = buildRepairOverlayData(dataset, ORIGIN);
  const projection = dataset.getProjection(ORIGIN);
  const shared = new SpatialAnalysisResources(ctx.device, 'repair-overlay-inputs');
  const buffersFor = (prefix: string, geometry: PackedPolygonData) => ({
    positions: shared.createBuffer(`${prefix}-positions`, geometry.positions),
    ringOffsets: shared.createBuffer(`${prefix}-ring-offsets`, geometry.ringOffsets),
    polygonOffsets: shared.createBuffer(`${prefix}-polygon-offsets`, geometry.polygonOffsets),
    featureOffsets: shared.createBuffer(`${prefix}-feature-offsets`, geometry.featureOffsets),
    sourceIds: shared.createBuffer(`${prefix}-source-ids`, geometry.sourceIds)
  });
  const sourceBuffers = buffersFor('source', data.source);
  const crossingBuffers = buffersFor('simulated-crossing', data.simulatedCrossing);
  const workAreaBuffers = buffersFor('work-area-scenario', data.workArea);
  const footprintBuffers = buffersFor('selected-footprints', data.footprints);
  const variants = new Map<string, Variant>();
  const graphSwap = shared.track(new GraphSwap());
  let active: Variant | null = null;
  let destroyed = false;
  let revision = 0;
  let currentPaths: [number, number][][] = [];
  const footprintLengths = {
    vertices: data.footprints.vertexCount,
    features: data.footprints.featureCount,
    polygons: data.footprints.polygonOffsets.length - 1,
    rings: data.footprints.ringCount
  };
  const oneLengths = (geometry: PackedPolygonData) => ({
    vertices: geometry.positions.length / 2,
    features: geometry.sourceIds.length,
    polygons: geometry.polygonOffsets.length - 1,
    rings: geometry.ringOffsets.length - 1
  });

  const makeVariant = async (
    operation: GPUPolygonOverlayOperation,
    inputVariant: RepairOverlayOptions['inputVariant']
  ): Promise<Variant> => {
    const variantKey = `${operation}-${inputVariant}`;
    const resources = new SpatialAnalysisResources(ctx.device, `repair-overlay-${variantKey}`);
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `repair-overlay-${variantKey}`});
    const imported = new Map<
      Buffer,
      GraphDataView<'float32x2' | 'float32x4' | 'float32' | 'uint32'>
    >();
    const importView = <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => {
      const existing = imported.get(buffer);
      if (existing) return existing as GraphDataView<Format>;
      const view = importGraphBuffer(graph, id, buffer, format, length);
      imported.set(buffer, view);
      return view;
    };
    const input = (
      prefix: string,
      buffers: ReturnType<typeof buffersFor>,
      geometry: PackedPolygonData
    ): GeometryViews => addGeometryViews(graph, buffers, oneLengths(geometry), prefix, importView);
    const source = input('repair-source', sourceBuffers, data.source);
    const crossing = input('repair-crossing', crossingBuffers, data.simulatedCrossing);
    const workArea = input('overlay-work-area', workAreaBuffers, data.workArea);
    const footprints = addGeometryViews(
      graph,
      footprintBuffers,
      footprintLengths,
      'overlay-footprints',
      importView
    );
    const validityBuffers = [
      resources.createBuffer('validity-source-mask', 4),
      resources.createBuffer('validity-crossing-mask', 4),
      resources.createBuffer('validity-repaired-mask', RING_CAPACITY * 4),
      resources.createBuffer('validity-source-overflow', 4),
      resources.createBuffer('validity-crossing-overflow', 4),
      resources.createBuffer('validity-repaired-overflow', 4)
    ];
    const validity = (
      id: string,
      polygons: GeometryViews,
      maskBuffer: Buffer,
      overflowBuffer: Buffer,
      featureCount: number,
      ringClosure: 'explicit' | 'implicit' = 'explicit'
    ) => ({
      id,
      spatialContext: projection.spatialContext,
      polygons,
      intersectionCapacity: INTERSECTION_CAPACITY,
      ringClosure,
      orientation: 'ignore' as const,
      output: {
        issueCode: importView(`${id}-mask`, maskBuffer, 'uint32', featureCount),
        status: {overflow: importView(`${id}-overflow`, overflowBuffer, 'uint32', 1)}
      }
    });
    graph.add(
      new GPUGeometryValidity(
        validity('validity-real-source', source, validityBuffers[0], validityBuffers[3], 1)
      )
    );
    graph.add(
      new GPUGeometryValidity(
        validity('validity-simulated-crossing', crossing, validityBuffers[1], validityBuffers[4], 1)
      )
    );
    const repairBundle = makePolygonizeOutput(
      graph,
      resources,
      importView,
      'repair',
      SEGMENT_CAPACITY,
      RING_CAPACITY,
      VERTEX_CAPACITY
    );
    const repairInput = inputVariant === 'source' ? source : crossing;
    graph.add(
      new GPUMakeValid({
        id: `repair-${inputVariant}`,
        spatialContext: projection.spatialContext,
        polygons: repairInput,
        intersectionCapacity: INTERSECTION_CAPACITY,
        precision: {vertexTolerance: 0.01, predicates: 'exact-or-uncertain', storage: 'float32'},
        output: repairBundle.output
      })
    );
    const repairedOutput = repairBundle.output.polygons;
    const repaired: GeometryViews = {
      kind: 'polygons',
      positions: repairedOutput.positions,
      featureOffsets: repairedOutput.featureOffsets,
      polygonOffsets: repairedOutput.polygonOffsets,
      ringOffsets: repairedOutput.ringOffsets,
      sourceIds: repairedOutput.sourceIds
    };
    graph.add(
      new GPUGeometryValidity(
        validity(
          'validity-repaired-output',
          repaired,
          validityBuffers[2],
          validityBuffers[5],
          RING_CAPACITY,
          'implicit'
        )
      )
    );
    const overlayPositions = resources.createBuffer(
      'overlay-output-positions',
      VERTEX_CAPACITY * 8
    );
    const overlayFeatureOffsets = resources.createBuffer(
      'overlay-output-features',
      (RING_CAPACITY + 1) * 4
    );
    const overlayPolygonOffsets = resources.createBuffer(
      'overlay-output-polygons',
      (RING_CAPACITY + 1) * 4
    );
    const overlayRingOffsets = resources.createBuffer(
      'overlay-output-rings',
      (RING_CAPACITY + 1) * 4
    );
    const overlayStatus = {
      count: resources.createBuffer('overlay-count', 4),
      requiredCount: resources.createBuffer('overlay-required', 4),
      overflow: resources.createBuffer('overlay-overflow', 4),
      candidateOverflow: resources.createBuffer('overlay-candidate-overflow', 4),
      uncertainCount: resources.createBuffer('overlay-uncertain', 4),
      nodedSegmentRequiredCount: resources.createBuffer('overlay-noded-required', 4)
    };
    const boundaryStatus = {
      count: resources.createBuffer('overlay-boundary-count', 4),
      requiredCount: resources.createBuffer('overlay-boundary-required', 4),
      overflow: resources.createBuffer('overlay-boundary-overflow', 4),
      candidateOverflow: resources.createBuffer('overlay-boundary-candidate-overflow', 4)
    };
    const boundaryEndpoints = resources.createBuffer('overlay-boundary-endpoints', 4096 * 16);
    const boundaryOperandIds = resources.createBuffer('overlay-boundary-operands', 4096 * 4);
    const boundarySourceFeatureIds = resources.createBuffer('overlay-boundary-sources', 4096 * 4);
    const overlayOutput: GPUPolygonOverlayOutput = {
      geometry: {
        kind: 'polygons' as const,
        positions: importView(
          'overlay-output-positions',
          overlayPositions,
          'float32x2',
          VERTEX_CAPACITY
        ),
        featureOffsets: importView(
          'overlay-output-features',
          overlayFeatureOffsets,
          'uint32',
          RING_CAPACITY + 1
        ),
        polygonOffsets: importView(
          'overlay-output-polygons',
          overlayPolygonOffsets,
          'uint32',
          RING_CAPACITY + 1
        ),
        ringOffsets: importView(
          'overlay-output-rings',
          overlayRingOffsets,
          'uint32',
          RING_CAPACITY + 1
        )
      },
      status: {
        count: importView('overlay-status-count', overlayStatus.count, 'uint32', 1),
        requiredRingCount: importView(
          'overlay-status-required',
          overlayStatus.requiredCount,
          'uint32',
          1
        ),
        overflow: importView('overlay-status-overflow', overlayStatus.overflow, 'uint32', 1),
        candidateOverflow: importView(
          'overlay-status-candidate-overflow',
          overlayStatus.candidateOverflow,
          'uint32',
          1
        ),
        uncertainCount: importView(
          'overlay-status-uncertain',
          overlayStatus.uncertainCount,
          'uint32',
          1
        ),
        nodedSegmentRequiredCount: importView(
          'overlay-status-noded-required',
          overlayStatus.nodedSegmentRequiredCount,
          'uint32',
          1
        )
      },
      boundary: {
        endpoints: importView('overlay-boundary-endpoints', boundaryEndpoints, 'float32x4', 4096),
        operandIds: importView('overlay-boundary-operands', boundaryOperandIds, 'uint32', 4096),
        sourceFeatureIds: importView(
          'overlay-boundary-sources',
          boundarySourceFeatureIds,
          'uint32',
          4096
        ),
        status: {
          count: importView('overlay-boundary-status-count', boundaryStatus.count, 'uint32', 1),
          requiredCount: importView(
            'overlay-boundary-status-required',
            boundaryStatus.requiredCount,
            'uint32',
            1
          ),
          overflow: importView(
            'overlay-boundary-status-overflow',
            boundaryStatus.overflow,
            'uint32',
            1
          ),
          candidateOverflow: importView(
            'overlay-boundary-status-candidate-overflow',
            boundaryStatus.candidateOverflow,
            'uint32',
            1
          )
        }
      }
    };
    graph.add(
      new GPUPolygonOverlay({
        id: 'selected-polygon-overlay',
        spatialContext: projection.spatialContext,
        left: operation === 'dissolve' ? footprints : source,
        ...(operation === 'dissolve' ? {} : {right: workArea}),
        operation,
        capacity: OVERLAY_CAPACITY,
        vertexTolerance: 0.01,
        output: overlayOutput
      })
    );
    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = resources.track(await graph.compileAsync());
    } catch (error) {
      resources.destroy();
      throw error;
    }
    const readBuffers = [
      ...validityBuffers,
      ...repairBundle.buffers,
      overlayPositions,
      overlayFeatureOffsets,
      overlayPolygonOffsets,
      overlayRingOffsets,
      boundaryOperandIds,
      boundarySourceFeatureIds,
      ...Object.values(overlayStatus),
      ...Object.values(boundaryStatus)
    ];
    let builtVariant: Variant | null = null;
    const reader = new SummaryReader(
      resources,
      `repair-overlay-${variantKey}`,
      readBuffers.map(buffer => ({buffer, size: buffer.byteLength})),
      bytes => {
        if (
          destroyed ||
          !builtVariant ||
          active !== builtVariant ||
          ctx.options.overlayOperation !== operation ||
          ctx.options.inputVariant !== inputVariant
        )
          return;
        const view = new DataView(bytes);
        let offset = 0;
        const readU32 = (count = 1) => {
          const values = new Uint32Array(count);
          for (let index = 0; index < count; index++)
            values[index] = view.getUint32(offset + index * 4, true);
          offset += count * 4;
          return values;
        };
        const sourceMask = readU32()[0];
        const crossingMask = readU32()[0];
        const repairedMasks = readU32(RING_CAPACITY);
        const sourceOverflow = readU32()[0];
        const crossingOverflow = readU32()[0];
        const repairedOverflow = readU32()[0];
        const repairCount = readU32()[0];
        const repairRequired = readU32()[0];
        const repairOverflow = readU32()[0];
        const repairCandidateOverflow = readU32()[0];
        const repairUncertain = readU32()[0];
        const repairPositions = new Float32Array(bytes, offset, VERTEX_CAPACITY * 2);
        offset += VERTEX_CAPACITY * 8;
        const repairFeatureOffsets = readU32(RING_CAPACITY + 1);
        const repairPolygonOffsets = readU32(RING_CAPACITY + 1);
        const repairRingOffsets = readU32(RING_CAPACITY + 1);
        const repairSources = readU32(RING_CAPACITY);
        const diagnosticCounts = readU32(5);
        const overlayPositionsRead = new Float32Array(bytes, offset, VERTEX_CAPACITY * 2);
        offset += VERTEX_CAPACITY * 8;
        const overlayFeatureRead = readU32(RING_CAPACITY + 1);
        const overlayPolygonRead = readU32(RING_CAPACITY + 1);
        const overlayRingRead = readU32(RING_CAPACITY + 1);
        const boundaryOperandIdsRead = readU32(4096);
        const boundarySourceFeatureIdsRead = readU32(4096);
        const overlayCounts = readU32(6);
        const boundaryCounts = readU32(4);
        const repairSafe =
          !repairOverflow && !repairCandidateOverflow && !repairUncertain && repairCount > 0;
        const overlaySafe =
          !overlayCounts[2] &&
          !overlayCounts[3] &&
          !overlayCounts[4] &&
          !boundaryCounts[2] &&
          !boundaryCounts[3];
        const repairRingCount = repairSafe
          ? repairPolygonOffsets[repairFeatureOffsets[repairCount]]
          : 0;
        const overlayRingCount = overlaySafe
          ? overlayPolygonRead[overlayFeatureRead[overlayCounts[0]]]
          : 0;
        const repairPaths = repairSafe
          ? unpackPaths(repairPositions, repairRingOffsets, repairRingCount, (x, y) =>
              projection.unproject(x, y)
            )
          : [];
        const overlayPaths = overlaySafe
          ? unpackPaths(overlayPositionsRead, overlayRingRead, overlayRingCount, (x, y) =>
              projection.unproject(x, y)
            )
          : [];
        builtVariant.repairPaths = repairPaths;
        builtVariant.overlayPaths = overlayPaths;
        currentPaths =
          ctx.options.mapView === 'repair'
            ? ctx.options.repairView === 'repaired'
              ? repairPaths
              : []
            : overlayPaths;
        ctx.requestLayers();
        ctx.setReadout(
          'validitySource',
          sourceOverflow ? 'incomplete check' : describeMask(sourceMask)
        );
        ctx.setReadout(
          'validityCrossing',
          crossingOverflow ? 'incomplete check' : describeMask(crossingMask)
        );
        ctx.setReadout(
          'validityRepaired',
          repairedOverflow
            ? 'incomplete check'
            : `${repairCount} returned face(s); ${Array.from(repairedMasks.slice(0, repairCount)).filter(mask => mask !== 0).length} with listed validity flags; first: ${describeMask(repairedMasks[0] ?? 0)}`
        );
        ctx.setReadout(
          'repairCounts',
          `${formatCount(repairCount)} kept polygons / ${formatCount(repairRequired)} ring slots required`
        );
        ctx.setReadout(
          'repairDiagnostics',
          `open ${diagnosticCounts[1]}, cut ${diagnosticCounts[2]}, dangle ${diagnosticCounts[3]}, unused ${diagnosticCounts[4]}; uncertainty ${repairUncertain}; overflow ${repairOverflow || repairCandidateOverflow}`
        );
        ctx.setReadout(
          'repairSources',
          repairSafe
            ? Array.from(repairSources.slice(0, repairCount)).join(', ')
            : 'withheld: bounded output incomplete'
        );
        ctx.setReadout(
          'overlayStatus',
          `polygons ${overlayCounts[0]}, rings required ${overlayCounts[1]}, overflow ${overlayCounts[2] || overlayCounts[3] || overlayCounts[4]}; boundary edges ${boundaryCounts[0]} / required ${boundaryCounts[1]}, boundary overflow ${boundaryCounts[2] || boundaryCounts[3]}`
        );
        const leftSourceIds =
          operation === 'dissolve' ? data.footprints.sourceIds : data.source.sourceIds;
        const contributingSourceIds = new Set<number>();
        if (overlaySafe) {
          for (
            let edge = 0;
            edge < Math.min(boundaryCounts[0], boundaryOperandIdsRead.length);
            edge++
          ) {
            const sourceFeature = boundarySourceFeatureIdsRead[edge];
            const sourceId =
              boundaryOperandIdsRead[edge] === 0
                ? leftSourceIds[sourceFeature]
                : data.workArea.sourceIds[sourceFeature];
            if (sourceId !== undefined) contributingSourceIds.add(sourceId);
          }
        }
        ctx.setReadout(
          'overlaySources',
          overlaySafe
            ? Array.from(contributingSourceIds).join(', ') || 'no selected boundary contributors'
            : 'withheld: bounded output incomplete'
        );
        ctx.setChart('overlayCounts', {
          kind: 'bars',
          labels: [
            'Polygons written',
            'Rings required',
            'Boundary edges written',
            'Boundary edges required'
          ],
          values: [overlayCounts[0], overlayCounts[1], boundaryCounts[0], boundaryCounts[1]],
          description:
            'Counts written by the selected GPU overlay and its boundary provenance output.'
        });
      }
    );
    builtVariant = {
      operation,
      inputVariant,
      graph: compiled,
      resources,
      reader,
      dirty: true,
      repairPaths: [],
      overlayPaths: []
    };
    return builtVariant;
  };

  const requestVariant = (
    operation: GPUPolygonOverlayOperation,
    inputVariant: RepairOverlayOptions['inputVariant']
  ) => {
    revision++;
    ctx.setReadout('overlayStatus', 'compiling GPU overlay variant…');
    const requestedRevision = revision;
    const variantKey = `${operation}-${inputVariant}`;
    currentPaths = [];
    ctx.requestLayers();
    ctx.setReadout('validitySource', null);
    ctx.setReadout('validityCrossing', null);
    ctx.setReadout('validityRepaired', null);
    ctx.setReadout('repairCounts', null);
    ctx.setReadout('repairDiagnostics', null);
    ctx.setReadout('repairSources', null);
    ctx.setReadout('overlaySources', null);
    void graphSwap
      .request({
        build: async () => variants.get(variantKey) ?? makeVariant(operation, inputVariant),
        commit: next => {
          if (destroyed || requestedRevision !== revision) return;
          variants.set(variantKey, next);
          active = next;
          next.dirty = true;
          currentPaths =
            ctx.options.mapView === 'repair'
              ? ctx.options.repairView === 'repaired'
                ? next.repairPaths
                : []
              : next.overlayPaths;
          ctx.requestLayers();
        },
        discard: next => {
          if (!variants.has(variantKey)) {
            next.reader.stop();
            next.resources.destroy();
          }
        }
      })
      .catch(error => {
        if (!destroyed && requestedRevision === revision)
          ctx.setReadout('overlayStatus', `GPU build failed: ${String(error)}`);
      });
  };
  requestVariant(ctx.options.overlayOperation, ctx.options.inputVariant);
  const layers = () => [
    new PathLayer({
      id: 'repair-overlay-source-footprints',
      data:
        ctx.options.mapView === 'overlay'
          ? ctx.options.overlayOperation === 'dissolve'
            ? data.footprints.sourcePaths
            : [data.sourcePath]
          : [data.sourcePath],
      getPath: (row: [number, number][]) => row,
      getColor: [40, 116, 122, 200],
      getWidth: 2,
      widthUnits: 'pixels',
      pickable: false
    }),
    ...(ctx.options.mapView === 'repair'
      ? [
          new PathLayer({
            id: 'repair-overlay-selected-input',
            data: [
              ctx.options.inputVariant === 'source' ? data.sourcePath : data.simulatedCrossingPath
            ],
            getPath: (row: [number, number][]) => row,
            getColor:
              ctx.options.inputVariant === 'source' ? [40, 116, 122, 220] : [202, 111, 60, 230],
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: false
          })
        ]
      : []),
    ...(ctx.options.mapView === 'overlay' && ctx.options.overlayOperation !== 'dissolve'
      ? [
          new PathLayer({
            id: 'repair-overlay-hypothetical-area',
            data: [data.workAreaPath],
            getPath: (row: [number, number][]) => row,
            getColor: [125, 102, 157, 210],
            getWidth: 2,
            widthUnits: 'pixels',
            dash: [5, 3],
            pickable: false
          })
        ]
      : []),
    ...(ctx.options.mapView === 'overlay' || ctx.options.repairView === 'repaired'
      ? [
          new PathLayer({
            id: 'repair-overlay-gpu-result',
            data: currentPaths,
            getPath: (row: [number, number][]) => row,
            getColor: ctx.options.mapView === 'repair' ? [220, 143, 70, 240] : [77, 148, 139, 240],
            getWidth: 3,
            widthUnits: 'pixels',
            pickable: false
          })
        ]
      : [])
  ];
  return {
    getCompiledGraphs: () => Array.from(variants.values(), variant => variant.graph),
    getLayers: layers,
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
    setOption: (id, _value, state) => {
      if (id === 'overlayOperation' || id === 'inputVariant')
        requestVariant(state.overlayOperation, state.inputVariant);
      currentPaths =
        state.mapView === 'repair'
          ? state.repairView === 'repaired'
            ? (active?.repairPaths ?? [])
            : []
          : (active?.overlayPaths ?? []);
      ctx.requestLayers();
    },
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
