// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPUCanonicalPolygonEquality,
  GPUNormalizePolygons,
  GPUPolygonPrecisionReduction,
  type GPUPolygonizeOutput
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {buildBoundaryForensicsGeometry} from './boundary-forensics-data';

type Options = {view: 'source' | 'encoding' | 'normalized' | 'precision'; gridSize: string};
const ORIGIN: [number, number] = [-87.6365, 41.882];
const TOPOLOGY_INTERSECTION_CAPACITY = 128;

type Variant = {
  gridSize: number;
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  dirty: boolean;
  normalizedPaths: [number, number][][];
  normalizedMarkers: [number, number][];
  outputPaths: [number, number][][];
};

function pathFromPacked(
  positions: Float32Array,
  offsets: Uint32Array,
  ringCount: number,
  unproject: (x: number, y: number) => [number, number]
): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let ring = 0; ring < ringCount; ring++) {
    const first = offsets[ring];
    const last = offsets[ring + 1];
    if (last - first < 3) continue;
    const path = Array.from({length: last - first}, (_, index) =>
      unproject(positions[(first + index) * 2], positions[(first + index) * 2 + 1])
    );
    if (path.length && (path[0][0] !== path.at(-1)![0] || path[0][1] !== path.at(-1)![1]))
      path.push(path[0]);
    paths.push(path);
  }
  return paths;
}

function polygonizeOutput(
  graph: GPUCommandGraph<void>,
  resources: SpatialAnalysisResources,
  importView: <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => GraphDataView<Format>,
  prefix: string,
  vertexCapacity: number,
  segmentCapacity: number
): {
  output: GPUPolygonizeOutput;
  positions: Buffer;
  ringOffsets: Buffer;
  polygonOffsets: Buffer;
  status: [Buffer, Buffer, Buffer, Buffer, Buffer];
} {
  const directedCapacity = segmentCapacity * 2;
  const ringCapacity = segmentCapacity;
  const status = (name: string) => {
    const count = resources.createBuffer(`${prefix}-${name}-count`, 4);
    const overflow = resources.createBuffer(`${prefix}-${name}-overflow`, 4);
    const requiredCount = resources.createBuffer(`${prefix}-${name}-required`, 4);
    const candidateOverflow = resources.createBuffer(`${prefix}-${name}-candidate-overflow`, 4);
    const uncertainCount = resources.createBuffer(`${prefix}-${name}-uncertain`, 4);
    return {
      views: {
        count: importView(`${prefix}-${name}-count`, count, 'uint32', 1),
        overflow: importView(`${prefix}-${name}-overflow`, overflow, 'uint32', 1),
        requiredCount: importView(`${prefix}-${name}-required`, requiredCount, 'uint32', 1),
        candidateOverflow: importView(
          `${prefix}-${name}-candidate-overflow`,
          candidateOverflow,
          'uint32',
          1
        ),
        uncertainCount: importView(`${prefix}-${name}-uncertain`, uncertainCount, 'uint32', 1)
      },
      buffers: [count, overflow, requiredCount, candidateOverflow, uncertainCount] as const
    };
  };
  const finalStatus = status(`${prefix}-status`);
  const nodedStatus = status(`${prefix}-noded-status`);
  const directedStatus = status(`${prefix}-directed-status`);
  const positions = resources.createBuffer(`${prefix}-polygon-positions`, vertexCapacity * 8);
  const ringOffsets = resources.createBuffer(`${prefix}-polygon-rings`, (ringCapacity + 1) * 4);
  const polygonOffsets = resources.createBuffer(`${prefix}-polygon-parts`, (ringCapacity + 1) * 4);
  const polygonOutput: GPUPolygonizeOutput = {
    nodedSegments: {
      endpoints: importView(
        `${prefix}-noded-endpoints`,
        resources.createBuffer(`${prefix}-noded-endpoints`, segmentCapacity * 16),
        'float32x4',
        segmentCapacity
      ),
      sourceFeatureIds: importView(
        `${prefix}-noded-features`,
        resources.createBuffer(`${prefix}-noded-features`, segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      sourceRingIds: importView(
        `${prefix}-noded-rings`,
        resources.createBuffer(`${prefix}-noded-rings`, segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      sourceEdgeIds: importView(
        `${prefix}-noded-edges`,
        resources.createBuffer(`${prefix}-noded-edges`, segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      sourceStartParameters: importView(
        `${prefix}-noded-start`,
        resources.createBuffer(`${prefix}-noded-start`, segmentCapacity * 4),
        'float32',
        segmentCapacity
      ),
      sourceEndParameters: importView(
        `${prefix}-noded-end`,
        resources.createBuffer(`${prefix}-noded-end`, segmentCapacity * 4),
        'float32',
        segmentCapacity
      ),
      status: nodedStatus.views
    },
    directedEdges: {
      endpoints: importView(
        `${prefix}-directed-endpoints`,
        resources.createBuffer(`${prefix}-directed-endpoints`, directedCapacity * 16),
        'float32x4',
        directedCapacity
      ),
      sourceSegmentIds: importView(
        `${prefix}-directed-segments`,
        resources.createBuffer(`${prefix}-directed-segments`, directedCapacity * 4),
        'uint32',
        directedCapacity
      ),
      sourceFeatureIds: importView(
        `${prefix}-directed-features`,
        resources.createBuffer(`${prefix}-directed-features`, directedCapacity * 4),
        'uint32',
        directedCapacity
      ),
      twinIds: importView(
        `${prefix}-directed-twins`,
        resources.createBuffer(`${prefix}-directed-twins`, directedCapacity * 4),
        'uint32',
        directedCapacity
      ),
      ringIds: importView(
        `${prefix}-directed-rings`,
        resources.createBuffer(`${prefix}-directed-rings`, directedCapacity * 4),
        'uint32',
        directedCapacity
      ),
      flags: importView(
        `${prefix}-directed-flags`,
        resources.createBuffer(`${prefix}-directed-flags`, directedCapacity * 4),
        'uint32',
        directedCapacity
      ),
      status: directedStatus.views
    },
    polygons: {
      kind: 'polygons',
      positions: importView(`${prefix}-polygon-positions`, positions, 'float32x2', vertexCapacity),
      featureOffsets: importView(
        `${prefix}-polygon-features`,
        resources.createBuffer(`${prefix}-polygon-features`, (ringCapacity + 1) * 4),
        'uint32',
        ringCapacity + 1
      ),
      polygonOffsets: importView(
        `${prefix}-polygon-parts`,
        polygonOffsets,
        'uint32',
        ringCapacity + 1
      ),
      ringOffsets: importView(`${prefix}-polygon-rings`, ringOffsets, 'uint32', ringCapacity + 1),
      sourceIds: importView(
        `${prefix}-polygon-sources`,
        resources.createBuffer(`${prefix}-polygon-sources`, ringCapacity * 4),
        'uint32',
        ringCapacity
      )
    },
    diagnostics: {
      edgeClasses: importView(
        `${prefix}-edge-classes`,
        resources.createBuffer(`${prefix}-edge-classes`, segmentCapacity * 4),
        'uint32',
        segmentCapacity
      ),
      closedRingCount: importView(
        `${prefix}-closed-rings`,
        resources.createBuffer(`${prefix}-closed-rings`, 4),
        'uint32',
        1
      ),
      openChainCount: importView(
        `${prefix}-open-chains`,
        resources.createBuffer(`${prefix}-open-chains`, 4),
        'uint32',
        1
      ),
      cutEdgeCount: importView(
        `${prefix}-cut-edges`,
        resources.createBuffer(`${prefix}-cut-edges`, 4),
        'uint32',
        1
      ),
      dangleCount: importView(
        `${prefix}-dangles`,
        resources.createBuffer(`${prefix}-dangles`, 4),
        'uint32',
        1
      ),
      unusedEdgeCount: importView(
        `${prefix}-unused-edges`,
        resources.createBuffer(`${prefix}-unused-edges`, 4),
        'uint32',
        1
      )
    },
    status: finalStatus.views
  };
  return {
    output: polygonOutput,
    positions,
    ringOffsets,
    polygonOffsets,
    status: [
      finalStatus.buffers[0],
      finalStatus.buffers[1],
      finalStatus.buffers[2],
      finalStatus.buffers[3],
      finalStatus.buffers[4]
    ]
  };
}

function ringArea(positions: Float32Array, offsets: Uint32Array, ring: number): number {
  const first = offsets[ring];
  const last = offsets[ring + 1];
  let ringArea = 0;
  const uniqueLast =
    last > first + 1 &&
    positions[first * 2] === positions[(last - 1) * 2] &&
    positions[first * 2 + 1] === positions[(last - 1) * 2 + 1]
      ? last - 1
      : last;
  for (let vertex = first; vertex < uniqueLast; vertex++) {
    const next = vertex + 1 < uniqueLast ? vertex + 1 : first;
    ringArea +=
      positions[vertex * 2] * positions[next * 2 + 1] -
      positions[next * 2] * positions[vertex * 2 + 1];
  }
  return Math.abs(ringArea) / 2;
}

function polygonArea(
  positions: Float32Array,
  ringOffsets: Uint32Array,
  polygonOffsets: Uint32Array,
  polygonCount: number
): number {
  let area = 0;
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const firstRing = polygonOffsets[polygon];
    const lastRing = polygonOffsets[polygon + 1];
    if (lastRing <= firstRing) continue;
    let polygonMeasure = ringArea(positions, ringOffsets, firstRing);
    for (let ring = firstRing + 1; ring < lastRing; ring++)
      polygonMeasure -= ringArea(positions, ringOffsets, ring);
    area += Math.max(0, polygonMeasure);
  }
  return area;
}

function formatGridSize(value: number): string {
  return `${Number(value.toFixed(3))} m`;
}

export async function createBoundaryForensics(
  ctx: SceneContext<Options>
): Promise<SceneInstance<Options>> {
  const dataset = ctx.datasets.get('chicago-buildings');
  const geometry = buildBoundaryForensicsGeometry(dataset, ORIGIN);
  const projection = dataset.getProjection(ORIGIN);
  const sourceArea = polygonArea(
    geometry.positions,
    geometry.ringOffsets,
    geometry.polygonOffsets,
    geometry.polygonOffsets.length - 1
  );
  const shared = new SpatialAnalysisResources(ctx.device, 'boundary-forensics-inputs');
  const sourcePositions = shared.createBuffer('source-positions', geometry.positions);
  const reencodedPositions = shared.createBuffer(
    'reencoded-positions',
    geometry.reencodedPositions
  );
  const ringOffsets = shared.createBuffer('ring-offsets', geometry.ringOffsets);
  const polygonOffsets = shared.createBuffer('polygon-offsets', geometry.polygonOffsets);
  const featureOffsets = shared.createBuffer('feature-offsets', geometry.featureOffsets);
  const sourceIds = shared.createBuffer('building-source-ids', geometry.sourceIds);
  ctx.setReadout(
    'sourceBuildings',
    `${formatCount(geometry.featureCount)} selected building records`
  );
  ctx.setReadout('sourceVertices', `${formatCount(geometry.vertexCount)} stored vertices`);
  ctx.setReadout('sourceIds', Array.from(geometry.sourceIds).join(', '));
  const variants = new Map<number, Variant>();
  const graphSwap = shared.track(new GraphSwap());
  let active: Variant | null = null;
  let destroyed = false;

  const makeVariant = async (gridSize: number): Promise<Variant> => {
    const resources = new SpatialAnalysisResources(
      ctx.device,
      `boundary-forensics-grid-${gridSize}`
    );
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `boundary-forensics-${gridSize}`});
    const imported = new Map<
      Buffer,
      GraphDataView<'float32x2' | 'float32x4' | 'float32' | 'uint32'>
    >();
    const importView = <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => {
      const cached = imported.get(buffer);
      if (cached) return cached as GraphDataView<Format>;
      const view = importGraphBuffer(graph, `import-${name}`, buffer, format, length);
      imported.set(buffer, view);
      return view;
    };
    const inputPositions = importView(
      'source-positions',
      sourcePositions,
      'float32x2',
      geometry.vertexCount
    );
    const variantPositions = importView(
      'reencoded-positions',
      reencodedPositions,
      'float32x2',
      geometry.vertexCount
    );
    const inputRings = importView('ring-offsets', ringOffsets, 'uint32', geometry.ringCount + 1);
    const inputPolygons = importView(
      'polygon-offsets',
      polygonOffsets,
      'uint32',
      geometry.polygonOffsets.length
    );
    const inputFeatures = importView(
      'feature-offsets',
      featureOffsets,
      'uint32',
      geometry.featureCount + 1
    );
    const inputSources = importView('source-ids', sourceIds, 'uint32', geometry.featureCount);
    const outputArrays = (prefix: string) => ({
      positions: resources.createBuffer(`${prefix}-positions`, geometry.vertexCount * 8),
      ringOffsets: resources.createBuffer(`${prefix}-rings`, (geometry.ringCount + 1) * 4),
      polygonOffsets: resources.createBuffer(
        `${prefix}-polygons`,
        geometry.polygonOffsets.length * 4
      )
    });
    const normalizedSourceBuffers = outputArrays('normalized-source');
    const normalizedReencodedBuffers = outputArrays('normalized-reencoded');
    const normalizeOutput = (prefix: string, buffers: typeof normalizedSourceBuffers) => ({
      positions: importView(
        `${prefix}-positions`,
        buffers.positions,
        'float32x2',
        geometry.vertexCount
      ),
      ringOffsets: importView(
        `${prefix}-rings`,
        buffers.ringOffsets,
        'uint32',
        geometry.ringCount + 1
      ),
      polygonOffsets: importView(
        `${prefix}-polygons`,
        buffers.polygonOffsets,
        'uint32',
        geometry.polygonOffsets.length
      )
    });
    const sourceGeometry = {
      positions: inputPositions,
      ringOffsets: inputRings,
      polygonOffsets: inputPolygons,
      featureOffsets: inputFeatures
    };
    const reencodedGeometry = {
      positions: variantPositions,
      ringOffsets: inputRings,
      polygonOffsets: inputPolygons,
      featureOffsets: inputFeatures
    };
    const normalizedSource = normalizeOutput('normalized-source', normalizedSourceBuffers);
    const normalizedReencoded = normalizeOutput('normalized-reencoded', normalizedReencodedBuffers);
    graph.add(
      new GPUNormalizePolygons({
        id: 'normalize-source-rings',
        ...sourceGeometry,
        output: normalizedSource
      })
    );
    graph.add(
      new GPUNormalizePolygons({
        id: 'normalize-reencoded-rings',
        ...reencodedGeometry,
        output: normalizedReencoded
      })
    );
    const equalityBuffer = resources.createBuffer('canonical-equality', geometry.featureCount * 4);
    const equalityView = importView(
      'canonical-equality',
      equalityBuffer,
      'uint32',
      geometry.featureCount
    );
    graph.add(
      new GPUCanonicalPolygonEquality({
        id: 'canonical-polygon-equality',
        left: {...normalizedSource, featureOffsets: inputFeatures},
        right: {...normalizedReencoded, featureOffsets: inputFeatures},
        output: equalityView
      })
    );
    const latticeBuffer = resources.createBuffer(
      'precision-lattice',
      Float32Array.of(0, 0, gridSize, 0)
    );
    const lattice = importView('precision-lattice', latticeBuffer, 'float32', 4);
    const segmentCapacity = geometry.vertexCount + TOPOLOGY_INTERSECTION_CAPACITY * 4;
    const vertexCapacity = Math.max(segmentCapacity * 2, 512);
    const precisionBundle = polygonizeOutput(
      graph,
      resources,
      importView,
      'precision',
      vertexCapacity,
      segmentCapacity
    );
    const precisionOutput = precisionBundle.output;
    graph.add(
      new GPUPolygonPrecisionReduction({
        id: 'precision-reduction',
        spatialContext: projection.spatialContext,
        polygons: {
          kind: 'polygons',
          ...normalizedSource,
          featureOffsets: inputFeatures,
          sourceIds: inputSources
        },
        intersectionCapacity: TOPOLOGY_INTERSECTION_CAPACITY,
        gridSize,
        lattice,
        output: precisionOutput
      })
    );
    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = resources.track(await graph.compileAsync());
    } catch (error) {
      resources.destroy();
      throw error;
    }
    const outputPositions = precisionBundle.positions;
    const outputRingOffsets = precisionBundle.ringOffsets;
    const outputPolygonOffsets = precisionBundle.polygonOffsets;
    const outputStatus = precisionBundle.status;
    const reader = new SummaryReader(
      resources,
      `boundary-forensics-${gridSize}`,
      [
        {buffer: equalityBuffer, size: geometry.featureCount * 4},
        {buffer: normalizedSourceBuffers.positions, size: geometry.vertexCount * 8},
        {buffer: normalizedSourceBuffers.ringOffsets, size: (geometry.ringCount + 1) * 4},
        {buffer: outputPositions, size: vertexCapacity * 8},
        {buffer: outputRingOffsets, size: (segmentCapacity + 1) * 4},
        {buffer: outputPolygonOffsets, size: (segmentCapacity + 1) * 4},
        ...outputStatus.map(buffer => ({buffer, size: 4})),
        {buffer: latticeBuffer, size: 16}
      ],
      bytes => {
        if (destroyed || active?.gridSize !== gridSize) return;
        const values = new DataView(bytes);
        const equality = new Uint32Array(bytes, 0, geometry.featureCount);
        const equalCount = equality.reduce((sum, result) => sum + Number(result === 1), 0);
        const failureCount = equality.reduce((sum, result) => sum + Number(result !== 1), 0);
        const normalizedPositionsOffset = geometry.featureCount * 4;
        const normalizedOffsetsOffset = normalizedPositionsOffset + geometry.vertexCount * 8;
        const normalizedPositions = new Float32Array(
          bytes,
          normalizedPositionsOffset,
          geometry.vertexCount * 2
        );
        const normalizedOffsets = new Uint32Array(
          bytes,
          normalizedOffsetsOffset,
          geometry.ringCount + 1
        );
        const normalizedPaths = pathFromPacked(
          normalizedPositions,
          normalizedOffsets,
          geometry.ringCount,
          (x, y) => projection.unproject(x, y)
        );
        const normalizedMarkers = Array.from({length: geometry.ringCount}, (_, ring) =>
          projection.unproject(
            normalizedPositions[normalizedOffsets[ring] * 2],
            normalizedPositions[normalizedOffsets[ring] * 2 + 1]
          )
        );
        const outputPositionsOffset = normalizedOffsetsOffset + (geometry.ringCount + 1) * 4;
        const outputRingOffsetsOffset = outputPositionsOffset + vertexCapacity * 8;
        const outputPolygonOffsetsOffset = outputRingOffsetsOffset + (segmentCapacity + 1) * 4;
        const statusOffset = outputPolygonOffsetsOffset + (segmentCapacity + 1) * 4;
        const writtenPolygons = values.getUint32(statusOffset, true);
        const overflow = values.getUint32(statusOffset + 4, true);
        const required = values.getUint32(statusOffset + 8, true);
        const candidateOverflow = values.getUint32(statusOffset + 12, true);
        const uncertainCount = values.getUint32(statusOffset + 16, true);
        const latticeOffset = statusOffset + 20;
        const actualGridSize = values.getFloat32(latticeOffset + 8, true);
        const positions = new Float32Array(bytes, outputPositionsOffset, vertexCapacity * 2);
        const offsets = new Uint32Array(bytes, outputRingOffsetsOffset, segmentCapacity + 1);
        const partOffsets = new Uint32Array(bytes, outputPolygonOffsetsOffset, segmentCapacity + 1);
        const validPolygons = Math.min(writtenPolygons, segmentCapacity);
        const validRings = Math.min(partOffsets[validPolygons] ?? 0, segmentCapacity);
        const writtenVertices = offsets[validRings] ?? 0;
        const area = polygonArea(positions, offsets, partOffsets, validPolygons);
        const paths = pathFromPacked(positions, offsets, validRings, (x, y) =>
          projection.unproject(x, y)
        );
        active.normalizedMarkers = normalizedMarkers;
        active.normalizedPaths = normalizedPaths;
        active.outputPaths = paths;
        ctx.setReadout(
          'canonicalEquality',
          `${formatCount(equalCount)} / ${formatCount(geometry.featureCount)} canonical pairs equal; ${formatCount(failureCount)} differ or are uncertain`
        );
        ctx.setReadout(
          'precisionStatus',
          overflow || candidateOverflow
            ? `Grid ${formatGridSize(actualGridSize)}; truncation flags: final ${overflow}, candidates ${candidateOverflow}; required ${formatCount(required)}, uncertain ${formatCount(uncertainCount)}`
            : `Complete at ${formatGridSize(actualGridSize)}: ${formatCount(validPolygons)} polygons, ${formatCount(uncertainCount)} uncertain predicates`
        );
        ctx.setReadout('precisionVertices', `${formatCount(writtenVertices)} GPU output vertices`);
        const areaDifference = area - sourceArea;
        const areaDifferencePercent = sourceArea ? (areaDifference / sourceArea) * 100 : 0;
        const signedAreaDifference = areaDifference >= 0 ? '+' : '−';
        ctx.setReadout(
          'precisionArea',
          `${formatCount(area)} m²; ${signedAreaDifference}${formatCount(Math.abs(areaDifference))} m² (${signedAreaDifference}${Math.abs(areaDifferencePercent).toFixed(2)}%) vs source ${formatCount(sourceArea)} m²`
        );
        ctx.setChart('precisionChart', {
          kind: 'bars',
          values: [geometry.vertexCount, writtenVertices],
          labels: ['source', `${gridSize} m grid`],
          height: 115,
          yLabel: 'vertices',
          formatY: value => formatCount(value),
          description:
            'Source vertex count compared with the count in the polygon geometry written by GPU precision reduction.'
        });
        ctx.requestLayers();
      }
    );
    return {
      gridSize,
      graph: compiled,
      resources,
      reader,
      dirty: true,
      normalizedPaths: [],
      normalizedMarkers: [],
      outputPaths: []
    };
  };

  const activateVariant = (variant: Variant) => {
    active = variant;
    variant.dirty = true;
    ctx.requestLayers();
  };
  const requestGrid = (gridSize: number) => {
    const cached = variants.get(gridSize);
    void graphSwap
      .request({
        build: () => (cached ? Promise.resolve(cached) : makeVariant(gridSize)),
        commit: variant => {
          if (!cached) variants.set(gridSize, variant);
          if (Number(ctx.options.gridSize) === gridSize) activateVariant(variant);
        },
        discard: variant => {
          if (!cached) variant.resources.destroy();
        }
      })
      .catch(error => ctx.setStatus(`Precision graph build failed: ${String(error)}`));
  };
  const initialGrid = Number(ctx.options.gridSize);
  const initial = await makeVariant(initialGrid);
  variants.set(initialGrid, initial);
  active = initial;
  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.graph),
    encode(encoder) {
      if (!active || destroyed) return;
      if (active.dirty) {
        active.graph.encode(encoder, {parameters: undefined});
        active.dirty = false;
        active.reader.markStale();
      }
      active.reader.flush(encoder);
    },
    getLayers() {
      const view = ctx.options.view;
      const normalizedPaths = active?.normalizedPaths ?? [];
      const precisionPaths = active?.outputPaths ?? [];
      return [
        ...(view === 'source' || view === 'encoding'
          ? [
              new PathLayer({
                id: 'forensics-source',
                data: geometry.sourcePaths,
                getPath: path => path,
                getColor: [27, 123, 130, 210],
                getWidth: 2,
                widthUnits: 'pixels',
                pickable: false
              })
            ]
          : []),
        ...(view === 'encoding'
          ? [
              new PathLayer({
                id: 'forensics-reencoded',
                data: geometry.reencodedPaths,
                getPath: path => path,
                getColor: [227, 119, 67, 190],
                getWidth: 1.35,
                widthUnits: 'pixels',
                pickable: false
              }),
              new ScatterplotLayer({
                id: 'forensics-source-starts',
                data: geometry.sourceMarkers,
                getPosition: point => point,
                getRadius: 4,
                radiusUnits: 'pixels',
                getFillColor: [27, 123, 130, 230],
                stroked: false,
                pickable: false
              }),
              new ScatterplotLayer({
                id: 'forensics-reencoded-starts',
                data: geometry.reencodedMarkers,
                getPosition: point => point,
                getRadius: 3,
                radiusUnits: 'pixels',
                getFillColor: [227, 119, 67, 230],
                stroked: true,
                getLineColor: [245, 236, 220, 255],
                getLineWidth: 1,
                lineWidthUnits: 'pixels',
                pickable: false
              })
            ]
          : []),
        ...(view === 'normalized'
          ? [
              new PathLayer({
                id: 'forensics-normalized',
                data: normalizedPaths,
                getPath: path => path,
                getColor: [178, 142, 202, 230],
                getWidth: 2.2,
                widthUnits: 'pixels',
                pickable: false
              }),
              new ScatterplotLayer({
                id: 'forensics-normalized-starts',
                data: active?.normalizedMarkers ?? [],
                getPosition: point => point,
                getRadius: 3,
                radiusUnits: 'pixels',
                getFillColor: [178, 142, 202, 245],
                stroked: true,
                getLineColor: [28, 33, 43, 255],
                getLineWidth: 1,
                lineWidthUnits: 'pixels',
                pickable: false
              })
            ]
          : []),
        ...(view === 'precision'
          ? [
              new PathLayer({
                id: 'forensics-precision-source',
                data: geometry.sourcePaths,
                getPath: path => path,
                getColor: [88, 112, 123, 115],
                getWidth: 1,
                widthUnits: 'pixels',
                pickable: false
              }),
              new PathLayer({
                id: 'forensics-precision-output',
                data: precisionPaths,
                getPath: path => path,
                getColor: [224, 129, 76, 235],
                getWidth: 2.4,
                widthUnits: 'pixels',
                pickable: false
              })
            ]
          : [])
      ];
    },
    setOption(id) {
      if (id === 'view') ctx.requestLayers();
      if (id === 'gridSize') {
        const gridSize = Number(ctx.options.gridSize);
        const cached = variants.get(gridSize);
        if (cached) activateVariant(cached);
        requestGrid(gridSize);
      }
    },
    destroy() {
      destroyed = true;
      shared.destroy();
      for (const variant of variants.values()) variant.resources.destroy();
    }
  };
}
