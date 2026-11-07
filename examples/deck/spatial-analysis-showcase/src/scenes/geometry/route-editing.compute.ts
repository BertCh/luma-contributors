// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, ScatterplotLayer, TextLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUAffineTransform,
  GPUCanonicalGeometryEquality,
  GPUGeometryCleanup,
  GPUGeometryMeasures,
  GPUGeometryOrientation,
  GPUGeometryPredicates,
  GPULineLoopRemoval,
  GPULinePrecisionReduction,
  GPUNormalizeGeometry,
  GPUSegmentSnap,
  GPUVertexSnap,
  GPU_CANONICAL_GEOMETRY_TYPE,
  getGPUAffineTransformParameterValues,
  getGPUGeometryCleanupParameterValues,
  getGPUGeometryPredicatesParameterValues,
  getGPUVertexSnapParameterValues
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GraphSwap} from '../../engine/graph-swap';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {LocalMetricProjection, projectLngLatArray} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {RouteEditingOptions} from './route-editing.types';

const RED_ROUTE_INDEX = 124;
const RED_SHAPE_ROW = 16958;
const ROUTE_VIEW = {longitude: -87.65, latitude: 41.88, zoom: 10.2};
const DOWNTOWN_VIEW = {longitude: -87.6298, latitude: 41.8781, zoom: 12.4};
const SOURCE_COLOR: [number, number, number, number] = [161, 174, 184, 165];
const EDITED_COLOR: [number, number, number, number] = [255, 186, 86, 245];
const REFERENCE_COLOR: [number, number, number, number] = [91, 208, 221, 230];
const DECLARED_TRANSLATION: readonly [number, number] = [64, -32];
const RECOVERED_COLOR: [number, number, number, number] = [73, 192, 204, 240];

type EditKind = RouteEditingOptions['edit'];
type RouteRecord = {id: string; name: string; type: string};
type RouteGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  edit: EditKind;
  outputPositions: Buffer;
  recoveredPositions: Buffer | null;
  affineMeasures: {sourceLength: Buffer; translatedLength: Buffer} | null;
  outputCapacity: number;
  inputLocal: Float32Array;
  inputLngLat: Float32Array;
  referenceLocal: Float32Array;
  referenceLngLat: Float32Array;
  buffers: {
    editStatus: Buffer[];
    editCounts: Buffer | null;
    snappedReferenceRows: Buffer | null;
  };
  settings: RouteEditingOptions;
};
type RouteResult = {
  reversedPositions: Float32Array;
  normalizeStatuses: Uint32Array;
  equality: number;
  outputPositions: Float32Array;
  outputOffsets: Uint32Array;
  editStatus: Uint32Array;
  editCounts: Uint32Array;
  snappedReferenceRows: Uint32Array;
  predicateValues: Uint32Array;
  writtenCount: number;
  requiredCount: number;
  overflow: number;
  invalid: number;
  recoveredPositions?: Float32Array;
  sourceLength?: number;
  translatedLength?: number;
};

function readUint32(bytes: ArrayBuffer, offset: {value: number}, count: number): Uint32Array {
  const values = new Uint32Array(count);
  const view = new DataView(bytes);
  for (let index = 0; index < count; index++)
    values[index] = view.getUint32(offset.value + index * 4, true);
  offset.value += count * 4;
  return values;
}

function readFloat32(bytes: ArrayBuffer, offset: {value: number}, count: number): Float32Array {
  const values = new Float32Array(count);
  const view = new DataView(bytes);
  for (let index = 0; index < count; index++)
    values[index] = view.getFloat32(offset.value + index * 4, true);
  offset.value += count * 4;
  return values;
}

function getRouteResult(graph: RouteGraph, bytes: ArrayBuffer): RouteResult {
  const offset = {value: 0};
  const routeVertexCount = graph.inputLocal.length / 2;
  const reversedPositions = readFloat32(bytes, offset, routeVertexCount * 2);
  const normalizeStatuses = readUint32(bytes, offset, 2);
  const equality = readUint32(bytes, offset, 1)[0];
  const predicateValues = readUint32(bytes, offset, 6);
  const outputPositions = readFloat32(bytes, offset, graph.outputCapacity * 2);
  const outputOffsets = readUint32(bytes, offset, 2);
  let editStatus: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
  let editCounts: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
  let snappedReferenceRows: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
  for (const statusBuffer of graph.buffers.editStatus)
    editStatus = Uint32Array.from([
      ...editStatus,
      ...readUint32(bytes, offset, statusBuffer.byteLength / 4)
    ]);
  if (graph.buffers.editCounts)
    editCounts = readUint32(bytes, offset, graph.buffers.editCounts.byteLength / 4);
  if (graph.buffers.snappedReferenceRows)
    snappedReferenceRows = readUint32(bytes, offset, routeVertexCount);
  const recoveredPositions =
    graph.edit === 'translation' ? readFloat32(bytes, offset, routeVertexCount * 2) : undefined;
  const sourceLength = graph.edit === 'translation' ? readFloat32(bytes, offset, 1)[0] : undefined;
  const translatedLength =
    graph.edit === 'translation' ? readFloat32(bytes, offset, 1)[0] : undefined;
  const writtenCount = Math.min(
    graph.edit === 'segment-snap'
      ? editCounts[0]
      : graph.edit === 'loop-revisit'
        ? editStatus[0]
        : outputOffsets[1],
    graph.outputCapacity
  );
  let requiredCount = outputOffsets[1];
  let overflow = 0;
  let invalid = 0;
  if (graph.edit === 'cleanup') {
    requiredCount = editStatus[2];
    overflow = editStatus[1];
  } else if (graph.edit === 'segment-snap') {
    requiredCount = editCounts[1];
    overflow = editCounts[2];
    invalid = editCounts[3];
  } else if (graph.edit === 'loop-revisit') {
    requiredCount = editStatus[1];
    overflow = editStatus[2];
    invalid = editStatus[3];
  } else if (graph.edit === 'precision') {
    invalid = editStatus[0] === 3 ? 1 : 0;
  }
  return {
    reversedPositions,
    normalizeStatuses,
    equality,
    predicateValues,
    outputPositions,
    outputOffsets,
    editStatus,
    editCounts,
    snappedReferenceRows,
    writtenCount,
    requiredCount,
    overflow,
    invalid,
    recoveredPositions,
    sourceLength,
    translatedLength
  };
}

function makePath(
  localPositions: Float32Array,
  projection: LocalMetricProjection,
  count: number
): [number, number][] {
  const path: [number, number][] = [];
  for (let row = 0; row < count; row++) {
    path.push(projection.unproject(localPositions[row * 2], localPositions[row * 2 + 1]));
  }
  return path;
}

function makeLoopRevisit(source: Float32Array): Float32Array {
  const rows = [0, 1, 2, 3, 2, 3, 4, 5];
  const result = new Float32Array(rows.length * 2);
  rows.forEach((row, index) => result.set(source.subarray(row * 2, row * 2 + 2), index * 2));
  return result;
}

export async function createRouteEditing(
  ctx: SceneContext<RouteEditingOptions>
): Promise<SceneInstance<RouteEditingOptions>> {
  const dataset = ctx.datasets.get('cta-transit');
  const routeRows = dataset.properties.routes as RouteRecord[];
  if (routeRows[RED_ROUTE_INDEX]?.id !== 'Red')
    throw new Error('CTA Red Line route row 124 is unavailable.');
  const shapeOffsets = dataset.column<Uint32Array>('shapePathOffsets');
  const shapeRoute = dataset.column<Uint16Array>('shapeRoute');
  const shapeDirection = dataset.column<Uint8Array>('shapeDirection');
  if (shapeRoute[RED_SHAPE_ROW] !== RED_ROUTE_INDEX || shapeDirection[RED_SHAPE_ROW] !== 0) {
    throw new Error('CTA Red Line source shape row 16958 no longer matches its catalog record.');
  }
  const vertices = dataset.column<Float32Array>('shapeVertices');
  const sourceStart = shapeOffsets[RED_SHAPE_ROW];
  const sourceEnd = shapeOffsets[RED_SHAPE_ROW + 1];
  const sourceLngLat = vertices.slice(sourceStart * 2, sourceEnd * 2);
  const projection = new LocalMetricProjection([-87.6298, 41.8781]);
  const sourceLocal = projectLngLatArray(projection, sourceLngLat);

  const stopPositions = dataset.column<Float32Array>('stopPosition');
  const stopRouteOffsets = dataset.column<Uint32Array>('stopRouteOffsets');
  const stopRoutes = dataset.column<Uint16Array>('stopRoutes');
  const stopModes = dataset.column<Uint8Array>('stopMode');
  const referenceRows: number[] = [];
  for (let stop = 0; stop + 1 < stopRouteOffsets.length; stop++) {
    if (stopModes[stop] !== 1) continue;
    for (let row = stopRouteOffsets[stop]; row < stopRouteOffsets[stop + 1]; row++) {
      if (stopRoutes[row] === RED_ROUTE_INDEX) {
        referenceRows.push(stop);
        break;
      }
    }
  }
  const referenceLngLat = new Float32Array(referenceRows.length * 2);
  referenceRows.forEach((stop, row) =>
    referenceLngLat.set(stopPositions.subarray(stop * 2, stop * 2 + 2), row * 2)
  );
  const referenceLocal = projectLngLatArray(projection, referenceLngLat);
  const loopLocal = makeLoopRevisit(sourceLocal);
  const loopLngLat = new Float32Array(makePath(loopLocal, projection, loopLocal.length / 2).flat());
  const revisitCoordinates = makePath(loopLocal, projection, loopLocal.length / 2);
  const revisitBounds: [number, number, number, number] = [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ];
  for (const coordinate of revisitCoordinates) {
    revisitBounds[0] = Math.min(revisitBounds[0], coordinate[0]);
    revisitBounds[1] = Math.min(revisitBounds[1], coordinate[1]);
    revisitBounds[2] = Math.max(revisitBounds[2], coordinate[0]);
    revisitBounds[3] = Math.max(revisitBounds[3], coordinate[1]);
  }
  const revisitView = {
    longitude: (revisitBounds[0] + revisitBounds[2]) / 2,
    latitude: (revisitBounds[1] + revisitBounds[3]) / 2,
    zoom: 13.2
  };
  const baseResources = new SpatialAnalysisResources(ctx.device, 'route-editing');
  const graphSwap = baseResources.track(new GraphSwap());
  let activeGraph: RouteGraph;
  let graphDirty = true;
  let latestResult: RouteResult | null = null;

  const compileGraph = async (settings: RouteEditingOptions): Promise<RouteGraph> => {
    const edit = settings.edit;
    const resources = new SpatialAnalysisResources(
      ctx.device,
      `route-editing-${edit}-${settings.tolerance}-${settings.gridSize}`
    );
    const isLoop = edit === 'loop-revisit';
    const inputLocal = isLoop ? loopLocal : sourceLocal;
    const inputLngLat = isLoop ? new Float32Array(loopLngLat) : sourceLngLat;
    const inputVertexCount = inputLocal.length / 2;
    const routeOffsetsData = Uint32Array.of(0, inputVertexCount);
    const sourcePositions = resources.createBuffer('source-positions', inputLocal);
    const lineOffsets = resources.createBuffer('source-line-offsets', routeOffsetsData);
    const references = resources.createBuffer('red-station-reference-positions', referenceLocal);
    const reversedPositions = resources.createBuffer(
      'reversed-source-positions',
      inputVertexCount * 8
    );
    const nodesData = Uint32Array.of(
      GPU_CANONICAL_GEOMETRY_TYPE.lineString,
      0,
      inputVertexCount,
      0
    );
    const childrenData = Uint32Array.of(0);
    const canonicalNodes = resources.createBuffer('canonical-line-tree', nodesData);
    const canonicalChildren = resources.createBuffer('canonical-line-children', childrenData);
    const normalizedSource = resources.createBuffer(
      'normalized-source-positions',
      inputVertexCount * 8
    );
    const normalizedReversed = resources.createBuffer(
      'normalized-reversed-positions',
      inputVertexCount * 8
    );
    const normalizedSourceChildren = resources.createBuffer('normalized-source-children', 4);
    const normalizedReversedChildren = resources.createBuffer('normalized-reversed-children', 4);
    const sourceNormalizeStatus = resources.createBuffer('source-normalize-status', 4);
    const reversedNormalizeStatus = resources.createBuffer('reversed-normalize-status', 4);
    const equality = resources.createBuffer('canonical-equality', 4);
    const exactOrderIdentity = resources.createBuffer('exact-order-identical', 4);
    const exactCoordinateEquality = resources.createBuffer('exact-coordinate-equality', 4);
    const sourceIsSimple = resources.createBuffer('source-is-simple', 4);
    const sourceIsClosed = resources.createBuffer('source-is-closed', 4);
    const predicateOverflow = resources.createBuffer('predicate-overflow', 4);
    const predicateIntersectionCount = resources.createBuffer('predicate-intersection-count', 4);
    const reversedOffsets = resources.createBuffer('reversed-line-offsets', routeOffsetsData);
    const predicateParameters = resources.createParameterBuffer(
      'predicate-parameters',
      'float32',
      4,
      getGPUGeometryPredicatesParameterValues({tolerance: 0})
    );
    const outputOffsets = resources.createBuffer(
      'edited-line-offsets',
      Uint32Array.of(0, inputVertexCount)
    );
    let outputCapacity = inputVertexCount;
    let outputPositions: Buffer = reversedPositions;
    let recoveredPositions: Buffer | null = null;
    let affineMeasures: {sourceLength: Buffer; translatedLength: Buffer} | null = null;
    let editStatus: Buffer[] = [];
    let editCounts: Buffer | null = null;
    let snappedReferenceRows: Buffer | null = null;
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `route-editing-${edit}`});
    const sourceView = importGraphBuffer(
      graph,
      'source-positions',
      sourcePositions,
      'float32x2',
      inputVertexCount
    );
    const lineOffsetsView = importGraphBuffer(
      graph,
      'source-line-offsets',
      lineOffsets,
      'uint32',
      2
    );
    const referenceView = importGraphBuffer(
      graph,
      'red-station-references',
      references,
      'float32x2',
      referenceRows.length
    );
    const reversedView = importGraphBuffer(
      graph,
      'reversed-source-positions',
      reversedPositions,
      'float32x2',
      inputVertexCount
    );
    const nodeView = importGraphBuffer(graph, 'canonical-line-tree', canonicalNodes, 'uint32x4', 1);
    const childrenView = importGraphBuffer(
      graph,
      'canonical-line-children',
      canonicalChildren,
      'uint32',
      1
    );
    const normalizedSourceView = importGraphBuffer(
      graph,
      'normalized-source',
      normalizedSource,
      'float32x2',
      inputVertexCount
    );
    const normalizedReversedView = importGraphBuffer(
      graph,
      'normalized-reversed',
      normalizedReversed,
      'float32x2',
      inputVertexCount
    );
    const normalizedSourceChildrenView = importGraphBuffer(
      graph,
      'normalized-source-children',
      normalizedSourceChildren,
      'uint32',
      1
    );
    const normalizedReversedChildrenView = importGraphBuffer(
      graph,
      'normalized-reversed-children',
      normalizedReversedChildren,
      'uint32',
      1
    );
    const sourceNormalizeStatusView = importGraphBuffer(
      graph,
      'source-normalize-status',
      sourceNormalizeStatus,
      'uint32',
      1
    );
    const reversedNormalizeStatusView = importGraphBuffer(
      graph,
      'reversed-normalize-status',
      reversedNormalizeStatus,
      'uint32',
      1
    );
    const equalityView = importGraphBuffer(graph, 'canonical-equality', equality, 'uint32', 1);
    const exactOrderIdentityView = importGraphBuffer(
      graph,
      'exact-order-identical',
      exactOrderIdentity,
      'uint32',
      1
    );
    const exactCoordinateEqualityView = importGraphBuffer(
      graph,
      'exact-coordinate-equality',
      exactCoordinateEquality,
      'uint32',
      1
    );
    const sourceIsSimpleView = importGraphBuffer(
      graph,
      'source-is-simple',
      sourceIsSimple,
      'uint32',
      1
    );
    const sourceIsClosedView = importGraphBuffer(
      graph,
      'source-is-closed',
      sourceIsClosed,
      'uint32',
      1
    );
    const predicateOverflowView = importGraphBuffer(
      graph,
      'predicate-overflow',
      predicateOverflow,
      'uint32',
      1
    );
    const predicateIntersectionCountView = importGraphBuffer(
      graph,
      'predicate-intersection-count',
      predicateIntersectionCount,
      'uint32',
      1
    );
    const reversedOffsetsView = importGraphBuffer(
      graph,
      'reversed-line-offsets',
      reversedOffsets,
      'uint32',
      2
    );
    const outputOffsetsView = importGraphBuffer(
      graph,
      'edited-line-offsets',
      outputOffsets,
      'uint32',
      2
    );
    graph.add(
      new GPUGeometryOrientation({
        id: 'reverse-source-order',
        spatialContext: projection.spatialContext,
        mode: 'reverse',
        positions: sourceView,
        lineOffsets: lineOffsetsView,
        output: {positions: reversedView}
      })
    );
    graph.add(
      new GPUNormalizeGeometry({
        id: 'normalize-original-line',
        positions: sourceView,
        nodes: nodeView,
        children: childrenView,
        output: {
          positions: normalizedSourceView,
          children: normalizedSourceChildrenView,
          status: sourceNormalizeStatusView
        }
      })
    );
    graph.add(
      new GPUNormalizeGeometry({
        id: 'normalize-reversed-line',
        positions: reversedView,
        nodes: nodeView,
        children: childrenView,
        output: {
          positions: normalizedReversedView,
          children: normalizedReversedChildrenView,
          status: reversedNormalizeStatusView
        }
      })
    );
    graph.add(
      new GPUCanonicalGeometryEquality({
        id: 'canonical-line-equality',
        left: {
          positions: normalizedSourceView,
          nodes: nodeView,
          children: normalizedSourceChildrenView
        },
        right: {
          positions: normalizedReversedView,
          nodes: nodeView,
          children: normalizedReversedChildrenView
        },
        pairs: importGraphBuffer(
          graph,
          'canonical-root-pairs',
          resources.createBuffer('canonical-root-pairs', Uint32Array.of(0, 0)),
          'uint32x2',
          1
        ),
        output: equalityView
      })
    );
    graph.add(
      new GPUGeometryPredicates({
        id: 'route-order-and-topology-predicates',
        spatialContext: projection.spatialContext,
        geometry: {kind: 'lines', positions: sourceView, lineOffsets: lineOffsetsView},
        other: {kind: 'lines', positions: reversedView, lineOffsets: reversedOffsetsView},
        parameters: predicateParameters.importToGraph(graph),
        intersectionCapacity: Math.max(1, (inputVertexCount * (inputVertexCount - 3)) / 2),
        output: {
          equalsIdentical: exactOrderIdentityView,
          equalsExact: exactCoordinateEqualityView,
          isSimple: sourceIsSimpleView,
          isClosed: sourceIsClosedView,
          status: {
            overflow: predicateOverflowView,
            intersectionCount: predicateIntersectionCountView
          }
        }
      })
    );

    if (edit === 'reverse') {
      outputCapacity = inputVertexCount;
      outputPositions = reversedPositions;
    } else if (edit === 'cleanup') {
      outputPositions = resources.createBuffer('cleanup-output-positions', inputVertexCount * 8);
      outputCapacity = inputVertexCount;
      const cleanupCount = resources.createBuffer('cleanup-count', 4);
      const cleanupOverflow = resources.createBuffer('cleanup-overflow', 4);
      const cleanupRequired = resources.createBuffer('cleanup-required', 4);
      editStatus = [cleanupCount, cleanupOverflow, cleanupRequired];
      const parameters = resources.createParameterBuffer(
        'cleanup-parameters',
        'float32',
        4,
        getGPUGeometryCleanupParameterValues({
          gridSize: Number(settings.gridSize),
          tolerance: Number(settings.tolerance)
        })
      );
      graph.add(
        new GPUGeometryCleanup({
          id: 'cleanup-route',
          geometryType: 'lines',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          lineOffsets: lineOffsetsView,
          parameters: parameters.importToGraph(graph),
          output: {
            positions: importGraphBuffer(
              graph,
              'cleanup-output',
              outputPositions,
              'float32x2',
              outputCapacity
            ),
            lineOffsets: outputOffsetsView,
            status: {
              count: importGraphBuffer(graph, 'cleanup-count', cleanupCount, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'cleanup-overflow', cleanupOverflow, 'uint32', 1),
              requiredCount: importGraphBuffer(
                graph,
                'cleanup-required',
                cleanupRequired,
                'uint32',
                1
              )
            }
          }
        })
      );
    } else if (edit === 'precision') {
      outputPositions = resources.createBuffer('precision-output-positions', inputVertexCount * 8);
      outputCapacity = inputVertexCount;
      const precisionStatus = resources.createBuffer('precision-status', 4);
      editStatus = [precisionStatus];
      graph.add(
        new GPULinePrecisionReduction({
          id: 'reduce-route-precision',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          lineOffsets: lineOffsetsView,
          gridSize: Number(settings.gridSize),
          mode: 'valid-output',
          output: {
            positions: importGraphBuffer(
              graph,
              'precision-output',
              outputPositions,
              'float32x2',
              outputCapacity
            ),
            lineOffsets: outputOffsetsView,
            status: importGraphBuffer(graph, 'precision-status', precisionStatus, 'uint32', 1)
          }
        })
      );
    } else if (edit === 'vertex-snap') {
      outputPositions = resources.createBuffer('vertex-snap-output', inputVertexCount * 8);
      outputCapacity = inputVertexCount;
      snappedReferenceRows = resources.createBuffer(
        'vertex-snap-reference-rows',
        inputVertexCount * 4
      );
      const parameters = resources.createParameterBuffer(
        'vertex-snap-parameters',
        'float32',
        4,
        getGPUVertexSnapParameterValues({tolerance: Number(settings.tolerance)})
      );
      graph.add(
        new GPUVertexSnap({
          id: 'snap-route-vertices',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          referencePositions: referenceView,
          parameters: parameters.importToGraph(graph),
          output: {
            positions: importGraphBuffer(
              graph,
              'vertex-snap-output',
              outputPositions,
              'float32x2',
              outputCapacity
            ),
            referenceRows: importGraphBuffer(
              graph,
              'vertex-snap-reference-rows',
              snappedReferenceRows,
              'uint32',
              inputVertexCount
            )
          }
        })
      );
    } else if (edit === 'segment-snap') {
      outputCapacity = inputVertexCount * (referenceRows.length + 1);
      outputPositions = resources.createBuffer('segment-snap-output', outputCapacity * 8);
      editCounts = resources.createBuffer('segment-snap-counts', 16);
      const segmentSnapValidity = resources.createBuffer('segment-snap-validity', 4);
      editStatus = [segmentSnapValidity];
      graph.add(
        new GPUSegmentSnap({
          id: 'snap-route-segments',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          lineOffsets: lineOffsetsView,
          referencePositions: referenceView,
          tolerance: Number(settings.tolerance),
          maximumVerticesPerLine: outputCapacity,
          output: {
            positions: importGraphBuffer(
              graph,
              'segment-snap-output',
              outputPositions,
              'float32x2',
              outputCapacity
            ),
            sourceEdges: importGraphBuffer(
              graph,
              'segment-snap-source-edges',
              resources.createBuffer('segment-snap-source-edges', outputCapacity * 4),
              'uint32',
              outputCapacity
            ),
            counts: importGraphBuffer(graph, 'segment-snap-counts', editCounts, 'uint32x4', 1),
            validity: importGraphBuffer(
              graph,
              'segment-snap-validity',
              segmentSnapValidity,
              'uint32',
              1
            )
          }
        })
      );
    } else if (edit === 'translation') {
      outputCapacity = inputVertexCount;
      outputPositions = resources.createBuffer('translated-route-positions', inputVertexCount * 8);
      recoveredPositions = resources.createBuffer(
        'inverse-recovered-route-positions',
        inputVertexCount * 8
      );
      const sourceLength = resources.createBuffer('source-route-length', 4);
      const translatedLength = resources.createBuffer('translated-route-length', 4);
      affineMeasures = {sourceLength, translatedLength};
      const translatedView = importGraphBuffer(
        graph,
        'translated-route',
        outputPositions,
        'float32x2',
        inputVertexCount
      );
      const recoveredView = importGraphBuffer(
        graph,
        'inverse-recovered-route',
        recoveredPositions,
        'float32x2',
        inputVertexCount
      );
      const forwardParameters = resources.createParameterBuffer(
        'forward-translation-parameters',
        'float32',
        12,
        getGPUAffineTransformParameterValues({translate: DECLARED_TRANSLATION})
      );
      const inverseParameters = resources.createParameterBuffer(
        'inverse-translation-parameters',
        'float32',
        12,
        getGPUAffineTransformParameterValues({
          translate: [-DECLARED_TRANSLATION[0], -DECLARED_TRANSLATION[1]]
        })
      );
      graph.add(
        new GPUAffineTransform({
          id: 'declare-metric-frame-translation',
          geometryType: 'lines',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          lineOffsets: lineOffsetsView,
          parameters: forwardParameters.importToGraph(graph),
          output: {positions: translatedView}
        })
      );
      graph.add(
        new GPUAffineTransform({
          id: 'apply-inverse-metric-frame-translation',
          geometryType: 'lines',
          spatialContext: projection.spatialContext,
          positions: translatedView,
          lineOffsets: lineOffsetsView,
          parameters: inverseParameters.importToGraph(graph),
          output: {positions: recoveredView}
        })
      );
      graph.add(
        new GPUGeometryMeasures({
          id: 'measure-source-line-length',
          geometryType: 'lines',
          spatialContext: projection.spatialContext,
          positions: sourceView,
          lineOffsets: lineOffsetsView,
          output: {
            lengths: importGraphBuffer(graph, 'source-line-length', sourceLength, 'float32', 1)
          }
        })
      );
      graph.add(
        new GPUGeometryMeasures({
          id: 'measure-translated-line-length',
          geometryType: 'lines',
          spatialContext: projection.spatialContext,
          positions: translatedView,
          lineOffsets: lineOffsetsView,
          output: {
            lengths: importGraphBuffer(
              graph,
              'translated-line-length',
              translatedLength,
              'float32',
              1
            )
          }
        })
      );
    } else if (edit === 'loop-revisit') {
      outputCapacity = loopLocal.length / 2 + 8 * 32;
      outputPositions = resources.createBuffer('loop-revisit-output', outputCapacity * 8);
      const loopRevisitStatus = resources.createBuffer('loop-revisit-status', 16);
      editStatus = [loopRevisitStatus];
      const loopPositions = resources.createBuffer('loop-revisit-input', loopLocal);
      const loopOffsets = resources.createBuffer(
        'loop-revisit-offsets',
        Uint32Array.of(0, loopLocal.length / 2)
      );
      graph.add(
        new GPULineLoopRemoval({
          id: 'erase-simulated-revisit',
          spatialContext: projection.spatialContext,
          lines: {
            kind: 'lines',
            positions: importGraphBuffer(
              graph,
              'loop-revisit-input',
              loopPositions,
              'float32x2',
              loopLocal.length / 2
            ),
            lineOffsets: importGraphBuffer(graph, 'loop-revisit-offsets', loopOffsets, 'uint32', 2)
          },
          intersectionCapacity: 32,
          output: {
            positions: importGraphBuffer(
              graph,
              'loop-revisit-output',
              outputPositions,
              'float32x2',
              outputCapacity
            ),
            lineOffsets: outputOffsetsView,
            sourceEdges: importGraphBuffer(
              graph,
              'loop-revisit-source-edges',
              resources.createBuffer('loop-revisit-source-edges', outputCapacity * 4),
              'uint32',
              outputCapacity
            ),
            status: importGraphBuffer(
              graph,
              'loop-revisit-status',
              loopRevisitStatus,
              'uint32x4',
              1
            )
          }
        })
      );
    }

    const compiled = resources.track(await graph.compileAsync());
    const outputBuffer = outputPositions;
    const editStatusBuffers = editStatus.map(buffer => ({buffer, size: buffer.byteLength}));
    const readerBuffers = [
      {buffer: reversedPositions, size: inputVertexCount * 8},
      {buffer: sourceNormalizeStatus, size: 4},
      {buffer: reversedNormalizeStatus, size: 4},
      {buffer: equality, size: 4},
      {buffer: exactOrderIdentity, size: 4},
      {buffer: exactCoordinateEquality, size: 4},
      {buffer: sourceIsSimple, size: 4},
      {buffer: sourceIsClosed, size: 4},
      {buffer: predicateOverflow, size: 4},
      {buffer: predicateIntersectionCount, size: 4},
      {buffer: outputBuffer, size: outputCapacity * 8},
      {buffer: outputOffsets, size: 8},
      ...editStatusBuffers,
      ...(edit === 'vertex-snap' && snappedReferenceRows
        ? [{buffer: snappedReferenceRows, size: inputVertexCount * 4}]
        : []),
      ...(edit === 'segment-snap' && editCounts ? [{buffer: editCounts, size: 16}] : []),
      ...(edit === 'translation' && recoveredPositions && affineMeasures
        ? [
            {buffer: recoveredPositions, size: inputVertexCount * 8},
            {buffer: affineMeasures.sourceLength, size: 4},
            {buffer: affineMeasures.translatedLength, size: 4}
          ]
        : [])
    ];
    const reader = new SummaryReader(resources, 'route-editing-summary', readerBuffers, bytes => {
      const result = getRouteResult(activeGraph, bytes);
      latestResult = result;
      publishResult(activeGraph, result);
    });
    return {
      resources,
      compiled,
      reader,
      edit,
      outputPositions: outputBuffer,
      recoveredPositions,
      affineMeasures,
      outputCapacity,
      inputLocal,
      inputLngLat,
      referenceLocal,
      referenceLngLat,
      buffers: {editStatus, editCounts, snappedReferenceRows},
      settings
    };
  };

  const publishResult = (graph: RouteGraph, result: RouteResult) => {
    const statusText = result.overflow
      ? `Output is truncated (${result.writtenCount} of ${result.requiredCount} required vertices).`
      : graph.edit === 'precision' && result.editStatus[0] === 1
        ? 'The selected grid collapsed the line; no valid path is shown.'
        : graph.edit === 'precision' && result.editStatus[0] === 2
          ? 'The line collapsed and was retained under the contributor policy.'
          : graph.edit === 'precision' && result.editStatus[0] === 3
            ? 'The contributor found non-finite coordinates; inspect the output cautiously.'
            : result.invalid
              ? `The selected contributor reported a status condition (${result.invalid}); inspect the caveat before reading the path.`
              : `${graph.edit} completed with ${result.writtenCount} output vertices.`;
    const inputCount = graph.inputLocal.length / 2;
    let sameIndexDifferenceCount = 0;
    for (let row = 0; row < Math.min(result.writtenCount, inputCount); row++) {
      if (
        result.outputPositions[row * 2] !== graph.inputLocal[row * 2] ||
        result.outputPositions[row * 2 + 1] !== graph.inputLocal[row * 2 + 1]
      )
        sameIndexDifferenceCount++;
    }
    ctx.setReadout('sourceVertexCount', inputCount);
    ctx.setReadout('editedVertexCount', result.writtenCount);
    ctx.setReadout('requiredVertexCount', result.requiredCount);
    ctx.setReadout(
      'sameIndexDifferenceCount',
      graph.edit === 'loop-revisit' ? 'Derived stress input' : sameIndexDifferenceCount
    );
    ctx.setReadout('removedVertexCount', Math.max(0, inputCount - result.writtenCount));
    ctx.setReadout(
      'exactOrderIdentical',
      result.predicateValues[0] === 1
        ? 'Yes'
        : result.predicateValues[0] === 0
          ? 'No'
          : `Status ${result.predicateValues[0]}`
    );
    ctx.setReadout(
      'exactCoordinateEquality',
      result.predicateValues[1] === 1
        ? 'Yes'
        : result.predicateValues[1] === 0
          ? 'No'
          : `Status ${result.predicateValues[1]}`
    );
    ctx.setReadout(
      'sourceSimple',
      result.predicateValues[2] === 1
        ? 'Yes'
        : result.predicateValues[2] === 0
          ? 'No'
          : `Status ${result.predicateValues[2]}`
    );
    ctx.setReadout(
      'sourceClosed',
      result.predicateValues[3] === 1
        ? 'Yes'
        : result.predicateValues[3] === 0
          ? 'No'
          : `Status ${result.predicateValues[3]}`
    );
    ctx.setReadout(
      'predicateStatus',
      `overflow ${result.predicateValues[4]}, intersections ${result.predicateValues[5]}`
    );
    ctx.setReadout('editStatus', statusText);
    ctx.setReadout(
      'canonicalStatus',
      result.normalizeStatuses[0] === 0 && result.normalizeStatuses[1] === 0
        ? result.equality === 1
          ? 'Original and reversed lines normalize to equal canonical geometry; exact vertex order differs.'
          : result.equality === 0
            ? 'Canonical line geometry differs.'
            : 'Canonical comparison invalid.'
        : `Normalization status: ${result.normalizeStatuses[0]} / ${result.normalizeStatuses[1]}`
    );
    ctx.setReadout('stationReferenceCount', referenceRows.length);
    ctx.setReadout(
      'snappedVertexCount',
      result.snappedReferenceRows.filter(row => row !== 0xffffffff).length
    );
    if (
      graph.edit === 'translation' &&
      result.recoveredPositions &&
      result.sourceLength !== undefined &&
      result.translatedLength !== undefined
    ) {
      let maximumRecoveryError = 0;
      for (let coordinate = 0; coordinate < graph.inputLocal.length; coordinate += 2) {
        const eastError = result.recoveredPositions[coordinate] - graph.inputLocal[coordinate];
        const northError =
          result.recoveredPositions[coordinate + 1] - graph.inputLocal[coordinate + 1];
        maximumRecoveryError = Math.max(maximumRecoveryError, Math.hypot(eastError, northError));
      }
      ctx.setReadout('declaredTranslation', 'East +64 m; north −32 m');
      ctx.setReadout('sourceLineLength', result.sourceLength);
      ctx.setReadout('translatedLineLength', result.translatedLength);
      ctx.setReadout(
        'translationRecovery',
        `${inputCount} rows; unchanged offsets; max error ${maximumRecoveryError.toFixed(6)} m`
      );
    }
    ctx.requestLayers();
  };

  const releaseGraph = (graph: RouteGraph) => {
    graph.reader.release();
    graph.resources.destroy();
  };
  const requestGraph = (settings: RouteEditingOptions) => {
    ctx.setReadout('editStatus', 'Compiling selected route edit…');
    void graphSwap
      .request({
        build: () => compileGraph(settings),
        commit: next => {
          const previous = activeGraph;
          activeGraph = next;
          latestResult = null;
          graphDirty = true;
          next.reader.markStale();
          for (const id of [
            'editedVertexCount',
            'requiredVertexCount',
            'sameIndexDifferenceCount',
            'removedVertexCount',
            'canonicalStatus',
            'snappedVertexCount'
          ])
            ctx.setReadout(id, 'Recomputing');
          releaseGraph(previous);
          const focusView =
            next.settings.focus === 'revisit'
              ? revisitView
              : next.settings.focus === 'downtown'
                ? DOWNTOWN_VIEW
                : ROUTE_VIEW;
          ctx.flyTo(focusView, {transitionMs: ctx.reducedMotion() ? 0 : 700});
          ctx.requestLayers();
        },
        discard: releaseGraph
      })
      .catch(error => ctx.setStatus(`Route edit graph failed: ${String(error)}`));
  };

  activeGraph = await compileGraph(ctx.options);
  activeGraph.reader.markStale();
  graphDirty = true;
  return {
    getCompiledGraphs: () => [activeGraph.compiled],
    setOption(id, _value, state) {
      if (id === 'edit' || id === 'tolerance' || id === 'gridSize') requestGraph(state);
      else if (id === 'focus')
        ctx.flyTo(
          state.focus === 'revisit'
            ? revisitView
            : state.focus === 'downtown'
              ? DOWNTOWN_VIEW
              : ROUTE_VIEW,
          {transitionMs: ctx.reducedMotion() ? 0 : 700}
        );
    },
    encode(commandEncoder) {
      if (graphDirty) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        activeGraph.reader.request(commandEncoder);
        graphDirty = false;
      } else activeGraph.reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const result = latestResult;
      const sourcePath = makePath(sourceLocal, projection, sourceLocal.length / 2);
      const inputPath = makePath(
        activeGraph.inputLocal,
        projection,
        activeGraph.inputLocal.length / 2
      );
      const editedPath = result
        ? makePath(result.outputPositions, projection, result.writtenCount)
        : [];
      const recoveredPath = result?.recoveredPositions
        ? makePath(result.recoveredPositions, projection, activeGraph.inputLocal.length / 2)
        : [];
      const vertexPoints =
        activeGraph.edit === 'translation'
          ? []
          : editedPath.map((coordinates, index) => ({coordinates, index}));
      const referencePoints = Array.from(
        {length: activeGraph.referenceLngLat.length / 2},
        (_, index) => ({
          coordinates: [
            activeGraph.referenceLngLat[index * 2],
            activeGraph.referenceLngLat[index * 2 + 1]
          ] as [number, number],
          index
        })
      );
      const endpoints = editedPath.length
        ? [
            {coordinates: editedPath[0], label: 'first'},
            {coordinates: editedPath[editedPath.length - 1], label: 'last'}
          ]
        : [];
      const loopDisplay = activeGraph.edit === 'loop-revisit';
      return [
        new PathLayer({
          id: 'cta-red-source-line',
          data: [{path: sourcePath}],
          getPath: (item: {path: [number, number][]}) => item.path,
          getColor: SOURCE_COLOR,
          getWidth: 3,
          widthUnits: 'pixels',
          widthMinPixels: 2,
          pickable: false,
          parameters: {depthCompare: 'always', depthWriteEnabled: false}
        }),
        ...(loopDisplay
          ? ([
              new PathLayer({
                id: 'derived-revisit-input',
                data: [{path: inputPath}],
                getPath: (item: {path: [number, number][]}) => item.path,
                getColor: [217, 133, 94, 235],
                getWidth: 5,
                widthUnits: 'pixels',
                widthMinPixels: 3,
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : []),
        ...(editedPath.length > 1
          ? ([
              new PathLayer({
                id: 'cta-red-edited-line',
                data: [{path: editedPath}],
                getPath: (item: {path: [number, number][]}) => item.path,
                getColor: EDITED_COLOR,
                getWidth: 6,
                widthUnits: 'pixels',
                widthMinPixels: 4,
                capRounded: true,
                jointRounded: true,
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : []),
        ...(recoveredPath.length > 1 && activeGraph.edit === 'translation'
          ? ([
              new PathLayer({
                id: 'inverse-recovered-route-line',
                data: [{path: recoveredPath}],
                getPath: (item: {path: [number, number][]}) => item.path,
                getColor: RECOVERED_COLOR,
                getWidth: 2,
                widthUnits: 'pixels',
                widthMinPixels: 2,
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : []),
        ...(activeGraph.edit === 'vertex-snap' || activeGraph.edit === 'segment-snap'
          ? ([
              new ScatterplotLayer({
                id: 'red-station-references',
                data: referencePoints,
                getPosition: (item: {coordinates: [number, number]}) => item.coordinates,
                getRadius: 4,
                radiusUnits: 'pixels',
                radiusMinPixels: 3,
                getFillColor: REFERENCE_COLOR,
                getLineColor: [18, 26, 35, 255],
                lineWidthMinPixels: 1,
                stroked: true,
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : []),
        ...(vertexPoints.length
          ? ([
              new ScatterplotLayer({
                id: 'edited-route-vertices',
                data: vertexPoints,
                getPosition: (item: {coordinates: [number, number]}) => item.coordinates,
                getRadius: 3,
                radiusUnits: 'pixels',
                radiusMinPixels: 2,
                getFillColor: [255, 246, 201, 230],
                getLineColor: [29, 36, 43, 230],
                lineWidthMinPixels: 0.8,
                stroked: true,
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : []),
        ...(endpoints.length
          ? ([
              new TextLayer({
                id: 'edited-route-direction-labels',
                data: endpoints,
                getPosition: (item: {coordinates: [number, number]}) => item.coordinates,
                getText: (item: {label: string}) => item.label,
                getColor: [24, 38, 52, 255],
                getSize: 13,
                sizeUnits: 'pixels',
                sizeMinPixels: 12,
                getPixelOffset: [8, -7],
                getTextAnchor: 'start',
                getAlignmentBaseline: 'center',
                characterSet: 'auto',
                fontSettings: {sdf: true},
                pickable: false,
                parameters: {depthCompare: 'always', depthWriteEnabled: false}
              })
            ] as Layer[])
          : [])
      ];
    },
    destroy() {
      graphSwap.destroy();
      releaseGraph(activeGraph);
      baseResources.destroy();
    }
  };
}
