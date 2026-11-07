// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUDelaunayTessellation,
  getDelaunayMaximumTriangleCount,
  GPUNaturalNeighbor
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {MEUSE_LOG_RESPONSE_RANGES} from './meuse-validation.constants';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';

export type MeuseNaturalNeighborOptions = {
  response: 'zinc' | 'copper' | 'lead' | 'cadmium';
};

const MAXIMUM_NEIGHBOR_COUNT = 64;

/** Indexed Sibson interpolation over every Meuse site and source query location. */
export async function createMeuseNaturalNeighbor(
  ctx: SceneContext<MeuseNaturalNeighborOptions>
): Promise<SceneInstance<MeuseNaturalNeighborOptions>> {
  const dataset = ctx.datasets.get('meuse-validation');
  const {device} = ctx;
  const sampleCount = dataset.column('sampleDist').length;
  const queryCount = dataset.column('gridDist').length;
  const triangleCapacity = getDelaunayMaximumTriangleCount(sampleCount);
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const sampleMapPositions = dataset.projectColumn('samplePosition', origin);
  const queryMapPositions = dataset.projectColumn('gridPosition', origin);
  const riverCoordinates =
    (dataset.geojson?.features[0]?.geometry as {coordinates: [number, number][]} | undefined)
      ?.coordinates ?? [];
  const riverSegments = new Float32Array(Math.max(0, riverCoordinates.length - 1) * 4);
  for (let index = 0; index + 1 < riverCoordinates.length; index++) {
    const first = projection.project(...riverCoordinates[index]);
    const second = projection.project(...riverCoordinates[index + 1]);
    riverSegments.set([first[0], first[1], second[0], second[1]], index * 4);
  }
  const sampleSource = dataset.column('samplePositionRD');
  const querySource = dataset.column('gridPositionRD');
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let row = 0; row < sampleCount; row++) {
    minX = Math.min(minX, sampleSource[row * 2]);
    maxX = Math.max(maxX, sampleSource[row * 2]);
    minY = Math.min(minY, sampleSource[row * 2 + 1]);
    maxY = Math.max(maxY, sampleSource[row * 2 + 1]);
  }
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const samplePositions = Float32Array.from(
    sampleSource,
    (value, index) => value - (index % 2 ? centerY : centerX)
  );
  const queryPositions = Float32Array.from(
    querySource,
    (value, index) => value - (index % 2 ? centerY : centerX)
  );
  const valuesByResponse = {
    zinc: dataset.column('logZinc'),
    copper: dataset.column('logCopper'),
    lead: dataset.column('logLead'),
    cadmium: dataset.column('logCadmium')
  };
  const resources = new SpatialAnalysisResources(device, 'meuse-natural-neighbor');
  const positionsBuffer = resources.createBuffer('sample-positions', samplePositions);
  const valuesBuffer = resources.createBuffer(
    'sample-log-metal',
    Float32Array.from(valuesByResponse[ctx.options.response])
  );
  const queriesBuffer = resources.createBuffer('query-positions', queryPositions);
  const sampleMapBuffer = resources.createBuffer(
    'sample-map-positions',
    sampleMapPositions,
    Buffer.VERTEX
  );
  const queryMapBuffer = resources.createBuffer(
    'query-map-positions',
    queryMapPositions,
    Buffer.VERTEX
  );
  const mappedValues = resources.createBuffer('natural-neighbor-map-values', queryCount * 4);
  const packedValues = resources.createBuffer('natural-neighbor-packed-values', queryCount * 16);
  const predictionStatus = resources.createBuffer('natural-neighbor-status', queryCount * 4);
  const rawTriangles = resources.createBuffer('delaunay-raw-triangles', triangleCapacity * 12);
  const adjacencyTriangles = resources.createBuffer(
    'delaunay-indexed-triangles',
    triangleCapacity * 6 * 4
  );
  const querySeeds = resources.createBuffer('query-triangle-seeds', queryCount * 4);
  const triangleCount = resources.createBuffer('delaunay-count', 4);
  const triangleRequiredCount = resources.createBuffer('delaunay-required-count', 4);
  const triangleOverflow = resources.createBuffer('delaunay-overflow', 4);
  const triangleInvalidCount = resources.createBuffer('delaunay-invalid-count', 4);
  const triangleDuplicateCount = resources.createBuffer('delaunay-duplicate-count', 4);
  const generationInput = resources.createBuffer(
    'natural-neighbor-generation-input',
    Uint32Array.of(0)
  );
  const generationOutput = resources.createBuffer('natural-neighbor-generation-output', 4);
  const riverBuffer = resources.createBuffer('river-outline', riverSegments);

  const graph = new GPUCommandGraph<void>(device, {id: 'meuse-natural-neighbor'});
  const positionView = importGraphBuffer(
    graph,
    'sample-positions',
    positionsBuffer,
    'float32x2',
    sampleCount
  );
  const responseView = importGraphBuffer(
    graph,
    'sample-log-metal',
    valuesBuffer,
    'float32',
    sampleCount
  );
  const queryView = importGraphBuffer(
    graph,
    'query-positions',
    queriesBuffer,
    'float32x2',
    queryCount
  );
  const rawTriangleView = importGraphBuffer(
    graph,
    'delaunay-raw-triangles',
    rawTriangles,
    'uint32x3',
    triangleCapacity
  );
  const countView = importGraphBuffer(graph, 'delaunay-count', triangleCount, 'uint32', 1);
  const requiredView = importGraphBuffer(
    graph,
    'delaunay-required-count',
    triangleRequiredCount,
    'uint32',
    1
  );
  const overflowView = importGraphBuffer(graph, 'delaunay-overflow', triangleOverflow, 'uint32', 1);
  const invalidView = importGraphBuffer(
    graph,
    'delaunay-invalid-count',
    triangleInvalidCount,
    'uint32',
    1
  );
  const duplicateView = importGraphBuffer(
    graph,
    'delaunay-duplicate-count',
    triangleDuplicateCount,
    'uint32',
    1
  );
  const adjacencyView = importGraphBuffer(
    graph,
    'delaunay-indexed-triangles',
    adjacencyTriangles,
    'uint32',
    triangleCapacity * 6
  );
  const seedView = importGraphBuffer(
    graph,
    'query-triangle-seeds',
    querySeeds,
    'uint32',
    queryCount
  );
  const predictionView = importGraphBuffer(
    graph,
    'natural-neighbor-packed-values',
    packedValues,
    'float32x4',
    queryCount
  );
  const mapValueView = importGraphBuffer(
    graph,
    'natural-neighbor-map-values',
    mappedValues,
    'float32',
    queryCount
  );
  const statusView = importGraphBuffer(
    graph,
    'natural-neighbor-status',
    predictionStatus,
    'uint32',
    queryCount
  );
  const generationInputView = importGraphBuffer(
    graph,
    'natural-neighbor-generation-input',
    generationInput,
    'uint32',
    1
  );
  const generationOutputView = importGraphBuffer(
    graph,
    'natural-neighbor-generation-output',
    generationOutput,
    'uint32',
    1
  );
  graph.add(
    new GPUDelaunayTessellation({
      id: 'meuse-delaunay',
      positions: positionView,
      output: {
        triangles: rawTriangleView,
        status: {
          count: countView,
          requiredCount: requiredView,
          overflow: overflowView,
          invalidCount: invalidView,
          duplicateCount: duplicateView
        }
      }
    })
  );
  addKernelPass(graph, {
    id: 'meuse-delaunay-adjacency',
    invocationCount: triangleCapacity,
    bindings: [
      {name: 'raw', view: rawTriangleView, type: 'u32', access: 'read'},
      {name: 'triangleCount', view: countView, type: 'u32', access: 'read'},
      {name: 'requiredCount', view: requiredView, type: 'u32', access: 'read'},
      {name: 'overflow', view: overflowView, type: 'u32', access: 'read'},
      {name: 'indexed', view: adjacencyView, type: 'u32', access: 'read_write'}
    ],
    body: `let required = requiredCount[requiredCountOffset]; let complete = overflow[overflowOffset] == 0u && required <= ${triangleCapacity}u && triangleCount[triangleCountOffset] == required; if (!complete || index >= required) { for (var slot = 0u; slot < 6u; slot++) { indexed[indexedOffset + index * 6u + slot] = 0xffffffffu; } return; } let row = rawOffset + index * 3u; let firstVertexIndex = raw[row]; let secondVertexIndex = raw[row + 1u]; let thirdVertexIndex = raw[row + 2u]; indexed[indexedOffset + index * 6u] = firstVertexIndex; indexed[indexedOffset + index * 6u + 1u] = secondVertexIndex; indexed[indexedOffset + index * 6u + 2u] = thirdVertexIndex; for (var edge = 0u; edge < 3u; edge++) { let firstEdgeVertex = select(select(firstVertexIndex, secondVertexIndex, edge == 1u), thirdVertexIndex, edge == 2u); let secondEdgeVertex = select(select(secondVertexIndex, thirdVertexIndex, edge == 1u), firstVertexIndex, edge == 2u); var adjacent = 0xffffffffu; for (var candidate = 0u; candidate < ${triangleCapacity}u; candidate++) { if (candidate == index || candidate >= required) { continue; } let other = rawOffset + candidate * 3u; let firstMatch = raw[other] == firstEdgeVertex || raw[other + 1u] == firstEdgeVertex || raw[other + 2u] == firstEdgeVertex; let secondMatch = raw[other] == secondEdgeVertex || raw[other + 1u] == secondEdgeVertex || raw[other + 2u] == secondEdgeVertex; if (firstMatch && secondMatch) { adjacent = candidate; break; } } indexed[indexedOffset + index * 6u + 3u + edge] = adjacent; }`
  });
  addKernelPass(graph, {
    id: 'meuse-query-triangle-seeds',
    invocationCount: queryCount,
    declarations:
      'fn crossProduct(first: vec2<f32>, second: vec2<f32>) -> f32 { return first.x * second.y - first.y * second.x; }',
    bindings: [
      {name: 'positions', view: positionView, type: 'f32', access: 'read'},
      {name: 'queries', view: queryView, type: 'f32', access: 'read'},
      {name: 'triangles', view: adjacencyView, type: 'u32', access: 'read'},
      {name: 'triangleCount', view: countView, type: 'u32', access: 'read'},
      {name: 'requiredCount', view: requiredView, type: 'u32', access: 'read'},
      {name: 'overflow', view: overflowView, type: 'u32', access: 'read'},
      {name: 'seeds', view: seedView, type: 'u32', access: 'read_write'}
    ],
    body: `let required = requiredCount[requiredCountOffset]; let complete = overflow[overflowOffset] == 0u && required <= ${triangleCapacity}u && triangleCount[triangleCountOffset] == required; if (!complete) { seeds[seedsOffset + index] = 0xffffffffu; return; } let query = vec2<f32>(queries[queriesOffset + index * 2u], queries[queriesOffset + index * 2u + 1u]); var seed = 0xffffffffu; for (var triangle = 0u; triangle < ${triangleCapacity}u; triangle++) { if (triangle >= required) { break; } let start = trianglesOffset + triangle * 6u; let firstVertexIndex = triangles[start]; let secondVertexIndex = triangles[start + 1u]; let thirdVertexIndex = triangles[start + 2u]; let firstVertex = vec2<f32>(positions[positionsOffset + firstVertexIndex * 2u], positions[positionsOffset + firstVertexIndex * 2u + 1u]); let secondVertex = vec2<f32>(positions[positionsOffset + secondVertexIndex * 2u], positions[positionsOffset + secondVertexIndex * 2u + 1u]); let thirdVertex = vec2<f32>(positions[positionsOffset + thirdVertexIndex * 2u], positions[positionsOffset + thirdVertexIndex * 2u + 1u]); let scale = max(max(length(secondVertex - firstVertex), length(thirdVertex - secondVertex)), length(firstVertex - thirdVertex)); let tolerance = max(scale * scale, 1.0) * 1e-6; if (crossProduct(secondVertex - firstVertex, query - firstVertex) >= -tolerance && crossProduct(thirdVertex - secondVertex, query - secondVertex) >= -tolerance && crossProduct(firstVertex - thirdVertex, query - thirdVertex) >= -tolerance) { seed = triangle; break; } } seeds[seedsOffset + index] = seed;`
  });
  graph.add(
    new GPUNaturalNeighbor({
      id: 'meuse-indexed-natural-neighbor',
      positions: positionView,
      values: responseView,
      queries: queryView,
      triangles: adjacencyView,
      queryTriangles: seedView,
      maximumNeighborCount: MAXIMUM_NEIGHBOR_COUNT,
      maximumVertexCount: 64,
      output: {values: predictionView, status: statusView}
    })
  );
  addKernelPass(graph, {
    id: 'meuse-natural-neighbor-map-values',
    invocationCount: queryCount,
    bindings: [
      {name: 'packed', view: predictionView, type: 'f32', access: 'read'},
      {name: 'status', view: statusView, type: 'u32', access: 'read'},
      {name: 'values', view: mapValueView, type: 'f32', access: 'read_write'}
    ],
    body: 'let invalid = status[statusOffset + index] != 0u; values[valuesOffset + index] = select(packed[packedOffset + index * 4u], bitcast<f32>(0x7fc00000u | (index & 0u)), invalid);'
  });
  addKernelPass(graph, {
    id: 'meuse-natural-neighbor-generation-snapshot',
    invocationCount: 1,
    bindings: [
      {name: 'generation', view: generationInputView, type: 'u32', access: 'read'},
      {name: 'snapshot', view: generationOutputView, type: 'u32', access: 'read_write'}
    ],
    body: 'snapshot[snapshotOffset] = generation[generationOffset];'
  });
  const compiled = resources.track(await graph.compileAsync());
  const mapReader = new SummaryReader(
    resources,
    'meuse-natural-neighbor-summary',
    [
      {buffer: predictionStatus, size: queryCount * 4},
      {buffer: mappedValues, size: queryCount * 4},
      {buffer: packedValues, size: queryCount * 16},
      {buffer: triangleCount, size: 4},
      {buffer: triangleRequiredCount, size: 4},
      {buffer: triangleOverflow, size: 4},
      {buffer: triangleInvalidCount, size: 4},
      {buffer: triangleDuplicateCount, size: 4},
      {buffer: generationOutput, size: 4}
    ],
    bytes => {
      const generationOffset = queryCount * 24 + 5 * 4;
      if (new Uint32Array(bytes, generationOffset, 1)[0] !== currentGeneration) return;
      const statuses = new Uint32Array(bytes, 0, queryCount);
      const valuesOffset = queryCount * 4;
      latestValues = new Float32Array(bytes.slice(valuesOffset, valuesOffset + queryCount * 4));
      latestStatuses = statuses;
      pending = false;
      const packedOffset = valuesOffset + queryCount * 4;
      const packed = new Float32Array(bytes, packedOffset, queryCount * 4);
      const statusOffset = packedOffset + queryCount * 16;
      const triangleRows = new Uint32Array(bytes, statusOffset, 5);
      const topologyIsComplete =
        triangleRows[2] === 0 &&
        triangleRows[0] === triangleRows[1] &&
        triangleRows[3] === 0 &&
        triangleRows[4] === 0;
      const validQueries = statuses.reduce((sum, value) => sum + Number(value === 0), 0);
      const invalidQueries = statuses.reduce((sum, value) => sum + Number(value === 1), 0);
      const degenerateQueries = statuses.reduce((sum, value) => sum + Number(value === 2), 0);
      const capacityFailures = statuses.reduce((sum, value) => sum + Number(value === 3), 0);
      let minimumPrediction = Infinity;
      let maximumPrediction = -Infinity;
      let predictionTotal = 0;
      let maximumNeighborCount = 0;
      let maximumWeightSumError = 0;
      for (let query = 0; query < queryCount; query++) {
        if (statuses[query] !== 0) continue;
        const prediction = packed[query * 4];
        minimumPrediction = Math.min(minimumPrediction, prediction);
        maximumPrediction = Math.max(maximumPrediction, prediction);
        predictionTotal += prediction;
        maximumWeightSumError = Math.max(
          maximumWeightSumError,
          Math.abs(packed[query * 4 + 2] - 1)
        );
        maximumNeighborCount = Math.max(maximumNeighborCount, packed[query * 4 + 3]);
      }
      const topologyLabel = topologyIsComplete
        ? `outside hull ${invalidQueries.toLocaleString()}`
        : `status 1 unavailable/invalid topology ${invalidQueries.toLocaleString()}`;
      ctx.setReadout(
        'triangulation',
        `Triangles ${triangleRows[0]}/${triangleRows[1]} · overflow ${triangleRows[2]} · invalid sites ${triangleRows[3]} · duplicate sites ${triangleRows[4]}`
      );
      ctx.setReadout(
        'naturalStatus',
        `${validQueries.toLocaleString()} valid · ${topologyLabel} · ${degenerateQueries.toLocaleString()} degenerate · ${capacityFailures.toLocaleString()} cavity-capacity failures`
      );
      ctx.setReadout(
        'naturalDiagnostics',
        validQueries > 0
          ? `log estimate mean ${(predictionTotal / validQueries).toFixed(3)} · range ${minimumPrediction.toFixed(3)}–${maximumPrediction.toFixed(3)} · max neighbors ${maximumNeighborCount} · max |weight sum − 1| ${maximumWeightSumError.toExponential(2)}`
          : 'No valid interpolation estimates to summarize.'
      );
      ctx.requestLayers();
      ctx.refreshTooltip();
    }
  );
  const mapValueBuffer = resources.createBuffer(
    'sample-log-metal-for-map',
    valuesByResponse[ctx.options.response],
    Buffer.VERTEX
  );
  let graphDirty = true;
  let pending = true;
  let currentGeneration = 0;
  let latestValues: Float32Array | null = null;
  let latestStatuses: Uint32Array | null = null;
  ctx.setReadout('sampleCount', sampleCount);
  ctx.setReadout('queryCount', queryCount);
  ctx.setReadout('triangulation', 'Delaunay triangulation pending GPU execution.');
  ctx.setReadout('naturalStatus', 'Indexed natural-neighbor interpolation pending GPU execution.');
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        mapReader.request(commandEncoder);
        graphDirty = false;
      } else {
        mapReader.flush(commandEncoder);
      }
    },
    setOption(id, _value, state) {
      if (id === 'response') {
        currentGeneration++;
        generationInput.write(Uint32Array.of(currentGeneration));
        const response = Float32Array.from(valuesByResponse[state.response]);
        valuesBuffer.write(response);
        mapValueBuffer.write(response);
        mapReader.markStale();
        graphDirty = true;
        pending = true;
        latestValues = null;
        latestStatuses = null;
        ctx.setReadout(
          'naturalStatus',
          'Recomputing exact index-based predictions for this metal.'
        );
        ctx.refreshLegends();
      }
      ctx.requestLayers();
      ctx.refreshTooltip();
    },
    getLayers(): Layer[] {
      const layers: Layer[] = [];
      if (!pending && latestValues) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'meuse-natural-neighbor-query-map',
            coordinateOrigin: [origin[0], origin[1], 0],
            positions: queryMapBuffer,
            values: mappedValues,
            valueFormat: 'float32',
            instanceCount: queryCount,
            radiusPixels: 4,
            colormap: 'cividis',
            valueRange: MEUSE_LOG_RESPONSE_RANGES[ctx.options.response],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'meuse-natural-neighbor-river',
          coordinateOrigin: [origin[0], origin[1], 0],
          segments: riverBuffer,
          instanceCount: riverSegments.length / 4,
          widthPixels: 2,
          color: [49, 113, 151, 230]
        }),
        new SpatialAnalysisPointLayer({
          id: 'meuse-natural-neighbor-samples',
          coordinateOrigin: [origin[0], origin[1], 0],
          positions: sampleMapBuffer,
          values: mapValueBuffer,
          valueFormat: 'float32',
          instanceCount: sampleCount,
          radiusPixels: 4.5,
          colormap: 'cividis',
          valueRange: MEUSE_LOG_RESPONSE_RANGES[ctx.options.response],
          outlineColor: [16, 18, 22, 255],
          outlineWidthPixels: 1
        })
      );
      return layers;
    },
    getTooltip(event: ScenePointerEvent): string | null {
      if (!event.coordinate) return null;
      const projected = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let distance = Infinity;
      for (let index = 0; index < queryCount; index++) {
        const candidateDistance = Math.hypot(
          queryMapPositions[index * 2] - projected[0],
          queryMapPositions[index * 2 + 1] - projected[1]
        );
        if (candidateDistance < distance) {
          nearest = index;
          distance = candidateDistance;
        }
      }
      if (nearest < 0 || distance > 35 || !latestValues || !latestStatuses) return null;
      const status = latestStatuses[nearest];
      const concentration = latestValues[nearest];
      const responseName = ctx.options.response;
      if (status !== 0 || !Number.isFinite(concentration)) {
        return `Meuse ${responseName} query · no prediction (status ${status})`;
      }
      return `Meuse ${responseName} natural-neighbor estimate · ${Math.exp(concentration).toFixed(2)} mg/kg geometric estimate · ${Math.round(distance)} m from grid point`;
    },
    destroy() {
      mapReader.stop();
      resources.destroy();
    }
  };
}
