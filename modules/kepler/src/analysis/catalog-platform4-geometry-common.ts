// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandNodeProducer, GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPULineGeometryOutput} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, type CatalogGpuSession} from './catalog-gpu-session';

/** All positions of a collection. */
export function collectCatalogPositions(
  collections: readonly CatalogFeatureCollection[]
): CatalogPosition[] {
  const positions: CatalogPosition[] = [];
  for (const collection of collections)
    for (const feature of collection.features) {
      positions.push(...getPointPositions(feature.geometry));
      for (const part of getLineParts(feature.geometry)) positions.push(...part);
      for (const polygon of getPolygonParts(feature.geometry))
        for (const ring of polygon) positions.push(...ring);
    }
  return positions;
}

/** One line part with its source feature. */
export type CatalogLinePart = {
  sourceIndex: number;
  partIndex: number;
  feature: CatalogFeature;
  coordinates: CatalogPosition[];
};

/** Line parts in feature order. */
export function listCatalogLineParts(
  lines: CatalogFeatureCollection,
  minimumVertices = 0
): CatalogLinePart[] {
  const parts: CatalogLinePart[] = [];
  lines.features.forEach((feature, sourceIndex) =>
    getLineParts(feature.geometry).forEach((coordinates, partIndex) => {
      if (coordinates.length >= minimumVertices)
        parts.push({sourceIndex, partIndex, feature, coordinates});
    })
  );
  return parts;
}

/** Flattens projected parts into one position array and line offsets. */
export function packCatalogParts(parts: readonly (readonly (readonly [number, number])[])[]): {
  positions: Float32Array;
  lineOffsets: Uint32Array;
} {
  const positions: number[] = [];
  const lineOffsets = [0];
  for (const part of parts) {
    for (const [x, y] of part) positions.push(x, y);
    lineOffsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), lineOffsets: Uint32Array.from(lineOffsets)};
}

export function getCatalogSourceColumns(feature: CatalogFeature, sourceIndex: number) {
  return {source_index: sourceIndex, source_id: getFeatureIdentity(feature)};
}

/** Result of one line-geometry contributor run. */
export type CatalogLineGeometryResult = {
  positions: Float32Array;
  lineOffsets: Uint32Array;
  sourceLineIndices: Uint32Array;
  lineCount: number;
};

/**
 * Runs a contributor that emits a `GPULineGeometryOutput` and reads it back. The factory receives
 * the uploaded inputs and the output views; overflow of either capacity is reported as unsupported
 * so the catalog falls back to the CPU reference.
 */
export async function runCatalogLineGeometry(
  session: CatalogGpuSession,
  packed: {positions: Float32Array; lineOffsets: Uint32Array},
  capacities: {vertices: number; lines: number},
  createContributor: (
    inputs: {positions: GraphDataView<'float32x2'>; lineOffsets: GraphDataView<'uint32'>},
    output: GPULineGeometryOutput
  ) => GPUCommandNodeProducer
): Promise<CatalogLineGeometryResult> {
  const lineCount = packed.lineOffsets.length - 1;
  const outputPositions = session.output('line-positions', 'float32x2', capacities.vertices);
  const outputOffsets = session.output('line-offsets', 'uint32', capacities.lines + 1);
  const outputSources = session.output('line-sources', 'uint32', capacities.lines);
  const count = session.output('line-count', 'uint32', 1);
  const overflow = session.output('line-overflow', 'uint32', 1);
  const outputLineCount = session.output('line-line-count', 'uint32', 1);
  session.graph.add(
    createContributor(
      {
        positions: session.input('line-input-positions', packed.positions, 'float32x2'),
        lineOffsets: session.input('line-input-offsets', packed.lineOffsets, 'uint32')
      },
      {
        positions: outputPositions.view,
        lineOffsets: outputOffsets.view,
        sourceLineIndices: outputSources.view,
        status: {count: count.view, overflow: overflow.view, lineCount: outputLineCount.view}
      }
    )
  );
  session.run();
  const [overflowFlag, producedLineCount, positions, lineOffsets, sourceLineIndices] = [
    await session.readScalar(overflow),
    await session.readScalar(outputLineCount),
    await session.readFloat32(outputPositions),
    await session.readUint32(outputOffsets),
    await session.readUint32(outputSources)
  ];
  // Capacities were sized from the exact requirement, so overflow is a contributor-side limit.
  if (overflowFlag) throw new CatalogGpuUnsupportedError('Line contributor output overflowed');
  return {positions, lineOffsets, sourceLineIndices, lineCount: producedLineCount || lineCount};
}

/** Reads line `index` of a result as positions mapped by `unproject`. */
export function readCatalogLine(
  result: CatalogLineGeometryResult,
  index: number,
  unproject: (x: number, y: number) => [number, number]
): [number, number][] {
  const line: [number, number][] = [];
  for (let vertex = result.lineOffsets[index]; vertex < result.lineOffsets[index + 1]; vertex++)
    line.push(unproject(result.positions[2 * vertex], result.positions[2 * vertex + 1]));
  return line;
}
