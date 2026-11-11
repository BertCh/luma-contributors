// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPULineChunk,
  GPULineMerge,
  GPULineSimplification,
  GPULineSmooth,
  getGPULineChunkParameterValues,
  getGPULineSimplificationParameterValues,
  getGPULineSmoothParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {type GPUSpatialContext} from '@luma-contributors/analysis/analysis-core';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {
  EARTH_RADIUS_METERS,
  addCatalogDiagnostic,
  appendCatalogRow,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureDistance,
  measurePathLength,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {
  collectCatalogPositions,
  getCatalogSourceColumns,
  listCatalogLineParts,
  packCatalogParts,
  runCatalogLineGeometry,
  readCatalogLine
} from './catalog-platform4-geometry-common';
import {makeLocalFrame} from './catalog-local-frame';

const LINE_COLUMNS = {
  line_id: 'string',
  source_index: 'int32',
  source_id: 'string',
  part_index: 'int32',
  vertices_before: 'int32',
  vertices_after: 'int32',
  geometry: 'string'
} as const;

function getPartsWarning(skipped: number, diagnostics: CatalogDiagnostic[]) {
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'part-skipped',
      'warning',
      'Line parts with fewer than two vertices were skipped.',
      skipped
    );
}

/**
 * GPULineSimplification (Douglas-Peucker, segment metric) on the local projected frame of the CPU
 * reference. Importance is computed once; the kept mask is `importance > tolerance`. Float32
 * distances differ from the float64 CPU reference only for vertices within about 1e-6 relative of
 * the tolerance.
 */
export async function runGpuLineSimplification(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {tolerance: number; coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  const parts = listCatalogLineParts(lines, 2);
  const skipped = listCatalogLineParts(lines).length - parts.length;
  if (!parts.length) throw new CatalogGpuUnsupportedError('No line part has two vertices');
  const frame = makeLocalFrame(parameters.coordinateSystem, collectCatalogPositions([lines]), {
    shift: 'center'
  });
  const packed = packCatalogParts(parts.map(part => part.coordinates.map(frame.forward)));
  const rowCount = packed.positions.length / 2;
  const {mask, converged} = await withCatalogGpuSession(
    context.device,
    'catalog-line-simplification',
    async session => {
      const importance = session.output('importance', 'float32', rowCount);
      const ids = session.output('ids', 'uint32', rowCount);
      const count = session.output('count', 'uint32', 1);
      const overflow = session.output('overflow', 'uint32', 1);
      const convergedOutput = session.output('converged', 'uint32', 1);
      const outputMask = session.output('mask', 'uint32', rowCount);
      session.graph.add(
        new GPULineSimplification({
          positions: session.input('positions', packed.positions, 'float32x2'),
          lineOffsets: session.input('line-offsets', packed.lineOffsets, 'uint32'),
          importance: importance.view,
          maximumIterations: 512,
          parameters: session.input(
            'parameters',
            getGPULineSimplificationParameterValues({tolerance: parameters.tolerance}),
            'float32'
          ),
          output: {
            ids: ids.view,
            outputMask: outputMask.view,
            status: {count: count.view, overflow: overflow.view, converged: convergedOutput.view}
          }
        })
      );
      session.run();
      return {
        mask: await session.readUint32(outputMask),
        converged: await session.readScalar(convergedOutput)
      };
    }
  );
  // Unconverged importance keeps a superset of the Douglas-Peucker vertices.
  if (!converged) throw new CatalogGpuUnsupportedError('GPULineSimplification did not converge');
  const table = makeCatalogTable('simplified-lines', 'Simplified lines', LINE_COLUMNS, 'geometry');
  let verticesAfter = 0;
  parts.forEach((part, partRow) => {
    const kept: CatalogPosition[] = [];
    for (let index = 0; index < part.coordinates.length; index++)
      if (mask[packed.lineOffsets[partRow] + index]) kept.push(part.coordinates[index]);
    verticesAfter += kept.length;
    appendCatalogRow(table, {
      line_id: `${part.sourceIndex}:${part.partIndex}`,
      ...getCatalogSourceColumns(part.feature, part.sourceIndex),
      part_index: part.partIndex,
      vertices_before: part.coordinates.length,
      vertices_after: kept.length,
      geometry: makeCatalogGeometryText('LineString', kept)
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  getPartsWarning(skipped, diagnostics);
  return {
    tables: [table],
    diagnostics,
    summary: {
      lineCount: table.rowCount,
      verticesBefore: parts.reduce((sum, part) => sum + part.coordinates.length, 0),
      verticesAfter
    },
    dispatchedContributors: ['GPULineSimplification']
  };
}

/**
 * GPULineSmooth: Chaikin is affine invariant, so lines are smoothed planar in the shifted
 * longitude/latitude frame. Open and closed parts run as two graphs (the contributor takes one
 * closed flag); closed parts follow the CPU rule of at least four vertices with equal ends.
 */
export async function runGpuLineSmooth(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {iterations: number}
): Promise<CatalogRunOutput> {
  const parts = listCatalogLineParts(lines, 2);
  const skipped = listCatalogLineParts(lines).length - parts.length;
  if (!parts.length) throw new CatalogGpuUnsupportedError('No line part has two vertices');
  const frame = makeLocalFrame('planar', collectCatalogPositions([lines]), {shift: 'center'});
  const isClosed = (coordinates: readonly CatalogPosition[]) => {
    const first = coordinates[0];
    const last = coordinates[coordinates.length - 1];
    return coordinates.length > 3 && first[0] === last[0] && first[1] === last[1];
  };
  const smoothed = new Map<number, [number, number][]>();
  for (const closed of [false, true]) {
    const group = parts
      .map((part, row) => ({part, row}))
      .filter(entry => isClosed(entry.part.coordinates) === closed);
    if (!group.length) continue;
    const packed = packCatalogParts(group.map(({part}) => part.coordinates.map(frame.forward)));
    const capacity =
      (packed.positions.length / 2) * 2 ** parameters.iterations + group.length * 2 + 16;
    const result = await withCatalogGpuSession(context.device, 'catalog-line-smooth', session =>
      runCatalogLineGeometry(
        session,
        packed,
        {vertices: capacity, lines: group.length},
        (inputs, output) =>
          new GPULineSmooth({
            ...inputs,
            iterationCount: parameters.iterations,
            closed,
            parameters: session.input('parameters', getGPULineSmoothParameterValues({}), 'float32'),
            output
          })
      )
    );
    group.forEach(({row}, index) =>
      smoothed.set(row, readCatalogLine(result, index, frame.inverse))
    );
  }
  const table = makeCatalogTable('smoothed-lines', 'Smoothed lines', LINE_COLUMNS, 'geometry');
  parts.forEach((part, row) => {
    const line = smoothed.get(row)!;
    appendCatalogRow(table, {
      line_id: `${part.sourceIndex}:${part.partIndex}`,
      ...getCatalogSourceColumns(part.feature, part.sourceIndex),
      part_index: part.partIndex,
      vertices_before: part.coordinates.length,
      vertices_after: line.length,
      geometry: makeCatalogGeometryText('LineString', line)
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  getPartsWarning(skipped, diagnostics);
  return {
    tables: [table],
    diagnostics,
    summary: {lineCount: table.rowCount},
    dispatchedContributors: ['GPULineSmooth']
  };
}

function getMeasureContext(coordinateSystem: CatalogCoordinateSystem): GPUSpatialContext {
  return coordinateSystem === 'planar'
    ? {coordinateSpace: 'planar', metric: 'native', units: 'native'}
    : {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        units: 'meters',
        sphereRadius: EARTH_RADIUS_METERS
      };
}

/**
 * GPULineChunk in chunk mode. Pieces follow great-circle interpolation for EPSG:4326, while the CPU
 * reference interpolates linearly in degrees; the piece count per line is the same.
 */
export async function runGpuLineChunk(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {maximumChunkLength: number; coordinateSystem: CatalogCoordinateSystem},
  maximumRows: number
): Promise<CatalogRunOutput> {
  const parts = listCatalogLineParts(lines, 2);
  const skipped = listCatalogLineParts(lines).length - parts.length;
  if (!parts.length) throw new CatalogGpuUnsupportedError('No line part has two vertices');
  const lengths = parts.map(part => {
    let total = 0;
    for (let index = 1; index < part.coordinates.length; index++)
      total += measureDistance(
        part.coordinates[index - 1],
        part.coordinates[index],
        parameters.coordinateSystem
      );
    return total;
  });
  const chunkCounts = lengths.map(total =>
    Math.max(1, Math.ceil(total / parameters.maximumChunkLength - 1e-9))
  );
  const lineCapacity = chunkCounts.reduce((sum, value) => sum + value, 0);
  // Chunk count exceeds the declared output capacity; the CPU engine raises the allocation error.
  if (lineCapacity > maximumRows) throw new CatalogGpuUnsupportedError('Chunk capacity exceeded');
  // Longitude/latitude interpolates on the sphere, so the raw (unshifted) degrees go in.
  const packed = packCatalogParts(
    parts.map(part =>
      part.coordinates.map(position => [position[0], position[1]] as [number, number])
    )
  );
  const spatialContext = getMeasureContext(parameters.coordinateSystem);
  const vertexCapacity = packed.positions.length / 2 + 2 * lineCapacity + 16;
  const result = await withCatalogGpuSession(context.device, 'catalog-line-chunk', session =>
    runCatalogLineGeometry(
      session,
      packed,
      {vertices: vertexCapacity, lines: lineCapacity},
      (inputs, output) =>
        new GPULineChunk({
          ...inputs,
          mode: 'chunk',
          spatialContext,
          parameters: session.input(
            'parameters',
            getGPULineChunkParameterValues({
              chunkLength: parameters.maximumChunkLength,
              spatialContext
            }),
            'float32'
          ),
          output
        })
    )
  );
  const table = makeCatalogTable(
    'line-chunks',
    'Line chunks',
    {
      chunk_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      chunk_index: 'int32',
      start_measure: 'float64',
      end_measure: 'float64',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const chunkIndices = new Array<number>(parts.length).fill(0);
  for (let piece = 0; piece < result.lineCount; piece++) {
    const row = result.sourceLineIndices[piece];
    const part = parts[row];
    const chunk = chunkIndices[row]++;
    const startMeasure = chunk * parameters.maximumChunkLength;
    const endMeasure = Math.min(lengths[row], (chunk + 1) * parameters.maximumChunkLength);
    appendCatalogRow(table, {
      chunk_id: `${part.sourceIndex}:${part.partIndex}:${chunk}`,
      ...getCatalogSourceColumns(part.feature, part.sourceIndex),
      part_index: part.partIndex,
      chunk_index: chunk,
      start_measure: startMeasure,
      end_measure: endMeasure,
      length: endMeasure - startMeasure,
      geometry: makeCatalogGeometryText(
        'LineString',
        readCatalogLine(result, piece, (x, y) => [x, y])
      )
    });
  }
  const diagnostics: CatalogDiagnostic[] = [];
  getPartsWarning(skipped, diagnostics);
  return {
    tables: [table],
    diagnostics,
    summary: {chunkCount: table.rowCount},
    dispatchedContributors: ['GPULineChunk']
  };
}

/**
 * GPULineMerge on coordinates shifted to the data center (equal endpoints stay bitwise equal after
 * the shift and gain float32 resolution). Chain membership, order and direction come from the GPU;
 * vertices are rebuilt from the float64 source parts. Chain order and direction follow the GPU
 * convention (ordered by leader line, path starting at the lower endpoint id), so a chain may be
 * reversed relative to the CPU reference.
 */
export async function runGpuLineMerge(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem}
): Promise<CatalogRunOutput> {
  const parts = listCatalogLineParts(lines, 2);
  if (!parts.length) throw new CatalogGpuUnsupportedError('No line part has two vertices');
  const frame = makeLocalFrame('planar', collectCatalogPositions([lines]), {shift: 'center'});
  const packed = packCatalogParts(parts.map(part => part.coordinates.map(frame.forward)));
  const partCount = parts.length;
  const {chainCount, lineChains, lineOrders, lineReversed} = await withCatalogGpuSession(
    context.device,
    'catalog-line-merge',
    async session => {
      const chainOffsets = session.output('chain-offsets', 'uint32', partCount + 1);
      const positions = session.output('chain-positions', 'float32x2', packed.positions.length / 2);
      const count = session.output('count', 'uint32', 1);
      const chains = session.output('line-chains', 'uint32', partCount);
      const orders = session.output('line-orders', 'uint32', partCount);
      const reversed = session.output('line-reversed', 'uint32', partCount);
      session.graph.add(
        new GPULineMerge({
          positions: session.input('positions', packed.positions, 'float32x2'),
          lineOffsets: session.input('line-offsets', packed.lineOffsets, 'uint32'),
          output: {
            chainOffsets: chainOffsets.view,
            positions: positions.view,
            status: {count: count.view},
            lineChains: chains.view,
            lineOrders: orders.view,
            lineReversed: reversed.view
          }
        })
      );
      session.run();
      return {
        chainCount: await session.readScalar(count),
        lineChains: await session.readUint32(chains),
        lineOrders: await session.readUint32(orders),
        lineReversed: await session.readUint32(reversed)
      };
    }
  );
  const members: number[][] = Array.from({length: chainCount}, () => []);
  for (let row = 0; row < partCount; row++) members[lineChains[row]].push(row);
  const table = makeCatalogTable(
    'merged-lines',
    'Merged lines',
    {
      merged_id: 'string',
      member_count: 'int32',
      member_source_indices: 'string',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (const chainMembers of members) {
    chainMembers.sort((first, second) => lineOrders[first] - lineOrders[second]);
    const chain: CatalogPosition[] = [];
    chainMembers.forEach((row, index) => {
      const coordinates = lineReversed[row]
        ? [...parts[row].coordinates].reverse()
        : parts[row].coordinates;
      chain.push(...(index === 0 ? coordinates : coordinates.slice(1)));
    });
    const sourceIndices = [...new Set(chainMembers.map(row => parts[row].sourceIndex))].sort(
      (first, second) => first - second
    );
    appendCatalogRow(table, {
      merged_id: `m${table.rowCount}`,
      member_count: chainMembers.length,
      member_source_indices: sourceIndices.join(','),
      length: measurePathLength(chain, parameters.coordinateSystem),
      geometry: makeCatalogGeometryText('LineString', chain)
    });
  }
  return {
    tables: [table],
    diagnostics: [],
    summary: {inputPartCount: partCount, mergedLineCount: table.rowCount},
    dispatchedContributors: ['GPULineMerge']
  };
}
