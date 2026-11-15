// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {toDegrees, toRadians} from '@math.gl/core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {getGeodesicDestination, getInitialBearingDegrees} from './catalog-geodesy';
import {makeProjector, type Projector} from './catalog-network';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  measureDistance,
  measurePathLength,
  measureRingSignedArea,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogGeometry,
  type CatalogOutputTable,
  type CatalogPosition
} from './catalog-geojson';
import {mapGeometryCoordinates} from './catalog-geometry-utils';

/** Tables, diagnostics and summary produced by one extra operation. */
export type ExtraRunOutput = {
  tables: CatalogOutputTable[];
  diagnostics: CatalogDiagnostic[];
  summary: Record<string, string | number>;
};

function collectPositions(collection: CatalogFeatureCollection): CatalogPosition[] {
  const positions: CatalogPosition[] = [];
  for (const feature of collection.features) {
    positions.push(...getPointPositions(feature.geometry));
    for (const part of getLineParts(feature.geometry)) positions.push(...part);
    for (const polygon of getPolygonParts(feature.geometry))
      for (const ring of polygon) positions.push(...ring);
  }
  return positions;
}

function makeTransformedGeometryText(
  geometry: NonNullable<CatalogGeometry>,
  mapPosition: (position: CatalogPosition) => CatalogPosition
): string {
  return makeCatalogGeometryText(
    geometry.type,
    mapGeometryCoordinates(geometry.coordinates, mapPosition)
  );
}

function makeSourceColumns(feature: CatalogFeature, sourceIndex: number) {
  return {source_index: sourceIndex, source_id: getFeatureIdentity(feature)};
}

/** Deterministic 32-bit generator (mulberry32). */
export function makeSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function makeFeatureSeed(seed: number, featureIndex: number): number {
  return (seed ^ Math.imul(featureIndex + 1, 0x9e3779b1)) >>> 0;
}

// ---------------------------------------------------------------------------------------------
// Line simplification (Douglas-Peucker)
// ---------------------------------------------------------------------------------------------

function simplifyProjectedLine(points: readonly [number, number][], tolerance: number): number[] {
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length) {
    const [first, last] = stack.pop()!;
    const [ax, ay] = points[first];
    const [bx, by] = points[last];
    const lengthSquared = (bx - ax) ** 2 + (by - ay) ** 2;
    let farthest = -1;
    let farthestDistance = tolerance;
    for (let index = first + 1; index < last; index++) {
      const [px, py] = points[index];
      const u =
        lengthSquared === 0
          ? 0
          : Math.max(
              0,
              Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / lengthSquared)
            );
      const distance = Math.hypot(px - (ax + u * (bx - ax)), py - (ay + u * (by - ay)));
      if (distance > farthestDistance) {
        farthestDistance = distance;
        farthest = index;
      }
    }
    if (farthest >= 0) {
      keep[farthest] = 1;
      stack.push([first, farthest], [farthest, last]);
    }
  }
  return Array.from(keep, (flag, index) => (flag ? index : -1)).filter(index => index >= 0);
}

export function runLineSimplification(
  lines: CatalogFeatureCollection,
  parameters: {tolerance: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, collectPositions(lines));
  const table = makeCatalogTable(
    'simplified-lines',
    'Simplified lines',
    {
      line_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      vertices_before: 'int32',
      vertices_after: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let skipped = 0;
  let verticesBefore = 0;
  let verticesAfter = 0;
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length < 2) {
        skipped++;
        return;
      }
      const kept = simplifyProjectedLine(part.map(project), parameters.tolerance);
      verticesBefore += part.length;
      verticesAfter += kept.length;
      appendCatalogRow(table, {
        line_id: `${sourceIndex}:${partIndex}`,
        ...makeSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        vertices_before: part.length,
        vertices_after: kept.length,
        geometry: makeCatalogGeometryText(
          'LineString',
          kept.map(index => part[index])
        )
      });
    });
  });
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'part-skipped',
      'warning',
      'Line parts with fewer than two vertices were skipped.',
      skipped
    );
  return {
    tables: [table],
    diagnostics,
    summary: {lineCount: table.rowCount, verticesBefore, verticesAfter}
  };
}

// ---------------------------------------------------------------------------------------------
// Line smoothing (Chaikin corner cutting)
// ---------------------------------------------------------------------------------------------

function smoothLine(part: readonly CatalogPosition[], closed: boolean): CatalogPosition[] {
  const cut = (first: CatalogPosition, second: CatalogPosition): CatalogPosition[] => [
    [0.75 * first[0] + 0.25 * second[0], 0.75 * first[1] + 0.25 * second[1]],
    [0.25 * first[0] + 0.75 * second[0], 0.25 * first[1] + 0.75 * second[1]]
  ];
  if (closed) {
    const ring = part.slice(0, -1);
    const result = ring.flatMap((position, index) =>
      cut(position, ring[(index + 1) % ring.length])
    );
    return [...result, result[0]];
  }
  const result: CatalogPosition[] = [part[0]];
  for (let index = 1; index < part.length; index++)
    result.push(...cut(part[index - 1], part[index]));
  result.push(part[part.length - 1]);
  return result;
}

export function runLineSmooth(
  lines: CatalogFeatureCollection,
  parameters: {iterations: number}
): ExtraRunOutput {
  const table = makeCatalogTable(
    'smoothed-lines',
    'Smoothed lines',
    {
      line_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      vertices_before: 'int32',
      vertices_after: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let skipped = 0;
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length < 2) {
        skipped++;
        return;
      }
      const first = part[0];
      const last = part[part.length - 1];
      const closed = part.length > 3 && first[0] === last[0] && first[1] === last[1];
      let current: CatalogPosition[] = [...part];
      for (let iteration = 0; iteration < parameters.iterations; iteration++)
        current = smoothLine(current, closed);
      appendCatalogRow(table, {
        line_id: `${sourceIndex}:${partIndex}`,
        ...makeSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        vertices_before: part.length,
        vertices_after: current.length,
        geometry: makeCatalogGeometryText('LineString', current)
      });
    });
  });
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'part-skipped',
      'warning',
      'Line parts with fewer than two vertices were skipped.',
      skipped
    );
  return {tables: [table], diagnostics, summary: {lineCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Line precision reduction
// ---------------------------------------------------------------------------------------------

export function runLinePrecisionReduction(
  lines: CatalogFeatureCollection,
  parameters: {decimals: number}
): ExtraRunOutput {
  const factor = 10 ** parameters.decimals;
  const round = (value: number) => Math.round(value * factor) / factor;
  const table = makeCatalogTable(
    'reduced-lines',
    'Precision-reduced lines',
    {
      line_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      vertices_before: 'int32',
      vertices_after: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let collapsed = 0;
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const reduced: CatalogPosition[] = [];
      for (const position of part) {
        const next: CatalogPosition = [round(position[0]), round(position[1])];
        const previous = reduced[reduced.length - 1];
        if (!previous || previous[0] !== next[0] || previous[1] !== next[1]) reduced.push(next);
      }
      if (reduced.length < 2) {
        collapsed++;
        return;
      }
      appendCatalogRow(table, {
        line_id: `${sourceIndex}:${partIndex}`,
        ...makeSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        vertices_before: part.length,
        vertices_after: reduced.length,
        geometry: makeCatalogGeometryText('LineString', reduced)
      });
    });
  });
  if (collapsed)
    addCatalogDiagnostic(
      diagnostics,
      'part-collapsed',
      'warning',
      'Line parts that collapsed below two distinct vertices were dropped.',
      collapsed
    );
  return {tables: [table], diagnostics, summary: {lineCount: table.rowCount}};
}

// ---------------------------------------------------------------------------------------------
// Line merge
// ---------------------------------------------------------------------------------------------

export function runLineMerge(
  lines: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const parts: {sourceIndex: number; coordinates: CatalogPosition[]}[] = [];
  lines.features.forEach((feature, sourceIndex) => {
    for (const part of getLineParts(feature.geometry))
      if (part.length >= 2) parts.push({sourceIndex, coordinates: part});
  });
  const keyOf = (position: CatalogPosition) => `${position[0]},${position[1]}`;
  const ends = new Map<string, {part: number; atStart: boolean}[]>();
  parts.forEach((part, partIndex) => {
    for (const [atStart, position] of [
      [true, part.coordinates[0]],
      [false, part.coordinates[part.coordinates.length - 1]]
    ] as const) {
      const key = keyOf(position);
      const list = ends.get(key) ?? [];
      list.push({part: partIndex, atStart});
      ends.set(key, list);
    }
  });
  const used = new Uint8Array(parts.length);
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
  const extend = (chain: CatalogPosition[], members: number[], atTail: boolean) => {
    for (;;) {
      const key = keyOf(atTail ? chain[chain.length - 1] : chain[0]);
      const incident = ends.get(key) ?? [];
      // Only a junction of exactly two line ends is merged.
      if (incident.length !== 2) return;
      const other = incident.find(candidate => !used[candidate.part]);
      if (!other) return;
      used[other.part] = 1;
      members.push(other.part);
      const addition = [...parts[other.part].coordinates];
      // Orient so the shared vertex meets the chain end.
      if (atTail === !other.atStart) addition.reverse();
      if (atTail) chain.push(...addition.slice(1));
      else chain.unshift(...addition.slice(0, -1));
    }
  };
  parts.forEach((part, partIndex) => {
    if (used[partIndex]) return;
    used[partIndex] = 1;
    const chain = [...part.coordinates];
    const members = [partIndex];
    extend(chain, members, true);
    extend(chain, members, false);
    const sourceIndices = [...new Set(members.map(member => parts[member].sourceIndex))].sort(
      (a, b) => a - b
    );
    appendCatalogRow(table, {
      merged_id: `m${table.rowCount}`,
      member_count: members.length,
      member_source_indices: sourceIndices.join(','),
      length: measurePathLength(chain, parameters.coordinateSystem),
      geometry: makeCatalogGeometryText('LineString', chain)
    });
  });
  return {
    tables: [table],
    diagnostics: [],
    summary: {inputPartCount: parts.length, mergedLineCount: table.rowCount}
  };
}

// ---------------------------------------------------------------------------------------------
// Linear referencing helpers, chunking, locate, point at measure
// ---------------------------------------------------------------------------------------------

function makeCumulativeLengths(
  part: readonly CatalogPosition[],
  coordinateSystem: CatalogCoordinateSystem
): number[] {
  const cumulative = [0];
  for (let index = 1; index < part.length; index++)
    cumulative.push(
      cumulative[index - 1] + measureDistance(part[index - 1], part[index], coordinateSystem)
    );
  return cumulative;
}

function getPositionAtMeasure(
  part: readonly CatalogPosition[],
  cumulative: readonly number[],
  measure: number
): CatalogPosition {
  const total = cumulative[cumulative.length - 1];
  if (measure <= 0) return part[0];
  if (measure >= total) return part[part.length - 1];
  let index = 1;
  while (cumulative[index] < measure) index++;
  const span = cumulative[index] - cumulative[index - 1];
  const u = span === 0 ? 0 : (measure - cumulative[index - 1]) / span;
  const start = part[index - 1];
  const end = part[index];
  return [start[0] + (end[0] - start[0]) * u, start[1] + (end[1] - start[1]) * u];
}

function sliceLineByMeasure(
  part: readonly CatalogPosition[],
  cumulative: readonly number[],
  startMeasure: number,
  endMeasure: number
): CatalogPosition[] {
  const result: CatalogPosition[] = [getPositionAtMeasure(part, cumulative, startMeasure)];
  for (let index = 1; index < part.length - 1; index++)
    if (cumulative[index] > startMeasure && cumulative[index] < endMeasure)
      result.push(part[index]);
  result.push(getPositionAtMeasure(part, cumulative, endMeasure));
  return result;
}

export function runLineChunk(
  lines: CatalogFeatureCollection,
  parameters: {maximumChunkLength: number; coordinateSystem: CatalogCoordinateSystem},
  maximumRows: number
): ExtraRunOutput {
  validateGeometryOutputCapacity('line-chunk', {lines}, parameters);
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
  let skipped = 0;
  let plannedRows = 0;
  const diagnostics: CatalogDiagnostic[] = [];
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length < 2) {
        skipped++;
        return;
      }
      const cumulative = makeCumulativeLengths(part, parameters.coordinateSystem);
      const total = cumulative[cumulative.length - 1];
      const chunkCount = Math.max(1, Math.ceil(total / parameters.maximumChunkLength - 1e-9));
      plannedRows += chunkCount;
      // Chunk count would exceed the declared output capacity; stop before allocating rows.
      if (plannedRows > maximumRows)
        throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
      for (let chunk = 0; chunk < chunkCount; chunk++) {
        const startMeasure = chunk * parameters.maximumChunkLength;
        const endMeasure = Math.min(total, (chunk + 1) * parameters.maximumChunkLength);
        appendCatalogRow(table, {
          chunk_id: `${sourceIndex}:${partIndex}:${chunk}`,
          ...makeSourceColumns(feature, sourceIndex),
          part_index: partIndex,
          chunk_index: chunk,
          start_measure: startMeasure,
          end_measure: endMeasure,
          length: endMeasure - startMeasure,
          geometry: makeCatalogGeometryText(
            'LineString',
            sliceLineByMeasure(part, cumulative, startMeasure, endMeasure)
          )
        });
      }
    });
  });
  if (skipped)
    addCatalogDiagnostic(
      diagnostics,
      'part-skipped',
      'warning',
      'Line parts with fewer than two vertices were skipped.',
      skipped
    );
  return {tables: [table], diagnostics, summary: {chunkCount: table.rowCount}};
}

export function runLineLocate(
  lines: CatalogFeatureCollection,
  points: CatalogFeatureCollection,
  parameters: {maximumOffset: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, [
    ...collectPositions(lines),
    ...collectPositions(points)
  ]);
  const prepared: {
    lineIndex: number;
    partIndex: number;
    part: CatalogPosition[];
    projected: [number, number][];
    cumulative: number[];
  }[] = [];
  lines.features.forEach((feature, lineIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length >= 2)
        prepared.push({
          lineIndex,
          partIndex,
          part,
          projected: part.map(project),
          cumulative: makeCumulativeLengths(part, parameters.coordinateSystem)
        });
    });
  });
  const table = makeCatalogTable(
    'line-locations',
    'Located points',
    {
      location_id: 'string',
      point_index: 'int32',
      point_source_id: 'string',
      line_index: 'int32',
      line_source_id: 'string',
      part_index: 'int32',
      status: 'string',
      distance_along: 'float64',
      fraction: 'float64',
      offset: 'float64',
      side: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let located = 0;
  let locationIndex = 0;
  points.features.forEach((feature, pointIndex) => {
    for (const position of getPointPositions(feature.geometry)) {
      const [px, py] = project(position);
      let best: {
        entry: (typeof prepared)[number];
        segment: number;
        u: number;
        offset: number;
        cross: number;
      } | null = null;
      for (const entry of prepared)
        for (let segment = 1; segment < entry.projected.length; segment++) {
          const [ax, ay] = entry.projected[segment - 1];
          const [bx, by] = entry.projected[segment];
          const lengthSquared = (bx - ax) ** 2 + (by - ay) ** 2;
          const u =
            lengthSquared === 0
              ? 0
              : Math.max(
                  0,
                  Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / lengthSquared)
                );
          const offset = Math.hypot(px - (ax + u * (bx - ax)), py - (ay + u * (by - ay)));
          if (!best || offset < best.offset)
            best = {
              entry,
              segment,
              u,
              offset,
              cross: (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            };
        }
      const base = {
        location_id: `p${locationIndex++}`,
        point_index: pointIndex,
        point_source_id: getFeatureIdentity(feature)
      };
      if (!best || best.offset > parameters.maximumOffset) {
        appendCatalogRow(table, {...base, status: 'beyond-maximum-offset'});
        continue;
      }
      const {entry, segment, u, offset, cross} = best;
      const along =
        entry.cumulative[segment - 1] +
        u * (entry.cumulative[segment] - entry.cumulative[segment - 1]);
      const total = entry.cumulative[entry.cumulative.length - 1];
      const start = entry.part[segment - 1];
      const end = entry.part[segment];
      located++;
      appendCatalogRow(table, {
        ...base,
        line_index: entry.lineIndex,
        line_source_id: getFeatureIdentity(lines.features[entry.lineIndex]),
        part_index: entry.partIndex,
        status: 'ok',
        distance_along: along,
        fraction: total > 0 ? along / total : 0,
        offset,
        side: offset === 0 || cross === 0 ? 'on' : cross > 0 ? 'left' : 'right',
        geometry: makeCatalogGeometryText('Point', [
          start[0] + (end[0] - start[0]) * u,
          start[1] + (end[1] - start[1]) * u
        ])
      });
    }
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (located < table.rowCount)
    addCatalogDiagnostic(
      diagnostics,
      'beyond-maximum-offset',
      'warning',
      'Points farther than the maximum offset from every line were not located.',
      table.rowCount - located
    );
  return {
    tables: [table],
    diagnostics,
    summary: {pointCount: table.rowCount, locatedCount: located}
  };
}

export function runLinearReferencing(
  lines: CatalogFeatureCollection,
  parameters: {
    measures: readonly number[];
    measureMode: 'distance' | 'fraction';
    coordinateSystem: CatalogCoordinateSystem;
  }
): ExtraRunOutput {
  const table = makeCatalogTable(
    'measure-points',
    'Points at measure',
    {
      measure_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      measure: 'float64',
      distance_along: 'float64',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let outside = 0;
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      if (part.length < 2) return;
      const cumulative = makeCumulativeLengths(part, parameters.coordinateSystem);
      const total = cumulative[cumulative.length - 1];
      parameters.measures.forEach((measure, measureIndex) => {
        const distance = parameters.measureMode === 'fraction' ? measure * total : measure;
        const inside = distance >= 0 && distance <= total;
        if (!inside) outside++;
        appendCatalogRow(table, {
          measure_id: `${sourceIndex}:${partIndex}:${measureIndex}`,
          ...makeSourceColumns(feature, sourceIndex),
          part_index: partIndex,
          measure,
          distance_along: inside ? distance : null,
          status: inside ? 'ok' : 'outside-line',
          geometry: inside
            ? makeCatalogGeometryText('Point', getPositionAtMeasure(part, cumulative, distance))
            : null
        });
      });
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (outside)
    addCatalogDiagnostic(
      diagnostics,
      'outside-line',
      'warning',
      'Measures outside the line extent produce no point; nothing is clamped.',
      outside
    );
  return {tables: [table], diagnostics, summary: {pointCount: table.rowCount - outside}};
}

// ---------------------------------------------------------------------------------------------
// Group convex hull
// ---------------------------------------------------------------------------------------------

function computeConvexHull(positions: readonly CatalogPosition[]): CatalogPosition[] {
  const sorted = [...positions].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const unique = sorted.filter(
    (position, index) =>
      index === 0 || position[0] !== sorted[index - 1][0] || position[1] !== sorted[index - 1][1]
  );
  if (unique.length < 3) return unique;
  const cross = (o: CatalogPosition, a: CatalogPosition, b: CatalogPosition) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const build = (points: readonly CatalogPosition[]) => {
    const chain: CatalogPosition[] = [];
    for (const point of points) {
      while (
        chain.length >= 2 &&
        cross(chain[chain.length - 2], chain[chain.length - 1], point) <= 0
      )
        chain.pop();
      chain.push(point);
    }
    chain.pop();
    return chain;
  };
  return [...build(unique), ...build([...unique].reverse())];
}

export function runGroupConvexHull(
  points: CatalogFeatureCollection,
  parameters: {groupProperty: string; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const groups = new Map<string, CatalogPosition[]>();
  for (const feature of points.features) {
    const value = parameters.groupProperty
      ? feature.properties?.[parameters.groupProperty]
      : undefined;
    const key = value === undefined || value === null ? '' : String(value);
    const list = groups.get(key) ?? [];
    list.push(...getPointPositions(feature.geometry));
    groups.set(key, list);
  }
  const table = makeCatalogTable(
    'group-hulls',
    'Group convex hulls',
    {
      hull_id: 'string',
      group_value: 'string',
      point_count: 'int32',
      hull_vertex_count: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let degenerate = 0;
  for (const [group, positions] of [...groups.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const hull = computeConvexHull(positions);
    if (hull.length < 3) {
      degenerate++;
      continue;
    }
    const ring = [...hull, hull[0]];
    appendCatalogRow(table, {
      hull_id: `h${table.rowCount}`,
      group_value: group,
      point_count: positions.length,
      hull_vertex_count: hull.length,
      area: Math.abs(measureRingSignedArea(ring, parameters.coordinateSystem)),
      geometry: makeCatalogGeometryText('Polygon', [ring])
    });
  }
  if (degenerate)
    addCatalogDiagnostic(
      diagnostics,
      'degenerate-group',
      'warning',
      'Groups with fewer than three distinct or only collinear points have no hull and are omitted.',
      degenerate
    );
  return {
    tables: [table],
    diagnostics,
    summary: {hullCount: table.rowCount, groupCount: groups.size}
  };
}

// ---------------------------------------------------------------------------------------------
// Spatial joins: distance to geometry, nearest feature, point in polygon, buffer selection
// ---------------------------------------------------------------------------------------------

type ProjectedGeometry = {
  points: [number, number][];
  lines: [number, number][][];
  polygons: [number, number][][][];
};

function prepareProjectedGeometry(
  project: Projector,
  geometry: CatalogGeometry
): ProjectedGeometry {
  return {
    points: getPointPositions(geometry).map(project),
    lines: getLineParts(geometry).map(part => part.map(project)),
    polygons: getPolygonParts(geometry).map(polygon => polygon.map(ring => ring.map(project)))
  };
}

function measurePointSegmentDistance(
  px: number,
  py: number,
  start: readonly number[],
  end: readonly number[]
): number {
  const lengthSquared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2;
  const u =
    lengthSquared === 0
      ? 0
      : Math.max(
          0,
          Math.min(
            1,
            ((px - start[0]) * (end[0] - start[0]) + (py - start[1]) * (end[1] - start[1])) /
              lengthSquared
          )
        );
  return Math.hypot(
    px - (start[0] + u * (end[0] - start[0])),
    py - (start[1] + u * (end[1] - start[1]))
  );
}

function isInsideRing(px: number, py: number, ring: readonly (readonly number[])[]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [xi, yi] = ring[index];
    const [xj, yj] = ring[previous];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function isInsideProjectedPolygon(
  px: number,
  py: number,
  polygon: readonly (readonly (readonly number[])[])[]
): boolean {
  if (!polygon.length || !isInsideRing(px, py, polygon[0])) return false;
  return !polygon.slice(1).some(hole => isInsideRing(px, py, hole));
}

function isOnRingBoundary(
  px: number,
  py: number,
  polygon: readonly (readonly (readonly number[])[])[]
): boolean {
  for (const ring of polygon)
    for (let index = 1; index < ring.length; index++)
      if (measurePointSegmentDistance(px, py, ring[index - 1], ring[index]) === 0) return true;
  return false;
}

/** Distance in projected units from a point to a prepared geometry; polygon interiors give 0. */
function measureDistanceToGeometry(prepared: ProjectedGeometry, px: number, py: number): number {
  let best = Infinity;
  for (const point of prepared.points)
    best = Math.min(best, Math.hypot(px - point[0], py - point[1]));
  for (const line of prepared.lines)
    for (let index = 1; index < line.length; index++)
      best = Math.min(best, measurePointSegmentDistance(px, py, line[index - 1], line[index]));
  for (const polygon of prepared.polygons) {
    if (isInsideProjectedPolygon(px, py, polygon)) return 0;
    for (const ring of polygon)
      for (let index = 1; index < ring.length; index++)
        best = Math.min(best, measurePointSegmentDistance(px, py, ring[index - 1], ring[index]));
  }
  return best;
}

export function runNearestFeatureJoin(
  points: CatalogFeatureCollection,
  targets: CatalogFeatureCollection,
  parameters: {maximumDistance: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, [
    ...collectPositions(points),
    ...collectPositions(targets)
  ]);
  const prepared = targets.features.map(feature =>
    prepareProjectedGeometry(project, feature.geometry)
  );
  const table = makeCatalogTable(
    'nearest-feature-join',
    'Nearest feature join',
    {
      join_id: 'string',
      point_index: 'int32',
      point_source_id: 'string',
      target_index: 'int32',
      target_source_id: 'string',
      status: 'string',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let joined = 0;
  let joinIndex = 0;
  points.features.forEach((feature, pointIndex) => {
    for (const position of getPointPositions(feature.geometry)) {
      const [px, py] = project(position);
      let bestIndex = -1;
      let bestDistance = Infinity;
      prepared.forEach((target, targetIndex) => {
        const distance = measureDistanceToGeometry(target, px, py);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = targetIndex;
        }
      });
      const found = bestIndex >= 0 && bestDistance <= parameters.maximumDistance;
      if (found) joined++;
      appendCatalogRow(table, {
        join_id: `j${joinIndex++}`,
        point_index: pointIndex,
        point_source_id: getFeatureIdentity(feature),
        target_index: found ? bestIndex : null,
        target_source_id: found ? getFeatureIdentity(targets.features[bestIndex]) : null,
        status: found ? 'ok' : 'no-target-within-distance',
        distance: found ? bestDistance : null,
        geometry: makeCatalogGeometryText('Point', position)
      });
    }
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (joined < table.rowCount)
    addCatalogDiagnostic(
      diagnostics,
      'no-target-within-distance',
      'warning',
      'Points with no target within the maximum distance keep a row with null target columns.',
      table.rowCount - joined
    );
  return {tables: [table], diagnostics, summary: {pointCount: table.rowCount, joinedCount: joined}};
}

export function runPointInPolygonJoin(
  points: CatalogFeatureCollection,
  polygons: CatalogFeatureCollection
): ExtraRunOutput {
  const identity = (position: CatalogPosition): [number, number] => [position[0], position[1]];
  const prepared = polygons.features.map(
    feature => prepareProjectedGeometry(identity, feature.geometry).polygons
  );
  const table = makeCatalogTable(
    'point-in-polygon-join',
    'Point in polygon join',
    {
      join_id: 'string',
      point_index: 'int32',
      point_source_id: 'string',
      polygon_index: 'int32',
      polygon_source_id: 'string',
      match_count: 'int32',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  let inside = 0;
  let joinIndex = 0;
  points.features.forEach((feature, pointIndex) => {
    for (const position of getPointPositions(feature.geometry)) {
      const [px, py] = position;
      const matches: number[] = [];
      prepared.forEach((parts, polygonIndex) => {
        // Boundary points count as inside, matching a closed-set containment test.
        if (
          parts.some(
            part => isInsideProjectedPolygon(px, py, part) || isOnRingBoundary(px, py, part)
          )
        )
          matches.push(polygonIndex);
      });
      if (matches.length) inside++;
      appendCatalogRow(table, {
        join_id: `j${joinIndex++}`,
        point_index: pointIndex,
        point_source_id: getFeatureIdentity(feature),
        polygon_index: matches.length ? matches[0] : null,
        polygon_source_id: matches.length
          ? getFeatureIdentity(polygons.features[matches[0]])
          : null,
        match_count: matches.length,
        status: matches.length ? 'inside' : 'outside',
        geometry: makeCatalogGeometryText('Point', position)
      });
    }
  });
  return {
    tables: [table],
    diagnostics: [],
    summary: {pointCount: table.rowCount, insideCount: inside}
  };
}

export function runBufferSelection(
  selectors: CatalogFeatureCollection,
  features: CatalogFeatureCollection,
  parameters: {distance: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, [
    ...collectPositions(selectors),
    ...collectPositions(features)
  ]);
  const centers = selectors.features.flatMap((feature, selectorIndex) =>
    getPointPositions(feature.geometry).map(position => ({
      selectorIndex,
      projected: project(position)
    }))
  );
  const table = makeCatalogTable(
    'buffer-selection',
    'Selected features',
    {
      source_index: 'int32',
      source_id: 'string',
      nearest_selector_index: 'int32',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  features.features.forEach((feature, sourceIndex) => {
    if (!feature.geometry) return;
    const prepared = prepareProjectedGeometry(project, feature.geometry);
    let bestDistance = Infinity;
    let bestSelector = -1;
    for (const center of centers) {
      const distance = measureDistanceToGeometry(
        prepared,
        center.projected[0],
        center.projected[1]
      );
      if (distance < bestDistance) {
        bestDistance = distance;
        bestSelector = center.selectorIndex;
      }
    }
    if (bestDistance <= parameters.distance)
      appendCatalogRow(table, {
        ...makeSourceColumns(feature, sourceIndex),
        nearest_selector_index: bestSelector,
        distance: bestDistance,
        geometry: JSON.stringify(feature.geometry)
      });
  });
  return {
    tables: [table],
    diagnostics: [],
    summary: {selectedCount: table.rowCount, featureCount: features.features.length}
  };
}

// ---------------------------------------------------------------------------------------------
// Geodesics
// ---------------------------------------------------------------------------------------------

function computeDestination(
  origin: CatalogPosition,
  bearingDegrees: number,
  distance: number,
  coordinateSystem: CatalogCoordinateSystem
): CatalogPosition {
  if (coordinateSystem === 'planar') {
    const bearing = toRadians(bearingDegrees);
    return [origin[0] + distance * Math.sin(bearing), origin[1] + distance * Math.cos(bearing)];
  }
  return getGeodesicDestination(origin[0], origin[1], bearingDegrees, distance);
}

function computeInitialBearing(
  from: CatalogPosition,
  to: CatalogPosition,
  coordinateSystem: CatalogCoordinateSystem
): number {
  if (coordinateSystem !== 'planar')
    return getInitialBearingDegrees(from[0], from[1], to[0], to[1]);
  return (toDegrees(Math.atan2(to[0] - from[0], to[1] - from[1])) + 360) % 360;
}

export function runGeodesicDestination(
  points: CatalogFeatureCollection,
  parameters: {bearing: number; distance: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const table = makeCatalogTable(
    'geodesic-destinations',
    'Destination points',
    {
      destination_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      bearing: 'float64',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  points.features.forEach((feature, sourceIndex) => {
    for (const position of getPointPositions(feature.geometry))
      appendCatalogRow(table, {
        destination_id: `d${table.rowCount}`,
        ...makeSourceColumns(feature, sourceIndex),
        bearing: parameters.bearing,
        distance: parameters.distance,
        geometry: makeCatalogGeometryText(
          'Point',
          computeDestination(
            position,
            parameters.bearing,
            parameters.distance,
            parameters.coordinateSystem
          )
        )
      });
  });
  return {tables: [table], diagnostics: [], summary: {destinationCount: table.rowCount}};
}

export function runGeodesicPairs(
  origins: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const flatten = (collection: CatalogFeatureCollection) =>
    collection.features.flatMap(feature => getPointPositions(feature.geometry));
  const from = flatten(origins);
  const to = flatten(destinations);
  const pairCount = Math.min(from.length, to.length);
  const table = makeCatalogTable(
    'geodesic-pairs',
    'Point pairs',
    {
      pair_id: 'string',
      origin_index: 'int32',
      destination_index: 'int32',
      distance: 'float64',
      initial_bearing: 'float64',
      final_bearing: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  for (let index = 0; index < pairCount; index++)
    appendCatalogRow(table, {
      pair_id: `${index}`,
      origin_index: index,
      destination_index: index,
      distance: measureDistance(from[index], to[index], parameters.coordinateSystem),
      initial_bearing: computeInitialBearing(from[index], to[index], parameters.coordinateSystem),
      final_bearing:
        (computeInitialBearing(to[index], from[index], parameters.coordinateSystem) + 180) % 360,
      geometry: makeCatalogGeometryText('LineString', [from[index], to[index]])
    });
  const diagnostics: CatalogDiagnostic[] = [];
  if (from.length !== to.length)
    addCatalogDiagnostic(
      diagnostics,
      'unpaired-points',
      'warning',
      'Origins and destinations are paired by position; surplus points on the longer side are ignored.',
      Math.abs(from.length - to.length)
    );
  return {tables: [table], diagnostics, summary: {pairCount}};
}

// ---------------------------------------------------------------------------------------------
// Editing: orientation, affine transform, vertex snap
// ---------------------------------------------------------------------------------------------

export function runGeometryOrientation(
  polygons: CatalogFeatureCollection,
  parameters: {shellOrientation: 'counterclockwise' | 'clockwise'}
): ExtraRunOutput {
  const table = makeCatalogTable(
    'oriented-polygons',
    'Oriented polygons',
    {source_index: 'int32', source_id: 'string', rings_reversed: 'int32', geometry: 'string'},
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let zeroArea = 0;
  let reversedTotal = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    let reversed = 0;
    const orientRing = (ring: CatalogPosition[], wantCounterclockwise: boolean) => {
      const area = measureRingSignedArea(ring, 'planar');
      if (area === 0) {
        zeroArea++;
        return ring;
      }
      if (area > 0 === wantCounterclockwise) return ring;
      reversed++;
      return [...ring].reverse();
    };
    const shellCounterclockwise = parameters.shellOrientation === 'counterclockwise';
    const oriented = getPolygonParts(feature.geometry).map(polygon =>
      polygon.map((ring, ringIndex) =>
        orientRing(ring, ringIndex === 0 ? shellCounterclockwise : !shellCounterclockwise)
      )
    );
    reversedTotal += reversed;
    const isMulti = feature.geometry?.type === 'MultiPolygon';
    appendCatalogRow(table, {
      ...makeSourceColumns(feature, sourceIndex),
      rings_reversed: reversed,
      geometry: isMulti
        ? makeCatalogGeometryText('MultiPolygon', oriented)
        : makeCatalogGeometryText('Polygon', oriented[0] ?? [])
    });
  });
  if (zeroArea)
    addCatalogDiagnostic(
      diagnostics,
      'zero-area-ring',
      'warning',
      'Rings with zero planar area have no orientation and are left unchanged.',
      zeroArea
    );
  return {tables: [table], diagnostics, summary: {ringsReversed: reversedTotal}};
}

export function runAffineTransform(
  features: CatalogFeatureCollection,
  parameters: {
    translateX: number;
    translateY: number;
    scaleX: number;
    scaleY: number;
    rotationDegrees: number;
    pivot: 'origin' | 'bounds-center';
  }
): ExtraRunOutput {
  let pivotX = 0;
  let pivotY = 0;
  if (parameters.pivot === 'bounds-center') {
    const positions = collectPositions(features);
    if (positions.length) {
      const xs = positions.map(position => position[0]);
      const ys = positions.map(position => position[1]);
      pivotX = (Math.min(...xs) + Math.max(...xs)) / 2;
      pivotY = (Math.min(...ys) + Math.max(...ys)) / 2;
    }
  }
  const angle = toRadians(parameters.rotationDegrees);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const mapPosition = (position: CatalogPosition): CatalogPosition => {
    const x = (position[0] - pivotX) * parameters.scaleX;
    const y = (position[1] - pivotY) * parameters.scaleY;
    return [
      pivotX + x * cos - y * sin + parameters.translateX,
      pivotY + x * sin + y * cos + parameters.translateY
    ];
  };
  const table = makeCatalogTable(
    'transformed-features',
    'Transformed features',
    {source_index: 'int32', source_id: 'string', geometry: 'string'},
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  let empty = 0;
  features.features.forEach((feature, sourceIndex) => {
    if (!feature.geometry) {
      empty++;
      return;
    }
    appendCatalogRow(table, {
      ...makeSourceColumns(feature, sourceIndex),
      geometry: makeTransformedGeometryText(feature.geometry, mapPosition)
    });
  });
  if (empty)
    addCatalogDiagnostic(
      diagnostics,
      'empty-geometry',
      'warning',
      'Features without geometry were skipped.',
      empty
    );
  return {tables: [table], diagnostics, summary: {featureCount: table.rowCount}};
}

export function runVertexSnap(
  lines: CatalogFeatureCollection,
  parameters: {tolerance: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, collectPositions(lines));
  const anchors: {position: CatalogPosition; projected: [number, number]}[] = [];
  const cellSize = parameters.tolerance > 0 ? parameters.tolerance : 1;
  const cells = new Map<string, number[]>();
  const cellKey = (x: number, y: number) => `${x},${y}`;
  const findAnchor = (position: CatalogPosition): number => {
    const projected = project(position);
    const cellX = Math.floor(projected[0] / cellSize);
    const cellY = Math.floor(projected[1] / cellSize);
    let best = -1;
    let bestDistance = Infinity;
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (const anchorIndex of cells.get(cellKey(cellX + dx, cellY + dy)) ?? []) {
          const anchor = anchors[anchorIndex].projected;
          const distance = Math.hypot(anchor[0] - projected[0], anchor[1] - projected[1]);
          if (distance <= parameters.tolerance && distance < bestDistance) {
            best = anchorIndex;
            bestDistance = distance;
          }
        }
    if (best >= 0) return best;
    anchors.push({position, projected});
    const key = cellKey(cellX, cellY);
    cells.set(key, [...(cells.get(key) ?? []), anchors.length - 1]);
    return anchors.length - 1;
  };
  const table = makeCatalogTable(
    'snapped-lines',
    'Vertex-snapped lines',
    {
      line_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      vertices_moved: 'int32',
      vertices_removed: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  let collapsed = 0;
  let movedTotal = 0;
  lines.features.forEach((feature, sourceIndex) => {
    getLineParts(feature.geometry).forEach((part, partIndex) => {
      const snapped: CatalogPosition[] = [];
      let moved = 0;
      let lastAnchor = -1;
      for (const position of part) {
        const anchorIndex = findAnchor(position);
        const anchor = anchors[anchorIndex].position;
        if (anchor[0] !== position[0] || anchor[1] !== position[1]) moved++;
        if (anchorIndex === lastAnchor) continue;
        lastAnchor = anchorIndex;
        snapped.push(anchor);
      }
      if (snapped.length < 2) {
        collapsed++;
        return;
      }
      movedTotal += moved;
      appendCatalogRow(table, {
        line_id: `${sourceIndex}:${partIndex}`,
        ...makeSourceColumns(feature, sourceIndex),
        part_index: partIndex,
        vertices_moved: moved,
        vertices_removed: part.length - snapped.length,
        geometry: makeCatalogGeometryText('LineString', snapped)
      });
    });
  });
  const diagnostics: CatalogDiagnostic[] = [];
  if (collapsed)
    addCatalogDiagnostic(
      diagnostics,
      'part-collapsed',
      'warning',
      'Line parts that collapsed below two distinct vertices were dropped.',
      collapsed
    );
  return {
    tables: [table],
    diagnostics,
    summary: {lineCount: table.rowCount, anchorCount: anchors.length, verticesMoved: movedTotal}
  };
}

// ---------------------------------------------------------------------------------------------
// Construction: grid, random points
// ---------------------------------------------------------------------------------------------

export function runGridGenerator(
  extent: CatalogFeatureCollection,
  parameters: {cellWidth: number; cellHeight: number},
  maximumCells: number
): ExtraRunOutput {
  validateGeometryOutputCapacity('grid-generator', {extent}, parameters);
  const positions = collectPositions(extent);
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'grid-cells',
    'Grid cells',
    {cell_id: 'string', row: 'int32', column: 'int32', geometry: 'string'},
    'geometry'
  );
  if (!positions.length) {
    addCatalogDiagnostic(
      diagnostics,
      'empty-extent',
      'error',
      'The extent input has no coordinates.'
    );
    return {tables: [table], diagnostics, summary: {cellCount: 0}};
  }
  const minX = Math.min(...positions.map(position => position[0]));
  const maxX = Math.max(...positions.map(position => position[0]));
  const minY = Math.min(...positions.map(position => position[1]));
  const maxY = Math.max(...positions.map(position => position[1]));
  const columns = Math.max(1, Math.ceil((maxX - minX) / parameters.cellWidth - 1e-9));
  const rows = Math.max(1, Math.ceil((maxY - minY) / parameters.cellHeight - 1e-9));
  // Cell count would exceed the declared output capacity; stop before allocating rows.
  if (columns * rows > maximumCells)
    throw new KeplerAllocationError('capacity', {message: 'Output capacity exceeded'});
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const x0 = minX + column * parameters.cellWidth;
      const y0 = minY + row * parameters.cellHeight;
      const x1 = x0 + parameters.cellWidth;
      const y1 = y0 + parameters.cellHeight;
      appendCatalogRow(table, {
        cell_id: `${row}:${column}`,
        row,
        column,
        geometry: makeCatalogGeometryText('Polygon', [
          [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
            [x0, y0]
          ]
        ])
      });
    }
  return {tables: [table], diagnostics, summary: {cellCount: table.rowCount, rows, columns}};
}

export function runRandomPointsInPolygon(
  polygons: CatalogFeatureCollection,
  parameters: {pointsPerPolygon: number; seed: number}
): ExtraRunOutput {
  validateGeometryOutputCapacity('random-points-in-polygon', {polygons}, parameters);
  const table = makeCatalogTable(
    'random-points',
    'Random points',
    {
      point_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      sample_index: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  const identity = (position: CatalogPosition): [number, number] => [position[0], position[1]];
  const diagnostics: CatalogDiagnostic[] = [];
  let shortfall = 0;
  polygons.features.forEach((feature, sourceIndex) => {
    const parts = prepareProjectedGeometry(identity, feature.geometry).polygons;
    const positions = parts.flatMap(polygon => polygon.flatMap(ring => ring));
    if (!positions.length) return;
    const minX = Math.min(...positions.map(position => position[0]));
    const maxX = Math.max(...positions.map(position => position[0]));
    const minY = Math.min(...positions.map(position => position[1]));
    const maxY = Math.max(...positions.map(position => position[1]));
    const random = makeSeededRandom(makeFeatureSeed(parameters.seed, sourceIndex));
    let produced = 0;
    for (
      let attempt = 0;
      attempt < parameters.pointsPerPolygon * 200 && produced < parameters.pointsPerPolygon;
      attempt++
    ) {
      const x = minX + random() * (maxX - minX);
      const y = minY + random() * (maxY - minY);
      if (!parts.some(polygon => isInsideProjectedPolygon(x, y, polygon))) continue;
      appendCatalogRow(table, {
        point_id: `${sourceIndex}:${produced}`,
        ...makeSourceColumns(feature, sourceIndex),
        sample_index: produced,
        geometry: makeCatalogGeometryText('Point', [x, y])
      });
      produced++;
    }
    shortfall += parameters.pointsPerPolygon - produced;
  });
  if (shortfall)
    addCatalogDiagnostic(
      diagnostics,
      'sampling-shortfall',
      'warning',
      'Rejection sampling stopped after 200 attempts per requested point; fewer points were produced.',
      shortfall
    );
  return {
    tables: [table],
    diagnostics,
    summary: {pointCount: table.rowCount, seed: parameters.seed}
  };
}

export function runRandomPointsOnLine(
  lines: CatalogFeatureCollection,
  parameters: {pointsPerLine: number; seed: number; coordinateSystem: CatalogCoordinateSystem}
): ExtraRunOutput {
  validateGeometryOutputCapacity('random-points-on-line', {lines}, parameters);
  const table = makeCatalogTable(
    'random-points',
    'Random points',
    {
      point_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      sample_index: 'int32',
      distance_along: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  lines.features.forEach((feature, sourceIndex) => {
    const random = makeSeededRandom(makeFeatureSeed(parameters.seed, sourceIndex));
    const parts = getLineParts(feature.geometry)
      .map((part, partIndex) => ({
        part,
        partIndex,
        cumulative: makeCumulativeLengths(part, parameters.coordinateSystem)
      }))
      .filter(entry => entry.part.length >= 2);
    const total = parts.reduce(
      (sum, entry) => sum + entry.cumulative[entry.cumulative.length - 1],
      0
    );
    if (!total) return;
    for (let sample = 0; sample < parameters.pointsPerLine; sample++) {
      // One measure across all parts of the feature keeps points uniform in length.
      let remaining = random() * total;
      let chosen = parts[parts.length - 1];
      for (const entry of parts) {
        const length = entry.cumulative[entry.cumulative.length - 1];
        if (remaining <= length) {
          chosen = entry;
          break;
        }
        remaining -= length;
      }
      appendCatalogRow(table, {
        point_id: `${sourceIndex}:${sample}`,
        ...makeSourceColumns(feature, sourceIndex),
        part_index: chosen.partIndex,
        sample_index: sample,
        distance_along: remaining,
        geometry: makeCatalogGeometryText(
          'Point',
          getPositionAtMeasure(chosen.part, chosen.cumulative, remaining)
        )
      });
    }
  });
  return {
    tables: [table],
    diagnostics: [],
    summary: {pointCount: table.rowCount, seed: parameters.seed}
  };
}

// ---------------------------------------------------------------------------------------------
// K-means
// ---------------------------------------------------------------------------------------------

export function runKMeans(
  points: CatalogFeatureCollection,
  parameters: {
    clusterCount: number;
    maximumIterations: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): ExtraRunOutput {
  const project = makeProjector(parameters.coordinateSystem, collectPositions(points));
  const entries = points.features.flatMap((feature, pointIndex) =>
    getPointPositions(feature.geometry).map(position => ({
      feature,
      pointIndex,
      position,
      projected: project(position)
    }))
  );
  const diagnostics: CatalogDiagnostic[] = [];
  const assignmentTable = makeCatalogTable(
    'kmeans-assignments',
    'Cluster assignments',
    {
      point_index: 'int32',
      point_source_id: 'string',
      cluster: 'int32',
      distance: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const centroidTable = makeCatalogTable(
    'kmeans-centroids',
    'Cluster centroids',
    {cluster: 'int32', point_count: 'int32', geometry: 'string'},
    'geometry'
  );
  const clusterCount = Math.min(parameters.clusterCount, entries.length);
  if (clusterCount < parameters.clusterCount)
    addCatalogDiagnostic(
      diagnostics,
      'cluster-count-reduced',
      'warning',
      'Fewer points than requested clusters; the cluster count was reduced to the point count.'
    );
  // Deterministic farthest-first initialization starting from the first point.
  const centroids: [number, number][] = entries.length ? [[...entries[0].projected]] : [];
  while (centroids.length < clusterCount) {
    let farthest = 0;
    let farthestDistance = -1;
    entries.forEach((entry, index) => {
      const nearest = Math.min(
        ...centroids.map(centroid =>
          Math.hypot(entry.projected[0] - centroid[0], entry.projected[1] - centroid[1])
        )
      );
      if (nearest > farthestDistance) {
        farthestDistance = nearest;
        farthest = index;
      }
    });
    centroids.push([...entries[farthest].projected]);
  }
  const assignment = new Int32Array(entries.length).fill(-1);
  let iterations = 0;
  for (; iterations < parameters.maximumIterations; iterations++) {
    let changed = false;
    entries.forEach((entry, index) => {
      let best = 0;
      let bestDistance = Infinity;
      centroids.forEach((centroid, cluster) => {
        const distance = Math.hypot(
          entry.projected[0] - centroid[0],
          entry.projected[1] - centroid[1]
        );
        if (distance < bestDistance) {
          bestDistance = distance;
          best = cluster;
        }
      });
      if (assignment[index] !== best) {
        assignment[index] = best;
        changed = true;
      }
    });
    if (!changed) break;
    const sums = centroids.map(() => [0, 0, 0]);
    entries.forEach((entry, index) => {
      const sum = sums[assignment[index]];
      sum[0] += entry.projected[0];
      sum[1] += entry.projected[1];
      sum[2]++;
    });
    sums.forEach((sum, cluster) => {
      // An empty cluster keeps its previous centroid.
      if (sum[2]) centroids[cluster] = [sum[0] / sum[2], sum[1] / sum[2]];
    });
  }
  let inertia = 0;
  const counts = new Int32Array(clusterCount);
  const centroidPositions = centroids.map((): [number, number, number] => [0, 0, 0]);
  entries.forEach((entry, index) => {
    const centroid = centroids[assignment[index]];
    const distance = Math.hypot(entry.projected[0] - centroid[0], entry.projected[1] - centroid[1]);
    inertia += distance * distance;
    counts[assignment[index]]++;
    const sum = centroidPositions[assignment[index]];
    sum[0] += entry.position[0];
    sum[1] += entry.position[1];
    sum[2]++;
    appendCatalogRow(assignmentTable, {
      point_index: entry.pointIndex,
      point_source_id: getFeatureIdentity(entry.feature),
      cluster: assignment[index],
      distance,
      geometry: makeCatalogGeometryText('Point', entry.position)
    });
  });
  centroidPositions.forEach((sum, cluster) =>
    appendCatalogRow(centroidTable, {
      cluster,
      point_count: counts[cluster],
      // Mean of member coordinates so the output stays in source coordinates.
      geometry: sum[2] ? makeCatalogGeometryText('Point', [sum[0] / sum[2], sum[1] / sum[2]]) : null
    })
  );
  return {
    tables: [assignmentTable, centroidTable],
    diagnostics,
    summary: {clusterCount, iterations, inertia}
  };
}

/** Validate projected output counts before row generation or GPU allocation. */
export function validateGeometryOutputCapacity(
  bindingId: string,
  inputs: Readonly<Record<string, CatalogFeatureCollection>>,
  parameters: Readonly<Record<string, unknown>>
): void {
  let count = 0;
  let maximum = 20000;
  let parameter = '';
  let remedy = '';
  if (bindingId === 'grid-generator') {
    const positions = collectPositions(inputs['extent']);
    if (!positions.length) return;
    let minimumX = Infinity;
    let maximumX = -Infinity;
    let minimumY = Infinity;
    let maximumY = -Infinity;
    for (const [x, y] of positions) {
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
    }
    count =
      Math.max(1, Math.ceil((maximumX - minimumX) / (parameters['cellWidth'] as number) - 1e-9)) *
      Math.max(1, Math.ceil((maximumY - minimumY) / (parameters['cellHeight'] as number) - 1e-9));
    parameter = 'cellWidth/cellHeight';
    remedy = 'increase cellWidth or cellHeight';
  } else if (bindingId === 'random-points-in-polygon') {
    count =
      inputs['polygons'].features.filter(
        feature => getPolygonParts(feature.geometry).flat(2).length > 0
      ).length * (parameters['pointsPerPolygon'] as number);
    parameter = 'pointsPerPolygon';
    remedy = 'reduce pointsPerPolygon or partition the input';
  } else if (bindingId === 'random-points-on-line') {
    count =
      inputs['lines'].features.filter(feature =>
        getLineParts(feature.geometry).some(
          part =>
            part.length >= 2 &&
            measurePathLength(part, parameters['coordinateSystem'] as CatalogCoordinateSystem) > 0
        )
      ).length * (parameters['pointsPerLine'] as number);
    parameter = 'pointsPerLine';
    remedy = 'reduce pointsPerLine or partition the input';
  } else if (bindingId === 'line-chunk' || bindingId === 'geometry-line-segmentize') {
    const coordinateSystem = parameters['coordinateSystem'] as CatalogCoordinateSystem;
    const segmentize = bindingId === 'geometry-line-segmentize';
    maximum = segmentize ? 500000 : 50000;
    parameter = segmentize ? 'maximumSegmentLength' : 'maximumChunkLength';
    const lengthLimit = parameters[parameter] as number;
    for (const feature of inputs['lines'].features)
      for (const part of getLineParts(feature.geometry)) {
        if (!part.length) continue;
        if (segmentize) {
          count++;
          for (let index = 1; index < part.length; index++)
            count += Math.max(
              1,
              Math.ceil(
                measureDistance(part[index - 1], part[index], coordinateSystem) / lengthLimit
              )
            );
        } else if (part.length >= 2) {
          count += Math.max(
            1,
            Math.ceil(measurePathLength(part, coordinateSystem) / lengthLimit - 1e-9)
          );
        }
      }
    remedy = `increase ${parameter} or partition the input`;
  } else return;
  if (count > maximum)
    throw new Error(
      `${bindingId}: ${parameter} projects ${count} output ${bindingId === 'geometry-line-segmentize' ? 'vertices' : 'rows'}, exceeding the ${maximum.toLocaleString('en-US')} limit; ${remedy}.`
    );
}
