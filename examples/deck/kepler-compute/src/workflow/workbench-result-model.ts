// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getWorkflowColumnNames,
  type WorkflowColumnSummary,
  type WorkflowData,
  type WorkflowDisplayStyle
} from '@luma-contributors/kepler';

const MAXIMUM_CELL_CHARACTERS = 120;

export function formatWorkbenchCell(value: unknown, type: string): string {
  if (value === null || value === undefined) return 'null';
  if (type === 'geometry') return describeWorkbenchGeometry(String(value));
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return 'NaN';
    if (!Number.isFinite(value)) return value < 0 ? '−∞' : '∞';
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, {maximumSignificantDigits: 4});
  }
  const text = String(value);
  return text.length > MAXIMUM_CELL_CHARACTERS
    ? `${text.slice(0, MAXIMUM_CELL_CHARACTERS - 1)}…`
    : text;
}

function isPosition(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    value.every(coordinate => typeof coordinate === 'number' && Number.isFinite(coordinate))
  );
}

function isLine(value: unknown): value is number[][] {
  return Array.isArray(value) && value.length >= 2 && value.every(isPosition);
}

function isRing(value: unknown): value is number[][] {
  if (!isLine(value) || value.length < 4) return false;
  const first = value[0];
  const last = value[value.length - 1];
  return (
    first.length === last.length && first.every((coordinate, index) => coordinate === last[index])
  );
}

function isPolygon(value: unknown): value is number[][][] {
  return Array.isArray(value) && value.length > 0 && value.every(isRing);
}

function describeGeometryPart(geometry: Record<string, unknown>): string | null {
  const {type, coordinates} = geometry;
  switch (type) {
    case 'Point':
      return isPosition(coordinates) ? 'Point' : null;
    case 'LineString':
    case 'MultiPoint': {
      const valid =
        type === 'LineString'
          ? isLine(coordinates)
          : Array.isArray(coordinates) && coordinates.length > 0 && coordinates.every(isPosition);
      if (!valid || !Array.isArray(coordinates)) return null;
      return `${type} · ${coordinates.length.toLocaleString()} ${coordinates.length === 1 ? 'vertex' : 'vertices'}`;
    }
    case 'Polygon': {
      if (!isPolygon(coordinates)) return null;
      const vertices = coordinates.reduce((total, ring) => total + ring.length, 0);
      return `Polygon · ${vertices.toLocaleString()} vertices`;
    }
    case 'MultiLineString':
    case 'MultiPolygon': {
      if (
        !Array.isArray(coordinates) ||
        coordinates.length === 0 ||
        !coordinates.every(type === 'MultiLineString' ? isLine : isPolygon)
      )
        return null;
      return `${type} · ${coordinates.length.toLocaleString()} ${coordinates.length === 1 ? 'part' : 'parts'}`;
    }
    default:
      return null;
  }
}

// Collection traversal is iterative and bounded independently of untrusted JSON nesting.
const MAXIMUM_GEOMETRY_COLLECTION_DEPTH = 32;
const MAXIMUM_GEOMETRY_COLLECTION_MEMBERS = 10000;

/** Called for visible geometry cells only; malformed values remain readable in the preview. */
export function describeWorkbenchGeometry(text: string): string {
  let geometry: unknown;
  try {
    geometry = JSON.parse(text);
  } catch {
    return 'Invalid geometry';
  }
  const pending: Array<{value: unknown; depth: number}> = [{value: geometry, depth: 0}];
  let description: string | null = null;
  let visited = 0;
  let scheduled = 1;
  while (pending.length) {
    const {value, depth} = pending.pop()!;
    if (
      !value ||
      typeof value !== 'object' ||
      Array.isArray(value) ||
      depth > MAXIMUM_GEOMETRY_COLLECTION_DEPTH ||
      ++visited > MAXIMUM_GEOMETRY_COLLECTION_MEMBERS
    ) {
      return 'Invalid geometry';
    }
    const member = value as Record<string, unknown>;
    const {type, geometries} = member;
    if (typeof type !== 'string') return 'Invalid geometry';
    if (type === 'GeometryCollection') {
      if (!Array.isArray(geometries)) return 'Invalid geometry';
      scheduled += geometries.length;
      if (scheduled > MAXIMUM_GEOMETRY_COLLECTION_MEMBERS) return 'Invalid geometry';
      description ??= `GeometryCollection · ${geometries.length.toLocaleString()} ${geometries.length === 1 ? 'part' : 'parts'}`;
      for (const child of geometries) pending.push({value: child, depth: depth + 1});
    } else {
      const partDescription = describeGeometryPart(member);
      if (partDescription === null) return 'Invalid geometry';
      description ??= partDescription;
    }
  }
  return description ?? 'Invalid geometry';
}

export type WorkbenchPreviewPage = {
  columns: string[];
  rows: Array<Array<string | number | boolean | null>>;
  pageIndex: number;
  pageCount: number;
  rowCount: number;
  startRow: number;
  endRow: number;
};

/** Page indexes are zero based; displayed row bounds are one based (zero for an empty table). */
export function getWorkbenchPreviewPage(
  data: WorkflowData,
  pageIndex: number,
  pageSize = 50
): WorkbenchPreviewPage {
  const size = Number.isFinite(pageSize) ? Math.max(1, Math.min(50, Math.floor(pageSize))) : 50;
  const rowCount = data.table.rowCount;
  const pageCount = Math.ceil(rowCount / size);
  const requestedPage = Number.isFinite(pageIndex) ? Math.floor(pageIndex) : 0;
  const currentPage = Math.max(0, Math.min(Math.max(0, pageCount - 1), requestedPage));
  const startIndex = currentPage * size;
  const endIndex = Math.min(rowCount, startIndex + size);
  const columns = getWorkflowColumnNames(data);
  if (data.identityColumn !== undefined && columns.includes(data.identityColumn)) {
    columns.splice(columns.indexOf(data.identityColumn), 1);
    columns.unshift(data.identityColumn);
  }
  if (
    data.table.geometryColumn !== undefined &&
    Object.hasOwn(data.table.columns, data.table.geometryColumn)
  ) {
    columns.push(data.table.geometryColumn);
  }
  return {
    columns,
    rows: Array.from({length: endIndex - startIndex}, (_, index) =>
      columns.map(name => data.table.columns[name][startIndex + index] ?? null)
    ),
    pageIndex: currentPage,
    pageCount,
    rowCount,
    startRow: rowCount ? startIndex + 1 : 0,
    endRow: endIndex
  };
}

export type WorkbenchHistogram = {edges: number[]; counts: number[]};

/** Equal-width, left-closed bins; the final bin includes the maximum. */
export function makeWorkbenchHistogram(
  values: readonly unknown[],
  binCount = 12
): WorkbenchHistogram {
  const count = Number.isFinite(binCount) ? Math.max(1, Math.floor(binCount)) : 12;
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  if (minimum === Infinity) return {edges: [], counts: []};
  const counts = new Array<number>(minimum === maximum ? 1 : count).fill(0);
  // Scaling avoids overflow when the finite domain crosses from -MAX_VALUE to MAX_VALUE.
  const scale = Math.max(Math.abs(minimum), Math.abs(maximum)) || 1;
  const scaledMinimum = minimum / scale;
  const scaledRange = maximum / scale - scaledMinimum;
  const edges =
    minimum === maximum
      ? [minimum, maximum]
      : Array.from({length: count + 1}, (_, index) =>
          index === 0
            ? minimum
            : index === count
              ? maximum
              : minimum * (1 - index / count) + maximum * (index / count)
        );
  for (const value of values) {
    if (typeof value !== 'number' || !Number.isFinite(value)) continue;
    let index =
      minimum === maximum
        ? 0
        : Math.max(
            0,
            Math.min(count - 1, Math.floor(((value / scale - scaledMinimum) / scaledRange) * count))
          );
    // Floating-point scaling may estimate the previous bin at an exact edge.
    // The returned edges define membership, including when narrow bins share an edge.
    while (index > 0 && value < edges[index]) index--;
    while (index < counts.length - 1 && value >= edges[index + 1]) index++;
    counts[index]++;
  }
  return {edges, counts};
}

export function formatWorkbenchDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return '0 ms';
  return milliseconds < 1000
    ? `${Math.round(milliseconds).toLocaleString()} ms`
    : `${(milliseconds / 1000).toLocaleString(undefined, {maximumFractionDigits: 1})} s`;
}

export function formatWorkbenchSummaryKey(key: string): string {
  const words = key
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([a-z\d])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim();
  return words ? words[0].toUpperCase() + words.slice(1).toLowerCase() : '';
}

export function makeWorkbenchDownloadName(
  workflowId: string,
  stepId: string,
  output: string,
  extension: string
): string {
  const safePart = (value: string) =>
    value.replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'output';
  return `${safePart(workflowId)}-${safePart(stepId)}-${safePart(output)}.${safePart(extension)}`;
}

/** Leave unspecified colour choices to the host, replacing any previous column's choices. */
export function getWorkbenchColumnDisplayStyle(
  summary: WorkflowColumnSummary
): WorkflowDisplayStyle {
  if (summary.type === 'string' || summary.type === 'boolean') {
    return {colorColumn: summary.name, colorScale: 'ordinal'};
  }
  return {
    colorColumn: summary.name,
    ...(summary.minimum !== undefined &&
    summary.maximum !== undefined &&
    summary.minimum < 0 &&
    summary.maximum > 0
      ? {palette: 'diverging' as const}
      : {})
  };
}
