// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {Feature, Geometry} from 'geojson';
import {
  materializeKeplerSelection,
  type KeplerComputeSession,
  type KeplerSessionExplorer
} from '@luma-contributors/kepler';
import type {KeplerComputeFilter} from './kepler-table-bridge';
import {
  makeConsumerState,
  makeReadyState,
  type ResidentConsumerState
} from './resident-consumer-state';
import {
  measureDrawnGeometry,
  validateDrawnPolygon,
  type DrawingIssue
} from './resident-drawing-geometry';
import {materializeResidentRows} from './resident-materialization';

/** A publishable source: retained Arrow table plus the explicit geometry role, no hidden state. */
export type DrawingPublishedSource = {
  id: string;
  label: string;
  source: arrow.Table;
  normalization: Record<string, never>;
  geometryColumn: string;
  isCurrent(): boolean;
};

export type DrawingToLayerResult =
  | {status: 'ready'; source: DrawingPublishedSource; featureCount: number}
  | {status: 'invalid'; rejected: {featureId: string; reasons: readonly DrawingIssue[]}[]}
  | {status: 'empty'; reason: string};

/**
 * Drawing-to-layer: every drawn feature becomes one row of a new dataset with its GeoJSON geometry
 * and measurements. Polygonal features are validated first; lines and points are measured only.
 * Nothing here touches resident datasets, so the result is an ordinary source to publish.
 */
export function makeDrawingLayerSource(
  features: readonly Feature<Geometry>[],
  props: {id: string; label: string; isCurrent?: () => boolean}
): DrawingToLayerResult {
  const rejected: {featureId: string; reasons: readonly DrawingIssue[]}[] = [];
  const rows = features.filter(feature => feature.geometry);
  if (!rows.length) return {status: 'empty', reason: 'Draw at least one shape first.'};
  for (const [index, feature] of rows.entries()) {
    if (feature.geometry.type !== 'Polygon' && feature.geometry.type !== 'MultiPolygon') continue;
    const validation = validateDrawnPolygon(feature);
    if (validation.status === 'invalid')
      rejected.push({featureId: String(feature.id ?? index), reasons: validation.reasons});
  }
  if (rejected.length) return {status: 'invalid', rejected};
  const measurements = rows.map(feature => measureDrawnGeometry(feature.geometry));
  const table = new arrow.Table({
    feature_id: arrow.vectorFromArray(
      rows.map((feature, index) => String(feature.id ?? `drawing-${index}`)),
      new arrow.Utf8()
    ),
    shape: arrow.vectorFromArray(
      rows.map(feature => {
        const shape = (feature.properties as {shape?: unknown} | null)?.shape;
        return typeof shape === 'string' ? shape : feature.geometry.type;
      }),
      new arrow.Utf8()
    ),
    length_meters: arrow.vectorFromArray(
      Float32Array.from(measurements, measurement => measurement.lengthMeters)
    ),
    area_square_meters: arrow.vectorFromArray(
      Float32Array.from(measurements, measurement => measurement.areaSquareMeters)
    ),
    geometry: arrow.vectorFromArray(
      rows.map(feature => JSON.stringify(feature.geometry)),
      new arrow.Utf8()
    )
  });
  return {
    status: 'ready',
    featureCount: rows.length,
    source: {
      id: props.id,
      label: props.label,
      source: table,
      normalization: {},
      geometryColumn: 'geometry',
      isCurrent: props.isCurrent ?? (() => true)
    }
  };
}

export type DrawingFilterResult =
  | {status: 'ready'; filter: KeplerComputeFilter; warnings: readonly DrawingIssue[]}
  | {status: 'invalid'; reasons: readonly DrawingIssue[]};

/**
 * Drawing-as-filter: the native polygon FeatureValue consumed by the per-layer mask planner.
 * Invalid shapes are refused with reasons instead of producing a filter that selects nothing.
 */
export function makeDrawingPolygonFilter(
  feature: Feature<Geometry>,
  props: {id: string; dataIds: readonly string[]; layerIds: readonly string[]; inverted?: boolean}
): DrawingFilterResult {
  const validation = validateDrawnPolygon(feature);
  if (validation.status === 'invalid') return {status: 'invalid', reasons: validation.reasons};
  return {
    status: 'ready',
    warnings: validation.warnings,
    filter: {
      id: props.id,
      dataId: [...props.dataIds],
      layerId: [...props.layerIds],
      name: [],
      type: 'polygon',
      value: {...feature, id: feature.id ?? props.id},
      enabled: true,
      inverted: Boolean(props.inverted)
    }
  };
}

export type ExtractedRows = {
  table: arrow.Table;
  datasetId: string;
  datasetRevision: number;
  selectionRevision: number;
  selectedCount: number;
  /** Ascending original source-row identities, aligned with the table rows. */
  sourceRowIndices: Uint32Array;
  sourceRowColumn: string;
};

export type RowExtractionOptions = {
  /** Selections larger than this are refused; extraction never truncates. */
  maximumRows: number;
  /** Bound on retained Arrow bytes for the extracted rows. */
  maximumOutputBytes: number;
  maximumReadbackBytes?: number;
  signal?: AbortSignal;
};

/**
 * Explicit bounded row extraction: the rows currently selected on the GPU (all filters, polygon
 * masks and brushes already applied) are materialized from the retained original source and returned
 * as a compact table. It refuses instead of truncating, reports a distinct state when the selection
 * is not yet submitted or empty, and never publishes if the dataset or selection changed meanwhile.
 */
export async function extractSelectedRows(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  options: RowExtractionOptions
): Promise<ResidentConsumerState<ExtractedRows>> {
  const dataset = explorer.dataset;
  const revision = explorer.currentRevision;
  const key = `${dataset.id}:${dataset.revision}:${revision}`;
  try {
    const count = await explorer.readSelectedCount();
    if (!count || count.revision !== revision)
      return makeConsumerState(
        'pending',
        'extraction',
        'The current selection is not submitted yet; try again after the map updates.',
        key
      );
    if (count.selectedCount === 0)
      return makeConsumerState('empty', 'extraction', 'No rows are selected.', key);
    if (count.selectedCount > options.maximumRows)
      return makeConsumerState(
        'unsupported',
        'extraction',
        `${count.selectedCount} rows are selected; extraction is limited to ${options.maximumRows}. Narrow the selection first.`,
        key
      );
    // Unmaterialized resident sources hold few or no CPU bytes, so the measured average alone
    // under-reports; the schema-width floor keeps the limit honest for them.
    const bytesPerRow = Math.max(
      estimateArrowTableBytes(dataset.value.source) / Math.max(1, dataset.value.source.numRows),
      estimateArrowSchemaRowBytes(dataset.value.source.schema)
    );
    const estimatedBytes = Math.ceil(bytesPerRow * count.selectedCount);
    if (estimatedBytes > options.maximumOutputBytes)
      return makeConsumerState(
        'unsupported',
        'extraction',
        `The selected rows need about ${estimatedBytes} bytes; the limit is ${options.maximumOutputBytes}.`,
        key
      );
    const indices = await materializeKeplerSelection(session, explorer, {
      mode: 'all-selected',
      maximumRows: options.maximumRows,
      columns: [],
      maximumReadbackBytes: options.maximumReadbackBytes ?? 64 * 1024 * 1024,
      maximumCpuBytes: options.maximumOutputBytes,
      signal: options.signal
    });
    const assertCurrent = () => {
      options.signal?.throwIfAborted();
      if (
        !explorer.isCurrent ||
        explorer.currentRevision !== revision ||
        session.registry.getDataset(dataset.id) !== dataset
      )
        throw new Error('The dataset or selection changed during extraction.');
    };
    try {
      await materializeResidentRows(dataset.value, indices.sourceRowIndices, {assertCurrent});
      assertCurrent();
      const sourceRowIndices = Uint32Array.from(indices.sourceRowIndices);
      const extracted = buildExtractedTable(dataset.value.source, sourceRowIndices, {
        datasetId: indices.datasetId,
        datasetRevision: indices.datasetRevision,
        selectionRevision: indices.selectionRevision,
        selectedCount: indices.selectedCount
      });
      return makeReadyState('extraction', key, {
        ...extracted,
        datasetId: indices.datasetId,
        datasetRevision: indices.datasetRevision,
        selectionRevision: indices.selectionRevision,
        selectedCount: indices.selectedCount,
        sourceRowIndices
      });
    } finally {
      indices.destroy();
    }
  } catch (failure) {
    const message = failure instanceof Error ? failure.message : String(failure);
    return makeConsumerState(
      /changed during extraction|stale/i.test(message) ? 'stale' : 'error',
      'extraction',
      message,
      key
    );
  }
}

/** Slices contiguous source runs (shared storage) and appends the original row identity. */
export function buildExtractedTable(
  source: arrow.Table,
  sourceRows: Uint32Array,
  metadata: Record<string, unknown>
): {table: arrow.Table; sourceRowColumn: string} {
  const usedNames = new Set(source.schema.fields.map(field => field.name));
  let sourceRowColumn = '__luma_source_row_index';
  while (usedNames.has(sourceRowColumn)) sourceRowColumn += '_';
  const fields = [
    ...source.schema.fields,
    new arrow.Field(sourceRowColumn, new arrow.Uint32(), false)
  ];
  const schemaMetadata = new Map(source.schema.metadata);
  schemaMetadata.set('luma:row-extraction', JSON.stringify({...metadata, sourceRowColumn}));
  const schema = new arrow.Schema(fields, schemaMetadata);
  const batches: arrow.RecordBatch[] = [];
  for (let index = 0; index < sourceRows.length; ) {
    if (index > 0 && sourceRows[index] <= sourceRows[index - 1])
      throw new Error('Extraction requires increasing unique source-row identities');
    let end = index + 1;
    while (end < sourceRows.length && sourceRows[end] === sourceRows[end - 1] + 1) end++;
    let cursor = index;
    for (const batch of source.slice(sourceRows[index], sourceRows[end - 1] + 1).batches) {
      const identities = arrow.makeVector(
        Uint32Array.from(sourceRows.subarray(cursor, cursor + batch.numRows))
      );
      const data = batch.data.clone(
        new arrow.Struct(fields),
        batch.data.offset,
        batch.numRows,
        batch.data.nullCount,
        batch.data.buffers,
        [...batch.data.children, identities.data[0]]
      );
      batches.push(new arrow.RecordBatch(schema, data));
      cursor += batch.numRows;
    }
    index = end;
  }
  return {table: new arrow.Table(schema, batches), sourceRowColumn};
}

/**
 * Publication of extracted rows as a new source. The rows are an explicit snapshot, so the
 * source stays current while its parent selection changes, and ends only with the parent dataset.
 */
export function makeExtractedPublishedSource(
  extracted: ExtractedRows,
  props: {
    id: string;
    label: string;
    geometryColumn?: string;
    isParentCurrent(): boolean;
  }
): DrawingPublishedSource & {parent: {datasetId: string; datasetRevision: number}} {
  return {
    id: props.id,
    label: props.label,
    source: extracted.table,
    normalization: {},
    geometryColumn: props.geometryColumn ?? '',
    isCurrent: props.isParentCurrent,
    parent: {datasetId: extracted.datasetId, datasetRevision: extracted.datasetRevision}
  };
}

/** Lower bound of bytes per row from the schema: fixed widths, 24 bytes for variable-length values. */
export function estimateArrowSchemaRowBytes(schema: arrow.Schema): number {
  return schema.fields.reduce((bytes, field) => {
    const type = field.type as {bitWidth?: number; precision?: number};
    // Arrow floats carry a precision (0 half, 1 single, 2 double) instead of a bit width.
    const bitWidth = type.bitWidth ?? (type.precision === undefined ? 0 : 16 << type.precision);
    return bytes + (bitWidth > 0 ? bitWidth / 8 : 24);
  }, 0);
}

function estimateArrowTableBytes(table: arrow.Table): number {
  const visit = (data: arrow.Data): number =>
    (data.values?.byteLength ?? 0) +
    (data.valueOffsets?.byteLength ?? 0) +
    (data.nullBitmap?.byteLength ?? 0) +
    data.children.reduce((bytes, child) => bytes + visit(child), 0);
  return table.batches.reduce(
    (bytes, batch) => bytes + batch.data.children.reduce((sum, child) => sum + visit(child), 0),
    0
  );
}
