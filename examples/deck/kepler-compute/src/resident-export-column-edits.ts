// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import type {ResidentColumnEdits} from './resident-column-edits';
import {exportSelectedSource, type SourceExportFormat} from './source-export';
import {formatCsvHeader, formatTableCsvRows} from './native-source-export';
import type {StreamingPageSource} from './streaming-export';

/** Edits that change exported headers or columns; formats and display types never do. */
export function hasResidentExportColumnEdits(edits: ResidentColumnEdits): boolean {
  return edits.deleted.length > 0 || Object.keys(edits.names).length > 0;
}

/**
 * Applies deleted columns (omitted) and renamed columns (alias as header) to an exported Arrow
 * table. Derived/metadata columns that the edits do not name pass through unchanged.
 */
export function applyColumnEditsToArrowTable(
  table: arrow.Table,
  edits: ResidentColumnEdits
): arrow.Table {
  if (!hasResidentExportColumnEdits(edits)) return table;
  const columns: Record<string, arrow.Vector> = Object.create(null);
  for (const [index, field] of table.schema.fields.entries()) {
    if (edits.deleted.includes(field.name)) continue;
    const header = edits.names[field.name] ?? field.name;
    // A rename onto an existing exported header keeps the source column under its own name.
    columns[header in columns ? field.name : header] = table.getChildAt(index)!;
  }
  return new arrow.Table(columns);
}

/** Streaming pages share one edited schema because every page is edited the same way. */
export function wrapPageSourceWithColumnEdits(
  source: StreamingPageSource,
  getEdits: () => ResidentColumnEdits
): StreamingPageSource {
  return {
    getTotalRows: () => source.getTotalRows(),
    readPage: async (offset, limit, signal) =>
      applyColumnEditsToArrowTable(await source.readPage(offset, limit, signal), getEdits())
  };
}

/**
 * Bounded source export with column edits. CSV is produced from the edited Arrow table so deleted
 * columns and aliased headers are identical in both formats; the source export remains the owner of
 * limits, staleness checks and memory reservations.
 */
export async function exportSelectedSourceWithColumnEdits(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  options: Parameters<typeof exportSelectedSource>[2],
  edits: ResidentColumnEdits
): ReturnType<typeof exportSelectedSource> {
  if (!hasResidentExportColumnEdits(edits)) return exportSelectedSource(session, explorer, options);
  const requestedFormat: SourceExportFormat = options.format;
  const result = await exportSelectedSource(session, explorer, {...options, format: 'arrow'});
  const edited = applyColumnEditsToArrowTable(arrow.tableFromIPC(result.bytes), edits);
  const bytes =
    requestedFormat === 'arrow'
      ? (arrow.tableToIPC(edited, 'stream') as Uint8Array<ArrayBuffer>)
      : (new TextEncoder().encode(
          `${formatCsvHeader(edited.schema)}${formatTableCsvRows(edited)}`
        ) as Uint8Array<ArrayBuffer>);
  if (bytes.byteLength > options.maximumOutputBytes) {
    result.destroy();
    // Output budget exceeded after applying column edits.
    throw new Error('Edited export exceeds the output byte limit');
  }
  return {...result, bytes, format: requestedFormat};
}
