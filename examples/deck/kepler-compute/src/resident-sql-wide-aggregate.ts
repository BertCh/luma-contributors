// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerArrowDataset, KeplerComputeSession} from '@luma-contributors/kepler';
import {
  executeWideSqlAggregate,
  listWideIntegerColumns,
  parseWideSqlAggregate,
  type WideSqlAggregatePlan
} from './wide-numeric-gpu-sql';

/** A planned exact Int64/Uint64 aggregate: the source table is the original (unnarrowed) Arrow table. */
export type ResidentSqlWideAggregate = {
  readonly plan: WideSqlAggregatePlan;
  readonly datasetId: string;
  readonly revision: number;
};

/**
 * Recognizes global COUNT/SUM/MIN/MAX statements (optionally with an exact integer range WHERE on
 * the aggregated column) over Int64 or Uint64 source columns. The resident copy of such a column is
 * narrowed to Float32, so these run on the exact wide-word kernels instead; anything else returns null.
 */
export function planResidentSqlWideAggregate(
  session: KeplerComputeSession,
  sql: string,
  findDatasetId: (tableName: string) => string | undefined
): ResidentSqlWideAggregate | null {
  let datasetId: string | undefined;
  const plan = parseWideSqlAggregate(sql, tableName => {
    datasetId = findDatasetId(tableName.toLowerCase());
    const snapshot = datasetId ? session.registry.getDataset<KeplerArrowDataset>(datasetId) : null;
    // Lazily paged GPU results have null cells in `source`; only complete tables are scanned.
    return snapshot && !snapshot.value.resident
      ? listWideIntegerColumns(snapshot.value.source)
      : undefined;
  });
  const snapshot = plan && datasetId ? session.registry.getDataset(datasetId) : null;
  return plan && datasetId && snapshot ? {plan, datasetId, revision: snapshot.revision} : null;
}

export async function executeResidentSqlWideAggregate(
  session: KeplerComputeSession,
  planned: ResidentSqlWideAggregate,
  resultId: string
) {
  const snapshot = session.registry.getDataset<KeplerArrowDataset>(planned.datasetId);
  if (!snapshot || snapshot.revision !== planned.revision)
    throw new Error('SQL sources changed during execution');
  const table = await executeWideSqlAggregate(session.device, planned.plan, snapshot.value.source);
  const dataset = session.importArrowDataset({id: resultId, table});
  return {
    dataset,
    rowCount: table.numRows,
    materialization: 'arrow' as const,
    fallbackReason: 'exact Int64/Uint64 GPU aggregate (values beyond Int32 are published as text)'
  };
}
