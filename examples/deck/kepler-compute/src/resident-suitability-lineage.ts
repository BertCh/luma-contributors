// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_RESIDENT_SUITABILITY_COLUMNS,
  MAXIMUM_RESIDENT_SUITABILITY_INDICATORS,
  type ResidentSuitabilityProvenance
} from './resident-suitability-plan';

/** Saved lineage of a resident Suitability result; the result table itself is the data. */
export type ResidentSuitabilityLineage = ResidentSuitabilityProvenance;

/** Validate and detach saved lineage. The restored table is a snapshot; the parent may be absent. */
export function parseResidentSuitabilityLineage(value: unknown): ResidentSuitabilityLineage {
  const record = readRecord(value);
  const columns = record['columns'];
  const fields = record['fields'];
  const indicators = record['indicators'];
  const weightStandardization = record['weightStandardization'];
  const dataStandardization = record['dataStandardization'];
  if (
    record['schemaVersion'] !== 1 ||
    record['kind'] !== 'suitability' ||
    (record['scorePolicy'] !== 'float32-composite-score-aligned-missing-values-null' &&
      record['scorePolicy'] !== 'float32-composite-score-renormalized-defined-weights') ||
    (record['missingValues'] !== undefined &&
      record['missingValues'] !== 'exclude' &&
      record['missingValues'] !== 'renormalize') ||
    (record['scorePolicy'] === 'float32-composite-score-renormalized-defined-weights') !==
      (record['missingValues'] === 'renormalize') ||
    record['rowOrder'] !== 'source-order' ||
    (weightStandardization !== 'normalize' && weightStandardization !== 'raw') ||
    (dataStandardization !== 'range' &&
      dataStandardization !== 'zScore' &&
      dataStandardization !== 'raw')
  )
    reject('Unsupported saved Suitability policy.');
  if (
    !Array.isArray(columns) ||
    !Array.isArray(fields) ||
    !Array.isArray(indicators) ||
    indicators.length < 1 ||
    indicators.length > MAXIMUM_RESIDENT_SUITABILITY_INDICATORS ||
    columns.length > MAXIMUM_RESIDENT_SUITABILITY_COLUMNS - 1 ||
    fields.length !== columns.length + 1
  )
    reject('Invalid saved Suitability columns.');
  const names = new Set<string>();
  const parsedFields = fields.map((field, index) => {
    const parsed = readRecord(field);
    const name = readText(parsed['name']);
    if (names.has(name) || (index < columns.length && name !== readRecord(columns[index])['name']))
      reject('Saved Suitability output schema is inconsistent.');
    names.add(name);
    return {
      name,
      type: readText(parsed['type']),
      analyzerType: readText(parsed['analyzerType'])
    };
  });
  const scoreColumn = readText(record['scoreColumn']);
  if (parsedFields[parsedFields.length - 1].name !== scoreColumn)
    reject('Saved Suitability score column is inconsistent.');
  const source = readRecord(record['source']);
  const revision = source['revision'];
  if (typeof revision !== 'number' || !Number.isSafeInteger(revision) || revision < 0)
    reject('Invalid saved Suitability source.');
  return {
    schemaVersion: 1,
    kind: 'suitability',
    source: {datasetId: readText(source['datasetId']), revision},
    operationId: readText(record['operationId']),
    resultId: readText(record['resultId']),
    resultLabel: readText(record['resultLabel']),
    indicators: indicators.map(indicator => {
      const parsed = readRecord(indicator);
      const weight = parsed['weight'];
      if (typeof weight !== 'number' || !Number.isFinite(weight) || weight < 0 || weight > 1)
        reject('Invalid saved Suitability weight.');
      return {column: readText(parsed['column']), weight};
    }),
    weightStandardization,
    dataStandardization,
    scoreColumn,
    columns: columns.map(column => {
      const parsed = readRecord(column);
      return {column: readText(parsed['column']), name: readText(parsed['name'])};
    }),
    fields: parsedFields,
    scorePolicy: record['scorePolicy'],
    ...(record['missingValues'] ? {missingValues: record['missingValues']} : {}),
    rowOrder: 'source-order',
    ...(Array.isArray(record['reclassifications'])
      ? {
          reclassifications: record['reclassifications'].map(entry => {
            const parsed = readRecord(entry);
            const scores = parsed['scores'];
            if (
              !Array.isArray(scores) ||
              scores.some(score => score !== null && !Number.isFinite(score))
            )
              reject('Invalid saved Suitability reclassification.');
            return {column: readText(parsed['column']), scores: scores as (number | null)[]};
          })
        }
      : {})
  };
}

/** Native `derivedDataset` metadata for the pinned Kepler dataset-ops lineage. */
export function getResidentSuitabilityDerivedDataset(lineage: ResidentSuitabilityLineage) {
  return {
    type: 'suitability',
    sourceDataIds: [lineage.source.datasetId],
    operationId: lineage.operationId
  };
}

function readRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    reject('Invalid saved Suitability metadata.');
  return value as Record<string, unknown>;
}
function readText(value: unknown): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096)
    reject('Invalid saved Suitability text.');
  return value;
}
function reject(message: string): never {
  throw new Error(message);
}
