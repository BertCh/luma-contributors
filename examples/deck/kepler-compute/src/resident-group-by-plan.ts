// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerGroupBySettings} from '@luma-contributors/kepler';

// Admission bounds for the whole operation. The GPU runtime splits work larger than one graph
// pass (4096+ dictionary slots, 16 measures) into key windows and measure batches.
export const MAXIMUM_RESIDENT_GROUP_BY_ROWS = 67_108_864;
export const MAXIMUM_RESIDENT_GROUP_BY_KEYS = 4_194_304;
export const MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS = 128;
export const MAXIMUM_RESIDENT_GROUP_BY_KEY_TEXT_BYTES = 32 * 1024 * 1024;
export const MAXIMUM_RESIDENT_GROUP_BY_RESULT_BYTES = 256 * 1024 * 1024;

export type ResidentGroupByTechnique = 'count' | 'sum' | 'average' | 'minimum' | 'maximum';
export type ResidentGroupByField = {name: string; type: string; analyzerType: string};
export type ResidentGroupByOperation = {
  id: string;
  dataId: string;
  fieldName: string | null;
  aggregations: Readonly<Record<string, string>>;
  resultId: string;
  resultLabel: string;
};
export type ResidentGroupBySourceMetadata = {
  datasetId: string;
  revision: number;
  rowCount: number;
  fields: readonly {name: string; type: string; analyzerType?: string}[];
  columns: Readonly<Record<string, {format?: string}>>;
  dictionaries: Readonly<Record<string, {values: readonly string[]; ordered: boolean}>>;
  nullCounts: Readonly<Record<string, readonly number[]>>;
};
export type ResidentGroupByProvenance = {
  schemaVersion: 1;
  kind: 'group-by';
  source: {datasetId: string; revision: number};
  population: 'all';
  operationId: string;
  resultId: string;
  resultLabel: string;
  keyColumn: string;
  aggregations: readonly {
    column: string;
    technique: ResidentGroupByTechnique;
    name: string;
  }[];
  fields: readonly ResidentGroupByField[];
  numericalPolicy: 'float32';
  rowOrder: 'first-source-occurrence';
};
export type ResidentGroupByPlan = {
  source: {datasetId: string; revision: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  key: {sourceColumn: string; outputColumn: string; type: string; analyzerType: string};
  measures: readonly {
    sourceColumn: string;
    outputColumn: string;
    technique: ResidentGroupByTechnique;
    type: string;
    analyzerType: string;
  }[];
  settings: KeplerGroupBySettings;
  /** Includes all dictionary labels and the dedicated null slot, even when unused. */
  keyCapacity: number;
  keyTextBytes: number;
  /** Conservative bound for materialized output values, validity and UTF-8 offsets/text. */
  resultByteLength: number;
  provenance: ResidentGroupByProvenance;
  derivedDataset: {type: 'groupBy'; sourceDataIds: [string]; operationId: string};
};

const SQL_GUIDANCE = ' Use SQL for a complementary query with unsupported keys or aggregations.';
const OPERATIONS = {
  count: 'count',
  sum: 'sum',
  average: 'mean',
  minimum: 'min',
  maximum: 'max'
} as const;

/** Admit metadata before allocating output storage or reading source rows. Active filters are ignored. */
export function createResidentGroupByPlan(
  operation: ResidentGroupByOperation,
  source: ResidentGroupBySourceMetadata
): ResidentGroupByPlan {
  if (source.datasetId !== operation.dataId) reject('The Group By source has changed.');
  const datasetId = readText(source.datasetId, 'source dataset');
  const revision = readInteger(source.revision, 0, Number.MAX_SAFE_INTEGER, 'source revision');
  readInteger(source.rowCount, 0, MAXIMUM_RESIDENT_GROUP_BY_ROWS, 'source row count');
  const operationId = readText(operation.id, 'operation ID');
  const resultId = readText(operation.resultId, 'result ID');
  const resultLabel = readText(operation.resultLabel, 'result name');
  if (resultId === datasetId) reject('Group By must create a separate result dataset.');
  const keyColumn = readText(operation.fieldName, 'grouping field');
  const keyField = source.fields.find(field => field.name === keyColumn);
  const dictionary = Object.hasOwn(source.dictionaries, keyColumn)
    ? source.dictionaries[keyColumn]
    : undefined;
  if (
    keyField?.type !== 'string' ||
    !dictionary ||
    source.columns[keyColumn]?.format !== 'uint32'
  ) {
    reject('Resident Group By requires a prepared string dictionary grouping field.');
  }
  if (dictionary.ordered) reject('Resident Group By requires an unordered string dictionary.');
  const keyCapacity = dictionary.values.length + 1;
  if (keyCapacity > MAXIMUM_RESIDENT_GROUP_BY_KEYS) {
    reject(
      `Group By supports at most ${MAXIMUM_RESIDENT_GROUP_BY_KEYS} dictionary slots including null.`
    );
  }
  let keyTextBytes = 0;
  for (const value of dictionary.values) {
    keyTextBytes += utf8ByteLength(value);
    if (keyTextBytes > MAXIMUM_RESIDENT_GROUP_BY_KEY_TEXT_BYTES) {
      reject('Group By dictionary text exceeds 32 MiB.');
    }
  }
  // The native reducer omits an aggregation on the grouping field itself.
  const entries = Object.entries(operation.aggregations).filter(([name]) => name !== keyColumn);
  if (entries.length > MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS) {
    reject(`Group By supports at most ${MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS} aggregations.`);
  }
  const usedNames = new Set([keyColumn]);
  const measures = entries.map(([column, technique]) => {
    const field = source.fields.find(candidate => candidate.name === column);
    if (!field) reject(`Group By field ${column} is unavailable.`);
    if (!isTechnique(technique))
      reject(`Group By aggregation ${technique} is unavailable on the GPU.`);
    if (
      technique !== 'count' &&
      (source.columns[column]?.format !== 'float32' || !['integer', 'real'].includes(field.type))
    ) {
      reject(`Group By ${technique} requires a prepared Float32 numeric field: ${column}.`);
    }
    const outputColumn = uniqueName(`${column}_${technique}`, usedNames);
    return {
      sourceColumn: column,
      outputColumn,
      technique,
      type: technique === 'count' ? 'integer' : technique === 'average' ? 'real' : field.type,
      analyzerType: technique === 'count' ? 'integer' : field.analyzerType || field.type
    };
  });
  const fields = [
    {name: keyColumn, type: keyField.type, analyzerType: keyField.analyzerType || keyField.type},
    ...measures.map(measure => ({
      name: measure.outputColumn,
      type: measure.type,
      analyzerType: measure.analyzerType
    }))
  ];
  const resultByteLength =
    keyTextBytes +
    4 * (keyCapacity + 1) +
    4 * keyCapacity * (measures.length + 1) +
    Math.ceil(keyCapacity / 8) * (measures.length + 1);
  if (resultByteLength > MAXIMUM_RESIDENT_GROUP_BY_RESULT_BYTES) {
    reject('Group By output exceeds the 256 MiB materialization bound.');
  }
  const provenance = parseResidentGroupByProvenance({
    schemaVersion: 1,
    kind: 'group-by',
    source: {datasetId, revision},
    population: 'all',
    operationId,
    resultId,
    resultLabel,
    keyColumn,
    aggregations: measures.map(measure => ({
      column: measure.sourceColumn,
      technique: measure.technique,
      name: measure.outputColumn
    })),
    fields,
    numericalPolicy: 'float32',
    rowOrder: 'first-source-occurrence'
  });
  return {
    source: {...provenance.source},
    operationId,
    resultId,
    resultLabel,
    key: {
      sourceColumn: keyColumn,
      outputColumn: keyColumn,
      type: keyField.type,
      analyzerType: keyField.analyzerType || keyField.type
    },
    measures,
    settings: {
      keyColumn,
      aggregations: measures.map(measure => ({
        column: measure.sourceColumn,
        operation: OPERATIONS[measure.technique],
        name: measure.outputColumn
      })),
      population: 'all',
      numericalPolicy: 'float32'
    },
    keyCapacity,
    keyTextBytes,
    resultByteLength,
    provenance,
    derivedDataset: {type: 'groupBy', sourceDataIds: [datasetId], operationId}
  };
}

/** Validate and detach saved lineage; re-admit against the restored source before execution. */
export function parseResidentGroupByProvenance(value: unknown): ResidentGroupByProvenance {
  const record = readRecord(value);
  const source = readRecord(record['source']);
  if (
    record['schemaVersion'] !== 1 ||
    record['kind'] !== 'group-by' ||
    record['population'] !== 'all' ||
    record['numericalPolicy'] !== 'float32' ||
    record['rowOrder'] !== 'first-source-occurrence'
  )
    reject('Unsupported saved Group By policy.');
  const keyColumn = readText(record['keyColumn'], 'grouping field');
  if (
    !Array.isArray(record['aggregations']) ||
    record['aggregations'].length > MAXIMUM_RESIDENT_GROUP_BY_AGGREGATIONS
  ) {
    reject('Invalid saved Group By aggregations.');
  }
  const usedNames = new Set([keyColumn]);
  const sourceColumns = new Set<string>();
  const aggregations = record['aggregations'].map(value => {
    const aggregation = readRecord(value);
    const column = readText(aggregation['column'], 'aggregation field');
    const technique = aggregation['technique'];
    if (!isTechnique(technique) || column === keyColumn || sourceColumns.has(column)) {
      reject('Invalid saved Group By aggregation.');
    }
    sourceColumns.add(column);
    const name = uniqueName(`${column}_${technique}`, usedNames);
    if (aggregation['name'] !== name)
      reject('Saved Group By output names do not match the native schema.');
    return {column, technique, name};
  });
  if (!Array.isArray(record['fields']) || record['fields'].length !== aggregations.length + 1) {
    reject('Invalid saved Group By output fields.');
  }
  const fields = record['fields'].map((value, index) => {
    const field = readRecord(value);
    const name = readText(field['name'], 'output field');
    const type = readText(field['type'], 'output type');
    const analyzerType = readText(field['analyzerType'], 'output analyzer type');
    const aggregation = aggregations[index - 1];
    if (
      name !== (index === 0 ? keyColumn : aggregation?.name) ||
      (index === 0 ? type !== 'string' : !['integer', 'real'].includes(type)) ||
      (aggregation?.technique === 'count' && (type !== 'integer' || analyzerType !== 'integer')) ||
      (aggregation?.technique === 'average' && type !== 'real')
    )
      reject('Saved Group By output schema is inconsistent.');
    return {name, type, analyzerType};
  });
  const datasetId = readText(source['datasetId'], 'source dataset');
  const resultId = readText(record['resultId'], 'result ID');
  if (datasetId === resultId) reject('Group By must create a separate result dataset.');
  return {
    schemaVersion: 1,
    kind: 'group-by',
    source: {
      datasetId,
      revision: readInteger(source['revision'], 0, Number.MAX_SAFE_INTEGER, 'source revision')
    },
    population: 'all',
    operationId: readText(record['operationId'], 'operation ID'),
    resultId,
    resultLabel: readText(record['resultLabel'], 'result name'),
    keyColumn,
    aggregations,
    fields,
    numericalPolicy: 'float32',
    rowOrder: 'first-source-occurrence'
  };
}

function isTechnique(value: unknown): value is ResidentGroupByTechnique {
  return typeof value === 'string' && Object.hasOwn(OPERATIONS, value);
}
function uniqueName(base: string, used: Set<string>): string {
  let name = base;
  let suffix = 1;
  while (used.has(name)) name = `${base}_${suffix++}`;
  used.add(name);
  return name;
}
function readRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    reject('Invalid saved Group By metadata.');
  return value as Record<string, unknown>;
}
function readText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096)
    reject(`Invalid Group By ${label}.`);
  return value;
}
function readInteger(value: unknown, minimum: number, maximum: number, label: string): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    reject(`Group By ${label} must be between ${minimum} and ${maximum}.`);
  }
  return value;
}
function reject(message: string): never {
  throw new Error(`${message}${SQL_GUIDANCE}`);
}
function utf8ByteLength(value: string): number {
  let bytes = 0;
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code < 0x80) bytes++;
    else if (code < 0x800) bytes += 2;
    else if (
      code >= 0xd800 &&
      code <= 0xdbff &&
      index + 1 < value.length &&
      value.charCodeAt(index + 1) >= 0xdc00 &&
      value.charCodeAt(index + 1) <= 0xdfff
    ) {
      bytes += 4;
      index++;
    } else bytes += 3;
  }
  return bytes;
}
