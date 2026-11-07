// vis.gl analysis
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  DataType,
  DateUnit,
  Dictionary,
  Field,
  Float32,
  Int32,
  Precision,
  RecordBatch,
  Schema,
  Struct,
  Table,
  TimeUnit,
  Uint32,
  Utf8,
  makeData,
  vectorFromArray,
  type Data,
  type Vector
} from 'apache-arrow';
import {convertWideNumericVector, isWideNumericType} from './arrow-analytics-wide-numeric';
import type {GPUAnalyticsDictionary} from '@luma.gl/graph/arrow';

export type ArrowAnalyticsTimeUnit = 'second' | 'millisecond' | 'microsecond' | 'nanosecond';

/** Float64 conversion is opt-in. Round mode enforces an absolute-plus-relative error budget. */
export type ArrowAnalyticsFloat64Policy =
  | {mode: 'exact'}
  | {mode: 'round'; maximumAbsoluteError: number; maximumRelativeError: number};

export type PrepareArrowTableForGPUAnalyticsOptions = {
  columns?: readonly string[];
  float64?: ArrowAnalyticsFloat64Policy;
  /** Previous unordered dictionaries. Existing codes remain stable; new labels append. Never mutated. */
  dictionaries?: Readonly<Record<string, GPUAnalyticsDictionary>>;
  /** Signed Int64 is interpreted as time only when its unit is explicitly declared here. */
  timeColumns?: Readonly<Record<string, ArrowAnalyticsTimeUnit>>;
};

export type PreparedArrowAnalyticsTimeColumn = {
  name: string;
  sourceType: string;
  unit: ArrowAnalyticsTimeUnit;
  timeZone: string | null;
  nullable: boolean;
  metadata: ReadonlyMap<string, string>;
  /** Owned CPU words, one independent chunk per original record batch; low word then high word. */
  chunks: readonly Uint32Array[];
  validity: readonly Uint32Array[];
};

export type ArrowAnalyticsColumnConversion = {
  name: string;
  sourceType: string;
  targetType: string;
  maximumAbsoluteError: number;
  maximumRelativeError: number;
};

export type PreparedArrowAnalyticsTable = {
  /** Portable scalar columns only. With no scalar columns Arrow reports zero rows; use sourceRowCount. */
  table: Table;
  /** Logical source row count, including time-only inputs. */
  sourceRowCount: number;
  dictionaries: Readonly<Record<string, GPUAnalyticsDictionary>>;
  timeColumns: Readonly<Record<string, PreparedArrowAnalyticsTimeColumn>>;
  conversions: readonly ArrowAnalyticsColumnConversion[];
  sourceBatches: readonly {
    sourceBatchIndex: number;
    sourceRowIndexOffset: number;
    sourceRowCount: number;
  }[];
};

/**
 * Explicit CPU preparation before upload. Widens Boolean/narrow integers exactly, reconciles
 * categories and separates lossless time words. Existing portable scalar data is borrowed;
 * converted data and time words are owned by the returned Arrow/typed-array objects.
 * No device allocation, submission, readback or implicit Float64 conversion occurs here.
 */
export function prepareArrowTableForGPUAnalytics(
  source: Table,
  options: PrepareArrowTableForGPUAnalyticsOptions = {}
): PreparedArrowAnalyticsTable {
  validateFloat64Policy(options.float64);
  const names = options.columns ?? source.schema.fields.map(field => field.name);
  if (new Set(names).size !== names.length) throw new Error('Duplicate selected analytics columns');
  if (new Set(source.schema.fields.map(field => field.name)).size !== source.schema.fields.length) {
    throw new Error('Duplicate source analytics columns');
  }
  const fields: Field[] = [];
  const columns: Data[][] = [];
  const dictionaries: Record<string, GPUAnalyticsDictionary> = Object.create(null);
  const timeColumns: Record<string, PreparedArrowAnalyticsTimeColumn> = Object.create(null);
  const conversions: ArrowAnalyticsColumnConversion[] = [];

  for (const name of names) {
    const fieldIndex = source.schema.fields.findIndex(field => field.name === name);
    if (fieldIndex < 0) throw new Error(`Unknown analytics column "${name}"`);
    const field = source.schema.fields[fieldIndex];
    const vectors = source.batches.map(batch => {
      const vector = batch.getChildAt(fieldIndex);
      if (!vector || String(vector.type) !== String(field.type)) {
        throw new Error(`Analytics column "${name}" changes type between batches`);
      }
      return vector;
    });
    const unit = getTimeUnit(field.type, options.timeColumns?.[name]);
    if (unit) {
      timeColumns[name] = prepareTimeColumn(field, vectors, unit);
      continue;
    }
    const report: ArrowAnalyticsColumnConversion = {
      name,
      sourceType: String(field.type),
      targetType: '',
      maximumAbsoluteError: 0,
      maximumRelativeError: 0
    };
    let type: DataType;
    let chunks: Data[];
    if (DataType.isUtf8(field.type) || DataType.isDictionary(field.type)) {
      const prepared = prepareCategories(
        field,
        vectors,
        options.dictionaries?.[name],
        fields.length
      );
      type = prepared.type;
      chunks = prepared.chunks;
      dictionaries[name] = prepared.dictionary;
    } else if (DataType.isBool(field.type)) {
      type = new Uint32();
      chunks = vectors.map(
        vector =>
          vectorFromArray(
            Array.from(vector, value => (value === null ? null : Number(value))),
            type
          ).data[0]
      );
    } else if (DataType.isInt(field.type) && field.type.bitWidth <= 32) {
      type = field.type.isSigned ? new Int32() : new Uint32();
      chunks = vectors.map(vector =>
        field.type.bitWidth === 32
          ? vector.data[0]
          : vectorFromArray(Array.from(vector), type).data[0]
      );
    } else if (DataType.isFloat(field.type) && field.type.precision === Precision.SINGLE) {
      type = field.type;
      chunks = vectors.map(vector => vector.data[0]);
    } else if (
      (DataType.isFloat(field.type) && field.type.precision === Precision.DOUBLE) ||
      isWideNumericType(field.type)
    ) {
      // Float64, Int64, Uint64 and Decimal narrow to Float32 only under the explicit error budget.
      if (!options.float64)
        throw new Error(`Float64 column "${name}" requires an explicit precision policy`);
      type = new Float32();
      chunks = vectors.map(
        vector =>
          vectorFromArray(
            convertWideNumericVector(field.type, vector, value =>
              convertFloat64(value, options.float64!, report)
            ),
            type
          ).data[0]
      );
    } else {
      throw new Error(`Unsupported analytics type ${field.type} for "${name}"`);
    }
    report.targetType = String(type);
    conversions.push(Object.freeze(report));
    fields.push(new Field(name, type, field.nullable, new Map(field.metadata)));
    columns.push(chunks);
  }
  const schema = new Schema(fields, new Map(source.schema.metadata));
  const batches = source.batches.map(
    (batch, batchIndex) =>
      new RecordBatch(
        schema,
        makeData({
          type: new Struct(fields),
          length: batch.numRows,
          children: columns.map(chunks => chunks[batchIndex])
        })
      )
  );
  let sourceRowIndexOffset = 0;
  const sourceBatches = source.batches.map((batch, sourceBatchIndex) => {
    const identity = {sourceBatchIndex, sourceRowIndexOffset, sourceRowCount: batch.numRows};
    sourceRowIndexOffset += batch.numRows;
    return Object.freeze(identity);
  });
  return {
    table: new Table(schema, batches),
    sourceRowCount: source.numRows,
    dictionaries: Object.freeze(dictionaries),
    timeColumns: Object.freeze(timeColumns),
    conversions: Object.freeze(conversions),
    sourceBatches: Object.freeze(sourceBatches)
  };
}

function validateFloat64Policy(policy: ArrowAnalyticsFloat64Policy | undefined): void {
  if (
    policy?.mode === 'round' &&
    ![policy.maximumAbsoluteError, policy.maximumRelativeError].every(
      value => Number.isFinite(value) && value >= 0
    )
  ) {
    throw new Error('Float64 rounding requires finite non-negative error budgets');
  }
}

function convertFloat64(
  value: number | null,
  policy: ArrowAnalyticsFloat64Policy,
  report: ArrowAnalyticsColumnConversion
): number | null {
  if (value === null) return null;
  const rounded = Math.fround(value);
  if (Object.is(value, rounded) || (Number.isNaN(value) && Number.isNaN(rounded))) return rounded;
  const absoluteError = Math.abs(value - rounded);
  const relativeError = value === 0 ? 0 : absoluteError / Math.abs(value);
  if (
    !Number.isFinite(rounded) ||
    policy.mode === 'exact' ||
    absoluteError > policy.maximumAbsoluteError + policy.maximumRelativeError * Math.abs(value)
  ) {
    throw new Error(`Float64 column "${report.name}" exceeds its precision policy`);
  }
  report.maximumAbsoluteError = Math.max(report.maximumAbsoluteError, absoluteError);
  report.maximumRelativeError = Math.max(report.maximumRelativeError, relativeError);
  return rounded;
}

function prepareCategories(
  field: Field,
  vectors: Vector[],
  previous: GPUAnalyticsDictionary | undefined,
  dictionaryId: number
): {type: Dictionary<Utf8, Uint32>; chunks: Data[]; dictionary: GPUAnalyticsDictionary} {
  if (DataType.isDictionary(field.type) && !DataType.isUtf8(field.type.dictionary)) {
    throw new Error(`Category column "${field.name}" requires UTF-8 labels`);
  }
  const ordered = DataType.isDictionary(field.type) && field.type.isOrdered;
  if (previous && previous.ordered !== ordered) throw new Error('Category ordering policy changed');
  const labels = [...(previous?.values ?? [])];
  const codes = new Map(labels.map((label, index) => [label, index]));
  if (codes.size !== labels.length)
    throw new Error('Category dictionary contains duplicate labels');
  // Ordered dictionaries must agree exactly; appending by encounter order would change semantics.
  if (ordered) {
    for (const vector of vectors) {
      const dictionary = vector.data[0].dictionary;
      const current = dictionary ? Array.from(dictionary) : [];
      if (current.some(value => typeof value !== 'string'))
        throw new Error('Category dictionary contains null or non-string labels');
      if (!previous && labels.length === 0) {
        for (const label of current) {
          codes.set(label, labels.length);
          labels.push(label);
        }
      }
      if (
        labels.length !== current.length ||
        labels.some((label, index) => label !== current[index])
      ) {
        throw new Error(`Ordered dictionary "${field.name}" changes across batches`);
      }
    }
  }
  const type = new Dictionary(new Utf8(), new Uint32(), dictionaryId, ordered);
  const encoded = vectors.map(vector => {
    const values = new Uint32Array(vector.length);
    const bitmap = new Uint8Array(Math.ceil(vector.length / 8));
    let nullCount = 0;
    for (let rowIndex = 0; rowIndex < vector.length; rowIndex++) {
      const value = vector.get(rowIndex);
      if (value === null) {
        nullCount++;
        continue;
      }
      if (typeof value !== 'string') throw new Error('Category values must be UTF-8 strings');
      let code = codes.get(value);
      if (code === undefined) {
        code = labels.length;
        codes.set(value, code);
        labels.push(value);
      }
      values[rowIndex] = code;
      bitmap[rowIndex >> 3] |= 1 << (rowIndex & 7);
    }
    return {values, bitmap, nullCount};
  });
  const dictionary = vectorFromArray(labels, new Utf8());
  return {
    type,
    chunks: encoded.map(({values, bitmap, nullCount}) =>
      makeData({
        type,
        data: values,
        length: values.length,
        nullBitmap: bitmap,
        nullCount,
        dictionary
      })
    ),
    dictionary: Object.freeze({values: Object.freeze(labels), ordered})
  };
}

function getTimeUnit(
  type: DataType,
  declared: ArrowAnalyticsTimeUnit | undefined
): ArrowAnalyticsTimeUnit | undefined {
  if (DataType.isTimestamp(type)) {
    const units: Record<TimeUnit, ArrowAnalyticsTimeUnit> = {
      [TimeUnit.SECOND]: 'second',
      [TimeUnit.MILLISECOND]: 'millisecond',
      [TimeUnit.MICROSECOND]: 'microsecond',
      [TimeUnit.NANOSECOND]: 'nanosecond'
    };
    if (declared && declared !== units[type.unit])
      throw new Error('Timestamp unit cannot be reinterpreted');
    return units[type.unit];
  }
  if (DataType.isDate(type)) {
    if (declared && declared !== 'millisecond')
      throw new Error('Date columns normalize to milliseconds');
    return 'millisecond';
  }
  if (declared && !(DataType.isInt(type) && type.bitWidth === 64 && type.isSigned)) {
    throw new Error('Explicit time units require a signed Int64 column');
  }
  return declared;
}

function prepareTimeColumn(
  field: Field,
  vectors: Vector[],
  unit: ArrowAnalyticsTimeUnit
): PreparedArrowAnalyticsTimeColumn {
  const validity: Uint32Array[] = [];
  const chunks = vectors.map(vector => {
    const words = new Uint32Array(vector.length * 2);
    const mask = new Uint32Array(vector.length);
    const data = vector.data[0];
    for (let rowIndex = 0; rowIndex < vector.length; rowIndex++) {
      if (!vector.isValid(rowIndex)) continue;
      // Arrow fixed-width slices already adjust values; timestamp get() would lose micro/nanoseconds.
      const raw = data.values[rowIndex];
      const value =
        DataType.isDate(field.type) && field.type.unit === DateUnit.DAY
          ? BigInt(raw) * 86400000n
          : BigInt(raw);
      const unsigned = BigInt.asUintN(64, value);
      words[rowIndex * 2] = Number(unsigned & 0xffffffffn);
      words[rowIndex * 2 + 1] = Number(unsigned >> 32n);
      mask[rowIndex] = 1;
    }
    validity.push(mask);
    return words;
  });
  return Object.freeze({
    name: field.name,
    sourceType: String(field.type),
    unit,
    timeZone: DataType.isTimestamp(field.type) ? (field.type.timezone ?? null) : null,
    nullable: field.nullable,
    metadata: new Map(field.metadata),
    chunks: Object.freeze(chunks),
    validity: Object.freeze(validity)
  });
}
