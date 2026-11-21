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
  Vector,
  makeData,
  vectorFromArray,
  type Data
} from 'apache-arrow';
import {isWideNumericType, narrowWideNumericValue} from './arrow-analytics-wide-numeric';
import {validateRequirement} from '../kernels/gpu-contributor-utils';
import type {GPUAnalyticsDictionary} from '@luma.gl/graph/arrow';

export type ArrowAnalyticsTimeUnit = 'second' | 'millisecond' | 'microsecond' | 'nanosecond';

/** Combined scalar Arrow vectors use uint32 chunk offsets; prepare larger tables batch by batch. */
export const MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT = 0xffffffff;

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
 * Combined scalar tables support at most 4294967295 source rows; use
 * prepareArrowBatchesForGPUAnalytics for larger sources.
 * Each rebuilt physical Utf8 category dictionary permits at most 2147483647 encoded bytes
 * across all its labels; prepare smaller independent dictionaries when that limit is exceeded.
 */
export function prepareArrowTableForGPUAnalytics(
  source: Table,
  options: PrepareArrowTableForGPUAnalyticsOptions = {}
): PreparedArrowAnalyticsTable {
  validateFloat64Policy(options.float64);
  validateTimeColumnOptions(source, options.timeColumns);
  const names = validateColumnSelection(source, options.columns);
  const hasScalars = names.some(name => {
    const field = source.schema.fields.find(candidate => candidate.name === name);
    return field && !getTimeUnit(field.type, getOwnOption(options.timeColumns, name));
  });
  validateRequirement(
    'prepareArrowTableForGPUAnalytics',
    !hasScalars || source.numRows <= MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT,
    `combined scalar sourceRowCount received ${source.numRows}; maximum ${MAXIMUM_ARROW_ANALYTICS_SCALAR_ROW_COUNT}; use prepareArrowBatchesForGPUAnalytics`
  );
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
    const unit = getTimeUnit(field.type, getOwnOption(options.timeColumns, name));
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
    if (
      DataType.isUtf8(field.type) ||
      DataType.isLargeUtf8(field.type) ||
      DataType.isDictionary(field.type)
    ) {
      const prepared = prepareCategories(
        field,
        vectors,
        getOwnOption(options.dictionaries, name),
        fields.length
      );
      type = prepared.type;
      chunks = prepared.chunks;
      if (prepared.dictionary) dictionaries[name] = prepared.dictionary;
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
    } else if (DataType.isFloat(field.type) && field.type.precision === Precision.HALF) {
      type = new Float32();
      chunks = vectors.map(vector => vectorFromArray(Array.from(vector), type).data[0]);
    } else if (DataType.isFloat(field.type) && field.type.precision === Precision.SINGLE) {
      type = field.type;
      chunks = vectors.map(vector => vector.data[0]);
    } else if (
      (DataType.isFloat(field.type) && field.type.precision === Precision.DOUBLE) ||
      isWideNumericType(field.type)
    ) {
      // Float64, Int64, Uint64 and Decimal narrow to Float32 only under the explicit error budget.
      if (!options.float64)
        throw new Error(`${field.type} column "${name}" requires an explicit precision policy`);
      type = new Float32();
      chunks = vectors.map(
        vector =>
          vectorFromArray(
            Array.from(vector, value =>
              isWideNumericType(field.type)
                ? narrowWideNumericValue(field.type, value, options.float64!, report)
                : convertFloat64(value as number | null, options.float64!, report)
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

/**
 * Independently addressable prepared record batches, retaining safe-integer source offsets.
 * Dictionary codes remain stable across yielded batches. The iterator never creates a combined
 * scalar vector, so totals above 4294967295 are supported (each scalar batch is limited to
 * 4294967295 rows). Retain the last dictionaries when continuing in a subsequent invocation.
 */
export function prepareArrowBatchesForGPUAnalytics(
  source: Table,
  options: PrepareArrowTableForGPUAnalyticsOptions = {}
): Generator<PreparedArrowAnalyticsTable, void, unknown> {
  validateFloat64Policy(options.float64);
  validateTimeColumnOptions(source, options.timeColumns);
  validateColumnSelection(source, options.columns);
  validateRequirement(
    'prepareArrowBatchesForGPUAnalytics',
    Number.isSafeInteger(source.numRows),
    `sourceRowCount must be a safe integer; received ${source.numRows}`
  );
  const capturedOptions = {
    columns: options.columns && [...options.columns],
    float64: options.float64 && {...options.float64},
    timeColumns:
      options.timeColumns &&
      Object.fromEntries(
        Object.getOwnPropertyNames(options.timeColumns).map(name => [
          name,
          options.timeColumns![name]
        ])
      )
  };
  const previousDictionaries: Record<string, GPUAnalyticsDictionary> = Object.create(null);
  for (const name of Object.getOwnPropertyNames(options.dictionaries ?? {})) {
    const dictionary = options.dictionaries![name];
    previousDictionaries[name] = {ordered: dictionary.ordered, values: [...dictionary.values]};
  }
  const batches = [...source.batches];
  const schema = source.schema;
  return prepareBatches();

  function* prepareBatches(): Generator<PreparedArrowAnalyticsTable, void, unknown> {
    let dictionaries: Readonly<Record<string, GPUAnalyticsDictionary>> = previousDictionaries;
    let sourceRowIndexOffset = 0;
    for (const [sourceBatchIndex, batch] of batches.entries()) {
      const prepared = prepareArrowTableForGPUAnalytics(new Table(schema, [batch]), {
        ...capturedOptions,
        dictionaries
      });
      dictionaries = prepared.dictionaries;
      yield {
        ...prepared,
        sourceBatches: Object.freeze([
          Object.freeze({sourceBatchIndex, sourceRowIndexOffset, sourceRowCount: batch.numRows})
        ])
      };
      sourceRowIndexOffset += batch.numRows;
    }
  }
}

function validateColumnSelection(
  source: Table,
  columns: readonly string[] | undefined
): readonly string[] {
  const operation = 'prepareArrowTableForGPUAnalytics';
  const names = columns ?? source.schema.fields.map(field => field.name);
  validateRequirement(
    operation,
    new Set(names).size === names.length,
    `Duplicate selected analytics columns; received ${JSON.stringify(names)}`
  );
  validateRequirement(
    operation,
    new Set(source.schema.fields.map(field => field.name)).size === source.schema.fields.length,
    'Duplicate source analytics columns'
  );
  for (const name of names) {
    validateRequirement(
      operation,
      source.schema.fields.some(field => field.name === name),
      `Unknown analytics column "${name}"; columns must name source fields`
    );
  }
  return names;
}

/**
 * Per-column option of a caller map. Only own properties count, so a column named `constructor`,
 * `toString` or `__proto__` never picks up an `Object.prototype` member from an ordinary `{}`.
 */
function getOwnOption<Value>(
  options: Readonly<Record<string, Value>> | undefined,
  name: string
): Value | undefined {
  return options && Object.hasOwn(options, name) ? options[name] : undefined;
}

function validateFloat64Policy(policy: ArrowAnalyticsFloat64Policy | undefined): void {
  validateRequirement(
    'prepareArrowTableForGPUAnalytics',
    policy === undefined || policy?.mode === 'exact' || policy?.mode === 'round',
    `float64.mode received ${String(policy?.mode)}; allowed choices: exact, round`
  );
  if (policy?.mode === 'round') {
    validateRequirement(
      'prepareArrowTableForGPUAnalytics',
      [policy.maximumAbsoluteError, policy.maximumRelativeError].every(
        value => Number.isFinite(value) && value >= 0
      ),
      `float64 rounding requires finite non-negative error budgets; received maximumAbsoluteError ${policy.maximumAbsoluteError}, maximumRelativeError ${policy.maximumRelativeError}`
    );
  }
}

function validateTimeColumnOptions(
  source: Table,
  columns: PrepareArrowTableForGPUAnalyticsOptions['timeColumns']
): void {
  const allowed = ['second', 'millisecond', 'microsecond', 'nanosecond'];
  for (const name of Object.getOwnPropertyNames(columns ?? {})) {
    const unit = columns![name];
    validateRequirement(
      'prepareArrowTableForGPUAnalytics',
      source.schema.fields.some(field => field.name === name),
      `timeColumns key "${name}" must name a source field; received "${name}"`
    );
    validateRequirement(
      'prepareArrowTableForGPUAnalytics',
      allowed.includes(unit),
      `timeColumns["${name}"] received ${String(unit)}; allowed choices: ${allowed.join(', ')}`
    );
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
  dictionaryId: number,
  forEquality = false
): {
  type: Dictionary<Utf8, Uint32>;
  chunks: Data[];
  dictionary: GPUAnalyticsDictionary | undefined;
} {
  if (
    DataType.isDictionary(field.type) &&
    !DataType.isUtf8(field.type.dictionary) &&
    !DataType.isLargeUtf8(field.type.dictionary)
  ) {
    throw new Error(`Category column "${field.name}" requires UTF-8 labels`);
  }
  const sourceOrdered = DataType.isDictionary(field.type) && field.type.isOrdered;
  const ordered = sourceOrdered && !forEquality;
  if (previous && previous.ordered !== ordered) throw new Error('Category ordering policy changed');
  const labels = [...(previous?.values ?? [])];
  const codes = validateCategoryLabels(field.name, labels);
  let ordinalLabels: readonly string[] | undefined = ordered && previous ? labels : undefined;
  // Ordered dictionaries must agree exactly; appending by encounter order would change semantics.
  if (sourceOrdered) {
    for (const vector of vectors) {
      const dictionary = vector.data[0].dictionary;
      const current = dictionary ? Array.from(dictionary) : [];
      validateCategoryLabels(field.name, current);
      if (ordinalLabels === undefined) {
        ordinalLabels = current;
        if (ordered) {
          for (const label of current) {
            codes.set(label, labels.length);
            labels.push(label);
          }
        }
      }
      if (
        ordinalLabels.length !== current.length ||
        ordinalLabels.some((label, index) => label !== current[index])
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
        validateRequirement(
          'prepareArrowTableForGPUAnalytics',
          labels.length < MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT,
          `column "${field.name}" category count received ${labels.length + 1}; maximum ${MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT} (all-ones code is reserved)`
        );
        code = labels.length;
        codes.set(value, code);
        labels.push(value);
      }
      values[rowIndex] = code;
      bitmap[rowIndex >> 3] |= 1 << (rowIndex & 7);
    }
    return {values, bitmap, nullCount};
  });
  let dictionaryByteLength = 0;
  for (const label of labels) {
    dictionaryByteLength += getUtf8ByteLength(label);
    validateArrowAnalyticsDictionaryByteLength(field.name, dictionaryByteLength);
  }
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
    // No observed dictionary is different from an observed dictionary containing zero labels.
    dictionary:
      ordered && ordinalLabels === undefined
        ? undefined
        : Object.freeze({values: Object.freeze(labels), ordered})
  };
}

/** Maximum category count per prepared codebook; uint32 all-ones is a reserved missing key. */
export const MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT = 0xffffffff;

/** Maximum encoded bytes in the single physical Utf8 category dictionary, across all labels. */
export const MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH = 0x7fffffff;

/** @internal Check the physical Utf8 offset limit before allocating its encoded dictionary. */
export function validateArrowAnalyticsDictionaryByteLength(name: string, byteLength: number): void {
  validateRequirement(
    'prepareArrowTableForGPUAnalytics',
    Number.isSafeInteger(byteLength) &&
      byteLength >= 0 &&
      byteLength <= MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH,
    `column "${name}" dictionary encoded byte length received ${byteLength}; maximum ${MAXIMUM_ARROW_ANALYTICS_DICTIONARY_BYTE_LENGTH}; split the source into separately prepared dictionaries`
  );
}

/** UTF-8 byte count without encoding a payload; unpaired UTF-16 surrogates become U+FFFD. */
function getUtf8ByteLength(value: string): number {
  let byteLength = 0;
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code < 0x80) byteLength++;
    else if (code < 0x800) byteLength += 2;
    else if (
      code >= 0xd800 &&
      code <= 0xdbff &&
      index + 1 < value.length &&
      value.charCodeAt(index + 1) >= 0xdc00 &&
      value.charCodeAt(index + 1) <= 0xdfff
    ) {
      byteLength += 4;
      index++;
    } else byteLength += 3;
  }
  return byteLength;
}

function validateCategoryLabels(name: string, labels: readonly unknown[]): Map<string, number> {
  const operation = 'prepareArrowTableForGPUAnalytics';
  validateRequirement(
    operation,
    labels.length <= MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT,
    `column "${name}" category count received ${labels.length}; maximum ${MAXIMUM_ARROW_ANALYTICS_CATEGORY_COUNT} (all-ones code is reserved)`
  );
  const codes = new Map<string, number>();
  labels.forEach((label, index) => {
    validateRequirement(
      operation,
      typeof label === 'string',
      `column "${name}" dictionary label at position ${index} must be a string; received ${String(label)}`
    );
    const firstPosition = codes.get(label);
    validateRequirement(
      operation,
      firstPosition === undefined,
      `column "${name}" dictionary contains duplicate labels at positions ${firstPosition} and ${index}`
    );
    codes.set(label, index);
  });
  return codes;
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

/** Owned key words and eligibility, chunked in the source table's record-batch order. */
export type PreparedArrowAnalyticsKeyColumn = {
  name: string;
  sourceType: string;
  chunks: readonly Uint32Array[];
  validity: readonly Uint32Array[];
};

export type PreparedArrowAnalyticsKeyPair = {
  left: PreparedArrowAnalyticsKeyColumn;
  right: PreparedArrowAnalyticsKeyColumn;
  /** Upload category codes as uint32, integer words as uint32x2 (low word first). */
  keyBits: 32 | 64;
  dictionary?: GPUAnalyticsDictionary;
  /** Decode integer pairs as signed only when true. */
  signed: boolean;
};

/**
 * Prepare equality keys together, including differently named fields. Categories share one
 * codebook; integer keys bypass floating-point conversion entirely. Integer signedness must
 * agree. The 64-bit all-ones key is reserved by GPUKeyLookup/GPUGroupStatistics and is rejected;
 * remap that identifier before preparation. Null keys get zero eligibility independently on
 * each side. Words and validity are owned; GPU uploads retain original batch boundaries.
 * Ordered dictionaries retain ordinal consistency within each source, then both sources are
 * reconciled by label into an unordered equality codebook (at most 2147483647 encoded bytes).
 */
export function prepareArrowKeyPairForGPUAnalytics(
  left: Table,
  leftColumn: string,
  right: Table,
  rightColumn: string
): PreparedArrowAnalyticsKeyPair {
  const operation = 'prepareArrowKeyPairForGPUAnalytics';
  const getField = (source: Table, name: string) => {
    const field = source.schema.fields.find(candidate => candidate.name === name);
    validateRequirement(
      operation,
      Boolean(field),
      `column "${name}" must name a source field; received "${name}"`
    );
    validateRequirement(
      operation,
      source.schema.fields.filter(candidate => candidate.name === name).length === 1,
      `column "${name}" must name exactly one source field`
    );
    return field!;
  };
  const leftField = getField(left, leftColumn);
  const rightField = getField(right, rightColumn);
  const isCategory = (type: DataType) =>
    DataType.isUtf8(type) || DataType.isLargeUtf8(type) || DataType.isDictionary(type);
  const category = isCategory(leftField.type) && isCategory(rightField.type);
  const integers = DataType.isInt(leftField.type) && DataType.isInt(rightField.type);
  validateRequirement(
    operation,
    category || integers,
    `keys must both be categories or integers; received ${leftColumn}: ${leftField.type}, ${rightColumn}: ${rightField.type}`
  );
  const signed = DataType.isInt(leftField.type) && leftField.type.isSigned;
  validateRequirement(
    operation,
    !integers || (DataType.isInt(rightField.type) && signed === rightField.type.isSigned),
    `key signedness must agree; received ${leftField.type} and ${rightField.type}`
  );
  let dictionary: GPUAnalyticsDictionary | undefined;
  const prepare = (source: Table, field: Field): PreparedArrowAnalyticsKeyColumn => {
    const vectors = source.batches.map(batch => batch.getChild(field.name)!);
    let validity = vectors.map(vector =>
      Uint32Array.from({length: vector.length}, (_, row) => Number(vector.isValid(row)))
    );
    let chunks: Uint32Array[];
    if (category) {
      const prepared = prepareCategories(field, vectors, dictionary, 0, true);
      dictionary = prepared.dictionary;
      chunks = prepared.chunks.map(chunk => new Uint32Array(chunk.values));
      validity = prepared.chunks.map(chunk => {
        const vector = new Vector([chunk]);
        return Uint32Array.from({length: vector.length}, (_, row) => Number(vector.isValid(row)));
      });
    } else {
      chunks = vectors.map(vector => {
        const words = new Uint32Array(vector.length * 2);
        for (let row = 0; row < vector.length; row++) {
          if (!vector.isValid(row)) continue;
          const value = BigInt.asUintN(64, BigInt(vector.get(row)));
          validateRequirement(
            operation,
            value !== 0xffffffffffffffffn,
            `column "${field.name}" row ${row} has reserved all-ones key ${vector.get(row)}; remap this identifier before preparation`
          );
          words[row * 2] = Number(value & 0xffffffffn);
          words[row * 2 + 1] = Number(value >> 32n);
        }
        return words;
      });
    }
    return Object.freeze({
      name: field.name,
      sourceType: String(field.type),
      chunks: Object.freeze(chunks),
      validity: Object.freeze(validity)
    });
  };
  const preparedLeft = prepare(left, leftField);
  const preparedRight = prepare(right, rightField);
  return Object.freeze({
    left: preparedLeft,
    right: preparedRight,
    keyBits: category ? 32 : 64,
    dictionary,
    signed
  });
}

/** Owned float measure chunks: null lanes become NaN, real zero lanes remain zero. */
export type PreparedArrowAnalyticsMeasureColumn = {
  name: string;
  sourceType: string;
  chunks: readonly Float32Array[];
  validity: readonly Uint32Array[];
};

/**
 * Materialize each selected float measure once for GPUGroupStatistics columns or GPUKeyLookup
 * gather/aggregate inputs. Each column uses its own Arrow validity; key eligibility is separate.
 * Float64 conversion still requires the ordinary explicit precision policy. Output chunks are
 * aligned to source record batches and can be uploaded without additional missing-value work.
 */
export function prepareArrowMeasureColumnsForGPUAnalytics(
  source: Table,
  columns: readonly string[],
  options: Omit<PrepareArrowTableForGPUAnalyticsOptions, 'columns'> = {}
): Readonly<Record<string, PreparedArrowAnalyticsMeasureColumn>> {
  for (const name of columns) {
    const field = source.schema.fields.find(candidate => candidate.name === name);
    validateRequirement(
      'prepareArrowMeasureColumnsForGPUAnalytics',
      Boolean(field && DataType.isFloat(field.type)),
      `column "${name}" must be a float measure; received ${field?.type}`
    );
  }
  const prepared = prepareArrowTableForGPUAnalytics(source, {...options, columns});
  const result: Record<string, PreparedArrowAnalyticsMeasureColumn> = Object.create(null);
  for (const name of columns) {
    const field = prepared.table.schema.fields.find(candidate => candidate.name === name);
    validateRequirement(
      'prepareArrowMeasureColumnsForGPUAnalytics',
      Boolean(field && DataType.isFloat(field.type)),
      `column "${name}" must be a float measure; received ${source.schema.fields.find(candidate => candidate.name === name)?.type}`
    );
    const vectors = prepared.table.batches.map(batch => batch.getChild(name)!);
    result[name] = Object.freeze({
      name,
      sourceType: String(source.schema.fields.find(candidate => candidate.name === name)!.type),
      chunks: Object.freeze(
        vectors.map(vector =>
          Float32Array.from({length: vector.length}, (_, row) =>
            vector.isValid(row) ? vector.get(row) : NaN
          )
        )
      ),
      validity: Object.freeze(
        vectors.map(vector =>
          Uint32Array.from({length: vector.length}, (_, row) => Number(vector.isValid(row)))
        )
      )
    });
  }
  return Object.freeze(result);
}
