// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Sentinel row index meaning "no row on this side"; the GPU gather writes zero and invalid. */
export const RESIDENT_JOIN_NO_ROW = 0xffffffff;
/** Sentinel key meaning "this row cannot match": null, empty label, or a label absent on the other side. */
export const RESIDENT_JOIN_NO_KEY = 0xffffffff;
/** Largest row index addressable by the uint32 row mappings (one value is the no-row sentinel). */
export const MAXIMUM_RESIDENT_JOIN_ROW_INDEX = 0xfffffffe;
/**
 * Structural limits only; side row counts and output row counts have no cap. Data-dependent
 * memory (key/sort working set, row mappings, result columns, retained Arrow copy) is admitted by
 * `reserveMemory` at execution time and fails with SQL guidance when it does not fit.
 */
export const MAXIMUM_RESIDENT_JOIN_COLUMNS = 32;
export const MAXIMUM_RESIDENT_JOIN_DICTIONARY_SLOTS = 65_536;
export const MAXIMUM_RESIDENT_JOIN_KEY_TEXT_BYTES = 4 * 1024 * 1024;

export type ResidentJoinType = 'LEFT' | 'INNER' | 'FULL';
export type ResidentJoinField = {name: string; type: string; analyzerType: string};
export type ResidentJoinOperation = {
  id: string;
  implementation?: string;
  leftDataId: string;
  rightDataId: string;
  leftField: string | null;
  rightField: string | null;
  type: string;
  leftColumns?: readonly string[];
  rightColumns?: readonly string[];
  /**
   * Which right rows join a left row with duplicated keys: every match (native, default), only
   * the first or only the last in right source order.
   */
  duplicateRightMatches?: ResidentJoinDuplicatePolicy;
  /**
   * Names colliding right columns `${name}${suffix}` (then a numeric counter) instead of the
   * native `${rightLabel}_${name}` prefix. Only columns that collide receive it.
   */
  rightColumnSuffix?: string;
  /** With a suffix policy, omit a right column that repeats the left key's name. */
  dropRightKey?: boolean;
  /** SQL self-joins read one dataset on both sides; the native Join panel still needs two. */
  allowSameDataset?: boolean;
  resultId: string;
  resultLabel: string;
};
export type ResidentJoinDuplicatePolicy = 'all' | 'first' | 'last';
export type ResidentJoinSourceMetadata = {
  datasetId: string;
  label: string;
  revision: number;
  rowCount: number;
  fields: readonly {name: string; type: string; analyzerType?: string}[];
  columns: Readonly<Record<string, {format?: string}>>;
  dictionaries: Readonly<Record<string, {values: readonly string[]; ordered: boolean}>>;
  nullCounts: Readonly<Record<string, readonly number[]>>;
};
export type ResidentJoinColumnFormat = 'float32' | 'sint32' | 'uint32';
export type ResidentJoinColumn = {
  side: 'left' | 'right';
  sourceColumn: string;
  outputColumn: string;
  type: string;
  analyzerType: string;
  format: ResidentJoinColumnFormat;
  /** Dictionary labels are shared with the source; only codes are gathered. */
  dictionary: boolean;
};
export type ResidentJoinKeyPolicy =
  | 'string-dictionary-exact-null-and-empty-unmatched'
  | 'numeric-string-form-integral-float32-null-unmatched';
export type ResidentJoinProvenance = {
  schemaVersion: 1;
  kind: 'join';
  left: {datasetId: string; revision: number};
  right: {datasetId: string; revision: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  joinType: ResidentJoinType;
  leftKey: string;
  rightKey: string;
  columns: readonly {side: 'left' | 'right'; column: string; name: string}[];
  fields: readonly ResidentJoinField[];
  keyPolicy: ResidentJoinKeyPolicy;
  rowOrder: 'left-source-order-right-source-order-then-unmatched-right';
  /** Absent means native: every duplicate right match and label-prefixed collisions. */
  duplicateRightMatches?: ResidentJoinDuplicatePolicy;
  rightColumnSuffix?: string;
};
export type ResidentJoinPlan = {
  left: {datasetId: string; revision: number; rowCount: number};
  right: {datasetId: string; revision: number; rowCount: number};
  operationId: string;
  resultId: string;
  resultLabel: string;
  joinType: ResidentJoinType;
  leftKey: string;
  rightKey: string;
  keyKind: 'string' | 'numeric';
  duplicateRightMatches: ResidentJoinDuplicatePolicy;
  columns: readonly ResidentJoinColumn[];
  provenance: ResidentJoinProvenance;
  derivedDataset: {type: 'join'; sourceDataIds: [string, string]; operationId: string};
};

export const RESIDENT_JOIN_SQL_GUIDANCE =
  ' Use SQL for a complementary query with unsupported keys, columns or sizes.';

/** Admit metadata before reading keys or allocating output storage. Active filters are ignored. */
export function createResidentJoinPlan(
  operation: ResidentJoinOperation,
  left: ResidentJoinSourceMetadata,
  right: ResidentJoinSourceMetadata
): ResidentJoinPlan {
  if (operation.implementation === 'spatial') reject('Spatial Join has no resident executor.');
  if (left.datasetId !== operation.leftDataId || right.datasetId !== operation.rightDataId)
    reject('The Join sources have changed.');
  if (left.datasetId === right.datasetId && !operation.allowSameDataset)
    reject('Join requires two different datasets.');
  const joinType = operation.type;
  if (joinType !== 'LEFT' && joinType !== 'INNER' && joinType !== 'FULL')
    reject(`Join type ${String(joinType)} is unavailable on the GPU.`);
  const operationId = readText(operation.id, 'operation ID');
  const resultId = readText(operation.resultId, 'result ID');
  const resultLabel = readText(operation.resultLabel, 'result name');
  if (resultId === left.datasetId || resultId === right.datasetId)
    reject('Join must create a separate result dataset.');
  for (const [side, source] of [
    ['left', left],
    ['right', right]
  ] as const) {
    readText(source.datasetId, `${side} dataset`);
    readInteger(source.revision, 0, Number.MAX_SAFE_INTEGER, `${side} revision`);
    if (
      !Number.isSafeInteger(source.rowCount) ||
      source.rowCount < 1 ||
      source.rowCount > MAXIMUM_RESIDENT_JOIN_ROW_INDEX
    )
      // Row indices are uint32 with one sentinel; there is no smaller row cap.
      reject(`Join ${side} input must have at least one row (found ${source.rowCount}).`);
  }
  const leftKey = readText(operation.leftField, 'left key');
  const rightKey = readText(operation.rightField, 'right key');
  const keyKind = checkKey(left, leftKey, 'left');
  if (keyKind !== checkKey(right, rightKey, 'right'))
    reject('Join keys must both be string dictionaries or both be numbers.');

  const leftNames = operation.leftColumns ?? left.fields.map(field => field.name);
  const rightNames = operation.rightColumns ?? right.fields.map(field => field.name);
  // Native Join silently drops unknown names and keeps duplicates.
  const suffix = operation.rightColumnSuffix;
  const duplicateRightMatches = operation.duplicateRightMatches ?? 'all';
  if (!['all', 'first', 'last'].includes(duplicateRightMatches))
    reject(`Join duplicate policy ${String(duplicateRightMatches)} is unavailable.`);
  // A suffix policy may drop the right key when the left key has the same name.
  const picked = [
    ...leftNames.flatMap(name => pick(left, name, 'left')),
    ...rightNames
      .filter(
        name =>
          !(suffix !== undefined && operation.dropRightKey && name === rightKey && name === leftKey)
      )
      .flatMap(name => pick(right, name, 'right'))
  ];
  if (!picked.length) reject('Join requires at least one output column.');
  if (picked.length > MAXIMUM_RESIDENT_JOIN_COLUMNS)
    reject(`Join supports at most ${MAXIMUM_RESIDENT_JOIN_COLUMNS} output columns.`);
  const usedNames = new Set<string>();
  // Left columns come first, so a suffixed right column sees every left name as used.
  const columns: ResidentJoinColumn[] = picked.map(column => ({
    ...column,
    outputColumn:
      column.side === 'left'
        ? uniqueName(column.sourceColumn, usedNames)
        : suffix !== undefined
          ? uniqueSuffixedName(column.sourceColumn, suffix, usedNames)
          : uniqueName(`${right.label}_${column.sourceColumn}`, usedNames)
  }));
  const fields = columns.map(column => ({
    name: column.outputColumn,
    type: column.type,
    analyzerType: column.analyzerType
  }));
  const provenance: ResidentJoinProvenance = {
    schemaVersion: 1,
    kind: 'join',
    left: {datasetId: left.datasetId, revision: left.revision},
    right: {datasetId: right.datasetId, revision: right.revision},
    operationId,
    resultId,
    resultLabel,
    joinType,
    leftKey,
    rightKey,
    columns: columns.map(column => ({
      side: column.side,
      column: column.sourceColumn,
      name: column.outputColumn
    })),
    fields,
    keyPolicy:
      keyKind === 'string'
        ? 'string-dictionary-exact-null-and-empty-unmatched'
        : 'numeric-string-form-integral-float32-null-unmatched',
    rowOrder: 'left-source-order-right-source-order-then-unmatched-right',
    ...(duplicateRightMatches !== 'all' ? {duplicateRightMatches} : {}),
    ...(suffix !== undefined ? {rightColumnSuffix: suffix} : {})
  };
  return {
    left: {datasetId: left.datasetId, revision: left.revision, rowCount: left.rowCount},
    right: {datasetId: right.datasetId, revision: right.revision, rowCount: right.rowCount},
    operationId,
    resultId,
    resultLabel,
    joinType,
    leftKey,
    rightKey,
    keyKind,
    duplicateRightMatches,
    columns,
    provenance,
    derivedDataset: {
      type: 'join',
      sourceDataIds: [left.datasetId, right.datasetId],
      operationId
    }
  };
}

/**
 * Native semantics (kepler.gl table/dataset-ops/join.js): keys compare as exact strings; null and
 * empty keys never match; each left row emits its matching right rows in right order; LEFT keeps
 * unmatched left rows; FULL appends unmatched right rows in right order. Pure and CPU-only over
 * dictionary codes, so the same function verifies against the pinned native implementation. It is
 * the test oracle only: the executor joins on the GPU (see resident-join-gpu-reference.ts for the
 * step-for-step CPU mirror of that pipeline).
 */
export function planResidentJoinPairs(props: {
  joinType: ResidentJoinType;
  leftCodes: Uint32Array;
  /** Nonzero means valid; omitted means every row is valid. */
  leftValidity?: Uint32Array;
  leftLabels: readonly string[];
  rightCodes: Uint32Array;
  rightValidity?: Uint32Array;
  rightLabels: readonly string[];
  /** Oracle-only output bound; the GPU executor has none. */
  maximumRows?: number;
  duplicateRightMatches?: ResidentJoinDuplicatePolicy;
}): {leftRows: Uint32Array; rightRows: Uint32Array; rowCount: number} {
  const {joinType, leftCodes, rightCodes, leftLabels, rightLabels} = props;
  const rightCodeByLabel = new Map<string, number>();
  for (const [code, label] of rightLabels.entries())
    if (label !== '' && !rightCodeByLabel.has(label)) rightCodeByLabel.set(label, code);
  const rightCodeForLeft = new Int32Array(leftLabels.length);
  for (const [code, label] of leftLabels.entries())
    rightCodeForLeft[code] = label === '' ? -1 : (rightCodeByLabel.get(label) ?? -1);
  // Stable counting sort of right rows by key code; the representative code owns duplicates.
  const offsets = new Uint32Array(rightLabels.length + 1);
  const rightKeyCode = new Int32Array(rightCodes.length);
  for (let row = 0; row < rightCodes.length; row++) {
    const code = rightCodes[row];
    let key = -1;
    if (!props.rightValidity || props.rightValidity[row]) {
      if (code >= rightLabels.length)
        throw new Error('Join right key has an invalid dictionary code');
      // CPU and GPU code agree on the first code for a repeated label.
      key = rightLabels[code] === '' ? -1 : rightCodeByLabel.get(rightLabels[code])!;
    }
    rightKeyCode[row] = key;
    if (key >= 0) offsets[key + 1]++;
  }
  for (let code = 0; code < rightLabels.length; code++) offsets[code + 1] += offsets[code];
  const cursor = offsets.slice(0, rightLabels.length);
  const sortedRows = new Uint32Array(offsets[rightLabels.length]);
  for (let row = 0; row < rightCodes.length; row++)
    if (rightKeyCode[row] >= 0) sortedRows[cursor[rightKeyCode[row]]++] = row;
  const matched = new Uint8Array(rightCodes.length);
  const leftKeyCode = new Int32Array(leftCodes.length);
  let rowCount = 0;
  for (let row = 0; row < leftCodes.length; row++) {
    let key = -1;
    if (!props.leftValidity || props.leftValidity[row]) {
      if (leftCodes[row] >= leftLabels.length)
        throw new Error('Join left key has an invalid dictionary code');
      key = rightCodeForLeft[leftCodes[row]];
    }
    leftKeyCode[row] = key;
    const matches = key >= 0 ? offsets[key + 1] - offsets[key] : 0;
    if (matches) {
      // Every right row of a matched key counts as matched, even when only one is emitted.
      for (let index = offsets[key]; index < offsets[key + 1]; index++)
        matched[sortedRows[index]] = 1;
      rowCount +=
        props.duplicateRightMatches && props.duplicateRightMatches !== 'all' ? 1 : matches;
    } else if (joinType !== 'INNER') rowCount++;
    if (rowCount > (props.maximumRows ?? Infinity))
      throw new Error(
        `Join output exceeds ${props.maximumRows} rows.${RESIDENT_JOIN_SQL_GUIDANCE}`
      );
  }
  if (joinType === 'FULL')
    for (let row = 0; row < rightCodes.length; row++) if (!matched[row]) rowCount++;
  if (rowCount > (props.maximumRows ?? Infinity))
    throw new Error(`Join output exceeds ${props.maximumRows} rows.${RESIDENT_JOIN_SQL_GUIDANCE}`);
  const leftRows = new Uint32Array(rowCount).fill(RESIDENT_JOIN_NO_ROW);
  const rightRows = new Uint32Array(rowCount).fill(RESIDENT_JOIN_NO_ROW);
  let output = 0;
  for (let row = 0; row < leftCodes.length; row++) {
    const key = leftKeyCode[row];
    let start = key >= 0 ? offsets[key] : 0;
    let end = key >= 0 ? offsets[key + 1] : 0;
    if (end > start && props.duplicateRightMatches === 'first') end = start + 1;
    else if (end > start && props.duplicateRightMatches === 'last') start = end - 1;
    if (end > start) {
      for (let index = start; index < end; index++) {
        leftRows[output] = row;
        rightRows[output++] = sortedRows[index];
      }
    } else if (joinType !== 'INNER') leftRows[output++] = row;
  }
  if (joinType === 'FULL')
    for (let row = 0; row < rightCodes.length; row++) if (!matched[row]) rightRows[output++] = row;
  return {leftRows, rightRows, rowCount};
}

function checkKey(
  source: ResidentJoinSourceMetadata,
  name: string,
  side: string
): 'string' | 'numeric' {
  const field = source.fields.find(candidate => candidate.name === name);
  const dictionary = Object.hasOwn(source.dictionaries, name)
    ? source.dictionaries[name]
    : undefined;
  const format = source.columns[name]?.format;
  if (!dictionary) {
    // Native compares String(value). Int32/Uint32 are exact; Float32 is verified integral at runtime.
    if (
      ((format === 'sint32' || format === 'uint32') && field?.type === 'integer') ||
      (format === 'float32' && (field?.type === 'integer' || field?.type === 'real'))
    )
      return 'numeric';
    reject(
      `Resident Join requires a prepared string dictionary or Int32/Uint32/integral Float32 ${side} key: ${name}.`
    );
  }
  if (field?.type !== 'string' || format !== 'uint32')
    reject(`Resident Join requires a prepared string dictionary ${side} key: ${name}.`);
  if (dictionary.ordered) reject('Resident Join requires unordered string dictionary keys.');
  if (dictionary.values.length > MAXIMUM_RESIDENT_JOIN_DICTIONARY_SLOTS)
    reject(
      `Join keys support at most ${MAXIMUM_RESIDENT_JOIN_DICTIONARY_SLOTS} dictionary labels.`
    );
  let bytes = 0;
  for (const value of dictionary.values) {
    bytes += value.length * 2;
    if (bytes > MAXIMUM_RESIDENT_JOIN_KEY_TEXT_BYTES)
      reject('Join key dictionary text exceeds 4 MiB.');
  }
  return 'string';
}

/**
 * Native compares String(value) of the original number. Int32/Uint32 are exact. A Float32 column
 * equals the original number only for integers up to 2^24 (a non-integer may be a downcast
 * Float64), so anything else is rejected rather than approximated. Output is dictionary codes with
 * the native string forms as labels, which feeds the same pair planner as string keys.
 */
export function makeResidentJoinNumericKey(
  words: Uint32Array,
  format: 'float32' | 'sint32' | 'uint32',
  validity?: Uint32Array
): {codes: Uint32Array; labels: string[]} {
  const view =
    format === 'float32'
      ? new Float32Array(words.buffer, words.byteOffset, words.length)
      : format === 'sint32'
        ? new Int32Array(words.buffer, words.byteOffset, words.length)
        : words;
  const codes = new Uint32Array(words.length);
  const labels: string[] = [];
  const codeByLabel = new Map<string, number>();
  for (let row = 0; row < view.length; row++) {
    if (validity && !validity[row]) continue;
    const value = view[row];
    if (format === 'float32' && !(Number.isInteger(value) && Math.abs(value) <= 16_777_216))
      reject('Join Float32 keys must be integers within +/-16777216 to match native string forms.');
    const label = String(value);
    let code = codeByLabel.get(label);
    if (code === undefined) {
      code = labels.length;
      labels.push(label);
      codeByLabel.set(label, code);
    }
    codes[row] = code;
  }
  return {codes, labels};
}

function pick(
  source: ResidentJoinSourceMetadata,
  name: string,
  side: 'left' | 'right'
): Omit<ResidentJoinColumn, 'outputColumn'>[] {
  const field = source.fields.find(candidate => candidate.name === name);
  if (!field) return [];
  const format = source.columns[name]?.format;
  const dictionary = Object.hasOwn(source.dictionaries, name)
    ? source.dictionaries[name]
    : undefined;
  const supported =
    (format === 'float32' && !dictionary && ['integer', 'real'].includes(field.type)) ||
    (format === 'sint32' && !dictionary && field.type === 'integer') ||
    (format === 'uint32' &&
      (dictionary ? field.type === 'string' && !dictionary.ordered : field.type === 'integer'));
  if (!supported)
    reject(
      `Resident Join cannot carry ${side} column ${name} (${field.type}); deselect it in Columns to include.`
    );
  return [
    {
      side,
      sourceColumn: name,
      type: field.type,
      analyzerType: field.analyzerType || field.type,
      format: format as ResidentJoinColumnFormat,
      dictionary: Boolean(dictionary)
    }
  ];
}

/** A colliding name gets the suffix, then a counter; a free name is kept as is. */
function uniqueSuffixedName(base: string, suffix: string, used: Set<string>): string {
  let name = used.has(base) ? `${base}${suffix}` : base;
  for (let counter = 2; used.has(name); counter++) name = `${base}${suffix}${counter}`;
  used.add(name);
  return name;
}
function uniqueName(base: string, used: Set<string>): string {
  let name = base;
  let suffix = 1;
  while (used.has(name)) name = `${base}_${suffix++}`;
  used.add(name);
  return name;
}
function readText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim() || value.length > 4096)
    reject(`Invalid Join ${label}.`);
  return value;
}
function readInteger(value: unknown, minimum: number, maximum: number, label: string): number {
  if (
    typeof value !== 'number' ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  )
    reject(`Join ${label} must be between ${minimum} and ${maximum}.`);
  return value;
}
function reject(message: string): never {
  throw new Error(`${message}${RESIDENT_JOIN_SQL_GUIDANCE}`);
}

/**
 * Dictionary-sized (not row-sized) key tables for the GPU string-key path. Both tables map a
 * dictionary code to a shared key: the first right code carrying the same label, or
 * `RESIDENT_JOIN_NO_KEY` for empty labels and labels with no right counterpart. This is the same
 * representative-code rule as `planResidentJoinPairs`.
 */
export function makeResidentJoinDictionaryKeyTables(
  leftLabels: readonly string[],
  rightLabels: readonly string[]
): {left: Uint32Array; right: Uint32Array} {
  const rightCodeByLabel = new Map<string, number>();
  for (const [code, label] of rightLabels.entries())
    if (label !== '' && !rightCodeByLabel.has(label)) rightCodeByLabel.set(label, code);
  return {
    left: Uint32Array.from(
      leftLabels,
      label => rightCodeByLabel.get(label) ?? RESIDENT_JOIN_NO_KEY
    ),
    right: Uint32Array.from(rightLabels, label =>
      label === '' ? RESIDENT_JOIN_NO_KEY : rightCodeByLabel.get(label)!
    )
  };
}

/**
 * Numeric keys compare as String(value), i.e. by mathematical integer. Int32 and integral Float32
 * keys share the int32 bit pattern, and Uint32 bits equal that pattern for values below 2^31. A
 * Uint32 value of 2^31 or more can only equal another Uint32 value, so when exactly one side is
 * Uint32 its high values must never match (they would alias negative Int32 bit patterns).
 */
export function shouldDropResidentJoinHighKeys(
  side: ResidentJoinColumnFormat,
  other: ResidentJoinColumnFormat
): boolean {
  return side === 'uint32' && other !== 'uint32';
}
