// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  canonicalizeResidentJoinKeys,
  type ResidentJoinKeyMode
} from './resident-join-gpu-reference';
import {addResidentJoinPass} from './resident-join-gpu-kernels';
import {
  createResidentJoinPlan,
  makeResidentJoinDictionaryKeyTables,
  RESIDENT_JOIN_NO_KEY,
  shouldDropResidentJoinHighKeys,
  type ResidentJoinColumnFormat,
  type ResidentJoinSourceMetadata
} from './resident-join-plan';

/**
 * Multi-column join keys. The executor joins on one 32-bit key word per row, so several ON
 * equalities are folded into that word on the GPU before the existing key pass runs:
 * - string components map their dictionary codes to a representative code shared by both sides
 *   and combine as a mixed radix number (exact);
 * - numeric components feed a 32-bit hash, reduced to `hashModulus` buckets and placed above the
 *   radix part. Hash buckets can collide, so numeric composites are only exact once the planner's
 *   residual equality check (the composed WHERE) removes false matches.
 */
export const MAXIMUM_RESIDENT_JOIN_COMPOSITE_KEYS = 4;
/** Hidden 0/1 result column of a hashed composite join: every numeric key part is equal. */
export const RESIDENT_JOIN_COMPOSITE_MATCH_COLUMN = '__luma_sql_join_match';
const HASH_SEED = 0x811c9dc5;
const HASH_PRIME = 0x01000193;

export type ResidentJoinCompositeComponent = {
  readonly kind: 'string' | 'numeric';
  readonly leftColumn: string;
  readonly rightColumn: string;
  readonly leftFormat: ResidentJoinColumnFormat;
  readonly rightFormat: ResidentJoinColumnFormat;
  /** String components: dictionary code to shared representative code, per side. */
  readonly leftTable?: Uint32Array;
  readonly rightTable?: Uint32Array;
  /** String components: number of distinct shared codes (the radix of this digit). */
  readonly radixSize: number;
  /** String components: place value of this digit in the radix part. */
  readonly radixStride: number;
  /**
   * Numeric components made exact: sorted distinct canonical key words of both sides. The digit is
   * the rank of the row's key in this list (shared by both sides), so equal digits mean equal keys.
   */
  readonly rankKeys?: Uint32Array;
};

export type ResidentJoinCompositeKeyPlan = {
  readonly components: readonly ResidentJoinCompositeComponent[];
  /** True when no numeric component is hashed: equal composite keys mean equal source keys. */
  readonly exact: boolean;
  /** Product of the string radixes (1 without strings). */
  readonly radixProduct: number;
  /** Hash buckets above the radix part (1 when exact). */
  readonly hashModulus: number;
};

export type ResidentJoinCompositeSideColumns = readonly {
  words: Uint32Array;
  validity?: Uint32Array;
}[];

/** Hash step shared by the WGSL pass and the CPU mirror. */
export function mixResidentJoinCompositeHash(hash: number, word: number): number {
  const mixed = Math.imul((hash ^ word) >>> 0, HASH_PRIME) >>> 0;
  return (mixed ^ (mixed >>> 15)) >>> 0;
}

/** Final avalanche applied once before reducing the hash to buckets. */
export function finishResidentJoinCompositeHash(hash: number): number {
  let value = (hash ^ (hash >>> 16)) >>> 0;
  value = Math.imul(value, 0x85ebca6b) >>> 0;
  return (value ^ (value >>> 13)) >>> 0;
}

/**
 * Plans the composite key for column pairs (left column, right column) of the join plan's sides.
 * Throws the refusal reason when a pair cannot be keyed or the string key space is too large.
 */
export function makeResidentJoinCompositeKeyPlan(props: {
  pairs: readonly {leftColumn: string; rightColumn: string}[];
  left: ResidentJoinSourceMetadata;
  right: ResidentJoinSourceMetadata;
}): ResidentJoinCompositeKeyPlan {
  const {pairs, left, right} = props;
  // Per-pair checks reuse the single key admission of the join planner.
  const components: ResidentJoinCompositeComponent[] = [];
  let radixProduct = 1;
  for (const pair of pairs) {
    createResidentJoinPlan(
      {
        id: 'composite-key-check',
        leftDataId: left.datasetId,
        rightDataId: right.datasetId,
        leftField: pair.leftColumn,
        rightField: pair.rightColumn,
        type: 'INNER',
        leftColumns: [pair.leftColumn],
        rightColumns: [],
        allowSameDataset: true,
        resultId: 'composite-key-check-result',
        resultLabel: 'composite key check'
      },
      left,
      right
    );
    const leftDictionary = Object.hasOwn(left.dictionaries, pair.leftColumn)
      ? left.dictionaries[pair.leftColumn]
      : undefined;
    const rightDictionary = Object.hasOwn(right.dictionaries, pair.rightColumn)
      ? right.dictionaries[pair.rightColumn]
      : undefined;
    const leftFormat = (left.columns[pair.leftColumn]?.format ??
      'uint32') as ResidentJoinColumnFormat;
    const rightFormat = (right.columns[pair.rightColumn]?.format ??
      'uint32') as ResidentJoinColumnFormat;
    if (Boolean(leftDictionary) !== Boolean(rightDictionary))
      throw new Error('Join keys must both be string dictionaries or both be numbers.');
    if (leftDictionary && rightDictionary) {
      const tables = makeResidentJoinDictionaryKeyTables(
        leftDictionary.values,
        rightDictionary.values
      );
      const radixSize = Math.max(1, rightDictionary.values.length);
      components.push({
        kind: 'string',
        leftColumn: pair.leftColumn,
        rightColumn: pair.rightColumn,
        leftFormat,
        rightFormat,
        leftTable: tables.left,
        rightTable: tables.right,
        radixSize,
        radixStride: radixProduct
      });
      radixProduct *= radixSize;
      if (radixProduct >= 0xffffffff)
        throw new Error(
          'Multi-column string join keys have too many label combinations to fold into one 32-bit key'
        );
    } else {
      components.push({
        kind: 'numeric',
        leftColumn: pair.leftColumn,
        rightColumn: pair.rightColumn,
        leftFormat,
        rightFormat,
        radixSize: 1,
        radixStride: 0
      });
    }
  }
  const exact = components.every(component => component.kind === 'string');
  return {
    components,
    exact,
    radixProduct,
    hashModulus: exact ? 1 : Math.floor(0xffffffff / radixProduct)
  };
}

/** CPU mirror of the composite key passes: one key word and an ok flag per row. */
export function computeResidentJoinCompositeKeys(
  plan: ResidentJoinCompositeKeyPlan,
  side: 'left' | 'right',
  columns: ResidentJoinCompositeSideColumns
): {key: Uint32Array; ok: Uint32Array; status: number} {
  const rowCount = columns[0].words.length;
  const ok = new Uint32Array(rowCount).fill(1);
  const radix = new Uint32Array(rowCount);
  const hash = new Uint32Array(rowCount).fill(HASH_SEED);
  let status = 0;
  plan.components.forEach((component, index) => {
    const {words, validity} = columns[index];
    const table = side === 'left' ? component.leftTable : component.rightTable;
    const format = side === 'left' ? component.leftFormat : component.rightFormat;
    const other = side === 'left' ? component.rightFormat : component.leftFormat;
    const numeric =
      component.kind === 'numeric'
        ? canonicalizeResidentJoinKeys({
            words,
            validity,
            mode: format,
            dropHighKeys: shouldDropResidentJoinHighKeys(format, other)
          })
        : undefined;
    if (numeric) status |= numeric.status;
    for (let row = 0; row < rowCount; row++) {
      let componentOk = 1;
      let value = 0;
      if (validity && !validity[row]) componentOk = 0;
      else if (numeric) {
        componentOk = numeric.ok[row];
        value = numeric.key[row];
        if (componentOk && component.rankKeys) {
          const rank = findResidentJoinRank(component.rankKeys, value);
          if (rank < 0) componentOk = 0;
          else value = rank;
        }
      } else if (words[row] >= table!.length) {
        status |= 2;
        componentOk = 0;
      } else if (table![words[row]] === RESIDENT_JOIN_NO_KEY) componentOk = 0;
      else value = table![words[row]];
      if (!componentOk) value = 0;
      ok[row] &= componentOk;
      if (numeric && !component.rankKeys)
        hash[row] = mixResidentJoinCompositeHash(hash[row], value);
      else radix[row] = (radix[row] + Math.imul(value, component.radixStride)) >>> 0;
    }
  });
  const key = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++)
    key[row] = plan.exact
      ? radix[row]
      : (Math.imul(
          finishResidentJoinCompositeHash(hash[row]) % plan.hashModulus,
          plan.radixProduct
        ) +
          radix[row]) >>>
        0;
  return {key, ok, status};
}

const literal = (value: number) => `${value >>> 0}u`;

/** Index of `key` in the ascending unsigned list, or -1. */
export function findResidentJoinRank(rankKeys: Uint32Array, key: number): number {
  let low = 0;
  let high = rankKeys.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (rankKeys[middle] < key) low = middle + 1;
    else high = middle;
  }
  return low < rankKeys.length && rankKeys[low] === key ? low : -1;
}

/**
 * Replaces the hashed numeric components with exact ranks. `rankKeys[index]` holds the sorted
 * distinct canonical keys of component `index` (undefined for string components). Returns undefined
 * when the mixed radix product no longer fits one 32-bit key word.
 */
export function makeResidentJoinRankedCompositeKeyPlan(
  plan: ResidentJoinCompositeKeyPlan,
  rankKeys: readonly (Uint32Array | undefined)[]
): ResidentJoinCompositeKeyPlan | undefined {
  let radixProduct = 1;
  const components: ResidentJoinCompositeComponent[] = [];
  for (const [index, component] of plan.components.entries()) {
    const keys = component.kind === 'numeric' ? rankKeys[index] : undefined;
    const radixSize = keys ? Math.max(1, keys.length) : component.radixSize;
    components.push({
      ...component,
      radixSize,
      radixStride: radixProduct,
      ...(keys ? {rankKeys: keys, leftTable: keys, rightTable: keys} : {})
    });
    radixProduct *= radixSize;
    if (radixProduct >= 0xffffffff) return undefined;
  }
  if (components.some(component => component.kind === 'numeric' && !component.rankKeys))
    return undefined;
  return {components, exact: true, radixProduct, hashModulus: 1};
}

/**
 * One component of a side's composite key: validity, decoding and accumulation into `keyWords`
 * (radix part), `hashes` (numeric parts) and `valid` (all components usable so far). Status bits
 * go to `summary[2]` like the single key pass: 1 non-integral Float32 key, 2 invalid code.
 */
export function addResidentJoinCompositeKeyPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    id: string;
    count: number;
    first: boolean;
    component: ResidentJoinCompositeComponent;
    side: 'left' | 'right';
    words: GraphDataView<'uint32'>;
    validity?: GraphDataView<'uint32'>;
    table?: GraphDataView<'uint32'>;
    keyWords: GraphDataView<'uint32'>;
    hashes?: GraphDataView<'uint32'>;
    valid: GraphDataView<'uint32'>;
    summary: GraphDataView<'uint32'>;
  }
): void {
  const {component, side} = props;
  const format: ResidentJoinKeyMode =
    side === 'left' ? component.leftFormat : component.rightFormat;
  const other = side === 'left' ? component.rightFormat : component.leftFormat;
  const table = side === 'left' ? component.leftTable : component.rightTable;
  let decode: string;
  if (component.kind === 'string')
    decode = `
    if (word >= ${literal(table!.length)}) { atomicOr(&summary[2], 2u); compOk = 0u; }
    else { comp = table[word]; if (comp == NO_KEY) { compOk = 0u; comp = 0u; } }`;
  else if (format === 'float32')
    decode = `
    let value = bitcast<f32>(word);
    if (abs(value) <= 16777216.0 && value == floor(value)) { comp = bitcast<u32>(i32(value)); }
    else { atomicOr(&summary[2], 1u); compOk = 0u; }`;
  else if (format === 'uint32' && shouldDropResidentJoinHighKeys(format, other))
    decode = `
    if (word >= 0x80000000u) { compOk = 0u; } else { comp = word; }`;
  else
    decode = `
    comp = word;`;
  const ranked = component.kind === 'numeric' && component.rankKeys;
  if (ranked)
    decode += `
    if (compOk == 1u) {
      var low = 0u;
      var high = ${literal(component.rankKeys!.length)};
      loop {
        if (low >= high) { break; }
        let middle = (low + high) / 2u;
        if (table[middle] < comp) { low = middle + 1u; } else { high = middle; }
      }
      if (low < ${literal(component.rankKeys!.length)} && table[low] == comp) { comp = low; } else { compOk = 0u; comp = 0u; }
    }`;
  const hashing = component.kind === 'numeric' && props.hashes && !ranked;
  addResidentJoinPass(graph, {
    id: props.id,
    count: props.count,
    bindings: {
      words: {view: props.words, access: 'read'},
      ...(props.validity ? {validity: {view: props.validity, access: 'read'}} : {}),
      ...(props.table ? {table: {view: props.table, access: 'read'}} : {}),
      keyWords: {view: props.keyWords, access: 'read-write'},
      ...(props.hashes ? {hashes: {view: props.hashes, access: 'read-write'}} : {}),
      valid: {view: props.valid, access: 'read-write'},
      summary: {view: props.summary, access: 'atomic'}
    },
    body: `
  var okValue = ${props.first ? '1u' : 'valid[index]'};
  var radix = ${props.first ? '0u' : 'keyWords[index]'};${
    props.hashes ? `\n  var hashValue = ${props.first ? literal(HASH_SEED) : 'hashes[index]'};` : ''
  }
  var compOk = 1u;
  var comp = 0u;
  // Static use keeps summary in the pipeline layout for components that never write it.
  let summaryWord = atomicLoad(&summary[3]);
  let word = words[index] + (summaryWord & 0u);
  if (${props.validity ? 'validity[index] == 0u' : 'false'}) { compOk = 0u; } else {${decode}
  }
  okValue = okValue * compOk;${
    hashing
      ? `\n  hashValue = (hashValue ^ comp) * ${literal(HASH_PRIME)};\n  hashValue = hashValue ^ (hashValue >> 15u);`
      : ''
  }${props.hashes ? '\n  hashes[index] = hashValue;' : ''}${
    component.kind === 'string' || ranked
      ? `\n  radix = radix + comp * ${literal(component.radixStride)};`
      : ''
  }
  keyWords[index] = radix;
  valid[index] = okValue;`
  });
}

/** Folds the hash into buckets above the radix part: key = bucket * radixProduct + radix. */
export function addResidentJoinCompositeFinishPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    id: string;
    count: number;
    plan: ResidentJoinCompositeKeyPlan;
    keyWords: GraphDataView<'uint32'>;
    hashes: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: props.id,
    count: props.count,
    bindings: {
      keyWords: {view: props.keyWords, access: 'read-write'},
      hashes: {view: props.hashes, access: 'read'}
    },
    body: `
  var hashValue = hashes[index];
  hashValue = hashValue ^ (hashValue >> 16u);
  hashValue = hashValue * 0x85ebca6bu;
  hashValue = hashValue ^ (hashValue >> 13u);
  keyWords[index] = (hashValue % ${literal(props.plan.hashModulus)}) * ${literal(props.plan.radixProduct)} + keyWords[index];`
  });
}
