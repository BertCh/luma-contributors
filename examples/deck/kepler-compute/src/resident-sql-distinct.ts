// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import type {GPUData, GPUVector} from '@luma-contributors/graph/gpu-data';
import {
  GPUCommandGraph,
  GPUScan,
  GPUSort,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {addResidentJoinPass} from './resident-join-gpu-kernels';
import {
  getResidentSqlOrderedKeySource,
  readResidentSqlOrderedKey
} from './resident-sql-cast-kernel';
import {SqlExecutionRefusal} from './sql-backend-router';
import {formatResidentSqlFloat} from './resident-sql-expressions';
import type {ResidentSqlScalarFormat} from './resident-sql-strings';

const STORAGE_USAGE = Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST | Buffer.VERTEX;

/** Upper bound on the distinct values a numeric column may have to become a dictionary. */
export const MAXIMUM_RESIDENT_SQL_DISTINCT_VALUES = 1_048_576;

export type ResidentSqlValueDictionary = {
  /** Sorted labels (code point order); the dictionary the cast column publishes. */
  readonly labels: readonly string[];
  /** Sorted ordered keys of the distinct values and the label code of each. */
  readonly keys: Uint32Array;
  readonly codes: Uint32Array;
};

function getBuffer(data: GPUData): Buffer {
  return data.buffer instanceof Buffer ? data.buffer : data.buffer.buffer;
}

/** DuckDB text of one raw cell (integers in decimal, floats shortest round trip). */
export function formatResidentSqlCell(format: ResidentSqlScalarFormat, word: number): string {
  const scratch = new DataView(new ArrayBuffer(4));
  scratch.setUint32(0, word >>> 0, true);
  if (format === 'float32') return formatResidentSqlFloat(scratch.getFloat32(0, true));
  return String(format === 'sint32' ? scratch.getInt32(0, true) : scratch.getUint32(0, true));
}

/** Builds the label dictionary for sorted ordered keys (pure; shared with the node oracle). */
export function makeResidentSqlValueDictionary(
  format: ResidentSqlScalarFormat,
  sortedKeys: readonly number[]
): ResidentSqlValueDictionary {
  const labelOfKey = sortedKeys.map(key =>
    formatResidentSqlCell(format, readResidentSqlOrderedKey(format, key))
  );
  const labels = [...new Set(labelOfKey)].sort((left, right) => {
    const a = Array.from(left);
    const b = Array.from(right);
    for (let index = 0; index < Math.min(a.length, b.length); index++) {
      const difference = a[index].codePointAt(0)! - b[index].codePointAt(0)!;
      if (difference) return difference;
    }
    return a.length - b.length;
  });
  const codeByLabel = new Map(labels.map((label, code) => [label, code]));
  return {
    labels,
    keys: Uint32Array.from(sortedKeys),
    codes: Uint32Array.from(labelOfKey, label => codeByLabel.get(label)!)
  };
}

/**
 * GPU distinct pass over a numeric column: per source chunk the cells become order-preserving
 * keys, are sorted (GPUSort), run starts are flagged and scanned (GPUScan) and unique keys are
 * compacted, so only the distinct keys are read back. Chunks are merged on the CPU. Rows with a
 * cleared validity bit sort as the maximum key, which can only add one unused dictionary label.
 * Chunks are processed one after another (each its own sort), so there is no per-chunk limit; throws
 * SqlExecutionRefusal when the merged dictionary exceeds {@link MAXIMUM_RESIDENT_SQL_DISTINCT_VALUES}.
 */
export async function findResidentSqlDistinctKeys(props: {
  device: Device;
  vector: GPUVector;
  validity?: GPUVector<'uint32'>;
  format: ResidentSqlScalarFormat;
  maximumCount?: number;
  /** WGSL defining `key: u32` from `word`; replaces the order-preserving key (equality use only). */
  keySource?: string;
}): Promise<number[]> {
  const {device, vector, validity, format} = props;
  const maximum = props.maximumCount ?? MAXIMUM_RESIDENT_SQL_DISTINCT_VALUES;
  const merged = new Set<number>();
  for (const [chunkIndex, chunk] of vector.data.entries()) {
    const count = chunk.length;
    if (!count) continue;
    const owned: Buffer[] = [];
    const graph = new GPUCommandGraph<undefined>(device, {
      id: `resident-sql-distinct-${chunkIndex}`
    });
    const allocate = (name: string, length: number) => {
      const buffer = device.createBuffer({
        id: `sql-distinct-${name}`,
        byteLength: Math.max(4, length * 4),
        usage: STORAGE_USAGE
      });
      owned.push(buffer);
      return buffer;
    };
    const handles = new Map<Buffer, ReturnType<typeof graph.importBuffer>>();
    const makeView = (buffer: Buffer, length: number, byteOffset = 0): GraphDataView<'uint32'> => {
      let handle = handles.get(buffer);
      if (!handle) {
        handle = graph.importBuffer(
          {id: `d-${handles.size}`, usage: buffer.usage, byteLength: buffer.byteLength},
          buffer
        );
        handles.set(buffer, handle);
      }
      return graph.createDataView(handle, {format: 'uint32', length, byteOffset});
    };
    const keys = allocate('keys', count);
    const values = allocate('values', count);
    const sortedKeys = allocate('sorted-keys', count);
    const sortedValues = allocate('sorted-values', count);
    const flags = allocate('flags', count);
    const included = allocate('included', count);
    const unique = allocate('unique', count);
    const validityChunk = validity?.data[chunkIndex];
    try {
      addResidentJoinPass(graph, {
        id: 'distinct-keys',
        count,
        bindings: {
          source: {view: makeView(getBuffer(chunk), count, chunk.byteOffset), access: 'read'},
          ...(validityChunk
            ? {
                validityIn: {
                  view: makeView(getBuffer(validityChunk), count, validityChunk.byteOffset),
                  access: 'read' as const
                }
              }
            : {}),
          keys: {view: makeView(keys, count), access: 'write'},
          values: {view: makeView(values, count), access: 'write'}
        },
        body: `  let word = source[index];
${props.keySource ?? getResidentSqlOrderedKeySource(format)}
${validityChunk ? '  if (validityIn[index] == 0u) { keys[index] = 0xffffffffu; } else { keys[index] = key; }' : '  keys[index] = key;'}
  values[index] = index;`
      });
      graph.add(
        new GPUSort({
          id: 'distinct-sort',
          keys: makeView(keys, count),
          values: makeView(values, count),
          outputKeys: makeView(sortedKeys, count),
          outputValues: makeView(sortedValues, count),
          keyBits: 32
        })
      );
      addResidentJoinPass(graph, {
        id: 'distinct-flags',
        count,
        bindings: {
          sorted: {view: makeView(sortedKeys, count), access: 'read'},
          flags: {view: makeView(flags, count), access: 'write'}
        },
        body: `  var flag: u32 = 1u;
  if (index > 0u && sorted[index] == sorted[index - 1u]) { flag = 0u; }
  flags[index] = flag;`
      });
      graph.add(
        new GPUScan({
          id: 'distinct-scan',
          input: makeView(flags, count),
          output: makeView(included, count),
          mode: 'inclusive'
        })
      );
      addResidentJoinPass(graph, {
        id: 'distinct-compact',
        count,
        bindings: {
          sorted: {view: makeView(sortedKeys, count), access: 'read'},
          flags: {view: makeView(flags, count), access: 'read'},
          included: {view: makeView(included, count), access: 'read'},
          unique: {view: makeView(unique, count), access: 'write'}
        },
        body: '  if (flags[index] == 1u) { unique[included[index] - 1u] = sorted[index]; }'
      });
      const compiled = graph.compile();
      try {
        const encoder = device.createCommandEncoder({id: 'resident-sql-distinct-encode'});
        compiled.encode(encoder, {parameters: undefined});
        device.submit(encoder.finish());
        const countBytes = await included.readAsync((count - 1) * 4, 4);
        const distinct = new Uint32Array(countBytes.buffer, countBytes.byteOffset, 1)[0];
        if (distinct > maximum)
          throw new SqlExecutionRefusal(
            `CAST to VARCHAR needs ${distinct} distinct values (GPU limit ${maximum})`
          );
        const uniqueBytes = await unique.readAsync(0, distinct * 4);
        const view = new Uint32Array(uniqueBytes.buffer, uniqueBytes.byteOffset, distinct);
        for (const key of view) merged.add(key);
        if (merged.size > maximum)
          throw new SqlExecutionRefusal(
            `CAST to VARCHAR needs more than ${maximum} distinct values (GPU limit)`
          );
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of owned) buffer.destroy();
    }
  }
  return [...merged].sort((left, right) => left - right);
}
