// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {
  makeResidentJoinRankedCompositeKeyPlan,
  type ResidentJoinCompositeKeyPlan
} from './resident-join-composite-key';
import {shouldDropResidentJoinHighKeys} from './resident-join-plan';
import {findResidentSqlDistinctKeys} from './resident-sql-distinct';
import {SqlExecutionRefusal} from './sql-backend-router';

type KeyFormat = 'float32' | 'sint32' | 'uint32';

/**
 * WGSL for the canonical join key of `word`, exactly as the join key pass decodes it: integral
 * Float32 become their Int32 bits, uint32 above Int32 range is dropped when the other side is
 * signed. Unusable cells become 0xffffffff, which only adds one unused rank.
 */
export function getResidentJoinCanonicalKeySource(format: KeyFormat, other: KeyFormat): string {
  if (format === 'float32')
    return `  var key: u32 = 0xffffffffu;
  let keyValue = bitcast<f32>(word);
  if (abs(keyValue) <= 16777216.0 && keyValue == floor(keyValue)) { key = bitcast<u32>(i32(keyValue)); }`;
  if (format === 'uint32' && shouldDropResidentJoinHighKeys(format, other))
    return '  var key: u32 = 0xffffffffu;\n  if (word < 0x80000000u) { key = word; }';
  return '  let key: u32 = word;';
}

/**
 * Makes numeric multi-column join keys exact. Per numeric component a GPU distinct pass (sort,
 * run flags, scan, compaction) finds the canonical keys of each side; their sorted union becomes
 * the rank table, so the composite key is a mixed radix of ranks and shared dictionary codes and
 * equal composite keys mean equal source keys. Returns undefined when a column has more than the
 * distinct limit or the radix product exceeds one 32-bit word (the caller then falls back to the
 * hashed INNER path or DuckDB).
 */
export async function rankResidentJoinCompositeKeyPlan(props: {
  device: Device;
  plan: ResidentJoinCompositeKeyPlan;
  getColumn: (
    side: 'left' | 'right',
    column: string
  ) => {vector: GPUVector; validity?: GPUVector<'uint32'>};
}): Promise<ResidentJoinCompositeKeyPlan | undefined> {
  const ranks: (Uint32Array | undefined)[] = [];
  for (const component of props.plan.components) {
    if (component.kind !== 'numeric') {
      ranks.push(undefined);
      continue;
    }
    const union = new Set<number>();
    for (const side of ['left', 'right'] as const) {
      const format = (side === 'left' ? component.leftFormat : component.rightFormat) as KeyFormat;
      const other = (side === 'left' ? component.rightFormat : component.leftFormat) as KeyFormat;
      const column = props.getColumn(
        side,
        side === 'left' ? component.leftColumn : component.rightColumn
      );
      try {
        const keys = await findResidentSqlDistinctKeys({
          device: props.device,
          vector: column.vector,
          validity: column.validity,
          format,
          keySource: getResidentJoinCanonicalKeySource(format, other)
        });
        for (const key of keys) union.add(key);
      } catch (error) {
        if (error instanceof SqlExecutionRefusal) return undefined;
        throw error;
      }
    }
    ranks.push(Uint32Array.from([...union].sort((a, b) => a - b)));
  }
  return makeResidentJoinRankedCompositeKeyPlan(props.plan, ranks);
}
