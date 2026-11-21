// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {WGSLKernelNodeRecord} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

/** Only reduction scheduling is serialized; retain emitted row maps and publication arithmetic.
 * These tiny fixtures have one chunk and one group. Device tests exercise cooperative trees.
 */
export function getSerialAuditBody(record: WGSLKernelNodeRecord): string {
  if (!record.body.includes('workgroupBarrier()')) return record.body;
  if (record.variant === 'response-anchor') {
    return `var minimum = vec2f(FLOAT32_MAXIMUM); var maximum = vec2f(-FLOAT32_MAXIMUM);
      for (var row = 0u; row < ITEM_COUNT; row++) {
        let item = readItem(row);
        if (item.z > 0.5) { minimum = min(minimum, item.xy); maximum = max(maximum, item.xy); }
      }
      let valid = minimum.x <= maximum.x && minimum.y <= maximum.y;
      boundsOut[boundsOutOffset] = select(0.0, minimum.x, valid);
      boundsOut[boundsOutOffset + 1u] = select(0.0, minimum.y, valid);
      boundsOut[boundsOutOffset + 2u] = select(0.0, maximum.x, valid);
      boundsOut[boundsOutOffset + 3u] = select(0.0, maximum.y, valid);`;
  }
  if (
    !/const GROUP_COUNT: u32 = 1u;/.test(record.declarations ?? '') ||
    !/const CHUNK_COUNT: u32 = 1u;/.test(record.declarations ?? '')
  )
    throw new Error(`${record.id}: serial audit supports only one group and chunk`);
  if (record.variant === 'grouped-partial') {
    const quantities = [...record.body.matchAll(/var sum(\d+) = 0.0;/g)].map(match =>
      Number(match[1])
    );
    const start = record.body.indexOf('if (row < end) {') + 'if (row < end) {'.length;
    const end = record.body.indexOf('\n    if (group >= GROUP_COUNT)');
    const rowMap = record.body.slice(start, end).replace(/}\s*$/, '');
    return `${quantities.map(quantity => `var sum${quantity} = 0.0;`).join('\n')}
      var count = 0u;
      for (var row = 0u; row < ROWS; row++) {
        var group = NO_GROUP;
        ${quantities.map(quantity => `var value${quantity} = 0.0;`).join('\n')}
        ${rowMap}
        if (group == 0u) {
          count++;
          ${quantities.map(quantity => `sum${quantity} += value${quantity};`).join('\n')}
        }
      }
      ${quantities.map(quantity => `partialSums[partialSumsOffset + ${quantity}u] = sum${quantity};`).join('\n')}
      ${record.bindings.some(binding => binding.name === 'partialCounts') ? 'partialCounts[partialCountsOffset] = count;' : ''}`;
  }
  if (record.variant === 'grouped-fold') {
    const start = record.body.indexOf('if (lane == 0u) {') + 'if (lane == 0u) {'.length;
    const body = record.body
      .slice(start)
      .replace(/}\s*$/, '')
      .replace(
        /sum(\d+) = reducedSums\[\d+u\];/g,
        (_, quantity) => `let sum${quantity} = partialSums[partialSumsOffset + ${quantity}u];`
      )
      .replace(
        'count = reducedCounts[0];',
        record.bindings.some(binding => binding.name === 'partialCounts')
          ? 'let count = partialCounts[partialCountsOffset];'
          : 'let count = 0u;'
      );
    return `let group = 0u; ${body}`;
  }
  throw new Error(`${record.id}: unsupported cooperative audit kernel`);
}
