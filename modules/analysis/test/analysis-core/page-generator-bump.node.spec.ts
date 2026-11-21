// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import typescript from 'typescript';
import {expect, it} from 'vitest';

const PACKED_ROW_FIXTURE = [1, 3, 5, 0];

// Execute the actual pure generator blocks with small independently worked fixtures.
// No lens record is read, no generator is launched, and no website file is written.
function readGenerator(page: string): string {
  return readFileSync(
    new URL(
      `../../../../website/src/components/meta/pages/${page}/generate-data.ts`,
      import.meta.url
    ),
    'utf8'
  );
}

function compileBlock(source: string, start: string, end: string): string {
  const first = source.indexOf(start);
  const last = source.indexOf(end, first);
  expect(first).toBeGreaterThanOrEqual(0);
  expect(last).toBeGreaterThan(first);
  return typescript.transpileModule(source.slice(first, last), {
    compilerOptions: {target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.None}
  }).outputText;
}

it('contract generator counts mixed findings and excludes baseline metadata', () => {
  const source = readGenerator('contract');
  const run = new Function(
    'baseline',
    `${compileBlock(source, 'const ruleEntries', '// Status ports:')}\nreturn rules;`
  );
  const result = run({
    _version: 2,
    _futureMetadata: {ignored: [1, 2, 3]},
    numeric: {'first.ts': 2, 'second.ts': 3},
    entries: {'third.ts': [{symbol: 'one'}, {symbol: 'two'}]},
    empty: {}
  });
  // Two numeric files have 2+3 findings, one structured list has two, metadata has none.
  expect(result).toEqual([
    {id: 'numeric', files: 2, findings: 5},
    {id: 'entries', files: 1, findings: 2},
    {id: 'empty', files: 0, findings: 0}
  ]);
});

it('building-blocks generator maps packed rows to slots and sums each mask slice', () => {
  const source = readGenerator('building-blocks');
  expect(source).toMatch(/const packedRows = getCapture\(timeWindowRecord, 'rows'\)/);
  const run = new Function(
    'outputMask',
    'timestamps',
    'packedRows',
    'keptCount',
    'TOY_ROW_STRIDE',
    'TOY_ROW_COUNT',
    'REDUCTION_LEAF_COUNT',
    `${compileBlock(source, 'const slotByRow', '// Cell aggregation')}\nreturn {toyRows, reductionLeaves};`
  );
  const result = run([0, 1, 0, 1, 0, 1], [10, 20, 30, 40, 50, 60], PACKED_ROW_FIXTURE, 3, 1, 6, 2);
  expect(result).toEqual({
    toyRows: [
      {row: 0, timestamp: 10, flag: 0, recordedSlot: -1},
      {row: 1, timestamp: 20, flag: 1, recordedSlot: 0},
      {row: 2, timestamp: 30, flag: 0, recordedSlot: -1},
      {row: 3, timestamp: 40, flag: 1, recordedSlot: 1},
      {row: 4, timestamp: 50, flag: 0, recordedSlot: -1},
      {row: 5, timestamp: 60, flag: 1, recordedSlot: 2}
    ],
    reductionLeaves: [
      {begin: 0, end: 3, kept: 1},
      {begin: 3, end: 6, kept: 2}
    ]
  });
});

it('R7 packed-row fixture catches reading capacity instead of keptCount', () => {
  const source = readGenerator('building-blocks');
  const block = compileBlock(source, 'const slotByRow', '// Cell aggregation');
  const mutant = block.replace('slot < keptCount', 'slot < packedRows.length');
  expect(mutant).not.toBe(block);
  const run = new Function(
    'outputMask',
    'timestamps',
    'packedRows',
    'keptCount',
    'TOY_ROW_STRIDE',
    'TOY_ROW_COUNT',
    'REDUCTION_LEAF_COUNT',
    `${mutant}\nreturn toyRows[0].recordedSlot;`
  );
  expect(run([0, 1, 0, 1, 0, 1], [10, 20, 30, 40, 50, 60], PACKED_ROW_FIXTURE, 3, 1, 6, 2)).toBe(3);
});
