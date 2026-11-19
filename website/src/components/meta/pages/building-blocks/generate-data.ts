// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes building-blocks.json, the recorded facts the building-blocks page draws beyond the view
// models. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/building-blocks/generate-data.ts
//
// - Twenty rows of the time-window-filter record (every 75th of 1,500), with their recorded
//   timestamps, recorded mask and the slot each kept row has in the recorded `ids` output.
// - The recorded mask summed over 16 slices of the 1,500 rows (the leaves of the reduction figure).
// - The 42 recorded kernel ids of cell aggregation, which its collapsed view model folds.
// - Per building block, how many of the records run it and how many source files construct it.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';

type RecordNode = {id: string; label: string; op?: string};
type RecordCapture = {name: string; values: ArrayLike<number>};
type RecordLike = {
  graphs: readonly {form: string; nodes: readonly RecordNode[]}[];
  captures?: {snapshots: readonly RecordCapture[]};
};

const pageDirectory = path.dirname(fileURLToPath(import.meta.url));
const repositoryDirectory = path.resolve(pageDirectory, '../../../../../..');
const recordsDirectory = path.join(repositoryDirectory, 'docs/lens-records');
const sourceDirectory = path.join(repositoryDirectory, 'modules/analysis/src');

function readRecord(scenarioId: string): RecordLike {
  const text = fs.readFileSync(path.join(recordsDirectory, `${scenarioId}.json`), 'utf8');
  return fromRecordedLensFile(JSON.parse(text)).record as unknown as RecordLike;
}

function getCompiledGraph(record: RecordLike) {
  return record.graphs.find(graph => graph.form === 'compiled') ?? record.graphs[0];
}

function getCapture(record: RecordLike, name: string): number[] {
  const capture = record.captures?.snapshots.find(snapshot => snapshot.name === name);
  if (!capture) throw new Error(`missing capture ${name}`);
  return Array.from(capture.values);
}

// Time window filter

const TOY_ROW_STRIDE = 75;
const TOY_ROW_COUNT = 20;
const REDUCTION_LEAF_COUNT = 16;

const timeWindowRecord = readRecord('gpu-dataframe.time-window-filter');
const timestamps = getCapture(timeWindowRecord, 'timestamps');
const outputMask = getCapture(timeWindowRecord, 'outputMask');
const packedRows = getCapture(timeWindowRecord, 'rows');
const [keptCount] = getCapture(timeWindowRecord, 'count');
const slotByRow = new Map<number, number>();
for (let slot = 0; slot < keptCount; slot++) slotByRow.set(packedRows[slot], slot);

const toyRows = Array.from({length: TOY_ROW_COUNT}, (_, index) => {
  const row = index * TOY_ROW_STRIDE;
  return {
    row,
    timestamp: timestamps[row],
    flag: outputMask[row] ? 1 : 0,
    // Slot of this row in the recorded, packed `rows` output; -1 when the row was not kept.
    recordedSlot: slotByRow.get(row) ?? -1
  };
});

const reductionLeaves = Array.from({length: REDUCTION_LEAF_COUNT}, (_, leaf) => {
  const begin = Math.round((leaf * outputMask.length) / REDUCTION_LEAF_COUNT);
  const end = Math.round(((leaf + 1) * outputMask.length) / REDUCTION_LEAF_COUNT);
  let kept = 0;
  for (let row = begin; row < end; row++) kept += outputMask[row] ? 1 : 0;
  return {begin, end, kept};
});

// Cell aggregation

const cellAggregationKernels = getCompiledGraph(
  readRecord('gpu-spatial-analysis.cell-aggregation')
).nodes.map(node => node.id);

// Building blocks over all records. Each block is recognised by the operator name the record
// carries or by the node ids the published luma.gl class emits (@luma.gl/graph/dist/gpu-core):
//   GPUScan       op `GPUScan`, or ids `<id>-level-<n>-scan` (gpu-scan.js:224)
//   GPUReduction  `<id>-level-0` beside `<id>-finalize`, optionally `-chunk-<n>` (gpu-reduction.js:139-209)
//   GPUCompaction op `GPUCompaction`, or `<id>-normalize-<n>` beside `<id>-scatter` (gpu-compaction.js:99,185)
//   GPUSort       `<id>-radix-digit-<bits>-histogram` or `<id>-bitonic-local` (gpu-sort.js:255,581)
//   GPUGridIndex  `<id>-initialize-ranks-<n>` beside `<id>-terminal` (gpu-grid-index-internals.js:30,106)

type BlockId = 'reduction' | 'scan' | 'compaction' | 'sort' | 'gridIndex';

function getBlocks(nodes: readonly RecordNode[]): Set<BlockId> {
  const nodeIds = new Set(nodes.map(node => node.id));
  const blocks = new Set<BlockId>();
  for (const node of nodes) {
    const {id, op} = node;
    if (op === 'GPUScan' || /-level-\d+-scan$/.test(id)) blocks.add('scan');
    if (op === 'GPUCompaction') blocks.add('compaction');
    const normalize = id.match(/^(.*)-normalize-\d+$/);
    if (normalize && nodeIds.has(`${normalize[1]}-scatter`)) blocks.add('compaction');
    const level = id.match(/^(.*?)(-chunk-\d+)?-level-0$/);
    if (level && nodeIds.has(`${level[1]}-finalize`)) blocks.add('reduction');
    if (/-radix-digit-\d+-histogram$/.test(id) || /-bitonic-local$/.test(id)) blocks.add('sort');
    const ranks = id.match(/^(.*)-initialize-ranks-\d+$/);
    if (ranks && nodeIds.has(`${ranks[1]}-terminal`)) blocks.add('gridIndex');
  }
  return blocks;
}

const recordCounts: Record<BlockId, number> = {
  reduction: 0,
  scan: 0,
  compaction: 0,
  sort: 0,
  gridIndex: 0
};
const recordExamples: Record<BlockId, string[]> = {
  reduction: [],
  scan: [],
  compaction: [],
  sort: [],
  gridIndex: []
};
let recordCount = 0;
for (const file of fs.readdirSync(recordsDirectory).sort()) {
  if (!file.endsWith('.json')) continue;
  recordCount++;
  const scenarioId = file.replace(/\.json$/, '');
  for (const block of getBlocks(getCompiledGraph(readRecord(scenarioId)).nodes)) {
    recordCounts[block]++;
    recordExamples[block].push(scenarioId);
  }
}

// Source files (tests excluded) that construct each class, and those that use the merge-path loop.
const SOURCE_PATTERNS: Record<BlockId | 'loadBalanced', RegExp> = {
  reduction: /new GPUReduction\(/,
  scan: /new GPUScan\(/,
  compaction: /new GPUCompaction\(/,
  sort: /new GPUSort\(/,
  gridIndex: /new GPUGridIndex\(/,
  loadBalanced: /from '[./]*(analysis-core\/kernels\/)?load-balanced-for-each'/
};
const sourceCounts = Object.fromEntries(
  Object.keys(SOURCE_PATTERNS).map(key => [key, 0])
) as Record<keyof typeof SOURCE_PATTERNS, number>;
function visitSource(directory: string): void {
  for (const entry of fs.readdirSync(directory, {withFileTypes: true})) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== 'test' && entry.name !== 'tests') visitSource(entryPath);
    } else if (/\.ts$/.test(entry.name) && !/\.(spec|test)\.ts$/.test(entry.name)) {
      const text = fs.readFileSync(entryPath, 'utf8');
      for (const [key, pattern] of Object.entries(SOURCE_PATTERNS)) {
        if (pattern.test(text)) sourceCounts[key as keyof typeof SOURCE_PATTERNS]++;
      }
    }
  }
}
visitSource(sourceDirectory);

const output = {
  timeWindow: {
    rowCount: timestamps.length,
    keptCount,
    stride: TOY_ROW_STRIDE,
    rows: toyRows
  },
  reductionLeaves,
  cellAggregationKernels,
  blocks: {
    recordCount,
    records: recordCounts,
    sourceFiles: sourceCounts
  }
};
fs.writeFileSync(path.join(pageDirectory, 'building-blocks.json'), `${JSON.stringify(output)}\n`);
process.stdout.write(`${JSON.stringify(output.blocks)}\n`);
for (const [block, examples] of Object.entries(recordExamples)) {
  process.stdout.write(`${block} ${examples.length}: ${examples.slice(0, 40).join(' ')}\n`);
}
