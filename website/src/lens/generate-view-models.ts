// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Regenerates src/lens/generated/*.json from docs/lens-records/*.json:
//   node --experimental-strip-types src/lens/generate-view-models.ts   (from website/)
// Only compact view models are written; the record files never ship.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';
import {extractLensViewModel, type LensRecordLike} from './extract-lens-view-model.ts';
import type {LensSummary} from './types.ts';

const lensDirectory = path.dirname(fileURLToPath(import.meta.url));
const recordsDirectory = path.resolve(lensDirectory, '../../../docs/lens-records');
const outputDirectory = path.join(lensDirectory, 'generated');

// Pass scenario ids to regenerate only those (nothing is deleted); no arguments rebuilds them all.
const onlyScenarioIds = new Set(process.argv.slice(2));
fs.mkdirSync(outputDirectory, {recursive: true});
if (onlyScenarioIds.size === 0) {
  for (const file of fs.readdirSync(outputDirectory)) {
    if (file.endsWith('.json')) fs.unlinkSync(path.join(outputDirectory, file));
  }
}

let written = 0;
let totalBytes = 0;
let recordBytes = 0;
const kernelCounts: {id: string; kernels: number}[] = [];
const summary: LensSummary = {
  recordCount: 0,
  packs: {},
  kernels: 0,
  copies: 0,
  gatedKernels: 0,
  dependencies: {raw: 0, war: 0, waw: 0},
  buffers: 0,
  bufferBytes: 0,
  snapshots: 0,
  formulas: 0,
  recordsWithFormulas: 0,
  kernelsPerRecord: {minimum: 0, median: 0, maximum: 0},
  smallest: {id: '', kernels: 0},
  largest: {id: '', kernels: 0},
  recordBytes: 0,
  viewModelBytes: 0
};
const failures: string[] = [];
for (const file of fs.readdirSync(recordsDirectory).sort()) {
  if (!file.endsWith('.json')) continue;
  const scenarioId = file.replace(/\.json$/, '');
  if (onlyScenarioIds.size > 0 && !onlyScenarioIds.has(scenarioId)) continue;
  try {
    const fileText = fs.readFileSync(path.join(recordsDirectory, file), 'utf8');
    const envelope = fromRecordedLensFile(JSON.parse(fileText));
    const record = envelope.record as unknown as LensRecordLike;
    const model = extractLensViewModel(scenarioId, record);
    recordBytes += fileText.length;
    summarizeRecord(scenarioId, record);
    const text = JSON.stringify(model);
    fs.writeFileSync(path.join(outputDirectory, `${scenarioId}.json`), text);
    written++;
    totalBytes += text.length;
  } catch (error) {
    failures.push(`${file}: ${(error as Error).message}`);
  }
}
if (onlyScenarioIds.size === 0) {
  kernelCounts.sort((left, right) => left.kernels - right.kernels);
  summary.kernelsPerRecord = {
    minimum: kernelCounts[0]?.kernels ?? 0,
    median: kernelCounts[Math.floor(kernelCounts.length / 2)]?.kernels ?? 0,
    maximum: kernelCounts.at(-1)?.kernels ?? 0
  };
  summary.smallest = kernelCounts[0] ?? summary.smallest;
  summary.largest = kernelCounts.at(-1) ?? summary.largest;
  summary.recordBytes = recordBytes;
  summary.viewModelBytes = totalBytes;
  fs.writeFileSync(
    path.join(lensDirectory, 'summary.json'),
    `${JSON.stringify(summary, null, 2)}\n`
  );
}

/** Adds one record to the totals in summary.json (the numbers the how-it-works pages quote). */
function summarizeRecord(scenarioId: string, record: LensRecordLike): void {
  const graph = record.graphs.find(candidate => candidate.form === 'compiled') ?? record.graphs[0];
  const pack = scenarioId.slice(0, scenarioId.indexOf('.'));
  const kernels = graph.nodes.filter(node => node.kind === 'compute').length;
  summary.recordCount++;
  summary.packs[pack] = (summary.packs[pack] ?? 0) + 1;
  summary.kernels += kernels;
  summary.copies += graph.nodes.filter(node => node.kind === 'copy').length;
  summary.gatedKernels += graph.nodes.filter(
    node => node.condition?.source === 'gpu' && node.condition.mode === 'indirect'
  ).length;
  for (const edge of graph.edges) {
    if (edge.kind === 'raw' || edge.kind === 'war' || edge.kind === 'waw') {
      summary.dependencies[edge.kind]++;
    }
  }
  summary.buffers += graph.resources?.length ?? 0;
  for (const resource of graph.resources ?? []) summary.bufferBytes += resource.bytes;
  summary.snapshots += record.captures?.snapshots.length ?? 0;
  const formulaCount = record.refs?.formulas?.length ?? 0;
  summary.formulas += formulaCount;
  if (formulaCount > 0) summary.recordsWithFormulas++;
  kernelCounts.push({id: scenarioId, kernels});
}

// biome-ignore lint/suspicious/noConsole: diagnostic output
console.log(`wrote ${written} view models, ${(totalBytes / 1e6).toFixed(2)} MB`);
if (failures.length) {
  // biome-ignore lint/suspicious/noConsole: diagnostic output
  console.error(failures.join('\n'));
  process.exitCode = 1;
}
