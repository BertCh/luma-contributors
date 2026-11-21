// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Regenerates src/lens/generated/*.json from docs/lens-records/*.json:
//   node --experimental-strip-types src/lens/generate-view-models.ts   (from website/)
// Also refreshes summary.json and the lenses page metadata; record files never ship.
// Decode and extract every requested record before replacing any previous artifact. A rejected
// input leaves the previous models and metadata available for a corrected retry.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Buffer} from 'node:buffer';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';
import {extractLensViewModel, type LensRecordLike} from './extract-lens-view-model.ts';
import type {LensSummary} from './types.ts';
import {getResourceBytesKind} from './format.ts';
import {
  createLensPageData,
  validateLensModel,
  validateLensFileNames,
  type LensPageEntry
} from './generation-metadata.ts';

const lensDirectory = path.dirname(fileURLToPath(import.meta.url));
const recordsDirectory = path.resolve(lensDirectory, '../../../docs/lens-records');
const outputDirectory = path.join(lensDirectory, 'generated');
const pageDataFile = path.resolve(
  lensDirectory,
  '../components/meta/pages/lenses/lenses-data.json'
);

// Pass scenario ids to replace only those models; both modes refresh full-dataset metadata.
// No arguments rebuilds all models. Selected updates require a complete retained model set.
const onlyScenarioIds = new Set(process.argv.slice(2));
const pendingModels = new Map<string, string>();
const sourceSnapshots = new Map<
  string,
  {text: string; record: LensRecordLike; model: ReturnType<typeof extractLensViewModel>}
>();

function readScenarioSource(scenarioId: string) {
  const previous = sourceSnapshots.get(scenarioId);
  if (previous) return previous;
  const text = fs.readFileSync(path.join(recordsDirectory, `${scenarioId}.json`), 'utf8');
  const envelope = fromRecordedLensFile(JSON.parse(text));
  if (envelope.scenarioId !== scenarioId) {
    throw new Error(
      `Record filename identifies ${scenarioId}, but envelope identifies ${envelope.scenarioId}; use the matching scenario filename`
    );
  }
  const record = envelope.record as unknown as LensRecordLike;
  const snapshot = {text, record, model: extractLensViewModel(scenarioId, record)};
  sourceSnapshots.set(scenarioId, snapshot);
  return snapshot;
}

let written = 0;
let totalBytes = 0;
let recordBytes = 0;
let proposedModelBytes = 0;
const kernelCounts: {id: string; kernels: number}[] = [];
const summary: LensSummary = {
  recordCount: 0,
  packs: {},
  kernels: 0,
  copies: 0,
  gatedKernels: 0,
  dependencies: {raw: 0, war: 0, waw: 0},
  resourceCount: 0,
  inventoryKnown: true,
  resourceBytes: 0,
  resourceBytesKind: 'logical',
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
const foundScenarioIds = new Set<string>();
const proposedModels = new Map<string, string>();
if (onlyScenarioIds.size > 0) {
  for (const file of fs.readdirSync(outputDirectory).sort()) {
    if (file.endsWith('.json'))
      proposedModels.set(file, fs.readFileSync(path.join(outputDirectory, file), 'utf8'));
  }
}
const sourceFiles = fs
  .readdirSync(recordsDirectory)
  .filter(file => file.endsWith('.json'))
  .sort();
for (const file of sourceFiles) {
  if (!file.endsWith('.json')) continue;
  const scenarioId = file.replace(/\.json$/, '');
  if (onlyScenarioIds.size > 0 && !onlyScenarioIds.has(scenarioId)) continue;
  foundScenarioIds.add(scenarioId);
  try {
    const {model} = readScenarioSource(scenarioId);
    const text = JSON.stringify(model);
    pendingModels.set(`${scenarioId}.json`, text);
    proposedModels.set(`${scenarioId}.json`, text);
    written++;
    totalBytes += Buffer.byteLength(text, 'utf8');
  } catch (error) {
    failures.push(`${file}: ${(error as Error).message}`);
  }
}
for (const scenarioId of onlyScenarioIds) {
  if (!foundScenarioIds.has(scenarioId))
    failures.push(
      `Requested scenario ${scenarioId} has no source record; restore the record or remove this selection`
    );
}
// Validate retained JSON and featured source requirements against the complete proposed output
// before replacing any files. Standalone page generation uses this same metadata calculation.
let pageDataText = '';
if (failures.length === 0) {
  try {
    const entries: LensPageEntry[] = [];
    validateLensFileNames(sourceFiles, [...proposedModels.keys()]);
    for (const [file, text] of [...proposedModels].sort(([left], [right]) =>
      left.localeCompare(right)
    )) {
      const id = file.slice(0, -5);
      const model = JSON.parse(text);
      if (model.id !== id) throw new Error(`${file}: proposed model id must be ${id}`);
      const snapshot = readScenarioSource(id);
      const recordText = snapshot.text;
      const record = snapshot.record;
      validateLensModel(id, text, snapshot.model);
      const graph = record.graphs.find(graph => graph.form === 'compiled') ?? record.graphs[0];
      summarizeRecord(id, record);
      recordBytes += Buffer.byteLength(recordText, 'utf8');
      proposedModelBytes += Buffer.byteLength(text, 'utf8');
      entries.push({
        id,
        kernels: graph.nodes.filter(node => node.kind === 'compute').length,
        recordBytes: Buffer.byteLength(recordText, 'utf8'),
        viewModelBytes: Buffer.byteLength(text, 'utf8')
      });
    }
    pageDataText = `${JSON.stringify(createLensPageData(entries), null, 2)}\n`;
  } catch (error) {
    failures.push((error as Error).message);
  }
}
{
  kernelCounts.sort((left, right) => left.kernels - right.kernels);
  summary.kernelsPerRecord = {
    minimum: kernelCounts[0]?.kernels ?? 0,
    median: kernelCounts[Math.floor(kernelCounts.length / 2)]?.kernels ?? 0,
    maximum: kernelCounts.at(-1)?.kernels ?? 0
  };
  summary.smallest = kernelCounts[0] ?? summary.smallest;
  summary.largest = kernelCounts.at(-1) ?? summary.largest;
  summary.recordBytes = recordBytes;
  summary.viewModelBytes = proposedModelBytes;
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
  summary.resourceCount += graph.resources?.length ?? 0;
  for (const resource of graph.resources ?? []) summary.resourceBytes += resource.bytes;
  if (graph.resources === undefined) summary.inventoryKnown = false;
  summary.resourceBytesKind =
    summary.inventoryKnown === false
      ? 'unavailable'
      : getResourceBytesKind([{bytesKind: summary.resourceBytesKind}, ...(graph.resources ?? [])]);
  summary.snapshots += record.captures?.snapshots.length ?? 0;
  const formulaCount = record.refs?.formulas?.length ?? 0;
  summary.formulas += formulaCount;
  if (formulaCount > 0) summary.recordsWithFormulas++;
  kernelCounts.push({id: scenarioId, kernels});
}

if (failures.length) {
  // biome-ignore lint/suspicious/noConsole: diagnostic output
  console.error(failures.join('\n'));
  process.exitCode = 1;
} else {
  fs.mkdirSync(outputDirectory, {recursive: true});
  if (onlyScenarioIds.size === 0) {
    for (const file of fs.readdirSync(outputDirectory)) {
      if (file.endsWith('.json')) fs.unlinkSync(path.join(outputDirectory, file));
    }
  }
  for (const [file, text] of pendingModels) {
    fs.writeFileSync(path.join(outputDirectory, file), text);
  }
  fs.writeFileSync(
    path.join(lensDirectory, 'summary.json'),
    `${JSON.stringify(summary, null, 2)}\n`
  );
  // biome-ignore lint/suspicious/noConsole: diagnostic output
  console.log(`wrote ${written} view models, ${(totalBytes / 1e6).toFixed(2)} MB`);
  fs.writeFileSync(pageDataFile, pageDataText);
}
