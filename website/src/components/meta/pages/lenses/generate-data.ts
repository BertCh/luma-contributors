// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes lenses-data.json: sizes of the featured records and their view models, and the spread of
// kernels per record. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/lenses/generate-data.ts
// Totals over all records are not repeated here: they live in src/lens/summary.json.

import fs from 'node:fs';
import {Buffer} from 'node:buffer';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';
import {
  createLensPageData,
  validateLensModel,
  validateLensFileNames,
  type LensPageEntry
} from '../../../../lens/generation-metadata.ts';
import {extractLensViewModel} from '../../../../lens/extract-lens-view-model.ts';

const directory = path.dirname(fileURLToPath(import.meta.url));
const generatedDirectory = path.resolve(directory, '../../../../lens/generated');
const recordsDirectory = path.resolve(directory, '../../../../../../docs/lens-records');

const entries: LensPageEntry[] = [];
const sourceFiles = fs
  .readdirSync(recordsDirectory)
  .filter(file => file.endsWith('.json'))
  .sort();
const modelFiles = fs
  .readdirSync(generatedDirectory)
  .filter(file => file.endsWith('.json'))
  .sort();
validateLensFileNames(sourceFiles, modelFiles);
for (const file of sourceFiles) {
  const id = file.replace(/\.json$/, '');
  const modelText = fs.readFileSync(path.join(generatedDirectory, file), 'utf8');
  const model = JSON.parse(modelText);
  if (model.id !== id) throw new Error(`${file}: model id must be ${id}`);
  const recordText = fs.readFileSync(path.join(recordsDirectory, file), 'utf8');
  const envelope = fromRecordedLensFile(JSON.parse(recordText));
  if (envelope.scenarioId !== id)
    throw new Error(
      `Record filename identifies ${id}, but envelope identifies ${envelope.scenarioId}; use the matching scenario filename`
    );
  const record = envelope.record;
  validateLensModel(id, modelText, extractLensViewModel(id, record));
  const graph = record.graphs.find(graph => graph.form === 'compiled') ?? record.graphs[0];
  // Display groups can mix compute and copy nodes; count original commands before grouping.
  const kernels = graph.nodes.filter(node => node.kind === 'compute').length;
  entries.push({
    id,
    kernels,
    recordBytes: Buffer.byteLength(recordText, 'utf8'),
    viewModelBytes: Buffer.byteLength(modelText, 'utf8')
  });
}
const page = createLensPageData(entries);
fs.writeFileSync(path.join(directory, 'lenses-data.json'), `${JSON.stringify(page, null, 2)}\n`);
process.stdout.write(`wrote ${entries.length} scenarios\n`);
