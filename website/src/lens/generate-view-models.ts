// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Regenerates src/lens/generated/*.json from docs/lens-records/*.json:
//   node --experimental-strip-types src/lens/generate-view-models.ts   (from website/)
// Only compact view models are written; the 80 MB of records never ship.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {fromRecordedLensFile} from '@luma-contributors/lenses/computation-lenses/registry';
import {extractLensViewModel, type LensRecordLike} from './extract-lens-view-model.ts';

const lensDirectory = path.dirname(fileURLToPath(import.meta.url));
const recordsDirectory = path.resolve(lensDirectory, '../../../docs/lens-records');
const outputDirectory = path.join(lensDirectory, 'generated');

fs.mkdirSync(outputDirectory, {recursive: true});
for (const file of fs.readdirSync(outputDirectory)) {
  if (file.endsWith('.json')) fs.unlinkSync(path.join(outputDirectory, file));
}

let written = 0;
let totalBytes = 0;
const failures: string[] = [];
for (const file of fs.readdirSync(recordsDirectory).sort()) {
  if (!file.endsWith('.json')) continue;
  const scenarioId = file.replace(/\.json$/, '');
  try {
    const envelope = fromRecordedLensFile(
      JSON.parse(fs.readFileSync(path.join(recordsDirectory, file), 'utf8'))
    );
    const model = extractLensViewModel(scenarioId, envelope.record as unknown as LensRecordLike);
    const text = JSON.stringify(model);
    fs.writeFileSync(path.join(outputDirectory, `${scenarioId}.json`), text);
    written++;
    totalBytes += text.length;
  } catch (error) {
    failures.push(`${file}: ${(error as Error).message}`);
  }
}
// biome-ignore lint/suspicious/noConsole: diagnostic output
console.log(`wrote ${written} view models, ${(totalBytes / 1e6).toFixed(2)} MB`);
if (failures.length) {
  // biome-ignore lint/suspicious/noConsole: diagnostic output
  console.error(failures.join('\n'));
  process.exitCode = 1;
}
