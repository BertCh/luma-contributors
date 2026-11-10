// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes lenses-data.json: sizes of the featured records and their view models, and the spread of
// kernels per record. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/lenses/generate-data.ts
// Totals over all records are not repeated here: they live in src/lens/summary.json.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const directory = path.dirname(fileURLToPath(import.meta.url));
const generatedDirectory = path.resolve(directory, '../../../../lens/generated');
const recordsDirectory = path.resolve(directory, '../../../../../../docs/lens-records');

const FEATURED = [
  'gpu-spatial-analysis.kernel-density',
  'gpu-spatial-analysis.local-moran',
  'gpu-network.network-service-areas'
];
// Lower bounds of the histogram bins, in kernels.
const BIN_STARTS = [1, 2, 5, 10, 20, 50, 100, 300];

type ModelLike = {nodes: {kind: string; count: number}[]};

const entries: {id: string; kernels: number}[] = [];
for (const file of fs.readdirSync(generatedDirectory).sort()) {
  if (!file.endsWith('.json')) continue;
  const id = file.replace(/\.json$/, '');
  const model = JSON.parse(
    fs.readFileSync(path.join(generatedDirectory, file), 'utf8')
  ) as ModelLike;
  const kernels = model.nodes
    .filter(node => node.kind === 'compute')
    .reduce((total, node) => total + node.count, 0);
  entries.push({id, kernels});
}
// Same ordering rule as generate-view-models.ts: sort by kernels, median is the middle element.
const sorted = [...entries].sort((left, right) => left.kernels - right.kernels);
const median = sorted[Math.floor(sorted.length / 2)];

const bins = BIN_STARTS.map((start, index) => {
  const next = BIN_STARTS[index + 1];
  const count = entries.filter(
    entry => entry.kernels >= start && (next === undefined || entry.kernels < next)
  ).length;
  const label =
    next === undefined ? `${start}+` : next - start === 1 ? `${start}` : `${start}-${next - 1}`;
  return {label, count};
});

const featured = FEATURED.map(id => ({
  id,
  recordBytes: fs.statSync(path.join(recordsDirectory, `${id}.json`)).size,
  viewModelBytes: fs.statSync(path.join(generatedDirectory, `${id}.json`)).size
}));

fs.writeFileSync(
  path.join(directory, 'lenses-data.json'),
  `${JSON.stringify({featured, median, bins}, null, 2)}\n`
);
process.stdout.write(`wrote ${entries.length} scenarios\n`);
