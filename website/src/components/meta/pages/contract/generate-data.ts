// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Writes contract-data.json: what the contributor-contract checker's baseline holds, and the
// named status ports of the contract. Run from website/:
//   node --experimental-strip-types src/components/meta/pages/contract/generate-data.ts

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(here, '../../../../../..');
const baselinePath = path.join(repositoryRoot, 'scripts/contributor-contract-baseline.json');
const sourceRoot = path.join(repositoryRoot, 'modules/analysis/src/analysis-core');

type BaselineValue = number | string[];
const baseline = JSON.parse(fs.readFileSync(baselinePath, 'utf8')) as Record<
  string,
  Record<string, BaselineValue>
>;

// A rule's findings: a count per file, or a list of symbols per file.
const rules = Object.entries(baseline).map(([id, files]) => ({
  id,
  files: Object.keys(files).length,
  findings: Object.values(files).reduce<number>(
    (total, value) => total + (typeof value === 'number' ? value : value.length),
    0
  )
}));

// Status ports: `name?: GPUStatusScalar;` fields with their doc comments, in the port types of
// contracts/status.ts and the cardinality and overflow ports of kernels/gpu-contributor-types.ts.
const portFiles = ['contracts/status.ts', 'kernels/gpu-contributor-types.ts'];
const ports: {name: string; doc: string; group: string}[] = [];
for (const file of portFiles) {
  const text = fs.readFileSync(path.join(sourceRoot, file), 'utf8');
  for (const block of text.matchAll(/export type (\w+Port) = (?:\w+ & )*\{([\s\S]*?)\n\};/g)) {
    const group = block[1];
    for (const field of block[2].matchAll(
      /\/\*\*((?:(?!\*\/)[\s\S])*?)\*\/\s*(\w+)\??: GPUStatusScalar;/g
    )) {
      const doc = field[1].replace(/\s*\n\s*\*\s?/g, ' ').trim();
      if (!ports.some(port => port.name === field[2])) {
        ports.push({name: field[2], doc, group});
      }
    }
  }
}

const output = {
  rules,
  totalFindings: rules.reduce((total, rule) => total + rule.findings, 0),
  ports
};
fs.writeFileSync(path.join(here, 'contract-data.json'), `${JSON.stringify(output, null, 1)}\n`);
process.stdout.write(`${rules.length} rules, ${ports.length} ports\n`);
