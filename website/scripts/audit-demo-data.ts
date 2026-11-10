// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU-only report for reviewing provenance before refreshing catalog media.
// Run from website: node --experimental-strip-types scripts/audit-demo-data.ts [--json]
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {buildDemoData} from '../build/demo-data-plugin.ts';

const gpuDirectory = fileURLToPath(new URL('../src/gpu/', import.meta.url));
const inventory = buildDemoData(path.resolve(gpuDirectory));
if (process.argv.includes('--json')) {
  process.stdout.write(JSON.stringify(inventory, null, 2) + '\n');
} else {
  const entries = Object.entries(inventory.entries);
  const constructed = entries.filter(([, entry]) => !entry.sources.length);
  const mixed = entries.filter(([, entry]) => entry.sources.length && entry.scenario);
  process.stdout.write(`${entries.length} scene modules. ${entries.length - constructed.length} use sourced inputs.\n`);
  process.stdout.write(`${mixed.length} sourced scenes have additional scenario notes.\n\n`);
  for (const [id, data] of constructed) process.stdout.write(`${id}: ${data.caption ?? data.scenario}\n`);
}
