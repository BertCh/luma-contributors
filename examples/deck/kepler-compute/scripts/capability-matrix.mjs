// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, writeFileSync} from 'node:fs';

const inventory = JSON.parse(readFileSync(new URL('../src/sandbox-capabilities.json', import.meta.url), 'utf8'));
const destination = new URL('../../../../dev-docs/reviews/kepler-sandbox-2026-10-28/capability-matrix.md', import.meta.url);
const labels = {
  partial: 'Partial host integration',
  adapter: 'Reusable building blocks',
  planned: 'Needs integration',
  host: 'Host or service workflow',
  integrated: 'Integrated and verified'
};
const lines = [
  '# Kepler sandbox capability matrix', '',
  `Reviewed against Kepler ${inventory.keplerVersion} on ${inventory.reviewDate}.`, '',
  inventory.scope, '',
  'Source: [sandbox-capabilities.json](../../../examples/deck/kepler-compute/src/sandbox-capabilities.json).', '',
  'A partial entry is not full native parity. A reusable building block may be a contributor, renderer or adapter that is not exposed in the host. Host/service entries still require compatibility checks. Priorities order integration work, not upstream feature importance.', ''
];
const escapeCell = value => value.replaceAll('|', '\\|').replaceAll('\n', ' ');
for (const group of new Set(inventory.capabilities.map(capability => capability.group))) {
  lines.push(`## ${group}`, '',
    '| Feature | Stage | Foundation | Remaining work | Acceptance | Priority |',
    '| --- | --- | --- | --- | --- | --- |');
  for (const capability of inventory.capabilities.filter(item => item.group === group)) {
    const cells = [
      `${capability.title} (\`${capability.id}\`)`, labels[capability.status],
      capability.foundation, capability.gap, capability.acceptance, capability.priority
    ];
    lines.push(`| ${cells.map(escapeCell).join(' | ')} |`);
  }
  lines.push('');
}
const markdown = lines.join('\n');
if (process.argv.includes('--write')) {
  writeFileSync(destination, markdown);
} else if (readFileSync(destination, 'utf8') !== markdown) {
  throw new Error('Capability matrix is stale. Run node examples/deck/kepler-compute/scripts/capability-matrix.mjs --write');
}
process.stdout.write(`Capability matrix matches ${inventory.capabilities.length} reviewed features.\n`);
