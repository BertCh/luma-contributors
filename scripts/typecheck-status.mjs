// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Read the shared incremental typecheck instead of starting another tsc process.
// One `yarn typecheck-watch` process collects every edit; this waits for it to settle and prints
// the latest cycle's errors, optionally filtered by a path regex.
// Usage: node scripts/typecheck-status.mjs ['<path regex>']
import fs from 'node:fs';

const LOG = new URL('../.typecheck/watch.log', import.meta.url);
const filter = process.argv[2] ? new RegExp(process.argv[2]) : null;
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

if (!fs.existsSync(LOG)) {
  process.stderr.write('No shared typecheck running. Start one with: yarn typecheck-watch\n');
  process.exit(2);
}
// Give the watcher time to notice edits made just before this call, then wait until it is idle.
await sleep(2000);
const deadline = Date.now() + 10 * 60 * 1000;
for (;;) {
  const text = fs.readFileSync(LOG, 'utf8');
  const idle = /Found \d+ errors?\. Watching for file changes\.\s*$/.test(text.trimEnd() + '\n');
  const quietFor = Date.now() - fs.statSync(LOG).mtimeMs;
  if (idle && quietFor > 1500) break;
  if (Date.now() > deadline) {
    process.stderr.write('Timed out waiting for the shared typecheck to settle\n');
    process.exit(3);
  }
  await sleep(1000);
}
const text = fs.readFileSync(LOG, 'utf8');
const cycleStart = Math.max(text.lastIndexOf('Starting compilation in watch mode'), text.lastIndexOf('Starting incremental compilation'));
const cycle = text.slice(cycleStart);
const errorLines = cycle.split('\n').filter(line => /error TS\d+/.test(line));
const selected = filter ? errorLines.filter(line => filter.test(line)) : errorLines;
const finishedAt = (cycle.match(/([0-9:]+ [AP]M) - Found \d+ error/) || [])[1] || 'unknown time';
process.stdout.write(selected.join('\n') + (selected.length ? '\n' : ''));
process.stdout.write(`${selected.length} matching errors (${errorLines.length} total) in cycle finished at ${finishedAt}\n`);
process.exitCode = selected.length ? 1 : 0;
