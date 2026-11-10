// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Summarizes the pointer gestures of every run from the reports of verify-runs.mjs, as the
// measures of dev-docs/roadmap.md section 7.2. Run `node scripts/verify-runs.mjs` first.
//
// Usage (from the website directory):
//   node scripts/audit-card-interactions.mjs [--reports .reports/runs] [--json]
//
// Exit code 1 when a run on disk has no report, fails verification, declares no gesture,
// recompiles under the pointer, or drives an input-data knob with it.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const websiteDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const entriesDirectory = path.join(websiteDirectory, 'src/gpu/runs/entries');
const argv = process.argv.slice(2);
let reports = path.join(websiteDirectory, '.reports/runs');
let asJson = false;
for (let index = 0; index < argv.length; index++) {
  if (argv[index] === '--reports') reports = path.resolve(argv[++index]);
  else if (argv[index] === '--json') asJson = true;
}

const runs = [];
for (const group of fs.readdirSync(entriesDirectory)) {
  const groupDirectory = path.join(entriesDirectory, group);
  if (!fs.statSync(groupDirectory).isDirectory()) continue;
  for (const file of fs.readdirSync(groupDirectory)) {
    if (file.endsWith('.ts') && !file.endsWith('.parameters.ts')) runs.push({group, id: file.slice(0, -3)});
  }
}
const results = new Map();
for (const file of fs.existsSync(reports) ? fs.readdirSync(reports) : []) {
  if (!file.endsWith('.json')) continue;
  for (const result of JSON.parse(fs.readFileSync(path.join(reports, file), 'utf8')).results ?? []) {
    if (result.pointer || !results.has(result.id) || result.sliders) results.set(result.id, result);
  }
}

const problems = [];
const kinds = {};
const summary = {
  runs: runs.length,
  reported: 0,
  passed: 0,
  withGesture: 0,
  recompiling: 0,
  drivingDataKnob: 0,
  fewerThanThreeDistinct: 0,
  withoutReadout: 0,
  restMismatch: 0,
  inertSliders: 0,
  warnings: 0
};
const changes = [];
for (const run of runs) {
  const result = results.get(run.id);
  if (!result) {
    problems.push(`${run.group}/${run.id}: no verify report`);
    continue;
  }
  summary.reported++;
  if (result.passed) summary.passed++;
  else problems.push(`${run.group}/${run.id}: ${result.failures.join('; ')}`);
  const warnings = result.warnings ?? [];
  summary.warnings += warnings.length;
  summary.inertSliders += warnings.filter(warning => warning.includes('does not change the picture') && warning.startsWith('slider')).length;
  if (!result.pointer) {
    problems.push(`${run.group}/${run.id}: no gesture declared`);
    continue;
  }
  summary.withGesture++;
  kinds[result.pointer.kind] = (kinds[result.pointer.kind] ?? 0) + 1;
  changes.push(Math.max(...result.pointer.changes, result.pointer.xEffect, result.pointer.yEffect));
  if (result.pointer.pointerRebuilds > 0) {
    summary.recompiling++;
    problems.push(`${run.group}/${run.id}: recompiles under the pointer`);
  }
  if (warnings.some(warning => warning.includes('input-data knob'))) {
    summary.drivingDataKnob++;
    problems.push(`${run.group}/${run.id}: gesture drives an input-data knob`);
  }
  if (result.pointer.distinct < 3) summary.fewerThanThreeDistinct++;
  if (result.pointer.readouts.every(readout => !readout)) summary.withoutReadout++;
  if (warnings.some(warning => warning.startsWith('rest position'))) summary.restMismatch++;
}
changes.sort((first, second) => first - second);
summary.medianPointerChange = changes.length ? changes[Math.floor(changes.length / 2)] : null;
summary.pointerChangeUnderThreePercent = changes.filter(change => change < 0.03).length;
summary.kinds = Object.fromEntries(Object.entries(kinds).sort((first, second) => second[1] - first[1]));

if (asJson) {
  console.log(JSON.stringify({summary, problems}, null, 2));
} else {
  console.log(`Runs on disk: ${summary.runs}; reported: ${summary.reported}; passed: ${summary.passed}`);
  console.log(`Declare a gesture: ${summary.withGesture} (${Object.entries(summary.kinds).map(([kind, count]) => `${kind} ${count}`).join(', ')})`);
  console.log(`Recompile under the pointer: ${summary.recompiling}; drive an input-data knob: ${summary.drivingDataKnob}`);
  console.log(`Fewer than 3 distinct pictures at 5 positions: ${summary.fewerThanThreeDistinct}; no readout: ${summary.withoutReadout}; rest differs from defaults: ${summary.restMismatch}`);
  console.log(`Largest pointer change, median: ${summary.medianPointerChange === null ? '-' : (summary.medianPointerChange * 100).toFixed(1) + '%'}; under 3%: ${summary.pointerChangeUnderThreePercent}`);
  console.log(`Inert sliders: ${summary.inertSliders}; warnings in total: ${summary.warnings}`);
  if (problems.length > 0) console.log(`\n${problems.length} problems:\n  ${problems.join('\n  ')}`);
}
process.exit(problems.length > 0 ? 1 : 0);
