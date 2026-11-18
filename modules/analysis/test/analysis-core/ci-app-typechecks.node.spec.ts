// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';

const workflow = readFileSync('.github/workflows/test.yml', 'utf8');

function getJobSteps(jobName: string): string[] {
  const jobHeaders = [...workflow.matchAll(/^  ([a-z][a-z-]*):\s*$/gm)];
  const index = jobHeaders.findIndex(header => header[1] === jobName);
  expect(index, `missing ${jobName} CI job`).toBeGreaterThanOrEqual(0);
  const job = workflow.slice(jobHeaders[index].index, jobHeaders[index + 1]?.index);
  return job.split(/^      - /m).slice(1);
}

test.each(['test', 'website'])('%s CI job builds its own packages before typechecking', jobName => {
  const steps = getJobSteps(jobName);
  const buildIndex = steps.findIndex(step => /^\s*run: yarn build\s*$/m.test(step));
  expect(
    buildIndex,
    `${jobName} needs fresh workspace package declarations`
  ).toBeGreaterThanOrEqual(0);
  expect(steps[buildIndex]).toContain('NODE_OPTIONS: --max-old-space-size=8192');
  const checks = steps.flatMap((step, index) => (/^name: Typecheck /m.test(step) ? [index] : []));
  expect(checks.length).toBeGreaterThan(0);
  for (const index of checks) {
    expect(
      buildIndex,
      `${jobName}: ${steps[index].split('\n')[0]} precedes its package build`
    ).toBeLessThan(index);
  }
});

test('CI checks all four apps using their unchanged project configurations', () => {
  const exampleSteps = getJobSteps('test').join('\n');
  for (const application of [
    'spatial-analysis-showcase',
    'spatial-analysis-explorer',
    'kepler-compute'
  ]) {
    expect(exampleSteps).toContain(`npx tsc --noEmit -p examples/deck/${application}`);
  }
  expect(getJobSteps('website').join('\n')).toContain('npx tsc -p website/tsconfig.json --noEmit');
});
