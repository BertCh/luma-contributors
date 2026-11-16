// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {WORKFLOW_TEMPLATES} from '../../../../../../website/src/components/meta/pages/workflows/workflows';

const scriptUrl = new URL(
  '../../../../../../website/scripts/verify-analysis-metadata.mjs',
  import.meta.url
);
const acceptance = (await import(/* @vite-ignore */ scriptUrl.href)) as {
  loadWorkflowTemplateIds(): Promise<string[]>;
  verifyWorkflowTemplateIds(actual: (string | null)[], expected: string[]): void;
};

it('accepts the current shared gallery including added templates, in any DOM order', async () => {
  const expected = WORKFLOW_TEMPLATES.map(template => template.id);
  const loaded = await acceptance.loadWorkflowTemplateIds();
  expect(loaded).toEqual(expected);
  expect(loaded).toContain('drive-time-bands');
  expect(() => acceptance.verifyWorkflowTemplateIds([...loaded].reverse(), expected)).not.toThrow();
  const page = readFileSync(
    new URL(
      '../../../../../../website/src/components/meta/pages/workflows/index.tsx',
      import.meta.url
    ),
    'utf8'
  );
  expect(page).toContain('data-template-id={template.id}');
});

it('rejects missing, duplicated, unknown and unlabelled templates even at the expected count', async () => {
  const expected = await acceptance.loadWorkflowTemplateIds();
  for (const actual of [
    expected.slice(0, -1),
    [...expected.slice(0, -1), expected[0]],
    [...expected.slice(0, -1), 'unknown-template'],
    [...expected.slice(0, -1), null]
  ]) {
    expect(() => acceptance.verifyWorkflowTemplateIds(actual, expected)).toThrow(
      'Curated template ids differ'
    );
  }
});
