// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('typechecks the owned areal-to-lag guide example with explicit imports and typed inputs', () => {
  const snippets = extractSnippets('gpu-spatial-analysis/operators-accessibility.md').filter(
    snippet => snippet.code.includes('new GPUArealInterpolation(')
  );
  expect(snippets).toHaveLength(1);
  expect(getSnippetDiagnostics(snippets)).toEqual([]);
}, 120_000);
