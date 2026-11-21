// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('rate mapping recipe example typechecks independently without implicit input declarations', () => {
  const snippets = extractSnippets('gpu-spatial-analysis/rate-mapping.md');
  expect(snippets).toHaveLength(1);
  // Explicit line mappings select the snippet checker's strict mode: no free-name prelude.
  const strict = snippets.map(snippet => ({
    ...snippet,
    markdownLines: snippet.code.split('\n').map((_, index) => snippet.fenceLine + 1 + index)
  }));
  expect(getSnippetDiagnostics(strict)).toEqual([]);
});
