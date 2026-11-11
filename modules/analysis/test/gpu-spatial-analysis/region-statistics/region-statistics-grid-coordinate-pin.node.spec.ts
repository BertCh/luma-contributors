// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

// Math correctness review, luma-official-02. The grid path of `GPURegionStatistics` turns a query
// rectangle into a range of cells of a grid that the published `GPUGridIndex` builds, with its own
// copy of the index's cell-coordinate function, because luma.gl does not export that function.
// The range is widened by one cell per side, so a one-cell disagreement cannot lose a candidate;
// a different rule could. This test reads both as text, so a luma.gl upgrade that changes the
// official rule fails here. Nothing is imported from the private file. Same convention as
// `trajectory-encounters/encounter-grid-coordinate-pin.node.spec.ts`.

const HERE = dirname(fileURLToPath(import.meta.url));
const REPOSITORY_ROOT = resolve(HERE, '../../../../..');

/** The text of WGSL `fn getCoordinate(...) { ... }` in `source`, with whitespace removed. */
function getCoordinateFunctionText(source: string, label: string): string {
  const start = source.indexOf('fn getCoordinate(');
  expect(start, `${label}: fn getCoordinate`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  for (let index = source.indexOf('{', start); index < source.length; index++) {
    if (source[index] === '{') {
      depth++;
    } else if (source[index] === '}') {
      depth--;
      if (depth === 0) {
        return source.slice(start, index + 1).replace(/\s+/g, '');
      }
    }
  }
  throw new Error(`${label}: unterminated fn getCoordinate`);
}

it('the region statistics grid query maps a bound to a cell with the arithmetic of the published GPUGridIndex', () => {
  const official = getCoordinateFunctionText(
    readFileSync(
      resolve(
        REPOSITORY_ROOT,
        'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-index-internals.js'
      ),
      'utf8'
    ),
    'published GPUGridIndex'
  );
  const local = getCoordinateFunctionText(
    readFileSync(
      resolve(
        REPOSITORY_ROOT,
        'modules/analysis/src/gpu-spatial-analysis/region-statistics/region-statistics-grid.ts'
      ),
      'utf8'
    ),
    'region-statistics-grid.ts'
  );
  expect(local).toBe(official);
  // The rule both hold today: bounds that straddle zero are divided by the larger magnitude first.
  expect(official).toContain('letscale=max(abs(minimum),abs(maximum));');
});
