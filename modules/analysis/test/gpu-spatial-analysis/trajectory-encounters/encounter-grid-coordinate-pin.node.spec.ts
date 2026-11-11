// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

// Math correctness review, luma-official-02. `GPUTrajectoryEncounters` scans a grid that the
// published `GPUGridIndex` builds, and decides the cell of a point with its own copy of the
// index's cell-coordinate function, because luma.gl does not export that function. The two only
// agree on a cell while the copy is the same arithmetic. This test reads both as text, so a
// luma.gl upgrade that changes the official rule fails here instead of silently dropping pairs
// next to cell edges. Nothing is imported from the private file.

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

it('the encounter scan maps a point to a cell with the arithmetic of the published GPUGridIndex', () => {
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
        'modules/analysis/src/gpu-spatial-analysis/trajectory-encounters/encounter-kernels.ts'
      ),
      'utf8'
    ),
    'encounter-kernels.ts'
  );
  expect(local).toBe(official);
  // The rule both hold today: bounds that straddle zero are divided by the larger magnitude first.
  expect(official).toContain('letscale=max(abs(minimum),abs(maximum));');
});
