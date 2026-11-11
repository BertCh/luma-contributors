// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

// Math correctness review, luma-official-02. The grid zones of `GPUFlowAggregation` use a local
// copy of the cell-coordinate function of the published `GPUGridBinning`, because luma.gl does
// not export that function and the local kernel also needs a per-frame grid size. A flow and the
// same points binned by `GPUGridBinning` land in the same cell only while the copy is the same
// arithmetic. This test reads both as text, so a luma.gl upgrade that changes the official rule
// fails here instead of silently moving points next to cell edges. Nothing is imported from the
// published file.

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

it('the flow zones map a point to a grid cell with the arithmetic of the published GPUGridBinning', () => {
  const official = getCoordinateFunctionText(
    readFileSync(
      resolve(REPOSITORY_ROOT, 'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-binning.js'),
      'utf8'
    ),
    'published GPUGridBinning'
  );
  const local = getCoordinateFunctionText(
    readFileSync(
      resolve(
        REPOSITORY_ROOT,
        'modules/analysis/src/gpu-network/flow-aggregation/flow-aggregation-kernels.ts'
      ),
      'utf8'
    ),
    'flow-aggregation-kernels.ts'
  );
  expect(local).toBe(official);
  // The rule both hold today: the plain quotient, with the maximum edge in the last cell.
  expect(official).toContain(
    'returnmin(u32((value-minimum)/(maximum-minimum)*f32(size)),size-1u);'
  );
});
