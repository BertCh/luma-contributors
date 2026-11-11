// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {expect, it} from 'vitest';

// Math correctness review, interpolation-density-01 and luma-official-02. `GPUPointDensity` takes
// its counts from the published `GPUGridBinning` and keys the weights of the default sum mode
// with its own copy of the binning's cell-coordinate function, because luma.gl does not export
// that function. Count and weight only land in one cell while the copy is the same arithmetic.
// This test reads both as text, so a luma.gl upgrade that changes the official rule fails here
// instead of silently moving weights next to cell edges. Nothing is imported from the private
// file.

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

function readCoordinateFunctionText(repositoryPath: string): string {
  return getCoordinateFunctionText(
    readFileSync(resolve(REPOSITORY_ROOT, repositoryPath), 'utf8'),
    repositoryPath
  );
}

it('the point density grid keys map a point to a cell with the arithmetic of the published GPUGridBinning and GPUGridAggregation', () => {
  const local = readCoordinateFunctionText(
    'modules/analysis/src/gpu-spatial-analysis/point-density/point-density-kernels.ts'
  );
  const binning = readCoordinateFunctionText(
    'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-binning.js'
  );
  const aggregation = readCoordinateFunctionText(
    'node_modules/@luma.gl/graph/dist/gpu-core/gpu-grid-aggregation.js'
  );
  expect(local).toBe(binning);
  // The `'atomic'` sum mode delegates to GPUGridAggregation, which must agree with the counts too.
  expect(aggregation).toBe(binning);
  // The rule all three hold today: no rescaling when the bounds straddle zero (GPUGridIndex has
  // one, and is the wrong sibling to copy for a kernel paired with GPUGridBinning counts).
  expect(binning).not.toContain('scale');
  expect(binning).toContain('u32((value-minimum)/(maximum-minimum)*f32(size))');
});
