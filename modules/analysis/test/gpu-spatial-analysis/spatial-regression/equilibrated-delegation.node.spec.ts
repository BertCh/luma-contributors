// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createHash} from 'node:crypto';
import {expect, it} from 'vitest';
import {getEquilibratedCholeskyWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/local-weighted-least-squares-wgsl';
import baseline from './equilibrated-source-baseline.json';

// Frozen independently from git show 85a29fcc, before either implementation was edited.
// Scratch reproduction: freeze-equilibrated.mjs extracts the baseline generator and float-literal
// formatter, evaluates only those functions with tolerance 0.001, and hashes sizes 1 through 25.
it.each(
  baseline
)('regression Cholesky preserves baseline WGSL through 25 columns: size %i', (size, digest) => {
  expect(
    createHash('sha256')
      .update(getEquilibratedCholeskyWGSL(Number(size)))
      .digest('hex')
  ).toBe(digest);
});
