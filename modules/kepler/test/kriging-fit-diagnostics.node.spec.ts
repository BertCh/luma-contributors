// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {renderKrigingFitDiagnostics} from '../../../examples/deck/kepler-compute/src/analysis-kriging-panel';
import type {KeplerVariogramFit} from '../src/analysis/kriging-variogram-fit';

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

test.each([
  'line-search-stall',
  'iteration-limit'
] as const)('suggested kriging fit displays %s alongside numeric diagnostics before Apply', stopReason => {
  const fit: KeplerVariogramFit = {
    model: 'exponential',
    nugget: 0.1,
    sill: 1,
    rangeMeters: 2,
    criterion: -1,
    method: 'vecchia-fisher',
    weighting: 'conditional-likelihood',
    candidates: [],
    binCount: 0,
    pairCount: 3,
    diagnostics: {budget_used: 1, converged: 0},
    stopReason,
    nonconvergenceDiagnostic: `Vecchia fit stopped: ${stopReason}; retained last usable covariance after 1 accepted steps`
  };
  const html = renderToStaticMarkup(renderKrigingFitDiagnostics(fit));
  expect(html).toContain('budget_used');
  expect(html).toContain(stopReason);
  expect(html).toContain(fit.nonconvergenceDiagnostic);
  // Text evidence is also shown when a producer supplies no numeric diagnostics.
  expect(
    renderToStaticMarkup(renderKrigingFitDiagnostics({...fit, diagnostics: undefined}))
  ).toContain(stopReason);
});

test('converged and WLS diagnostics never fabricate a nonconvergence warning', () => {
  const fit: KeplerVariogramFit = {
    model: 'spherical',
    nugget: 0,
    sill: 1,
    rangeMeters: 2,
    criterion: 0,
    method: 'weighted-least-squares',
    weighting: 'cressie',
    candidates: [],
    binCount: 2,
    pairCount: 3
  };
  expect(renderKrigingFitDiagnostics(fit)).toBeNull();
  expect(
    renderToStaticMarkup(renderKrigingFitDiagnostics({...fit, stopReason: 'converged'}))
  ).toContain('converged');
});

test('kriging display retains the additive stall cause', () => {
  const html = renderToStaticMarkup(
    renderKrigingFitDiagnostics({
      model: 'exponential',
      nugget: 0.1,
      sill: 1,
      rangeMeters: 2,
      criterion: 1,
      method: 'vecchia-fisher',
      weighting: 'conditional-likelihood',
      candidates: [],
      binCount: 0,
      pairCount: 3,
      stopReason: 'line-search-stall',
      stallReason: 'direction-failure'
    })
  );
  expect(html).toContain('line-search-stall');
  expect(html).toContain('direction-failure');
});
