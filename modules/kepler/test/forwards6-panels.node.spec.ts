// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import type {ReactElement, ReactNode} from 'react';
import {OLSResults} from '../../../examples/deck/kepler-compute/src/analysis-ols-panel';
import type {AnalysisRuntime} from '../../../examples/deck/kepler-compute/src/analysis-runtime';
import {RasterTimeSeriesPanel} from '../../../examples/deck/kepler-compute/src/raster-timeseries-panel';
import type {RasterTimeSeriesSettings} from '../src/analysis/raster-timeseries';

const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const react = hostRequire('react') as typeof import('react') & {
  __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE: {H: unknown};
};
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

function elements(node: ReactNode): ReactElement<Record<string, unknown>>[] {
  if (Array.isArray(node)) return node.flatMap(elements);
  return react.isValidElement<Record<string, unknown>>(node)
    ? [node, ...elements(node.props['children'] as ReactNode)]
    : [];
}

test.each([
  'classical-homoskedastic',
  'penalized-descriptive',
  'unavailable'
] as const)('OLS report labels %s inference explicitly', inference => {
  const analysis = {
    summary: {
      ols: {
        inference,
        inferenceMethod: inference === 'penalized-descriptive' ? 'ridge-approximate' : undefined,
        rowCount: 8,
        degreesOfFreedom: 6,
        coefficients: [],
        diagnostics: {},
        conditioning: {illConditioned: true, conditionNumber: 10, correlationConditionNumber: 10},
        ridgeLambda: 1
      }
    }
  } as unknown as AnalysisRuntime;
  const html = renderToStaticMarkup(react.createElement(OLSResults, {analysis}));
  expect(html).toContain('Coefficient inference:');
  if (inference === 'unavailable') {
    expect(html).toContain('Least-squares fit (inference metadata unavailable)');
    expect(html).not.toContain('residual degrees of freedom');
    expect(html).not.toContain('two-sided Student t tests');
  } else if (inference === 'penalized-descriptive') {
    expect(html).toContain('Ridge descriptive fit');
    expect(html).toContain('approximate inverse-penalized-normal-matrix');
    expect(html).not.toContain('two-sided Student t tests');
  } else {
    expect(html).toContain('Ordinary least squares fit');
    expect(html).toContain('6 residual degrees of freedom');
    expect(html).toContain('two-sided Student t tests');
  }
  expect(html).toContain('does not establish that a coefficient-inference convention is available');
});

test('time-series panel submits distribution defaults and explicit atom booleans', async () => {
  let state: unknown[] = [
    'standardized-index',
    'raster',
    'zones',
    '[{"time":0,"bands":[1]}]',
    {},
    '',
    0,
    false,
    '',
    [],
    {}
  ];
  const submitted: RasterTimeSeriesSettings[] = [];
  const host = {
    run: async (request: {settings: RasterTimeSeriesSettings}) => {
      submitted.push(request.settings);
      return {
        columns: {},
        rowCount: 0,
        skippedRowCount: 0,
        contributors: [],
        outputName: 'index',
        outputComponent: 0,
        readbackBytes: 0,
        diagnostics: {}
      };
    }
  } as unknown as Parameters<typeof RasterTimeSeriesPanel>[0]['host'];
  const render = () => {
    let cursor = 0;
    const internals = react.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
    const previous = internals.H;
    internals.H = {
      useState: () => {
        const index = cursor++;
        return [
          state[index],
          (value: unknown) => {
            state[index] = value;
          }
        ];
      }
    };
    try {
      return elements(
        RasterTimeSeriesPanel({
          host,
          rasterLayers: [{id: 'raster', label: 'Raster'}],
          polygonDatasets: [{id: 'zones', label: 'Zones'}]
        })
      );
    } finally {
      internals.H = previous;
    }
  };
  for (const distribution of ['gamma', 'fisk']) {
    for (const zeroAtom of ['default', 'on', 'off']) {
      state = [...state];
      state[4] = {distribution, zeroAtom};
      const tree = render();
      const control = tree.find(
        element => element.props['data-testid'] === 'timeseries-setting-zeroAtom'
      );
      expect(control?.props['value']).toBe(zeroAtom);
      const button = tree.find(element => element.props['data-testid'] === 'timeseries-run');
      await (button!.props['onClick'] as () => Promise<void>)();
      const settings = submitted.at(-1)!;
      expect(settings['distribution']).toBe(distribution);
      if (zeroAtom === 'default') expect(Object.hasOwn(settings, 'zeroAtom')).toBe(false);
      else expect(settings['zeroAtom']).toBe(zeroAtom === 'on');
    }
  }
});
