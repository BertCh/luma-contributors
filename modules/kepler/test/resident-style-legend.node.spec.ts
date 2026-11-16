// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {afterAll, describe, expect, test, vi} from 'vitest';
import {LayerColorLegendFactory} from '@kepler.gl/components';
import {
  createResidentStyleLegendFactory,
  getResidentStyleLegendRows,
  getResidentStyleLegendTitle,
  matchesResidentStyleLegend,
  type ResidentStyleLegendSnapshot
} from '../../../examples/deck/kepler-compute/src/resident-style-legend';
import type {KeplerResidentStyleLegend} from '../src/styles/resident-style';

// Shared Node workers may have loaded the legend with another suite's component factory mock.
vi.hoisted(() => vi.resetModules());
afterAll(() => {
  vi.doUnmock('@kepler.gl/components');
  vi.resetModules();
});

// The host intentionally pins React 19 independently of the repository's React 18 tools.
const hostRequire = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
const {createElement} = hostRequire('react') as typeof import('react');
const {renderToStaticMarkup} = hostRequire('react-dom/server') as typeof import('react-dom/server');

vi.mock('@kepler.gl/components', () => {
  const factory = Object.assign(() => () => createElement('div', null, 'Native legend'), {
    deps: [() => undefined, () => undefined, () => undefined]
  });
  return {LayerColorLegendFactory: factory};
});

function legend(): KeplerResidentStyleLegend {
  return {
    styleId: 'points:fillColor',
    revision: 4,
    source: {datasetId: 'sample', revision: 2},
    selectionRevision: 3,
    scale: 'quantile',
    domain: [0, 4, 4, 100],
    domainStatus: 'valid',
    finiteCount: 100,
    palette: [0xff0000ff, 0xff00ff00, 0xffff0000],
    numericalPolicy: 'float32'
  };
}
function snapshot(): ResidentStyleLegendSnapshot {
  const result = legend();
  return {expected: {...result}, legend: result};
}
function render(state: ResidentStyleLegendSnapshot | null, resident = true, stroke = false) {
  const getLegend = vi.fn(() => state);
  const factory = createResidentStyleLegendFactory({isResidentDataset: () => resident, getLegend});
  const component = factory(...([] as unknown as Parameters<typeof LayerColorLegendFactory>));
  type Props = Parameters<typeof component>[0];
  const config = {
    dataId: 'sample',
    colorField: {name: 'Value'},
    strokeColorField: {name: 'Stroke'}
  };
  const props = {
    layer: {id: 'points', config},
    config,
    description: {measure: 'Value'},
    colorChannel: {property: stroke ? 'strokeColor' : 'color'}
  } as unknown as Props;
  return {markup: renderToStaticMarkup(createElement(component, props)), getLegend, factory};
}

describe('resident style legend injection', () => {
  test('exact encoded classes name their source field, retaining ordinary legend titles', () => {
    expect(
      getResidentStyleLegendTitle(
        'analysis_result_class (classification)',
        'analysis_result_class (classification)',
        {type: 'customOrdinal', name: 'classification'}
      )
    ).toBe('classification');
    expect(
      getResidentStyleLegendTitle('value', 'Measured value', {
        type: 'customOrdinal',
        name: 'Palette'
      })
    ).toBe('Measured value');
  });
  test('retains exact native factory dependencies and delegates ordinary datasets', () => {
    const result = render(null, false);
    expect(result.factory.deps).toBe(LayerColorLegendFactory.deps);
    expect(result.markup).toContain('Native legend');
    expect(result.getLegend).not.toHaveBeenCalled();
  });
  test('selects independent stroke channels and never delegates a pending resident mapped legend', () => {
    const result = render(null, true, true);
    expect(result.getLegend).toHaveBeenCalledWith('points', 'strokeColor');
    expect(result.markup).toContain('Computing GPU legend');
    expect(result.markup).not.toContain('Native legend');
  });
  test('rejects every stale provenance dimension before exposing old GPU values', () => {
    for (const change of [
      {styleId: 'other'},
      {revision: 5},
      {selectionRevision: 4},
      {source: {datasetId: 'other', revision: 2}},
      {source: {datasetId: 'sample', revision: 3}}
    ]) {
      const state = snapshot();
      state.expected = {...state.expected, ...change};
      expect(matchesResidentStyleLegend(state)).toBe(false);
      expect(render(state).markup).toContain('Updating GPU legend');
      expect(render(state).markup).not.toContain('value &lt;');
    }
  });
  test('renders GPU quantile edges directly including tied buckets, bounded to requested entries', () => {
    expect(getResidentStyleLegendRows(legend(), 2)).toEqual([
      {color: 0xff0000ff, label: '< 4'},
      {color: 0xff00ff00, label: '4 ≤ value < 4'}
    ]);
    expect(render(snapshot()).markup).toContain('≥ 4');
    const categorical = {...legend(), scale: 'ordinal' as const, labels: ['a', 'b'], codes: [2, 0]};
    expect(getResidentStyleLegendRows(categorical)).toEqual([
      {label: 'a', color: 0xffff0000},
      {label: 'b', color: 0xff0000ff}
    ]);
  });
  test('renders signed log domains continuously and custom threshold bookends never become labels', () => {
    const state = snapshot();
    state.legend = {...state.legend!, scale: 'log', domain: [-100, -10, -1]};
    const markup = render(state).markup;
    expect(markup).toContain('log color scale');
    expect(markup).toContain('50%');
    expect(markup).not.toContain('value &lt;');
    const custom = {...legend(), domain: [10, 10, 20, 20]};
    expect(getResidentStyleLegendRows(custom).map(row => row.label)).toEqual([
      '< 10',
      '10 ≤ value < 20',
      '≥ 20'
    ]);
    expect(getResidentStyleLegendRows({...custom, domain: [0, 0], palette: [0xff0000ff]})).toEqual([
      {color: 0xff0000ff, label: 'All finite values'}
    ]);
  });
  test('shows empty domain and explicit unsupported reasons without native recomputation', () => {
    const state = snapshot();
    state.legend = {...state.legend!, domain: null, domainStatus: 'empty', finiteCount: 0};
    expect(render(state).markup).toContain('No finite values');
    state.legend = undefined;
    state.reason = 'This native scale is unsupported.';
    expect(render(state).markup).toContain(state.reason);
  });
  test('shows invalid log-domain diagnostics without confusing invalid and empty populations', () => {
    const state = snapshot();
    state.legend = {
      ...state.legend!,
      scale: 'log',
      domain: [-10, 100],
      domainStatus: 'invalid-log-domain'
    };
    const markup = render(state).markup;
    expect(markup).toContain('crosses or touches zero');
    expect(markup).toContain('entirely above or below zero');
    expect(markup).not.toContain('No finite values');
    expect(markup).not.toContain('linear-gradient');
    expect(getResidentStyleLegendRows(state.legend)).toEqual([]);
    state.expected.selectionRevision = 4;
    expect(render(state).markup).toContain('Updating GPU legend');
    expect(render(state).markup).not.toContain('crosses or touches zero');
  });
});
