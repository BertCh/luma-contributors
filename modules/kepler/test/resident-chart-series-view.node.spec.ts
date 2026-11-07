// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, test} from 'vitest';
import {
  getResidentChartSeriesColor,
  makeResidentChartLineSegments,
  makeResidentChartSeriesLayout
} from '../../../examples/deck/kepler-compute/src/resident-chart-series-layout';
import {
  ResidentChartSeriesView,
  type ResidentChartSeriesViewProps
} from '../../../examples/deck/kepler-compute/src/resident-chart-series-view';

const require = createRequire(
  new URL('../../../examples/deck/kepler-compute/package.json', import.meta.url)
);
// Match the example's React 19 renderer; the repository root retains React 18 for other examples.
const {createElement} = require('react');
const {renderToStaticMarkup} = require('react-dom/server');
type ChartBin = ResidentChartSeriesViewProps['bins'][number];

const measure = (key: string, value: number, count = 1): ChartBin => ({
  key,
  value,
  count,
  color: 'unused-local-native-color'
});
const bins: ChartBin[] = [
  {
    key: 'first',
    value: 8,
    count: 3,
    color: 'white',
    filterValue: [4294967293, 4294967293.5],
    series: [measure('Beta', 8, 2), measure('Alpha', 0)]
  },
  {
    key: 'second',
    value: -5,
    count: 1,
    color: 'white',
    filterValue: ['second'],
    series: [measure('Alpha', -5)]
  },
  {key: 'empty', value: 0, count: 0, color: 'white', filterValue: ['empty'], series: []},
  {
    key: 'last',
    value: 2,
    count: 1,
    color: 'white',
    filterValue: ['last'],
    series: [measure('Beta', 2)]
  },
  {key: 'Other', value: 20, count: 4, color: 'gray'}
];

test('series layout preserves parent filters, distinguishes absence from zero, and includes negative and aggregate values', () => {
  const layout = makeResidentChartSeriesLayout(bins);
  expect(layout.groups).toBe(bins);
  expect(layout.groups[0].filterValue).toBe(bins[0].filterValue);
  expect(layout.series.map(series => series.key)).toEqual(['Alpha', 'Beta']);
  expect(layout.series[0].values[0]?.value).toBe(0);
  expect(layout.series[0].values[2]).toBeUndefined();
  expect(layout.series[0].segments).toEqual([
    [
      {groupIndex: 0, value: 0},
      {groupIndex: 1, value: -5}
    ]
  ]);
  expect(layout.series[1].segments).toEqual([
    [{groupIndex: 0, value: 8}],
    [{groupIndex: 3, value: 2}]
  ]);
  expect(layout.aggregateGroups).toEqual([{bin: bins[4], groupIndex: 4}]);
  expect(layout.domain).toEqual([-5, 20]);
  expect(bins[0].series?.[0].color).toBe('unused-local-native-color');
});

test('series identity remains stable after category reorder and removal, including user palettes', () => {
  const palette = ['#1a2b3c', '#4d5e6f'];
  const complete = makeResidentChartSeriesLayout(bins, palette);
  const filtered = makeResidentChartSeriesLayout([bins[3], bins[0]], palette);
  for (const series of complete.series) {
    expect(filtered.series.find(candidate => candidate.key === series.key)?.color).toBe(
      series.color
    );
    expect(series.color).toBe(getResidentChartSeriesColor(series.key, palette));
  }
  expect(getResidentChartSeriesColor('Beta', [])).toBe(getResidentChartSeriesColor('Beta'));
  // A finite palette may collide, but legend keys and accessible labels remain distinct.
  expect(
    makeResidentChartSeriesLayout(bins, ['red']).series.map(series => [series.key, series.color])
  ).toEqual([
    ['Alpha', 'red'],
    ['Beta', 'red']
  ]);
});

test('line segmentation excludes nonfinite values and preserves an explicitly supplied zero-count zero', () => {
  expect(
    makeResidentChartLineSegments([
      measure('A', 1),
      measure('A', NaN),
      measure('A', 0, 0),
      undefined,
      measure('A', Infinity)
    ])
  ).toEqual([[{groupIndex: 0, value: 1}], [{groupIndex: 2, value: 0}]]);
  expect(
    makeResidentChartSeriesLayout([{...bins[0], series: [measure('Alpha', 0)]}]).domain
  ).toEqual([-1, 1]);
  expect(makeResidentChartSeriesLayout([]).series).toEqual([]);
});

test('SVG renders separate segments and keyboard marks without inventing an Other selection', () => {
  const markup = renderToStaticMarkup(
    createElement(ResidentChartSeriesView, {
      bins,
      type: 'lineChart',
      onSelect: () => {},
      selectedKey: 'first'
    })
  );
  // Alpha connects its two real observations; Beta's distant observations do not bridge missing groups.
  expect((markup.match(/<polyline /g) ?? []).length).toBe(1);
  expect((markup.match(/role="button"/g) ?? []).length).toBe(4);
  expect((markup.match(/aria-pressed="true"/g) ?? []).length).toBe(2);
  expect(markup).toContain('first · Alpha: 0 (1 row)');
  expect(markup).toContain('Combined groups (aggregate only)');
  expect(markup).toContain('role="group"');
  expect(markup).not.toMatch(/role="button"[^>]*aria-label="Other/);
  expect(markup).not.toMatch(/(?:cx|cy|x|y|width|height)="(?:NaN|Infinity|-Infinity)"/);
});

test('both bar orientations keep zero and negative marks, while read-only export removes interactions', () => {
  for (const type of ['barChart', 'horizontalBar'] as const) {
    const markup = renderToStaticMarkup(
      createElement(ResidentChartSeriesView, {bins, type, onSelect: () => {}})
    );
    expect((markup.match(/role="button"/g) ?? []).length).toBe(4);
    expect(markup).toContain('second · Alpha: -5 (1 row)');
    expect(markup).not.toMatch(/(?:width|height)="-/);
    const exported = renderToStaticMarkup(
      createElement(ResidentChartSeriesView, {bins, type, readOnly: true, onSelect: () => {}})
    );
    expect(exported).not.toContain('role="button"');
    expect(exported).not.toContain('tabindex=');
    expect(exported).not.toContain('min-width:');
    expect(exported).toContain('data-read-only="true"');
    expect(exported).toContain('Chart series legend');
  }
});

test('empty or invalid series expose a readable state instead of invalid SVG coordinates', () => {
  const markup = renderToStaticMarkup(
    createElement(ResidentChartSeriesView, {
      bins: [{...bins[0], series: [measure('Alpha', NaN)]}],
      type: 'lineChart'
    })
  );
  expect(markup).toContain('No series values in this population.');
  expect(markup).not.toContain('<svg');
});

test('bounded labels retain distinct weekly dates and reserve room in plots and read-only exports', () => {
  const keys = ['Week of 2024-03-10', 'Week of 2024-03-17', 'abcdefghijklmnopqrstuvwxyz'];
  const labeledBins = keys.map(key => ({...bins[1], key, filterValue: [key]}));
  for (const type of ['barChart', 'horizontalBar', 'lineChart'] as const) {
    for (const readOnly of [false, true]) {
      const markup = renderToStaticMarkup(
        createElement(ResidentChartSeriesView, {
          bins: labeledBins,
          type,
          readOnly,
          onSelect: () => {}
        })
      );
      expect(markup).toContain('</title>Week of 2024-03-10</text>');
      expect(markup).toContain('</title>Week of 2024-03-17</text>');
      expect(markup).toContain('</title>abcdefghijklmnopqrstuvw…</text>');
      expect(markup).toContain('<title>abcdefghijklmnopqrstuvwxyz</title>');
      expect(markup).toContain('aria-label="abcdefghijklmnopqrstuvwxyz · Alpha: -5 (1 row)"');
      expect(markup).not.toMatch(/(?:width|height)="-/);
      const viewBox = markup.match(/class="resident-series-plot" viewBox="0 0 (\d+) (\d+)"/);
      expect(Number(viewBox?.[type === 'horizontalBar' ? 1 : 2])).toBeGreaterThan(
        type === 'horizontalBar' ? 400 : 250
      );
      if (readOnly) expect(markup).not.toContain('min-width:');
    }
  }
});
